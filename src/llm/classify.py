from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from llm.utils import _reader, finals, flatten, four_class


def call_llm(client, msgs: list, llm_cfg: dict, prm: dict) -> str:
    model = llm_cfg["model"]
    retries = int(llm_cfg["max_retries"])
    json_models = set(llm_cfg["json_models"])
    no_temp_models = set(llm_cfg["no_temperature_models"])
    for i in range(1, retries + 1):
        try:
            kw = {"model": model, "messages": msgs}
            if model not in no_temp_models:
                kw["temperature"] = float(prm.get("temperature", 0.0))
            if model in json_models:
                kw["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kw).choices[0].message.content
        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
        ) as e:
            if isinstance(e, openai.RateLimitError) and (
                "insufficient_quota" in str(e) or "billing" in str(e).lower()
            ):
                raise RuntimeError(
                    f"Quota exhausted — add credits at platform.openai.com/settings/organization/billing\n{e}"
                ) from e
            if i == retries:
                raise
            time.sleep(int(llm_cfg["retry_wait"]) * i)


def evaluate_labels(
    llm_out: Path, gt_path: Path, llm_cfg: dict, classes: list[str]
) -> pd.DataFrame:
    llm_df = pd.read_csv(llm_out, keep_default_na=False, na_values=[""])
    gt_cols = llm_cfg["ground_truth_columns"]
    gt = (
        _reader(gt_path)(gt_path)[gt_cols]
        .reset_index()
        .rename(columns={"index": "original_index"})
    )
    merged = llm_df.merge(gt, on="original_index", how="inner")
    if merged.empty:
        raise ValueError(
            "No rows matched between LLM output and ground truth — check original_index alignment."
        )
    class_col = next(
        (c for c in ("Final_Class", "Final_HFACS_Code") if c in merged.columns), None
    )
    if class_col is None:
        raise ValueError(
            f"Missing Final_Class or Final_HFACS_Code in LLM output. Columns: {list(llm_df.columns)}"
        )

    _lookup = {
        alias: cls for cls in classes for alias in (cls.lower(), cls.lower().split()[0])
    }
    _lookup |= {"multiple": _lookup.get("both"), "none": _lookup.get("neither")}

    merged["y_true"] = merged.apply(
        lambda r: four_class(*[r[c] == c for c in gt_cols], classes), axis=1
    )
    merged["y_pred"] = merged[class_col].apply(
        lambda v: _lookup.get(str(v).strip().lower()) if not pd.isna(v) else None
    )
    valid = merged["y_pred"].isin(classes)
    invalid = merged.loc[~valid, ["original_index", class_col]]
    y_true_v = merged.loc[valid, "y_true"].tolist()
    y_pred_v = merged.loc[valid, "y_pred"].tolist()

    cm = confusion_matrix(y_true_v, y_pred_v, labels=classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True: {c}" for c in classes],
        columns=[f"Pred: {c}" for c in classes],
    )

    print("\n=== Four-Class Evaluation (Final_Class vs. Ground Truth) ===")
    print(f"Total rows: {len(merged)}")
    if not invalid.empty:
        print(f"Excluded (blank/invalid Final_Class): {len(invalid)}")
        print(invalid.to_string(index=False))
    print(f"Evaluated rows: {len(y_true_v)}")
    print(f"Accuracy: {accuracy_score(y_true_v, y_pred_v):.4f}")
    print("\n--- Confusion Matrix ---")
    print(cm_df.to_string())
    print("\n--- Classification Report ---")
    print(classification_report(y_true_v, y_pred_v, labels=classes, zero_division=0))
    return cm_df


# ---------------------------------------------------------------------------
# Sync mode
# ---------------------------------------------------------------------------

def _run_sync(
    client, df: pd.DataFrame, tasks: list, llm_cfg: dict, prm: dict,
    narr: str, final_cols: set, compact: bool, prompt_name: str, out: Path,
    inp_path: Path, classes: list[str],
) -> None:
    sys_p = prm.get("system_prompt", prm.get("system", ""))
    step2_t = prm.get("step2_prompt")
    usr_t = prm.get(
        "step1_prompt", prm.get("user_prompt_template", prm.get("prompt_template", ""))
    )

    def build_msgs(content: str) -> list:
        return ([{"role": "system", "content": sys_p}] if sys_p else []) + [
            {"role": "user", "content": content}
        ]

    def process(i: int, n: str) -> tuple:
        if not n or n.lower() in {"nan", "none", ""}:
            return i, {"original_index": i, "skip_reason": "empty_narrative"}
        try:
            step1_out = call_llm(
                client, build_msgs(usr_t.replace("{narrative}", n)), llm_cfg, prm
            )
            raw = (
                call_llm(
                    client,
                    build_msgs(step2_t.replace("{preconditions}", step1_out)),
                    llm_cfg,
                    prm,
                )
                if step2_t
                else step1_out
            )
            x = flatten(json.loads(raw))
        except RuntimeError:
            raise
        except Exception as e:
            x = {"error": str(e)}
        x |= {"original_index": i, narr: n}
        return i, finals(x, narr, final_cols) if compact else x

    results = {}
    with ThreadPoolExecutor(max_workers=int(llm_cfg["workers"])) as executor:
        futures = {executor.submit(process, *t): t[0] for t in tasks}
        for f in tqdm(
            as_completed(futures), total=len(futures), desc=f"[{prompt_name}]"
        ):
            i, row = f.result()
            results[i] = row

    _save_output(results, out)
    if llm_cfg["evaluate"]:
        evaluate_labels(out, inp_path, llm_cfg, classes)


# ---------------------------------------------------------------------------
# Batch submit
# ---------------------------------------------------------------------------

def _run_batch_submit(
    client, tasks: list, llm_cfg: dict, prm: dict,
    narr: str, compact: bool, prompt_name: str, out: Path,
) -> None:
    if prm.get("step2_prompt"):
        raise NotImplementedError(
            "Batch mode does not support two-step prompts. Use mode: sync."
        )

    model = llm_cfg["model"]
    json_models = set(llm_cfg["json_models"])
    no_temp_models = set(llm_cfg["no_temperature_models"])
    sys_p = prm.get("system_prompt", prm.get("system", ""))
    usr_t = prm.get(
        "step1_prompt", prm.get("user_prompt_template", prm.get("prompt_template", ""))
    )

    def build_msgs(content: str) -> list:
        return ([{"role": "system", "content": sys_p}] if sys_p else []) + [
            {"role": "user", "content": content}
        ]

    lines = []
    for i, n in tasks:
        if not n or n.lower() in {"nan", "none", ""}:
            continue
        body: dict = {
            "model": model,
            "messages": build_msgs(usr_t.replace("{narrative}", n)),
        }
        if model not in no_temp_models:
            body["temperature"] = float(prm.get("temperature", 0.0))
        if model in json_models:
            body["response_format"] = {"type": "json_object"}
        lines.append(
            json.dumps({"custom_id": f"row-{i}", "method": "POST",
                        "url": "/v1/chat/completions", "body": body})
        )

    jsonl_path = out.parent / f"{out.stem}_batch_input.jsonl"
    jsonl_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[batch] Written {len(lines)} requests to {jsonl_path}")

    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"[batch] Uploaded input file: {uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    batch_id_path = out.parent / f"{out.stem}_batch_id.txt"
    batch_id_path.write_text(batch.id, encoding="utf-8")

    print(f"\n[batch] Submitted. Batch ID: {batch.id}")
    print(f"[batch] Status: {batch.status}")
    print(f"[batch] Batch ID saved to: {batch_id_path}")
    print(f"[batch] Set llm.batch_id: {batch.id} in config.yaml, then run with mode: batch_retrieve")


# ---------------------------------------------------------------------------
# Batch retrieve
# ---------------------------------------------------------------------------

def _run_batch_retrieve(
    client, df: pd.DataFrame, llm_cfg: dict, narr: str,
    final_cols: set, compact: bool, out: Path,
    inp_path: Path, classes: list[str],
) -> None:
    batch_id = llm_cfg.get("batch_id") or None
    if not batch_id:
        batch_id_path = out.parent / f"{out.stem}_batch_id.txt"
        if not batch_id_path.exists():
            raise RuntimeError(
                f"No batch_id found. Set llm.batch_id in config.yaml or ensure "
                f"{batch_id_path} exists from a prior batch_submit run."
            )
        batch_id = batch_id_path.read_text(encoding="utf-8").strip()

    batch = client.batches.retrieve(batch_id)
    print(f"[batch] ID: {batch_id}  status: {batch.status}")

    if batch.status in ("validating", "in_progress", "finalizing"):
        completed = batch.request_counts.completed if batch.request_counts else "?"
        total = batch.request_counts.total if batch.request_counts else "?"
        print(f"[batch] Progress: {completed}/{total} — check back later.")
        return

    if batch.status != "completed":
        raise RuntimeError(
            f"Batch ended with status '{batch.status}'. "
            f"Error file ID: {batch.error_file_id}"
        )

    raw_bytes = client.files.content(batch.output_file_id).content
    results: dict[int, dict] = {}
    for line in raw_bytes.decode("utf-8").splitlines():
        item = json.loads(line)
        i = int(item["custom_id"].removeprefix("row-"))
        if item.get("error"):
            results[i] = {"original_index": i, "error": str(item["error"])}
            continue
        content = item["response"]["body"]["choices"][0]["message"]["content"]
        n = str(df.loc[i, narr]).strip() if i in df.index else ""
        try:
            x = flatten(json.loads(content))
        except Exception as e:
            x = {"error": str(e)}
        x |= {"original_index": i, narr: n}
        results[i] = finals(x, narr, final_cols) if compact else x

    _save_output(results, out)
    if llm_cfg["evaluate"]:
        evaluate_labels(out, inp_path, llm_cfg, classes)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _save_output(results: dict, out: Path) -> None:
    rows = [v for _, v in sorted(results.items())]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved {len(rows)} rows to: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config: dict) -> None:
    llm_cfg = config["llm"]
    classes = llm_cfg["classes"]
    prompt_name = llm_cfg["prompt"]
    compact = llm_cfg["compact"]

    inp_path = Path(config["paths"]["ghfacs_data_dir"]) / llm_cfg["input"]
    out_filename = (
        f"{inp_path.stem}_LLM_Output_{prompt_name}{'_compact' if compact else ''}.csv"
    )
    out = Path(config["paths"]["llm_output_dir"]) / out_filename
    out.parent.mkdir(parents=True, exist_ok=True)

    mode = llm_cfg["mode"]

    if mode == "evaluate":
        evaluate_labels(out, inp_path, llm_cfg, classes)
        return

    key = (llm_cfg["api_key"] or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key:
        raise RuntimeError(
            "Missing API key. Set llm.api_key in config or OPENAI_API_KEY env var."
        )
    client = openai.OpenAI(api_key=key)

    if mode == "batch_retrieve":
        narr = llm_cfg["narrative_column"]
        df = _reader(inp_path)(inp_path)
        _run_batch_retrieve(
            client, df, llm_cfg, narr,
            set(llm_cfg["output_columns"]), compact, out, inp_path, classes,
        )
        return

    prompts_dir = Path(config["paths"]["prompts_dir"])
    prm = yaml.safe_load(
        (prompts_dir / f"{prompt_name}.yaml").read_text(encoding="utf-8")
    )
    narr = llm_cfg["narrative_column"]
    df = _reader(inp_path)(inp_path)
    if narr not in df.columns:
        raise ValueError(f"Missing narrative column '{narr}' in {inp_path}.")

    _lim = llm_cfg["limit"]
    subset = df.head(int(_lim)) if _lim not in (None, "", "none", "null") else df
    tasks = [(i, str(r[narr]).strip()) for i, r in subset.iterrows()]
    final_cols = set(llm_cfg["output_columns"])

    if mode == "batch_submit":
        _run_batch_submit(client, tasks, llm_cfg, prm, narr, compact, prompt_name, out)
    else:
        _run_sync(
            client, df, tasks, llm_cfg, prm, narr,
            final_cols, compact, prompt_name, out, inp_path, classes,
        )
