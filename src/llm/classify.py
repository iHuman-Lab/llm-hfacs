from __future__ import annotations

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from llm.utils import _reader, finals, flatten, resolve_openai_api_key
from models.random_forest import make_four_class_target_from_config


def _retry_delay(error: Exception, attempt: int, llm_cfg: dict) -> float:
    """Prefer OpenAI's suggested wait, then add a small stagger for parallel calls."""
    retry_wait = float(llm_cfg["retry_wait"])
    match = re.search(r"try again in ([0-9.]+)s", str(error), flags=re.IGNORECASE)
    if match:
        return float(match.group(1)) + retry_wait + random.uniform(0.25, 1.25)
    return retry_wait * attempt + random.uniform(0.25, 1.25)


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
        except openai.AuthenticationError as e:
            raise RuntimeError(
                "OpenAI authentication failed. Check the API key available to this "
                "Python process; the current run is not using a valid key."
            ) from e
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
            time.sleep(_retry_delay(e, i, llm_cfg))


def _render_prompt(template: str, narrative: str) -> str:
    return template.replace("{narrative}", narrative)


def evaluate_llm_predictions(
    llm_out: Path,
    gt_path: Path,
    classes: list[str],
    config: dict,
    eval_indices: set | None = None,
) -> pd.DataFrame:
    llm_df = pd.read_csv(llm_out, keep_default_na=False, na_values=[""])
    gt_df = _reader(gt_path)(gt_path)

    gt = gt_df.reset_index().rename(columns={"index": "original_index"})
    merged = llm_df.merge(gt, on="original_index", how="inner")
    if eval_indices is not None:
        merged = merged[merged["original_index"].isin(eval_indices)].reset_index(
            drop=True
        )
    if merged.empty:
        raise ValueError(
            "No rows matched between LLM output and ground truth — check original_index alignment."
        )
    label_names = {0: "Neither", 1: "Error", 2: "Violation", 3: "Both"}
    merged["y_true"] = make_four_class_target_from_config(merged, config).map(
        label_names
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
    merged["y_pred"] = merged[class_col].apply(
        lambda v: _lookup.get(str(v).strip().lower()) if not pd.isna(v) else None
    )
    valid = merged["y_pred"].isin(classes)
    invalid_cols = ["original_index", class_col]
    if "error" in merged.columns:
        invalid_cols.append("error")
    invalid = merged.loc[~valid, invalid_cols]
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
    client,
    df: pd.DataFrame,
    tasks: list,
    llm_cfg: dict,
    prm: dict,
    narr: str,
    final_cols: set,
    compact: bool,
    prompt_name: str,
    out: Path,
    inp_path: Path,
    classes: list[str],
    config: dict | None = None,
    eval_indices: set | None = None,
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
                client,
                build_msgs(_render_prompt(usr_t, n)),
                llm_cfg,
                prm,
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
        evaluate_llm_predictions(out, inp_path, classes, config, eval_indices)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def _submit_chunk(client, jsonl_path: Path) -> str:
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id


def _parse_batch_results(
    raw_bytes: bytes, df: pd.DataFrame, narr: str, final_cols: set, compact: bool
) -> dict[int, dict]:
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
    return results


# ---------------------------------------------------------------------------
# Batch submit
# ---------------------------------------------------------------------------


def _run_batch_submit(
    client,
    tasks: list,
    llm_cfg: dict,
    prm: dict,
    narr: str,
    compact: bool,
    prompt_name: str,
    out: Path,
) -> None:
    if prm.get("step2_prompt"):
        raise NotImplementedError(
            "Batch mode does not support two-step prompts. Use mode: sync."
        )

    state_path = out.parent / f"{out.stem}_batch_state.json"

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        chunks = state["chunks"]
        next_idx = next(
            (i for i, c in enumerate(chunks) if c["batch_id"] is None), None
        )
        if next_idx is None:
            print(
                "[batch] All chunks already submitted. Run with mode: batch_retrieve."
            )
            return
        batch_id = _submit_chunk(client, Path(chunks[next_idx]["input_file"]))
        state["chunks"][next_idx]["batch_id"] = batch_id
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(
            f"\n[batch] Submitted chunk {next_idx + 1}/{len(chunks)}. Batch ID: {batch_id}"
        )
        print("[batch] Run with mode: batch_retrieve to check status.")
        return

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
            "messages": build_msgs(_render_prompt(usr_t, n)),
        }
        if model not in no_temp_models:
            body["temperature"] = float(prm.get("temperature", 0.0))
        if model in json_models:
            body["response_format"] = {"type": "json_object"}
        lines.append(
            json.dumps(
                {
                    "custom_id": f"row-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
        )

    chunk_size = int(llm_cfg.get("batch_chunk_size") or len(lines))
    raw_chunks = [lines[j : j + chunk_size] for j in range(0, len(lines), chunk_size)]
    total = len(raw_chunks)
    print(f"[batch] {len(lines)} requests → {total} chunk(s) of ≤{chunk_size}")

    state: dict = {"chunks": []}
    for idx, chunk_lines in enumerate(raw_chunks):
        jsonl_path = out.parent / f"{out.stem}_chunk_{idx:03d}_batch_input.jsonl"
        jsonl_path.write_text("\n".join(chunk_lines), encoding="utf-8")
        state["chunks"].append(
            {
                "input_file": str(jsonl_path),
                "batch_id": None,
                "done": False,
                "result_file": None,
            }
        )

    batch_id = _submit_chunk(client, Path(state["chunks"][0]["input_file"]))
    state["chunks"][0]["batch_id"] = batch_id
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print(f"\n[batch] Submitted chunk 1/{total}. Batch ID: {batch_id}")
    print(f"[batch] State saved to: {state_path}")
    print("[batch] Run with mode: batch_retrieve to check status and advance.")


# ---------------------------------------------------------------------------
# Batch retrieve
# ---------------------------------------------------------------------------


def _run_batch_retrieve(
    client,
    df: pd.DataFrame,
    llm_cfg: dict,
    narr: str,
    final_cols: set,
    compact: bool,
    out: Path,
    inp_path: Path,
    classes: list[str],
    config: dict | None = None,
    eval_indices: set | None = None,
) -> None:
    state_path = out.parent / f"{out.stem}_batch_state.json"

    if not state_path.exists():
        # Legacy single-batch path
        batch_id = llm_cfg.get("batch_id") or None
        if not batch_id:
            batch_id_path = out.parent / f"{out.stem}_batch_id.txt"
            if not batch_id_path.exists():
                raise RuntimeError(
                    "No batch state found. Run with mode: batch_submit first, "
                    "or set llm.batch_id in config.yaml."
                )
            batch_id = batch_id_path.read_text(encoding="utf-8").strip()
        batch = client.batches.retrieve(batch_id)
        print(f"[batch] ID: {batch_id}  status: {batch.status}")
        if batch.status in ("validating", "in_progress", "finalizing"):
            completed = batch.request_counts.completed if batch.request_counts else "?"
            total_req = batch.request_counts.total if batch.request_counts else "?"
            print(f"[batch] Progress: {completed}/{total_req} — check back later.")
            return
        if batch.status != "completed":
            raise RuntimeError(
                f"Batch ended with status '{batch.status}'. "
                f"Error file ID: {batch.error_file_id}"
            )
        results = _parse_batch_results(
            client.files.content(batch.output_file_id).content,
            df,
            narr,
            final_cols,
            compact,
        )
        _save_output(results, out)
        if llm_cfg["evaluate"]:
            evaluate_llm_predictions(out, inp_path, classes, config, eval_indices)
        return

    state = json.loads(state_path.read_text(encoding="utf-8"))
    chunks = state["chunks"]
    total = len(chunks)

    active_idx = next(
        (i for i, c in enumerate(chunks) if c["batch_id"] and not c["done"]),
        None,
    )

    if active_idx is None:
        if all(c["done"] for c in chunks):
            print("[batch] All chunks already complete and merged.")
        else:
            pending = sum(1 for c in chunks if not c["batch_id"])
            print(
                f"[batch] {pending} chunk(s) not yet submitted. Run with mode: batch_submit."
            )
        return

    batch_id = chunks[active_idx]["batch_id"]
    batch = client.batches.retrieve(batch_id)
    print(
        f"[batch] Chunk {active_idx + 1}/{total}  ID: {batch_id}  status: {batch.status}"
    )

    if batch.status in ("validating", "in_progress", "finalizing"):
        completed = batch.request_counts.completed if batch.request_counts else "?"
        total_req = batch.request_counts.total if batch.request_counts else "?"
        print(f"[batch] Progress: {completed}/{total_req} — check back later.")
        return

    if batch.status != "completed":
        raise RuntimeError(
            f"Chunk {active_idx + 1}/{total} ended with status '{batch.status}'. "
            f"Error file ID: {batch.error_file_id}"
        )

    result_file = out.parent / f"{out.stem}_chunk_{active_idx:03d}_result.jsonl"
    result_file.write_bytes(client.files.content(batch.output_file_id).content)
    state["chunks"][active_idx]["done"] = True
    state["chunks"][active_idx]["result_file"] = str(result_file)
    print(f"[batch] Chunk {active_idx + 1}/{total} complete. Results: {result_file}")

    next_idx = next((i for i, c in enumerate(chunks) if not c["batch_id"]), None)
    if next_idx is not None:
        next_batch_id = _submit_chunk(client, Path(chunks[next_idx]["input_file"]))
        state["chunks"][next_idx]["batch_id"] = next_batch_id
        print(
            f"[batch] Auto-submitted chunk {next_idx + 1}/{total}. Batch ID: {next_batch_id}"
        )
        print("[batch] Run with mode: batch_retrieve again to check status.")

    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    if all(c["done"] for c in state["chunks"]):
        print(f"\n[batch] All {total} chunks complete. Merging results...")
        results: dict[int, dict] = {}
        for chunk in state["chunks"]:
            results |= _parse_batch_results(
                Path(chunk["result_file"]).read_bytes(), df, narr, final_cols, compact
            )
        _save_output(results, out)
        if llm_cfg["evaluate"]:
            evaluate_llm_predictions(out, inp_path, classes, config, eval_indices)


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


def run_llm_classification(config: dict) -> None:
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

    # Compute eval indices — same 50-per-class balanced split as Qwen LoRA experiments
    lora_cfg = config.get("lora_qwen3", {})
    train_per_class = int(lora_cfg.get("train_per_class", 125))
    test_per_class = int(lora_cfg.get("test_per_class", 50))
    random_state = config["models"]["random_state"]

    _df_full = _reader(inp_path)(inp_path)
    eval_indices = None
    if "category" in _df_full.columns:
        test_idxs = []
        for _, group in _df_full.groupby("category"):
            sample = group.sample(
                n=train_per_class + test_per_class, random_state=random_state
            )
            test_idxs.extend(sample.index[train_per_class:].tolist())
        eval_indices = set(test_idxs)
        print(
            f"[dataset] Evaluating on {len(eval_indices)} rows ({test_per_class} per class)"
        )

    if mode == "evaluate":
        eval_prompts = llm_cfg.get("eval_prompts", [prompt_name])
        for p in eval_prompts:
            out_p = (
                Path(config["paths"]["llm_output_dir"])
                / f"{inp_path.stem}_LLM_Output_{p}.csv"
            )
            if not out_p.exists():
                print(f"\n[{p}] Output file not found — skipping: {out_p}")
                continue
            print(f"\n{'='*60}\nEvaluating: {p}\n{'='*60}")
            evaluate_llm_predictions(out_p, inp_path, classes, config, eval_indices)
        return

    key = resolve_openai_api_key(llm_cfg)
    client = openai.OpenAI(api_key=key)

    if mode == "batch_retrieve":
        narr = llm_cfg["narrative_column"]
        df = _reader(inp_path)(inp_path)
        _run_batch_retrieve(
            client,
            df,
            llm_cfg,
            narr,
            set(llm_cfg["output_columns"]),
            compact,
            out,
            inp_path,
            classes,
            config,
            eval_indices,
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

    if eval_indices is not None:
        subset = df.loc[sorted(eval_indices)]
    else:
        _lim = llm_cfg["limit"]
        subset = df.head(int(_lim)) if _lim not in (None, "", "none", "null") else df

    tasks = [(i, str(r[narr]).strip()) for i, r in subset.iterrows()]
    final_cols = set(llm_cfg["output_columns"])

    if mode == "batch_submit":
        _run_batch_submit(client, tasks, llm_cfg, prm, narr, compact, prompt_name, out)
    else:
        _run_sync(
            client,
            df,
            tasks,
            llm_cfg,
            prm,
            narr,
            final_cols,
            compact,
            prompt_name,
            out,
            inp_path,
            classes,
            config,
            eval_indices,
        )
