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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from llm.utils import resolve_openai_api_key
from models.random_forest import make_four_class_target_from_config

_EXCLUDED = {"Error", "Violation"}


def _get_items(config: dict) -> list[str]:
    return [
        item
        for cat, items in config["hfacs_categories"].items()
        if cat not in _EXCLUDED
        for item in items
    ]


def _category_map(config: dict) -> dict[str, list[str]]:
    return {
        cat: items
        for cat, items in config["hfacs_categories"].items()
        if cat not in _EXCLUDED
    }


def _retry_delay(error: Exception, attempt: int, llm_cfg: dict) -> float:
    retry_wait = float(llm_cfg["retry_wait"])
    match = re.search(r"try again in ([0-9.]+)s", str(error), flags=re.IGNORECASE)
    if match:
        return float(match.group(1)) + retry_wait + random.uniform(0.25, 1.25)
    return retry_wait * attempt + random.uniform(0.25, 1.25)


def _call_llm(client, msgs: list, llm_cfg: dict) -> str:
    model = llm_cfg["model"]
    retries = int(llm_cfg["max_retries"])
    json_models = set(llm_cfg["json_models"])
    no_temp_models = set(llm_cfg["no_temperature_models"])
    for attempt in range(1, retries + 1):
        try:
            kw = {"model": model, "messages": msgs}
            if model not in no_temp_models:
                kw["temperature"] = 0.0
            if model in json_models:
                kw["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kw).choices[0].message.content
        except openai.AuthenticationError as e:
            raise RuntimeError("OpenAI authentication failed.") from e
        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
        ) as e:
            if isinstance(e, openai.RateLimitError) and (
                "insufficient_quota" in str(e) or "billing" in str(e).lower()
            ):
                raise RuntimeError(f"Quota exhausted\n{e}") from e
            if attempt == retries:
                raise
            time.sleep(_retry_delay(e, attempt, llm_cfg))


def _evaluate_precondition_extraction(
    out_df: pd.DataFrame, subset: pd.DataFrame, config: dict
) -> None:
    from sklearn.metrics import classification_report

    items = _get_items(config)
    cat_map = _category_map(config)

    # Per-item report
    print("\n=== Per-Item Extraction Evaluation (LLM vs Ground Truth) ===")
    item_accs: dict[str, float] = {}
    item_f1s: dict[str, float] = {}
    for item in items:
        llm_col = f"llm_{item}"
        if item not in subset.columns or llm_col not in out_df.columns:
            continue
        y_pred = out_df[llm_col].fillna(0).astype(int)
        y_true = (
            subset[item].apply(pd.to_numeric, errors="coerce").fillna(0) > 0
        ).astype(int)
        item_accs[item] = accuracy_score(y_true, y_pred)
        item_f1s[item] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(
            f"\n{item}  (accuracy={item_accs[item]:.4f}  macro-f1={item_f1s[item]:.4f})"
        )
        print(classification_report(y_true, y_pred, labels=[0, 1], zero_division=0))

    # Category-level summary
    print("\n=== Category-Level Summary ===")
    header = f"{'Category':<40} {'Items':>5}  {'Avg Acc':>8}  {'Avg F1':>8}"
    print(header)
    print("-" * len(header))
    all_accs, all_f1s = [], []
    for cat, cat_items in cat_map.items():
        accs = [item_accs[i] for i in cat_items if i in item_accs]
        f1s = [item_f1s[i] for i in cat_items if i in item_f1s]
        if not accs:
            continue
        avg_acc = sum(accs) / len(accs)
        avg_f1 = sum(f1s) / len(f1s)
        all_accs.extend(accs)
        all_f1s.extend(f1s)
        print(f"{cat:<40} {len(accs):>5}  {avg_acc:>8.4f}  {avg_f1:>8.4f}")
    if all_accs:
        print("-" * len(header))
        print(
            f"{'Overall':<40} {len(all_accs):>5}  {sum(all_accs)/len(all_accs):>8.4f}  {sum(all_f1s)/len(all_f1s):>8.4f}"
        )


def run_extract_preconditions(config: dict) -> None:
    """Extract individual HFACS precondition columns from narratives via LLM.

    Factor descriptions are read from ASRS_extract_preconditions.yaml and injected
    into the prompt so the LLM knows what to look for in the narrative text.
    All subcategory items from hfacs_categories (excluding Error/Violation) are
    extracted as binary llm_{item} columns. Evaluation compares each against the
    existing ground truth column and prints per-item and category-level summaries.
    """
    llm_cfg = config["llm"]
    inp_path = Path(config["paths"]["ghfacs_data_dir"]) / llm_cfg["input"]
    out_path = Path(config["paths"]["llm_output_dir"]) / (
        inp_path.stem + "_preconditions.csv"
    )

    df = pd.read_csv(inp_path, low_memory=False)
    narr_col = llm_cfg.get("narrative_column", "Report 1_Narrative")
    if narr_col not in df.columns:
        raise ValueError(f"Missing column '{narr_col}' in {inp_path}")

    prompts_dir = Path(config["paths"]["prompts_dir"])
    prm = yaml.safe_load(
        (prompts_dir / "ASRS_extract_preconditions.yaml").read_text(encoding="utf-8")
    )
    sys_p = prm["system_prompt"].strip()
    usr_t = prm["user_prompt_template"]
    factor_defs: dict[str, str] = prm.get("factors", {})

    items = _get_items(config)
    # Build factors block: use description from YAML if available, else just the name
    factors_block = "\n".join(
        f"  - {item}: {factor_defs[item].strip()}"
        if item in factor_defs
        else f"  - {item}"
        for item in items
    )
    template_json = json.dumps({item: 0 for item in items}, indent=2)

    print(
        f"[INFO] Extracting {len(items)} individual HFACS precondition columns via LLM"
    )

    key = resolve_openai_api_key(llm_cfg)
    client = openai.OpenAI(api_key=key)

    def process(i: int, narrative: str) -> tuple:
        if not narrative or narrative.lower() in {"nan", "none", ""}:
            return i, {item: 0 for item in items}
        prompt = (
            usr_t.replace("{factors}", factors_block)
            .replace("{template}", template_json)
            .replace("{narrative}", narrative)
        )
        msgs = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = _call_llm(client, msgs, llm_cfg)
            data = json.loads(raw)
            return i, {item: int(bool(data.get(item, 0))) for item in items}
        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
        ):
            raise
        except Exception as e:
            print(f"\n[WARN] Row {i} failed ({type(e).__name__}): {e}")
            return i, {item: 0 for item in items}

    lim = llm_cfg.get("limit")
    subset = df if lim in (None, "", "none", "null") else df.head(int(lim))

    eval_indices = None
    eval_split = config.get("dataset", {}).get("eval_split")
    if eval_split is not None:
        y = make_four_class_target_from_config(subset, config)
        _, eval_df = train_test_split(
            subset,
            test_size=float(eval_split),
            stratify=y,
            random_state=config["models"]["random_state"],
        )
        eval_indices = set(eval_df.index.tolist())
        print(
            f"[dataset] Evaluating on {len(eval_indices)} rows "
            f"({float(eval_split):.0%} of {len(subset)})"
        )

    tasks = [(i, str(row[narr_col]).strip()) for i, row in subset.iterrows()]

    results: dict = {}
    with ThreadPoolExecutor(max_workers=int(llm_cfg["workers"])) as executor:
        futures = {executor.submit(process, *t): t[0] for t in tasks}
        for f in tqdm(
            as_completed(futures), total=len(futures), desc="[extract_preconditions]"
        ):
            i, row_data = f.result()
            results[i] = row_data

    out_df = subset.copy()
    for item in items:
        out_df[f"llm_{item}"] = out_df.index.map(
            lambda idx, c=item: results.get(idx, {}).get(c, 0)
        )

    if eval_indices is not None:
        eval_mask = out_df.index.isin(eval_indices)
        _evaluate_precondition_extraction(out_df[eval_mask], subset[eval_mask], config)
    else:
        _evaluate_precondition_extraction(out_df, subset, config)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(
        f"\n[INFO] Saved {len(out_df)} rows × {len(out_df.columns)} columns to: {out_path}"
    )
    print(f"[INFO] LLM feature columns: {[f'llm_{c}' for c in items]}")
