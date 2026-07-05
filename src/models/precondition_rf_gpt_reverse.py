from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from features.utils import (
    ensure_dir,
    get_hfacs_feature_cols,
    load_dataset,
    print_eval_metrics,
    save_predictions,
)


def precondition_rf_gpt_reverse(config):
    """RF trained on GPT-4o mini llm_* features (500 lora-train rows), tested on ground truth features (200 lora-test rows)."""
    out_dir = config["paths"]["rf_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["rf"]
    random_state = config["models"]["random_state"]

    # split_source_df only supplies the train/test split assignment; gpt_df has llm_* for all 700 rows
    split_source_df = load_dataset(config["paths"]["rf_lora_preconditions_csv"])
    gpt_df = load_dataset(config["paths"]["rf_gpt_preconditions_csv"])

    hfacs_cols = get_hfacs_feature_cols(config)
    bare_cols = [c for c in hfacs_cols if c in split_source_df.columns]
    llm_cols = ["llm_" + c for c in bare_cols if "llm_" + c in gpt_df.columns]
    aligned_bare = [c for c in bare_cols if "llm_" + c in gpt_df.columns]

    if not bare_cols:
        raise ValueError("No ground truth HFACS columns found.")
    if not llm_cols:
        raise ValueError("No llm_* columns found in GPT preconditions CSV.")
    if "lora_split" not in split_source_df.columns:
        raise ValueError("lora_split column not found in split source CSV.")

    # apply the same train/test split used for the LoRA experiments to gpt_df
    split_map = split_source_df.set_index("_ACN")["lora_split"]
    gpt_df = gpt_df.copy()
    gpt_df["lora_split"] = gpt_df["_ACN"].map(split_map)

    train_df = gpt_df[gpt_df["lora_split"] == "train"]
    test_df = gpt_df[gpt_df["lora_split"] == "test"]

    X_train = train_df[llm_cols].astype(float)
    y_train = train_df["category"]

    X_test = test_df[aligned_bare].astype(float)
    X_test.columns = llm_cols  # align feature names with training
    y_test = test_df["category"]

    print(
        f"\n[RF-GPT-Reverse] Train: {len(X_train)} rows (GPT-4o mini llm_*) | Test: {len(X_test)} rows (ground truth)"
    )
    print(f"Features: {len(llm_cols)} precondition columns")

    print("\nClass distribution in train:")
    for cls, count in y_train.value_counts().sort_index().items():
        print(f"  {cls}: {count}")
    print("\nClass distribution in test:")
    for cls, count in y_test.value_counts().sort_index().items():
        print(f"  {cls}: {count}")

    param_dist = {
        "n_estimators": cfg["n_estimators"],
        "max_depth": cfg["max_depth"],
        "min_samples_split": cfg["min_samples_split"],
        "min_samples_leaf": cfg["min_samples_leaf"],
        "max_features": cfg["max_features"],
    }

    n_splits = min(cfg["cv_splits"], int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_dist,
        n_iter=cfg["n_iter"],
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
    )
    grid.fit(X_train, y_train)

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV f1_macro: {grid.best_score_:.4f}")

    y_pred = grid.predict(X_test)
    print_eval_metrics(y_test, y_pred)
    save_predictions(
        y_test, y_pred, out_dir, "rf_gpt_reverse_preconditions_predictions.csv"
    )


if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    precondition_rf_gpt_reverse(config)
