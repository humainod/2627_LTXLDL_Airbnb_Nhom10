import pandas as pd
import yaml
import argparse
from pathlib import Path

def run_evaluation(config_path: str, snapshot: str):
    base_dir = Path(__file__).resolve().parent.parent
    
    # 1. Đường dẫn file theo đúng luồng
    llm_file = base_dir / "data" / "processed" / f"reviews_llm_{snapshot}.csv"
    base_file = base_dir / "data" / "processed" / f"reviews_baseline_{snapshot}.csv"
    out_file = base_dir / "reports" / f"eval_report_{snapshot}.csv"
    
    if not llm_file.exists() or not base_file.exists():
        print(f"Bỏ qua {snapshot} do thiếu file đầu vào.")
        return

    # 2. Đọc và chuẩn hóa ID
    df_llm = pd.read_csv(llm_file)
    df_base = pd.read_csv(base_file)
    
    if "id" in df_base.columns and "review_id" not in df_base.columns:
        df_base = df_base.rename(columns={"id": "review_id"})

    # 3. Gộp dữ liệu (Inner Join)
    merged = pd.merge(df_llm, df_base, on="review_id", how="inner")
    
    if merged.empty:
        print(f"Không có review chung để so sánh trong mốc {snapshot}.")
        return

    # 4. Đánh giá cơ bản (Sentiment Match)
    merged["sentiment_match"] = (
        merged["llm_sentiment"].astype(str).str.strip().str.lower() == 
        merged["baseline_sentiment"].astype(str).str.strip().str.lower()
    )
    
    accuracy = merged["sentiment_match"].mean() * 100
    print(f"Mốc {snapshot} | Số mẫu: {len(merged)} | Tỷ lệ khớp Sentiment: {accuracy:.2f}%")

    # 5. Lưu báo cáo
    out_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_file, index=False, encoding="utf-8")


LABELS = ("positive", "negative", "mixed")


def read_sentiments(path, column, *, allow_legacy_id=False, allow_missing=False):
    table = pd.read_csv(path, dtype="string", encoding="utf-8-sig", keep_default_na=False)
    if allow_legacy_id and "review_id" not in table and "id" in table:
        table = table.rename(columns={"id": "review_id"})
    missing_columns = {"review_id", column} - set(table.columns)
    if missing_columns:
        raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing_columns))}")

    table = table[["review_id", column]].copy()
    table["review_id"] = table["review_id"].str.strip()
    table[column] = table[column].str.strip().str.lower()
    if table["review_id"].eq("").any() or table["review_id"].duplicated().any():
        raise ValueError(f"{path}: review_id must be nonempty and unique")
    valid_labels = (*LABELS, "") if allow_missing else LABELS
    invalid = table.loc[~table[column].isin(valid_labels), column].unique()
    if len(invalid):
        raise ValueError(f"{path}: invalid {column} labels: {', '.join(map(repr, invalid))}")
    return table


def evaluate_sample(project_root, snapshot, labels_path):
    processed = project_root / "data" / "processed"
    predictions = {
        "llm": processed / f"reviews_llm_{snapshot}.csv",
        "baseline": processed / f"reviews_baseline_{snapshot}.csv",
    }
    missing_files = [str(p) for p in (labels_path, *predictions.values()) if not p.is_file()]
    if missing_files:
        raise ValueError("Missing input files:\n" + "\n".join(missing_files))

    report = read_sentiments(labels_path, "sentiment").rename(
        columns={"sentiment": "manual_sentiment"}
    )
    total = len(report)
    if not total:
        raise ValueError(f"{labels_path}: no labeled reviews")

    for model, path in predictions.items():
        column = f"{model}_sentiment"
        table = read_sentiments(path, column, allow_legacy_id=model == "baseline", allow_missing=True)
        report = report.merge(table, on="review_id", how="left", validate="one_to_one")
        present = report[column].notna() & report[column].ne("").fillna(False)
        correct = report[column].eq(report["manual_sentiment"]).where(present)
        report[f"{model}_correct"] = correct
        n_correct = int(correct.sum())
        n_present = int(present.sum())
        accuracy = f"{n_correct / n_present:.2%}" if n_present else "n/a"
        print(f"{model}: {n_correct}/{n_present} correct among predictions (accuracy: {accuracy})")
        print(f"  Coverage: {n_present}/{total}; correct over full sample: {n_correct}/{total}")
        for label in LABELS:
            group = report["manual_sentiment"].eq(label)
            if group.any():
                print(f"  {label}: {int(correct[group].sum())}/{int(group.sum())} correct")
        if n_present != total:
            print("  Missing review IDs: " + ", ".join(report.loc[~present, "review_id"]))

    output = project_root / "reports" / f"eval_sample_{snapshot}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Report: {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="CSV with review_id and manual sentiment")
    parser.add_argument("--snapshot", help="Snapshot to evaluate, e.g. 2025-09-22")
    args = parser.parse_args()
    if args.labels and not args.snapshot:
        parser.error("--labels requires --snapshot")

    with args.config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    snapshots = [str(snapshot) for snapshot in config.get("snapshots", [])]
    if args.snapshot and args.snapshot not in snapshots:
        parser.error(f"snapshot not listed in config: {args.snapshot}")

    if args.labels:
        try:
            evaluate_sample(Path(__file__).resolve().parent.parent, args.snapshot, args.labels)
        except (ValueError, OSError) as exc:
            parser.exit(1, f"{exc}\n")
    else:
        for snapshot in ([args.snapshot] if args.snapshot else snapshots):
            run_evaluation(args.config, snapshot)


if __name__ == "__main__":
    main()
