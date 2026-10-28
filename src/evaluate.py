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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn file yaml")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    print("--- BẮT ĐẦU CHẠY ĐÁNH GIÁ (EVALUATE) ---")
    for snap in config.get("snapshots", []):
        run_evaluation(args.config, snap)