"""
kpi.py - Tính toán 6-8 chỉ số KPI chuẩn Airbnb sau bước QA và tích hợp kết quả LLM.

Tiêu chuẩn các KPI được tính toán:
  1. Tỷ lệ lấp đầy (Occupancy Rate) - Đo hiệu suất sử dụng phòng (%)
  2. Giá thuê trung bình ngày (ADR) - Đo chiến lược giá (USD $)
  3. RevPAR (Doanh thu trên phòng khả dụng) - Đo hiệu quả kinh doanh tổng thể (USD $)
  4. Tổng doanh thu ước tính (Estimated Revenue) - Đo quy mô tài chính (USD $)
  5. Doanh thu trung bình theo khu vực (Neighborhood Revenue) - Phân tích theo neighbourhood_cleansed (USD $)
  6. Tỷ lệ đánh giá tích cực (Positive Sentiment Rate) - Tích hợp kết quả LLM (%)

Cách chạy:
    python src/kpi.py --config configs/florence.yaml
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

def calculate_snapshot_kpis(config: dict, snapshot: str):
    city = config.get("city", "unknown")
    clean_dir = Path("data/processed/clean") / snapshot
    processed_dir = Path("data/processed")
    
    listings_path = clean_dir / "listings.csv"
    calendar_path = clean_dir / "calendar.csv"
    llm_review_path = processed_dir / f"reviews_llm_{snapshot}.csv"
    
    if not listings_path.exists() or not calendar_path.exists():
        print(f"⚠️ Cảnh báo: Không tìm thấy dữ liệu clean tại {clean_dir}. Bỏ qua snapshot {snapshot}.")
        return

    print(f"\n📊 --- ĐANG TÍNH TOÁN 6 KPI CHO SNAPSHOT: {snapshot} (Thành phố: {city}) ---")
    
    # 1. Đọc dữ liệu listings và calendar đã qua QA
    df_listings = pd.read_csv(listings_path)
    df_calendar = pd.read_csv(calendar_path)
    
    id_col = 'id' if 'id' in df_listings.columns else 'listing_id'
    
    # 2. Xử lý trạng thái đặt phòng trên calendar
    if 'available' in df_calendar.columns:
        if df_calendar['available'].dtype == object:
            df_calendar['is_booked'] = df_calendar['available'].isin(['f', 'F', False, 'False', 'FALSE'])
        else:
            df_calendar['is_booked'] = ~df_calendar['available']
    else:
        df_calendar['is_booked'] = False

    # 3. Xử lý cột giá trên calendar (đã qua QA hoặc lấy bổ sung từ listings)
    if 'price' in df_calendar.columns:
        df_calendar['price_clean'] = pd.to_numeric(df_calendar['price'], errors='coerce')
    else:
        df_calendar['price_clean'] = np.nan

    if df_calendar['price_clean'].isna().all() or df_calendar['price_clean'].sum() == 0:
        if 'price' in df_listings.columns:
            listing_price_map = df_listings.set_index(id_col)['price'].to_dict()
            df_calendar['price_clean'] = df_calendar['listing_id'].map(listing_price_map)

    # Dự phòng an toàn nếu dữ liệu giá bị khuyết
    df_calendar['price_clean'] = pd.to_numeric(df_calendar['price_clean'], errors='coerce')
    mean_fallback = df_calendar['price_clean'].mean()
    if pd.isna(mean_fallback) or mean_fallback == 0:
        mean_fallback = 100.0
    df_calendar['price_clean'] = df_calendar['price_clean'].fillna(mean_fallback)

    # Tính doanh thu theo từng ngày
    df_calendar['revenue'] = np.where(df_calendar['is_booked'], df_calendar['price_clean'], 0.0)

    # 4. Gom nhóm calendar theo từng listing
    calendar_agg = df_calendar.groupby('listing_id').agg(
        total_days=('date', 'count'),
        booked_days=('is_booked', 'sum'),
        total_revenue=('revenue', 'sum')
    ).reset_index()

    # Tính các chỉ số KPI cấp listing
    calendar_agg['occupancy_rate'] = (calendar_agg['booked_days'] / calendar_agg['total_days'].replace(0, np.nan)).fillna(0)
    calendar_agg['adr'] = (calendar_agg['total_revenue'] / calendar_agg['booked_days'].replace(0, np.nan)).fillna(0)
    calendar_agg['revpar'] = (calendar_agg['total_revenue'] / calendar_agg['total_days'].replace(0, np.nan)).fillna(0)

    # 5. Tích hợp kết quả LLM (nếu tồn tại file reviews_llm_*.csv)
    if llm_review_path.exists():
        print(f"🤖 Đã tìm thấy dữ liệu LLM: {llm_review_path.name}. Đang tích hợp KPI cảm xúc...")
        df_llm = pd.read_csv(llm_review_path)
        if 'llm_sentiment' in df_llm.columns:
            df_llm['is_positive'] = (df_llm['llm_sentiment'] == 'positive').astype(int)
            llm_agg = df_llm.groupby('listing_id').agg(
                total_reviews_analyzed=('review_id', 'count'),
                positive_sentiment_count=('is_positive', 'sum')
            ).reset_index()
            llm_agg['positive_sentiment_rate'] = llm_agg['positive_sentiment_count'] / llm_agg['total_reviews_analyzed']
        else:
            llm_agg = pd.DataFrame(columns=['listing_id', 'positive_sentiment_rate'])
    else:
        print(f"⚠️ Chưa tìm thấy file LLM cho snapshot {snapshot}. Đặt positive_sentiment_rate = 0.0")
        llm_agg = pd.DataFrame(columns=['listing_id', 'positive_sentiment_rate'])

    # 6. Merge dữ liệu vào bảng listings chính
    df_merged = pd.merge(df_listings, calendar_agg, left_on=id_col, right_on='listing_id', how='left')
    if not llm_agg.empty and 'listing_id' in llm_agg.columns:
        df_merged = pd.merge(df_merged, llm_agg, left_on=id_col, right_on='listing_id', how='left')
    
    for col in ['total_days', 'booked_days', 'total_revenue', 'occupancy_rate', 'adr', 'revpar', 'positive_sentiment_rate']:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)

    # 7. Tính KPI theo khu vực (neighbourhood_cleansed)
    if 'neighbourhood_cleansed' in df_merged.columns:
        neighborhood_kpi = df_merged.groupby('neighbourhood_cleansed').agg(
            neighborhood_avg_revenue=('total_revenue', 'mean'),
            neighborhood_avg_occupancy=('occupancy_rate', 'mean')
        ).reset_index()
        df_merged = pd.merge(df_merged, neighborhood_kpi, on='neighbourhood_cleansed', how='left')
        print(f"🏙️ Đã tính KPI theo khu vực (neighbourhood_cleansed) cho {len(neighborhood_kpi)} khu vực.")

    df_merged['snapshot'] = snapshot
    df_merged['city'] = city

    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = processed_dir / f"kpi_{snapshot}.csv"
    df_merged.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ THÀNH CÔNG: Đã lưu file KPI vào: {output_path}")

    # 8. Báo cáo tổng kết
    print(f"\n📈 --- TỔNG KẾT 6 CHỈ SỐ KPI CHO SNAPSHOT {snapshot} ---")
    print(f"  1. Tổng số phòng (Listings): {len(df_merged)}")
    print(f"  2. Tỷ lệ lấp đầy trung bình (Occupancy Rate): {df_merged['occupancy_rate'].mean() * 100:.2f}%")
    print(f"  3. Giá thuê trung bình (ADR): ${df_merged['adr'].replace(0, np.nan).mean():.2f} USD")
    print(f"  4. RevPAR trung bình: ${df_merged['revpar'].mean():.2f} USD")
    print(f"  5. Tổng doanh thu ước tính: ${df_merged['total_revenue'].sum():,.2f} USD")
    if 'positive_sentiment_rate' in df_merged.columns:
        print(f"  6. Tỷ lệ đánh giá tích cực từ LLM: {df_merged['positive_sentiment_rate'].mean() * 100:.2f}%")

    summary_metrics = {
        "snapshot": snapshot,
        "total_listings": len(df_merged),
        "avg_occupancy_rate": df_merged['occupancy_rate'].mean(),
        "avg_adr": df_merged['adr'].replace(0, np.nan).mean(),
        "avg_revpar": df_merged['revpar'].mean(),
        "total_revenue": df_merged['total_revenue'].sum(),
        "avg_positive_sentiment_rate": df_merged['positive_sentiment_rate'].mean() if 'positive_sentiment_rate' in df_merged.columns else 0.0
    }
    summary_df = pd.DataFrame([summary_metrics])
    summary_path = processed_dir / f"kpi_summary_{snapshot}.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Tính toán hệ thống KPI Airbnb nâng cao.")
    parser.add_argument("--config", required=True, help="Đường dẫn tới file cấu hình YAML")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for snapshot in config["snapshots"]:
        calculate_snapshot_kpis(config, str(snapshot))

    print("\n🎉 HOÀN THÀNH TOÀN BỘ CÁC MỐC SNAPSHOT CHO MODULE KPI NÂNG CAO!")

if __name__ == "__main__":
    main()