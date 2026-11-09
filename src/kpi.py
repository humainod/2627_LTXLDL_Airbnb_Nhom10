"""
kpi.py - Tính toán 6 chỉ số KPI chuẩn Airbnb sau bước QA và tích hợp kết quả LLM.
Có thêm tính toán & in ra chỉ số Sau Khi Lọc Giá Trị Ngoại Lai (Outliers) 
và lọc/gắn dữ liệu theo đúng yêu cầu phân vùng (analysis_area).

Quy định lọc dữ liệu nhóm (theo yêu cầu trong ảnh):
  1. listings: Giữ 1.000 dòng có id nằm trong danh sách ID của nhóm. Gắn cột `analysis_area`.
  2. calendar: Giữ các dòng có listing_id thuộc 1.000 listing này (365.000 dòng/mốc).
  3. Xử lý giá trị ngoại lai (Outliers):
     - Sử dụng phương pháp IQR (Interquartile Range) để phát hiện và loại bỏ giá trị ngoại lai cho ADR / Revenue / Price.
     - Tính toán và in song song cả 2 phiên bản: [Gốc (Raw)] và [Sau khi lọc ngoại lai (Clean/Filtered)].

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

def remove_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.Series:
    """
    Hàm hỗ trợ lọc giá trị ngoại lai bằng phương pháp IQR.
    Trả về Series chứa True (hợp lệ) hoặc False (ngoại lai).
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return (df[column] >= lower_bound) & (df[column] <= upper_bound)

def calculate_snapshot_kpis(config: dict, snapshot: str):
    city = config.get("city", "unknown")
    clean_dir = Path("data/processed/clean") / snapshot
    processed_dir = Path("data/processed")
    id_dir = Path("data/id")
    
    listings_path = clean_dir / "listings.csv"
    calendar_path = clean_dir / "calendar.csv"
    llm_review_path = processed_dir / f"reviews_llm_{snapshot}.csv"
    
    # File chứa danh sách ID của nhóm và mapping analysis_area
    listing_id_path = id_dir / "listing_id.csv"
    
    if not listings_path.exists() or not calendar_path.exists():
        print(f"⚠️ Cảnh báo: Không tìm thấy dữ liệu clean tại {clean_dir}. Bỏ qua snapshot {snapshot}.")
        return

    print(f"\n==================================================================")
    print(f" --- ĐANG TÍNH TOÁN KPI CHO SNAPSHOT: {snapshot} (Thành phố: {city}) ---")
    print(f"==================================================================")
    
    # 1. Đọc dữ liệu listings và calendar đã qua QA
    df_listings = pd.read_csv(listings_path)
    df_calendar = pd.read_csv(calendar_path)
    
    id_col = 'id' if 'id' in df_listings.columns else 'listing_id'
    
    # --- YÊU CẦU TRONG ẢNH: LỌC THEO DANH SÁCH ID VÀ GẮN CỘT analysis_area ---
    if listing_id_path.exists():
        df_group_ids = pd.read_csv(listing_id_path)
        group_id_col = 'id' if 'id' in df_group_ids.columns else 'listing_id'
        
        valid_ids = set(df_group_ids[group_id_col])
        print(f" Đã đọc danh sách ID nhóm từ file: {len(valid_ids)} ID mẫu.")
        
        # Lọc 1.000 listing của nhóm
        df_listings = df_listings[df_listings[id_col].isin(valid_ids)].copy()
        
        # Gắn cột analysis_area nếu có trong file mapping
        if 'analysis_area' in df_group_ids.columns:
            area_map = df_group_ids.set_index(group_id_col)['analysis_area'].to_dict()
            df_listings['analysis_area'] = df_listings[id_col].map(area_map)
            print(f"🗺️ Đã gắn thành công cột `analysis_area` cho {df_listings['analysis_area'].nunique()} vùng.")
        
        # Lọc 365.000 dòng calendar tương ứng với 1.000 listing
        df_calendar = df_calendar[df_calendar['listing_id'].isin(valid_ids)].copy()
        print(f" Số dòng calendar sau khi lọc theo ID nhóm: {len(df_calendar):,} dòng.")
    else:
        print(f" Không thấy {listing_id_path}, sử dụng toàn bộ {len(df_listings)} listing hiện có.")

    # 2. Xử lý trạng thái đặt phòng trên calendar
    if 'available' in df_calendar.columns:
        if df_calendar['available'].dtype == object:
            df_calendar['is_booked'] = df_calendar['available'].isin(['f', 'F', False, 'False', 'FALSE'])
        else:
            df_calendar['is_booked'] = ~df_calendar['available']
    else:
        df_calendar['is_booked'] = False

    # 3. Xử lý giá tiền
    if 'price' in df_calendar.columns:
        df_calendar['price_clean'] = pd.to_numeric(df_calendar['price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
    else:
        df_calendar['price_clean'] = np.nan

    if df_calendar['price_clean'].isna().all() or df_calendar['price_clean'].sum() == 0:
        if 'price' in df_listings.columns:
            price_clean_listings = pd.to_numeric(df_listings['price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
            listing_price_map = dict(zip(df_listings[id_col], price_clean_listings))
            df_calendar['price_clean'] = df_calendar['listing_id'].map(listing_price_map)

    # Dự phòng giá trung bình
    mean_fallback = df_calendar['price_clean'].mean()
    if pd.isna(mean_fallback) or mean_fallback == 0:
        mean_fallback = 100.0
    df_calendar['price_clean'] = df_calendar['price_clean'].fillna(mean_fallback)

    # Tính doanh thu từng ngày
    df_calendar['revenue'] = np.where(df_calendar['is_booked'], df_calendar['price_clean'], 0.0)

    # 4. Gom nhóm calendar theo từng listing
    calendar_agg = df_calendar.groupby('listing_id').agg(
        total_days=('date', 'count'),
        booked_days=('is_booked', 'sum'),
        total_revenue=('revenue', 'sum')
    ).reset_index()

    # Tính KPI cấp listing
    calendar_agg['occupancy_rate'] = (calendar_agg['booked_days'] / calendar_agg['total_days'].replace(0, np.nan)).fillna(0)
    calendar_agg['adr'] = (calendar_agg['total_revenue'] / calendar_agg['booked_days'].replace(0, np.nan)).fillna(0)
    calendar_agg['revpar'] = (calendar_agg['total_revenue'] / calendar_agg['total_days'].replace(0, np.nan)).fillna(0)

    # 5. Tích hợp kết quả LLM sentiment (nếu có)
    if llm_review_path.exists():
        print(f" Đã tìm thấy dữ liệu LLM: {llm_review_path.name}. Đang tích hợp...")
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
        llm_agg = pd.DataFrame(columns=['listing_id', 'positive_sentiment_rate'])

    # 6. Merge dữ liệu vào df_listings
    df_merged = pd.merge(df_listings, calendar_agg, left_on=id_col, right_on='listing_id', how='left')
    if not llm_agg.empty and 'listing_id' in llm_agg.columns:
        df_merged = pd.merge(df_merged, llm_agg, left_on=id_col, right_on='listing_id', how='left')
    
    for col in ['total_days', 'booked_days', 'total_revenue', 'occupancy_rate', 'adr', 'revpar', 'positive_sentiment_rate']:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)

    # --- 7. TÍNH TOÁN THÊM GIÁ TRỊ SAU KHI LỌC NGOẠI LAI (OUTLIERS) ---
    valid_adr_mask = df_merged['adr'] > 0
    non_zero_adr_df = df_merged[valid_adr_mask]
    
    if not non_zero_adr_df.empty:
        non_outlier_mask = remove_outliers_iqr(non_zero_adr_df, 'adr')
        valid_listing_ids = set(non_zero_adr_df[non_outlier_mask][id_col])
        df_merged['is_outlier_adr'] = df_merged[id_col].apply(lambda x: x in set(non_zero_adr_df[~non_outlier_mask][id_col]))
    else:
        df_merged['is_outlier_adr'] = False

    df_clean = df_merged[~df_merged['is_outlier_adr']].copy()
    num_outliers = df_merged['is_outlier_adr'].sum()

    # 8. Tính KPI theo khu vực (neighbourhood_cleansed & analysis_area)
    area_col = 'analysis_area' if 'analysis_area' in df_merged.columns else ('neighbourhood_cleansed' if 'neighbourhood_cleansed' in df_merged.columns else None)
    if area_col:
        # Trước lọc
        neighborhood_kpi = df_merged.groupby(area_col).agg(
            neighborhood_avg_revenue=('total_revenue', 'mean'),
            neighborhood_avg_occupancy=('occupancy_rate', 'mean')
        ).reset_index()
        df_merged = pd.merge(df_merged, neighborhood_kpi, on=area_col, how='left')

        # Sau lọc ngoại lai
        clean_neighborhood_kpi = df_clean.groupby(area_col).agg(
            clean_neighborhood_avg_revenue=('total_revenue', 'mean'),
            clean_neighborhood_avg_occupancy=('occupancy_rate', 'mean')
        ).reset_index()
        df_merged = pd.merge(df_merged, clean_neighborhood_kpi, on=area_col, how='left')

    df_merged['snapshot'] = snapshot
    df_merged['city'] = city

    # Lưu dữ liệu kết quả
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / f"kpi_{snapshot}.csv"
    df_merged.to_csv(output_path, index=False, encoding='utf-8')
    print(f" Đã lưu kết quả chi tiết KPI vào: {output_path}")

    # --- 9. IN BÁO CÁO TỔNG KẾT SO SÁNH TRƯỚC VÀ SAU KHI BỎ NGOẠI LAI ---
    avg_occ_raw = df_merged['occupancy_rate'].mean() * 100
    avg_occ_clean = df_clean['occupancy_rate'].mean() * 100
    
    avg_adr_raw = df_merged['adr'].replace(0, np.nan).mean()
    avg_adr_clean = df_clean['adr'].replace(0, np.nan).mean()
    
    avg_revpar_raw = df_merged['revpar'].mean()
    avg_revpar_clean = df_clean['revpar'].mean()

    tot_rev_raw = df_merged['total_revenue'].sum()
    tot_rev_clean = df_clean['total_revenue'].sum()

    avg_sent_raw = df_merged['positive_sentiment_rate'].mean() * 100 if 'positive_sentiment_rate' in df_merged.columns else 0.0
    avg_sent_clean = df_clean['positive_sentiment_rate'].mean() * 100 if 'positive_sentiment_rate' in df_clean.columns else 0.0

    print(f"\n ==================== TỔNG KẾT KPI SNAPSHOT {snapshot} ====================")
    print(f"   Tổng số Listing: {len(df_merged)} (Loại bỏ {num_outliers} ngoại lai -> Còn lại: {len(df_clean)})")
    print(f"----------------------------------------------------------------------------------")
    print(f"  CHỈ SỐ KPI                            | GỐC (RAW)       | SAU BỎ NGOẠI LAI (CLEAN)")
    print(f"----------------------------------------------------------------------------------")
    print(f"  1. Tỷ lệ lấp đầy (Occupancy Rate)     | {avg_occ_raw:8.2f}%       | {avg_occ_clean:8.2f}%")
    print(f"  2. Giá trung bình ngày (ADR)          | ${avg_adr_raw:7.2f} USD   | ${avg_adr_clean:7.2f} USD")
    print(f"  3. Doanh thu/phòng (RevPAR)          | ${avg_revpar_raw:7.2f} USD   | ${avg_revpar_clean:7.2f} USD")
    print(f"  4. Tổng doanh thu ước tính            | ${tot_rev_raw:11,.2f} | ${tot_rev_clean:11,.2f}")
    print(f"  5. Tỷ lệ cảm xúc tích cực (LLM)      | {avg_sent_raw:8.2f}%       | {avg_sent_clean:8.2f}%")
    print(f"==================================================================================\n")

    # Lưu file tóm tắt KPI
    summary_metrics = {
        "snapshot": snapshot,
        "total_listings_raw": len(df_merged),
        "total_listings_clean": len(df_clean),
        "outliers_removed": num_outliers,
        "avg_occupancy_rate_raw": avg_occ_raw,
        "avg_occupancy_rate_clean": avg_occ_clean,
        "avg_adr_raw": avg_adr_raw,
        "avg_adr_clean": avg_adr_clean,
        "avg_revpar_raw": avg_revpar_raw,
        "avg_revpar_clean": avg_revpar_clean,
        "total_revenue_raw": tot_rev_raw,
        "total_revenue_clean": tot_rev_clean,
        "avg_positive_sentiment_raw": avg_sent_raw,
        "avg_positive_sentiment_clean": avg_sent_clean
    }
    summary_df = pd.DataFrame([summary_metrics])
    summary_path = processed_dir / f"kpi_summary_{snapshot}.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Tính toán hệ thống KPI Airbnb có lọc ngoại lai & phân vùng.")
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

    print("\n HOÀN THÀNH TÍNH TOÁN & IN BÁO CÁO KPI!")

if __name__ == "__main__":
    main()