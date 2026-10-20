"""
kpi.py - Tính toán các chỉ số KPI hiệu suất kinh doanh từ dữ liệu Airbnb đã làm sạch.

Chức năng:
  1. Đọc file config YAML để biết thành phố và các snapshot.
  2. Đọc dữ liệu clean từ `data/processed/clean/<snapshot>/listings.csv` và `calendar.csv`.
  3. Tính toán các KPI quan trọng:
     - Tỷ lệ lấp đầy (Occupancy Rate)
     - Giá thuê trung bình ngày (ADR - Average Daily Rate)
     - Doanh thu ước tính (Estimated Revenue)
     - RevPAR (Revenue Per Available Room)
  4. Lưu kết quả ra file `data/processed/kpi_<snapshot>.csv`.

Cách chạy:
    python src/kpi.py --config configs/<thành phố>.yaml
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

def calculate_snapshot_kpis(config: dict, snapshot: str):
    """
    Tính toán các KPI cho một snapshot cụ thể từ thư mục dữ liệu đã clean.
    """
    city = config.get("city", "unknown")
    clean_dir = Path("data/processed/clean") / snapshot
    
    listings_path = clean_dir / "listings.csv"
    calendar_path = clean_dir / "calendar.csv"
    
    if not listings_path.exists() or not calendar_path.exists():
        print(f"⚠️ Cảnh báo: Không tìm thấy dữ liệu clean tại {clean_dir}. Bỏ qua snapshot {snapshot}.")
        return

    print(f"\n📊 --- ĐANG TÍNH TOÁN KPI CHO SNAPSHOT: {snapshot} (Thành phố: {city}) ---")
    
    # 1. Đọc dữ liệu listings và calendar đã làm sạch từ qa.py
    df_listings = pd.read_csv(listings_path)
    df_calendar = pd.read_csv(calendar_path)
    
    # 2. Xử lý trạng thái đặt phòng trên calendar (available: 't'/'f' hoặc True/False)
    if 'available' in df_calendar.columns:
        if df_calendar['available'].dtype == object:
            df_calendar['is_booked'] = df_calendar['available'].isin(['f', 'F', False, 'False', 'FALSE'])
        else:
            df_calendar['is_booked'] = ~df_calendar['available']
    else:
        df_calendar['is_booked'] = False

    # 3. Xử lý cột giá trên calendar
    if 'price' in df_calendar.columns:
        #if df_calendar['price'].dtype == object:
        #    df_calendar['price_clean'] = df_calendar['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
        #else:
            df_calendar['price_clean'] = df_calendar['price']
    else:
        df_calendar['price_clean'] = 0.0

    # Tính doanh thu từng ngày (nếu được đặt thì lấy giá, ngược lại bằng 0)
    df_calendar['revenue'] = np.where(df_calendar['is_booked'], df_calendar['price_clean'], 0.0)

    # 4. Gom nhóm theo listing_id trên calendar
    calendar_agg = df_calendar.groupby('listing_id').agg(
        total_days=('date', 'count'),
        booked_days=('is_booked', 'sum'),
        total_revenue=('revenue', 'sum'),
        avg_booked_price=('price_clean', lambda x: x[df_calendar.loc[x.index, 'is_booked']].mean() if df_calendar.loc[x.index, 'is_booked'].sum() > 0 else 0)
    ).reset_index()

    # Tính các chỉ số KPI theo từng listing
    calendar_agg['occupancy_rate'] = calendar_agg['booked_days'] / calendar_agg['total_days'].replace(0, np.nan)
    calendar_agg['adr'] = calendar_agg['total_revenue'] / calendar_agg['booked_days'].replace(0, np.nan)
    calendar_agg['revpar'] = calendar_agg['total_revenue'] / calendar_agg['total_days'].replace(0, np.nan)

    # 5. Kết hợp (Merge) với bảng listings
    id_col = 'id' if 'id' in df_listings.columns else 'listing_id'
    df_merged = pd.merge(df_listings, calendar_agg, left_on=id_col, right_on='listing_id', how='left')

    # Điền giá trị 0 cho các listing không xuất hiện trong calendar
    for col in ['total_days', 'booked_days', 'total_revenue', 'occupancy_rate', 'adr', 'revpar']:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)

    # Thêm thông tin metadata
    df_merged['snapshot'] = snapshot
    df_merged['city'] = city

    # 6. Xuất file kết quả chi tiết kpi_<snapshot>.csv
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"kpi_{snapshot}.csv"
    df_merged.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ THÀNH CÔNG: Đã xuất KPI chi tiết ra file: {output_path}")
    
    # 7. In tổng kết nhanh (Summary)
    print(f"📈 --- TỔNG KẾT NHANH SNAPSHOT {snapshot} ---")
    print(f"  - Tổng số phòng (listings): {len(df_merged)}")
    print(f"  - Tỷ lệ lấp đầy trung bình: {df_merged['occupancy_rate'].mean() * 100:.2f}%")
    print(f"  - Giá thuê trung bình (ADR): ${df_merged['adr'].replace(0, np.nan).mean():.2f}")
    print(f"  - RevPAR trung bình: ${df_merged['revpar'].mean():.2f}")
    print(f"  - Tổng doanh thu ước tính: ${df_merged['total_revenue'].sum():,.2f}")

    # Lưu tệp summary gọn nhẹ nếu cần phục vụ cho module vẽ biểu đồ (viz.py sau này)
    summary_metrics = {
        "snapshot": snapshot,
        "total_listings": len(df_merged),
        "avg_occupancy_rate": df_merged['occupancy_rate'].mean(),
        "avg_adr": df_merged['adr'].replace(0, np.nan).mean(),
        "avg_revpar": df_merged['revpar'].mean(),
        "total_revenue": df_merged['total_revenue'].sum()
    }

    summary_df = pd.DataFrame([summary_metrics])
    summary_path = output_dir / f"kpi_summary_{snapshot}.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Tính toán các chỉ số KPI Airbnb từ dữ liệu clean.")
    parser.add_argument(
        "--config",
        required=True,
        help="Đường dẫn tới file cấu hình YAML (ví dụ: configs/florence.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config.get("city"):
        print("[LỖI] Trường 'city' trong config đang trống.")
        sys.exit(1)
    if not config.get("snapshots"):
        print("[LỖI] Trường 'snapshots' trong config đang trống.")
        sys.exit(1)

    # Chạy lần lượt cho tất cả các snapshot được cấu hình
    for snapshot in config["snapshots"]:
        calculate_snapshot_kpis(config, str(snapshot))

    print("\n🎉 HOÀN THÀNH TOÀN BỘ CÁC MỐC SNAPSHOT CHO MODULE KPI!")

if __name__ == "__main__":
    main()