import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Tối ưu hóa cấu hình hiển thị đồ thị và ép cấu hình Font hệ thống hỗ trợ tiếng Việt Unicode
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Segoe UI', 'DejaVu Sans', 'Calibri', 'Tahoma']
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def load_config(config_path):
    """Đọc tệp cấu hình YAML động"""
    if not os.path.exists(config_path):
        print(f"[Lỗi] Không tìm thấy tệp cấu hình tại: {config_path}")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def plot_spatial_density(clean_dir, latest_snapshot, output_dir, city_name, currency):
    """
    1. TRỰC QUAN HÓA KHÔNG GIAN (Spatial Visualization) theo ô tọa độ của Giảng viên
    Chỉ hiển thị các listings sạch thuộc 1000 ID được giao và vẽ kèm ranh giới 8 vùng hình chữ nhật.
    """
    print(f"[Viz] Đang trực quan hóa Không gian theo ô tọa độ nhóm cho mốc {latest_snapshot}...")
    listings_path = os.path.join(clean_dir, latest_snapshot, "listings.csv")
    listing_id_path = os.path.join("data", "id", "listing_id.csv")
    region_coord_path = os.path.join("data", "id", "region_coordinate.csv")
    
    # Kiểm tra sự tồn tại của các tệp cốt lõi
    if not os.path.exists(listings_path):
        print(f"[Cảnh báo] Thiếu dữ liệu phòng sạch: {listings_path}")
        return
    if not os.path.exists(listing_id_path) or not os.path.exists(region_coord_path):
        print("[Cảnh báo] Thiếu các tệp bổ trợ danh sách ID phân công hoặc tọa độ vùng vuông.")
        return

    # Đọc và đồng bộ dữ liệu theo ID phân công
    df_listings = pd.read_csv(listings_path)
    df_assigned = pd.read_csv(listing_id_path)
    df_regions = pd.read_csv(region_coord_path)
    
    # Đảm bảo đồng bộ kiểu dữ liệu ID phòng để thực hiện phép lọc chính xác
    df_listings['id'] = df_listings['id'].astype(str)
    df_assigned['listing_id'] = df_assigned['listing_id'].astype(str)
    
    # Lọc nghiêm ngặt: Chỉ giữ lại các phòng nằm trong tập 1.000 ID được cấp
    df_plot = df_listings[df_listings['id'].isin(df_assigned['listing_id'])].copy()
    
    if df_plot.empty:
        print("[Cảnh báo] Không tìm thấy phòng nào khớp với tập ID được phân công trong mốc này.")
        return

    if 'longitude' in df_plot.columns and 'latitude' in df_plot.columns:
        fig, ax = plt.subplots(figsize=(11, 9))
        
        # Tiền xử lý cột giá phòng (Yêu cầu: chỉ dùng giá mốc cuối cùng để trực quan hóa)
        if 'price' in df_plot.columns:
            if df_plot['price'].dtype == 'object':
                df_plot['price'] = df_plot['price'].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)
            
            # Loại bỏ biên ngoại lai cực đại (98%) để tránh bóp méo dải màu hiển thị
            q_high = df_plot['price'].quantile(0.98)
            df_spatial = df_plot[df_plot['price'] <= q_high]
            
            # Chấm các điểm phòng lên hệ tọa độ dựa trên mức giá
            scatter = ax.scatter(df_spatial['longitude'], df_spatial['latitude'], 
                                 c=df_spatial['price'], cmap='viridis', 
                                 alpha=0.6, s=30, edgecolors='none', zorder=2)
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(f"Mức giá thuê phòng ({currency})", fontsize=11, fontweight='bold')
        else:
            ax.scatter(df_plot['longitude'], df_plot['latitude'], alpha=0.6, s=25, color='#6c5ce7', zorder=2)
            
        # Vẽ các khối hình chữ nhật đại diện cho 8 vùng phân tích (A01 -> A08) lên đồ thị
        for _, row in df_regions.iterrows():
            area_code = row['analysis_area']
            lon_min, lat_min = row['lon_min'], row['lat_min']
            lon_max, lat_max = row['lon_max'], row['lat_max']
            
            width = lon_max - lon_min
            height = lat_max - lat_min
            
            # Khởi tạo khung nét đứt màu đỏ bao quanh ô chữ nhật
            rect = patches.Rectangle((lon_min, lat_min), width, height, 
                                     linewidth=1.5, linestyle='--', 
                                     edgecolor='#e74c3c', facecolor='none', alpha=0.7, zorder=3)
            ax.add_patch(rect)
            
            # Gắn nhãn tên vùng (ví dụ: A01) ngay tại tâm của hình chữ nhật tương ứng
            ax.text(lon_min + width/2, lat_min + height/2, area_code, 
                    color='#d63031', fontsize=12, fontweight='bold',
                    ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5), zorder=4)

        ax.set_title(f"Bản đồ phân bố mật độ & ranh giới 8 ô vùng phân tích tại {city_name.upper()}\nMốc thời gian: {latest_snapshot} (Tập mẫu 1.000 phòng phân công)", 
                     fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Kinh độ (Longitude)", fontsize=11, labelpad=10)
        ax.set_ylabel("Vĩ độ (Latitude)", fontsize=11, labelpad=10)
        ax.grid(True, linestyle=':', alpha=0.5, zorder=1)
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "spatial_listings_density.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("Đã lưu hình ảnh không gian: figures/spatial_listings_density.png")

def plot_temporal_trends(processed_dir, snapshots, output_dir, currency):
    """
    2. TRỰC QUAN HÓA THỜI GIAN (Temporal Visualization)
    Quét qua các file kpi_summary_<snapshot>.csv để lấy cột avg_adr vẽ xu hướng giá trung bình.
    """
    print("[Viz] Đang trực quan hóa xu hướng KPI theo Thời gian qua 4 mốc...")
    trend_data = []
    
    for snap in snapshots:
        kpi_summary_path = os.path.join(processed_dir, f"kpi_summary_{snap}.csv")
        
        if os.path.exists(kpi_summary_path):
            df_kpi = pd.read_csv(kpi_summary_path)
            
            if 'avg_adr' in df_kpi.columns:
                val = df_kpi['avg_adr'].mean()
                if pd.notna(val) and np.isfinite(val):
                    trend_data.append({
                        'Snapshot': snap,
                        'Giá trung bình': float(val)
                    })
            
    if trend_data:
        df_trend = pd.DataFrame(trend_data)
        df_trend = df_trend.dropna(subset=['Giá trung bình'])
        
        if not df_trend.empty:
            plt.figure(figsize=(9, 5))
            sns.lineplot(data=df_trend, x='Snapshot', y='Giá trung bình', marker='o', 
                         color='#e84393', linewidth=2.5, markersize=8)
            
            plt.title("Biến động mức giá thuê phòng trung bình (ADR) qua các quý", pad=15, fontweight='bold')
            plt.xlabel("Mốc trích xuất dữ liệu (Snapshot)", labelpad=10)
            plt.ylabel(f"Giá thuê phòng trung bình ({currency})", labelpad=10)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            for x, y in zip(df_trend['Snapshot'], df_trend['Giá trung bình']):
                plt.text(str(x), float(y) + (float(y) * 0.01), f"{y:.2f}", 
                         ha='center', va='bottom', fontsize=9, fontweight='bold')
                
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "temporal_price_trend.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print("Đã lưu hình ảnh xu hướng: figures/temporal_price_trend.png")
    else:
        print("[Cảnh báo] Không tìm thấy dữ liệu số hợp lệ trong các file kpi_summary_*.csv để vẽ đồ thị đường.")

def plot_llm_sentiment_analysis(processed_dir, snapshots, output_dir):
    """
    3. TRỰC QUAN HÓA LLM (LLM-based Visualization)
    Sử dụng biểu đồ tròn đại diện phân phối cảm xúc từ tệp dữ liệu mẫu cố định mới nhất.
    """
    print("[Viz] Đang trực quan hóa phân tích cảm xúc từ dữ liệu LLM...")
    latest_snap = snapshots[-1]
    llm_file_path = os.path.join(processed_dir, f"reviews_llm_{latest_snap}.csv")
    
    if os.path.exists(llm_file_path):
        print(f" -> [Đọc dữ liệu LLM] Lấy dữ liệu đại diện từ tệp: {os.path.basename(llm_file_path)}")
        df_llm = pd.read_csv(llm_file_path)
        
        if 'llm_sentiment' in df_llm.columns:
            df_llm['llm_sentiment'] = df_llm['llm_sentiment'].astype(str).str.lower().str.strip()
            counts = df_llm['llm_sentiment'].value_counts()
            
            label_mapping = {
                'positive': 'Tích cực (Positive)',
                'tích cực': 'Tích cực (Positive)',
                'mixed': 'Hỗn hợp (Mixed)',
                'hỗn hợp': 'Hỗn hợp (Mixed)',
                'negative': 'Tiêu cực (Negative)',
                'tiêu cực': 'Tiêu cực (Negative)'
            }
            counts.index = [label_mapping.get(x, x.capitalize()) for x in counts.index]
            
            color_map = {
                'Tích cực (Positive)': '#2ecc71',
                'Hỗn hợp (Mixed)': '#f1c40f',
                'Tiêu cực (Negative)': '#e74c3c'
            }
            colors = [color_map.get(lbl, '#95a5a6') for lbl in counts.index]
            
            plt.figure(figsize=(7, 7))
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140,
                    colors=colors, textprops={'fontsize': 11, 'fontweight': 'bold'},
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'antialiased': True})
            
            plt.title("Tỷ lệ phân bố sắc thái đánh giá của khách hàng\n(Phân tích bằng LLM trên tập dữ liệu mẫu cố định)", 
                      fontsize=13, fontweight='bold', pad=20)
            
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "llm_sentiment_distribution.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print("Đã lưu hình ảnh biểu đồ cảm xúc: figures/llm_sentiment_distribution.png")
    else:
        print(f"[Cảnh báo] Không tìm thấy file dữ liệu mẫu: reviews_llm_{latest_snap}.csv để vẽ biểu đồ tròn.")

def generate_all_figures(config):
    """Hàm tổng phối hợp chạy toàn bộ các góc độ trực quan hóa từ cấu hình động"""
    city_name = config.get('city', 'florence')
    snapshots = config.get('snapshots', [])
    currency = config.get('currency', 'EUR')
    
    base_processed_dir = "data/processed"
    clean_data_dir = os.path.join(base_processed_dir, "clean")
    output_figures_dir = "figures"
    
    if not snapshots:
        print("[Lỗi] Không tìm thấy danh sách mốc thời gian 'snapshots' trong file cấu hình.")
        return

    print("========================================================")
    print(f"Bắt đầu tạo biểu đồ tự động cho thành phố: {city_name.upper()}")
    print("========================================================")
    
    # Thực hiện gọi hàm xử lý không gian và truyền mốc chụp mới nhất theo yêu cầu của thầy cô
    plot_spatial_density(clean_data_dir, snapshots[-1], output_figures_dir, city_name, currency)
    plot_temporal_trends(base_processed_dir, snapshots, output_figures_dir, currency)
    plot_llm_sentiment_analysis(base_processed_dir, snapshots, output_figures_dir)
    
    print("\nHoàn thành! Toàn bộ hình ảnh đã được lưu tại thư mục 'figures/'.")
    print("========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/florence.yaml',
                        help='Đường dẫn tới tệp cấu hình định dạng .yaml')
    args = parser.parse_args()
    print(f"[Hệ thống] Đang chạy kiểm thử độc lập mô-đun viz.py...")
    config_data = load_config(args.config)
    generate_all_figures(config_data)
