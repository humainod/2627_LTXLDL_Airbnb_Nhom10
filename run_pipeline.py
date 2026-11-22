"""
run_pipeline.py - File điều phối tổng toàn bộ quy trình Pipeline của dự án Airbnb.

Thứ tự chạy các bước:
  1. download.py : Tải dữ liệu thô từ Inside Airbnb.
  2. qa.py       : Kiểm tra và làm sạch dữ liệu thô, xuất báo cáo QA.
  3. kpi.py      : Tính toán các chỉ số KPI hiệu suất kinh doanh (có tích hợp LLM).
  4. llm.py      : Gọi Gemini API phân tích cảm xúc các bài đánh giá (reviews).
  5. baseline.py : Đếm từ khóa baseline.
  6. evaluate.py : So sánh đánh giá.
  7. viz.py      : Vẽ biểu đồ trực quan hóa.

Cách chạy:
    python run_pipeline.py --config configs/florence.yaml
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path

def run_step(command: list[str], step_name: str):
    """
    Hàm hỗ trợ thực thi một lệnh trong terminal và kiểm tra lỗi.
    """
    print(f"\n==================================================")
    print(f" BẮT ĐẦU BƯỚC: {step_name}")
    print(f"Lệnh chạy: {' '.join(command)}")
    print(f"==================================================")
    
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"\n[LỖI] Bước '{step_name}' gặp sự cố (mã lỗi: {result.returncode}). Dừng pipeline.")
        sys.exit(result.returncode)
    else:
        print(f"\n[HOÀN THÀNH] Bước '{step_name}' thành công!")

def main():
    parser = argparse.ArgumentParser(description="Chạy toàn bộ Data Pipeline cho dự án Airbnb.")
    parser.add_argument(
        "--config",
        required=True,
        help="Đường dẫn tới file cấu hình YAML (ví dụ: configs/florence.yaml)"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
        sys.exit(1)

    # Đọc config để kiểm tra nhanh
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    city = config.get("city", "unknown")
    print(f" BẮT ĐẦU CHẠY PIPELINE CHO THÀNH PHỐ: {city.upper()} ")

    # --- BƯỚC 1: Tải dữ liệu thô ---
    run_step([sys.executable, "src/download.py", "--config", str(config_path)], "1. Tải dữ liệu (Download)")

    # --- BƯỚC 2: Kiểm tra và làm sạch dữ liệu ---
    # (Lưu ý: qa.py thường đọc config trực tiếp hoặc qua code nội bộ)
    run_step([sys.executable, "src/qa.py"], "2. Kiểm tra & Làm sạch (QA & Data Cleaning)")

    # --- BƯỚC 3a: Gọi LLM phân tích cảm xúc reviews ---
    run_step([sys.executable, "src/llm.py", "--config", str(config_path)], "3a. Phân tích LLM (Gemini API)")

    # --- BƯỚC 3b: Tính toán các chỉ số KPI ---
    run_step([sys.executable, "src/kpi.py", "--config", str(config_path)], "3b. Tính toán KPI (KPI Calculation)")

    # --- BƯỚC 4a: Chạy Baseline (đếm từ khóa) ---
    baseline_script = Path("src/baseline.py")
    if baseline_script.exists():
        run_step([sys.executable, "src/baseline.py", "--config", str(config_path)], "4a. Chạy Baseline Keyword")

    # --- BƯỚC 4b: Đánh giá và so sánh (Evaluate) ---
    eval_script = Path("src/evaluate.py")
    if eval_script.exists():
        run_step([sys.executable, "src/evaluate.py", "--config", str(config_path)], "4b. Đánh giá kết quả (Evaluate)")

    # --- BƯỚC 5: Vẽ biểu đồ trực quan hóa (Viz) ---
    viz_script = Path("src/viz.py")
    if viz_script.exists():
        run_step([sys.executable, "src/viz.py", "--config", str(config_path)], "5. Vẽ biểu đồ trực quan (Visualization)")

    print(f"\n CHÚC MỪNG! TOÀN BỘ PIPELINE CHO {city.upper()} ĐÃ CHẠY XONG XUẤT SẮC! ")

if __name__ == "__main__":
    main()