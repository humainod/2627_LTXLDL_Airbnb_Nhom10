"""
download.py - Tải dữ liệu Airbnb thô từ Inside Airbnb.

Chức năng:
  1. Đọc file config YAML để biết thành phố, snapshot, URL.
  2. Với mỗi snapshot, tải 3 file (listings, calendar, reviews).
  3. Nếu file đã tồn tại thì bỏ qua (không tải lại).
  4. Bắt lỗi: mạng, 404, 500, timeout.

Cách chạy:
    python src/download.py --config configs/<thành phố>.yaml
"""

import argparse
import sys
from pathlib import Path
import requests
import yaml

def tai_snapshot(config: dict, snapshot: str, raw_dir: Path) -> bool:
    """
    Tải 3 file của 1 snapshot vào thư mục:
        data/raw/<city>/<snapshot>/
    Trả về True nếu tất cả tải thành công.
    """
    print(f"\n=== Snapshot {snapshot} ===")

    # Thư mục lưu: data/raw/<city>/<snapshot>/
    save_dir = raw_dir / config["city"] / snapshot

    tat_ca_ok = True
    for filename in config["files"]:
        # Tạo URL từ template
        url = config["url_template"].format(
            country=config["country"],
            region=config["region"],
            city=config["city"],
            snapshot=snapshot,
            filename=filename,
        )
        save_path = save_dir / filename

        ok = tai_file(url, save_path)
        if not ok:
            tat_ca_ok = False
    return tat_ca_ok

def tai_file(url: str, save_path: Path) -> bool:
    """
    Tải 1 file từ `url` và lưu vào `save_path`.
    Trả về True nếu tải thành công, False nếu lỗi.
    """
    # Bước 1: Nếu file đã tồn tại → bỏ qua, không tải lại
    if save_path.exists():
        print(f"  [bỏ qua] Đã có: {save_path.name}")
        return True

    # Bước 2: Tạo thư mục cha nếu chưa có
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Bước 3: Gửi request tải file
    print(f"  [tải] {url}")
    try:
        r = requests.get(url, timeout=60)
    except requests.exceptions.Timeout:
        print(f"  [LỖI] Hết thời gian chờ (timeout) khi tải: {url}")
        return False
    except requests.exceptions.ConnectionError:
        print(f"  [LỖI] Không kết nối được tới server: {url}")
        return False

    # Bước 4: Kiểm tra mã trạng thái HTTP
    if r.status_code == 404:
        print(f"  [LỖI] File không tồn tại (404): {url}")
        return False
    if r.status_code == 403:
        print(f"  [LỖI] Bị chặn truy cập (403): {url}")
        return False
    if r.status_code >= 500:
        print(f"  [LỖI] Server lỗi ({r.status_code}): {url}")
        return False
    if r.status_code != 200:
        print(f"  [LỖI] Mã lỗi không xác định ({r.status_code}): {url}")
        return False

    # Bước 5: Ghi file xuống ổ cứng theo từng phần nhỏ (8 KB)
    try:
        with save_path.open("wb") as f:
                f.write(r.content)
    except OSError as e:
        print(f"  [LỖI] Không ghi được file {save_path}: {e}")
        return False

    # Bước 6: Kiểm tra file sau khi tải — phải có dung lượng > 0
    size_mb = save_path.stat().st_size / (1024 * 1024)
    if size_mb < 0.001:
        print(f"  [LỖI] File tải về rỗng: {save_path}")
        return False

    print(f"  [xong] {save_path.name} ({size_mb:.1f} MB)")
    return True
    


def main():

    # Bước 4: Tải lần lượt từng snapshot
    raw_dir = Path("data/raw")
    tat_ca_ok = True

    for snapshot in config["snapshots"]:
        ok = tai_snapshot(config, snapshot, raw_dir)
        if not ok:
            tat_ca_ok = False

    # Bước 5: Tổng kết
    print("\n=== Kết quả ===")
    if tat_ca_ok:
        print("Tải thành công toàn bộ.")
    else:
        print("Có lỗi xảy ra. Xem log phía trên.")
        sys.exit(1)

if __name__ == "__main__":
    # Bước 1: Đọc tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Tải dữ liệu Airbnb thô.")
    parser.add_argument(
            "--config",
            required=True,
            help="Đường dẫn tới file cấu hình YAML",
        )
    args = parser.parse_args()

# Bước 2: Đọc file config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Bước 3: Kiểm tra config có đủ thông tin chưa
    if not config.get("city"):
        print("[LỖI] Trường 'city' trong config đang trống.")
        sys.exit(1)
    if not config.get("snapshots") or config["snapshots"][0] == "":
        print("[LỖI] Trường 'snapshots' trong config đang trống.")
        sys.exit(1)

    main()

