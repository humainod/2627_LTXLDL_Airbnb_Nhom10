import argparse
import yaml
import pandas as pd
import re
from pathlib import Path

def get_root() -> Path:
    """Lấy đường dẫn gốc của project"""
    return Path(__file__).resolve().parents[1]

def load_config(config_path: Path) -> dict:
    """Đọc file cấu hình YAML"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def clean_text(text: str) -> set:
    """Đưa về chữ thường, xoá dấu câu và tách thành set các từ"""
    if pd.isna(text):
        return set()
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text) # Xoá dấu câu
    return set(text.split())

def classify_review(text: str, pos_words: set, neg_words: set, aspect_kws: dict) -> tuple:
    """Phân loại 1 review bằng phương pháp keyword matching"""
    words = clean_text(text)
    if not words:
        return "mixed", "", "" # Mặc định nếu rỗng

    # 1. Đếm từ tích cực / tiêu cực
    pos_count = len(words.intersection(pos_words))
    neg_count = len(words.intersection(neg_words))

    # 2. Xác định Sentiment
    if pos_count > 0 and neg_count > 0:
        sentiment = "mixed"
    elif pos_count > neg_count:
        sentiment = "positive"
    elif neg_count > pos_count:
        sentiment = "negative"
    else:
        sentiment = "mixed" # Không có từ khoá -> coi như mixed/neutral

    # 3. Trích xuất khía cạnh (Aspects)
    pos_aspects = []
    neg_aspects = []
    
    for aspect, keywords in aspect_kws.items():
        # Nếu có bất kỳ từ khoá nào của aspect xuất hiện trong câu
        if words.intersection(set(keywords)):
            if sentiment == "positive":
                pos_aspects.append(aspect)
            elif sentiment == "negative":
                neg_aspects.append(aspect)
            elif sentiment == "mixed":
                pos_aspects.append(aspect)
                neg_aspects.append(aspect)

    return sentiment, ",".join(pos_aspects), ",".join(neg_aspects)

def main():
    parser = argparse.ArgumentParser(description="Chạy Baseline không dùng LLM (Keyword Matching)")
    parser.add_argument("--config", required=True, help="Đường dẫn tới file cấu hình YAML")
    args = parser.parse_args()

    root = get_root()
    config_path = root / args.config
    
    if not config_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file config tại {config_path}")
        return

    config = load_config(config_path)

    # Lấy cấu hình baseline từ file yaml
    b_config = config.get("baseline_config", {})
    pos_words = set(b_config.get("positive_words", []))
    neg_words = set(b_config.get("negative_words", []))
    aspect_kws = b_config.get("aspect_keywords", {})

    city = config.get("city", "florence")
    
    print(f"====== BẮT ĐẦU CHẠY BASELINE CHO {city.upper()} ======")

    # Lặp qua tất cả các snapshot giống hệt llm.py
    for snapshot in config.get("snapshots", []):
        snapshot = str(snapshot)
        
        review_id_map_path = root / "data" / "id" / "review_id.csv"
        input_file = root / "data" / "processed" / "clean" / snapshot / "reviews.csv"
        output_file = root / "data" / "processed" / f"reviews_baseline_{snapshot}.csv"

        if not review_id_map_path.exists():
            print(f"❌ Lỗi: Không tìm thấy file ID mẫu tại {review_id_map_path}")
            return

        if not input_file.exists():
            print(f"⚠️ Cảnh báo: Không tìm thấy dữ liệu sạch tại {input_file}. Bỏ qua mốc {snapshot}.")
            continue

        # 1. Đọc danh sách ID được chọn cho LLM
        df_map = pd.read_csv(review_id_map_path)
        # Ép kiểu về string để tránh lỗi so sánh số và chữ
        df_map['Mẫu LLM'] = df_map['Mẫu LLM'].astype(str).str.strip().str.lower()
        target_ids = set(df_map[df_map['Mẫu LLM'] == 'x']['review_id'].astype(str).str.strip())

        # 2. Đọc dữ liệu sạch
        df = pd.read_csv(input_file, encoding="utf-8")
        df['id'] = df['id'].astype(str).str.strip()
        
        # Lọc đúng 1000 dòng có trong danh sách mẫu
        df_sample = df[df['id'].isin(target_ids)].copy()
        
        if len(df_sample) == 0:
            print(f" -> Không có ID mẫu nào thuộc mốc {snapshot}. Chuyển mốc tiếp theo.")
            continue
            
        print(f"[{snapshot}] Đang chạy Baseline trên {len(df_sample)} reviews...")

        # 3. Chạy phân tích Baseline
        results = df_sample["comments"].apply(
            lambda x: classify_review(x, pos_words, neg_words, aspect_kws)
        )

        # Tách tuple thành 3 cột
        df_sample[["sentiment", "aspects_positive", "aspects_negative"]] = pd.DataFrame(
            results.tolist(), index=df_sample.index
        )

        # Đổi tên cột cho giống hệt format của LLM để dễ evaluate
        df_sample = df_sample.rename(columns={
            "sentiment": "baseline_sentiment",
            "aspects_positive": "baseline_praised_aspects",
            "aspects_negative": "baseline_criticized_aspects"
        })

        # 4. Xuất kết quả (chỉ lấy các cột cần thiết giống llm.py)
        cols_to_export = ["id", "listing_id", "date", "reviewer_name", "comments", 
                          "baseline_sentiment", "baseline_praised_aspects", "baseline_criticized_aspects"]
        
        # Đề phòng trường hợp file clean thiếu cột reviewer_name
        cols_to_export = [c for c in cols_to_export if c in df_sample.columns]
        
        df_sample[cols_to_export].to_csv(output_file, index=False, encoding="utf-8")
        print(f" ✅ Đã lưu kết quả Baseline tại: {output_file}")

if __name__ == "__main__":
    main()