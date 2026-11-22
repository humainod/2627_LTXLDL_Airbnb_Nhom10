import argparse
import yaml
import pandas as pd
import re
from pathlib import Path

# ==========================================
# 1. KHAI BÁO TỪ ĐIỂN (LIỆT KÊ NGAY TRONG CODE)
# ==========================================
POSITIVE_WORDS = {
    "great", "good", "clean", "nice", "perfect", "amazing", "comfortable", 
    "excellent", "wonderful", "friendly", "convenient", "beautiful", "love", 
    "best", "enjoy", "helpful", "spacious", "recommend", "lovely", "fantastic", 
    "awesome", "super", "kind", "responsive"
}

NEGATIVE_WORDS = {
    "bad", "dirty", "noisy", "terrible", "awful", "smelly", "rude", "poor", 
    "disappointed", "worst", "hate", "broken", "cold", "smell", "stain", 
    "uncomfortable", "issue", "problem", "hard", "difficult", "avoid", 
    "loud", "annoying", "bug", "mold", "cockroach"
}

NEGATION_WORDS = {
    "not", "no", "never", "dont", "don't", "didnt", "didn't", "isnt", "isn't", 
    "wasnt", "wasn't", "cant", "can't", "couldnt", "couldn't", "wont", "won't"
}

ASPECT_KEYWORDS = {
    "vị trí": ["location", "near", "center", "subway", "walk", "close", "distance", "area", "neighborhood", "station", "transport", "accessible"],
    "sạch sẽ": ["clean", "dirty", "smell", "tidy", "mess", "hygiene", "dust", "stain", "spotless", "fresh"],
    "chủ nhà": ["host", "owner", "communication", "responsive", "helpful", "friendly", "reply", "message", "support", "welcome"],
    "giá trị": ["price", "value", "money", "cheap", "expensive", "cost", "worth", "pay", "budget"],
    "tiện nghi": ["amenity", "amenities", "wifi", "kitchen", "bathroom", "shower", "bed", "tv", "air conditioning", "heater", "furniture", "pool", "gym"],
    "độ ồn": ["noisy", "quiet", "loud", "street", "traffic", "neighbor", "night", "sound", "peaceful", "silent"],
    "an toàn": ["safe", "dangerous", "security", "scary", "lock", "neighborhood"]
}

# Chuyển list thành set để tìm kiếm nhanh hơn (O(1) thay vì O(n))
POS_SET = set(POSITIVE_WORDS)
NEG_SET = set(NEGATIVE_WORDS)
NEGATION_SET = set(NEGATION_WORDS)
ASPECTS_DICT = {k: set(v) for k, v in ASPECT_KEYWORDS.items()}

# ==========================================
# 2. CÁC HÀM XỬ LÝ
# ==========================================
def get_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def classify_review(text: str) -> tuple:
    """Phân loại 1 review dựa trên từ điển có sẵn"""
    if pd.isna(text):
        return "mixed", "", ""

    # 1. Làm sạch và tách từ
    text_str = str(text).lower()
    text_clean = re.sub(r"[^\w\s]", "", text_str)
    words = text_clean.split()
    
    if not words:
        return "mixed", "", ""

    pos_count = 0
    neg_count = 0

    # 2. Đếm từ khen/chê (xử lý phủ định)
    for i, word in enumerate(words):
        is_pos = word in POS_SET
        is_neg = word in NEG_SET
        
        if not (is_pos or is_neg):
            continue

        # Check 2 từ phía trước xem có từ phủ định không (VD: "not clean")
        is_negated = False
        for j in range(max(0, i-2), i):
            if words[j] in NEGATION_SET:
                is_negated = True
                break
                
        if is_pos:
            if is_negated: neg_count += 1
            else: pos_count += 1
        elif is_neg:
            if is_negated: pos_count += 1
            else: neg_count += 1

    # 3. Xác định cảm xúc tổng quát
    if pos_count == 0 and neg_count == 0:
        sentiment = "mixed"
    elif pos_count > 0 and neg_count > 0:
        sentiment = "mixed"
    elif pos_count > neg_count:
        sentiment = "positive"
    else:
        sentiment = "negative"

    # 4. Trích xuất Khía cạnh (Aspects) theo từng mệnh đề
    pos_aspects = set()
    neg_aspects = set()
    
    # Tách câu theo dấu câu và chữ "but"
    clauses = re.split(r'[.,!?;]|\bbut\b', text_str)
    
    for clause in clauses:
        clause_words = set(re.sub(r"[^\w\s]", "", clause).split())
        if not clause_words:
            continue
            
        has_pos = bool(clause_words.intersection(POS_SET))
        has_neg = bool(clause_words.intersection(NEG_SET))
        
        for aspect, keywords in ASPECTS_DICT.items():
            if clause_words.intersection(keywords):
                # Nếu mệnh đề chứa từ khen -> gán khía cạnh được khen
                if has_pos and not has_neg:
                    pos_aspects.add(aspect)
                # Nếu chứa từ chê -> gán khía cạnh bị chê
                elif has_neg and not has_pos:
                    neg_aspects.add(aspect)
                # Nếu mệnh đề phức tạp -> gán cả 2
                elif sentiment == "mixed":
                    pos_aspects.add(aspect)
                    neg_aspects.add(aspect)

    return sentiment, ",".join(sorted(list(pos_aspects))), ",".join(sorted(list(neg_aspects)))

# ==========================================
# 3. HÀM CHẠY CHÍNH
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Chạy Baseline (Keyword Matching) trên 1000 mẫu LLM")
    parser.add_argument("--config", required=True, help="Đường dẫn tới file cấu hình YAML")
    args = parser.parse_args()

    root = get_root()
    config = load_config(root / args.config)
    city = config.get("city", "florence")

    print(f"====== BẮT ĐẦU CHẠY BASELINE CHO {city.upper()} ======")
    
    review_id_map_path = root / "data" / "id" / "review_id.csv"
    if not review_id_map_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file ID mẫu tại {review_id_map_path}")
        return

    # Đọc file ID và LỌC CHUẨN XÁC THEO CỘT 'Mẫu LLM' (Giống hệt llm.py)
    df_map = pd.read_csv(review_id_map_path)
    df_map['Mẫu LLM'] = df_map['Mẫu LLM'].astype(str).str.strip().str.lower()
    target_ids = set(df_map[df_map['Mẫu LLM'] == 'x']['review_id'].astype(str).str.strip())
    
    print(f"🔍 Tìm thấy {len(target_ids)} ID được chọn làm Mẫu LLM.")

    # Lặp qua các snapshot
    for snapshot in config.get("snapshots", []):
        snapshot = str(snapshot)
        input_file = root / "data" / "processed" / "clean" / snapshot / "reviews.csv"
        output_file = root / "data" / "processed" / f"reviews_baseline_{snapshot}.csv"

        if not input_file.exists():
            print(f"⚠️ Không tìm thấy dữ liệu sạch tại {input_file}. Bỏ qua mốc {snapshot}.")
            continue

        # Đọc dữ liệu sạch
        df = pd.read_csv(input_file, encoding="utf-8")
        df['id'] = df['id'].astype(str).str.strip()
        
        # Lọc đúng tập ID Mẫu LLM
        df_sample = df[df['id'].isin(target_ids)].copy()
        
        if len(df_sample) == 0:
            print(f" -> Không có ID mẫu nào thuộc mốc {snapshot}. Chuyển mốc tiếp theo.")
            continue
            
        print(f"[{snapshot}] Đang chạy Baseline trên {len(df_sample)} reviews...")

        # Chạy phân tích
        results = df_sample["comments"].apply(classify_review)

        # Tách tuple thành 3 cột
        df_sample[["sentiment", "aspects_positive", "aspects_negative"]] = pd.DataFrame(
            results.tolist(), index=df_sample.index
        )

        # Đổi tên cột chuẩn format
        df_sample = df_sample.rename(columns={
            "sentiment": "baseline_sentiment",
            "aspects_positive": "baseline_praised_aspects",
            "aspects_negative": "baseline_criticized_aspects"
        })

        # Chỉ lấy các cột cần thiết để xuất
        cols_to_export = ["id", "listing_id", "date", "reviewer_name", "comments", 
                          "baseline_sentiment", "baseline_praised_aspects", "baseline_criticized_aspects"]
        cols_to_export = [c for c in cols_to_export if c in df_sample.columns]
        
        df_sample[cols_to_export].to_csv(output_file, index=False, encoding="utf-8")
        print(f" ✅ Đã lưu kết quả Baseline tại: {output_file}\n")

if __name__ == "__main__":
    main()