import argparse
import yaml
import pandas as pd
import re
from pathlib import Path

def get_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ==========================================
# KHAI BÁO TỪ ĐIỂN TRỰC TIẾP TRONG CODE
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
    "uncomfortable", "issue", "problem", "hard", "difficult", "avoid", "loud", 
    "annoying", "bug", "mold", "cockroach"
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
# ==========================================

def clean_text(text: str) -> list:
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def classify_review(text: str) -> tuple:
    words = clean_text(text)
    if not words:
        return "neutral", "", ""

    pos_count = 0
    neg_count = 0

    # 1. Đếm từ khen/chê (có xử lý phủ định)
    for i, word in enumerate(words):
        is_pos = word in POSITIVE_WORDS
        is_neg = word in NEGATIVE_WORDS
        
        if not (is_pos or is_neg):
            continue

        is_negated = any(words[j] in NEGATION_WORDS for j in range(max(0, i-2), i))
                
        if is_pos:
            if is_negated: neg_count += 1
            else: pos_count += 1
        elif is_neg:
            if is_negated: pos_count += 1
            else: neg_count += 1

    # 2. Xác định sentiment tổng quát
    if pos_count == 0 and neg_count == 0:
        sentiment = "neutral"
    elif pos_count > 0 and neg_count > 0:
        sentiment = "mixed"
    elif pos_count > neg_count:
        sentiment = "positive"
    else:
        sentiment = "negative"

    # 3. Trích xuất Aspects
    pos_aspects = set()
    neg_aspects = set()
    
    clauses = re.split(r'[.,!?;]|\bbut\b', str(text).lower())
    for clause in clauses:
        clause_words = set(re.sub(r"[^\w\s]", "", clause).split())
        if not clause_words:
            continue
            
        has_pos = bool(clause_words.intersection(POSITIVE_WORDS))
        has_neg = bool(clause_words.intersection(NEGATIVE_WORDS))
        
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if clause_words.intersection(set(keywords)):
                if has_pos and not has_neg:
                    pos_aspects.add(aspect)
                elif has_neg and not has_pos:
                    neg_aspects.add(aspect)
                elif sentiment == "mixed":
                    pos_aspects.add(aspect)
                    neg_aspects.add(aspect)

    return sentiment, ",".join(sorted(list(pos_aspects))), ",".join(sorted(list(neg_aspects)))


def main():
    parser = argparse.ArgumentParser(description="Chạy Baseline (Keyword Matching) trên tập Nhãn tay")
    parser.add_argument("--config", required=True, help="Đường dẫn tới file cấu hình YAML")
    args = parser.parse_args()

    root = get_root()
    config = load_config(root / args.config)
    city = config.get("city", "florence")

    print(f"====== CHẠY BASELINE CHO {city.upper()} ======")

    for snapshot in config.get("snapshots", []):
        snapshot = str(snapshot)
        review_id_map_path = root / "data" / "id" / "review_id.csv"
        input_file = root / "data" / "processed" / "clean" / snapshot / "reviews.csv"
        output_file = root / "data" / "processed" / f"reviews_baseline_{snapshot}.csv"

        if not review_id_map_path.exists() or not input_file.exists():
            continue

        # Lấy tập ID mẫu DỰA TRÊN CỘT NHÃN TAY
        df_map = pd.read_csv(review_id_map_path)
        df_map['Nhãn tay'] = df_map['Nhãn tay'].astype(str).str.strip().str.lower()
        
        # CHỈ LẤY NHỮNG DÒNG CÓ NHÃN TAY == 'x'
        target_ids = set(df_map[df_map['Nhãn tay'] == 'x']['review_id'].astype(str).str.strip())
        
        if len(target_ids) == 0:
            print("❌ Không tìm thấy ID nào có 'Nhãn tay' = 'x' trong file ID.")
            return

        df = pd.read_csv(input_file, encoding="utf-8")
        df['id'] = df['id'].astype(str).str.strip()
        
        # Lọc data
        df_sample = df[df['id'].isin(target_ids)].copy()
        if len(df_sample) == 0:
            continue
            
        print(f"[{snapshot}] Đang chạy Baseline trên {len(df_sample)} reviews (Nhãn tay)...")

        results = df_sample["comments"].apply(classify_review)

        df_sample[["sentiment", "aspects_positive", "aspects_negative"]] = pd.DataFrame(
            results.tolist(), index=df_sample.index
        )

        df_sample = df_sample.rename(columns={
            "sentiment": "baseline_sentiment",
            "aspects_positive": "baseline_praised_aspects",
            "aspects_negative": "baseline_criticized_aspects"
        })

        cols_to_export = ["id", "comments", "baseline_sentiment", "baseline_praised_aspects", "baseline_criticized_aspects"]
        cols_to_export = [c for c in cols_to_export if c in df_sample.columns]
        
        df_sample[cols_to_export].to_csv(output_file, index=False, encoding="utf-8")
        print(f" ✅ Đã lưu kết quả Baseline tại: {output_file}")

if __name__ == "__main__":
    main()