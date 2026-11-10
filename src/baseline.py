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

def clean_text(text: str) -> list:
    """Đưa về chữ thường, xoá dấu câu và trả về list các từ (giữ nguyên thứ tự để check phủ định)"""
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def classify_review(text: str, pos_words: set, neg_words: set, negations: set, aspect_kws: dict) -> tuple:
    words = clean_text(text)
    if not words:
        return "neutral", "", ""

    pos_count = 0
    neg_count = 0

    # 1. Đếm từ khen/chê (có xử lý phủ định)
    for i, word in enumerate(words):
        is_pos = word in pos_words
        is_neg = word in neg_words
        
        if not (is_pos or is_neg):
            continue

        # Check xem 2 từ trước đó có từ phủ định không
        is_negated = False
        for j in range(max(0, i-2), i):
            if words[j] in negations:
                is_negated = True
                break
                
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

    # 3. Trích xuất Aspects (Chia câu theo dấu phẩy/chấm để xét ngữ cảnh từng cụm)
    pos_aspects = set()
    neg_aspects = set()
    
    # Tách câu theo dấu chấm, phẩy, chữ "but" để xét cục bộ
    clauses = re.split(r'[.,!?;]|\bbut\b', str(text).lower())
    
    for clause in clauses:
        clause_words = set(re.sub(r"[^\w\s]", "", clause).split())
        if not clause_words:
            continue
            
        # Xem mệnh đề này chứa từ khen hay chê
        has_pos = bool(clause_words.intersection(pos_words))
        has_neg = bool(clause_words.intersection(neg_words))
        
        for aspect, keywords in aspect_kws.items():
            if clause_words.intersection(set(keywords)):
                # Nếu mệnh đề này chứa từ khen -> Aspect là được khen
                if has_pos and not has_neg:
                    pos_aspects.add(aspect)
                # Nếu chứa từ chê -> Aspect bị chê
                elif has_neg and not has_pos:
                    neg_aspects.add(aspect)
                # Nếu mệnh đề vừa khen vừa chê, hoặc sentiment tổng là mixed -> cho vào cả 2
                elif sentiment == "mixed":
                    pos_aspects.add(aspect)
                    neg_aspects.add(aspect)

    return sentiment, ",".join(sorted(list(pos_aspects))), ",".join(sorted(list(neg_aspects)))

def main():
    parser = argparse.ArgumentParser(description="Chạy Baseline không dùng LLM (Keyword Matching)")
    parser.add_argument("--config", required=True, help="Đường dẫn tới file cấu hình YAML")
    args = parser.parse_args()

    root = get_root()
    config = load_config(root / args.config)

    b_config = config.get("baseline_config", {})
    pos_words = set(b_config.get("positive_words", []))
    neg_words = set(b_config.get("negative_words", []))
    negations = set(b_config.get("negation_words", []))
    aspect_kws = b_config.get("aspect_keywords", {})
    city = config.get("city", "florence")

    for snapshot in config.get("snapshots", []):
        snapshot = str(snapshot)
        review_id_map_path = root / "data" / "id" / "review_id.csv"
        input_file = root / "data" / "processed" / "clean" / snapshot / "reviews.csv"
        output_file = root / "data" / "processed" / f"reviews_baseline_{snapshot}.csv"

        if not review_id_map_path.exists() or not input_file.exists():
            continue

        # Lấy tập ID mẫu
        df_map = pd.read_csv(review_id_map_path)
        df_map['Mẫu LLM'] = df_map['Mẫu LLM'].astype(str).str.strip().str.lower()
        target_ids = set(df_map[df_map['Mẫu LLM'] == 'x']['review_id'].astype(str).str.strip())

        df = pd.read_csv(input_file, encoding="utf-8")
        df['id'] = df['id'].astype(str).str.strip()
        
        df_sample = df[df['id'].isin(target_ids)].copy()
        if len(df_sample) == 0:
            continue
            
        print(f"[{snapshot}] Đang chạy Baseline trên {len(df_sample)} reviews...")

        results = df_sample["comments"].apply(
            lambda x: classify_review(x, pos_words, neg_words, negations, aspect_kws)
        )

        df_sample[["sentiment", "aspects_positive", "aspects_negative"]] = pd.DataFrame(
            results.tolist(), index=df_sample.index
        )

        df_sample = df_sample.rename(columns={
            "sentiment": "baseline_sentiment",
            "aspects_positive": "baseline_praised_aspects",
            "aspects_negative": "baseline_criticized_aspects"
        })

        cols_to_export = ["id", "listing_id", "date", "reviewer_name", "comments", 
                          "baseline_sentiment", "baseline_praised_aspects", "baseline_criticized_aspects"]
        cols_to_export = [c for c in cols_to_export if c in df_sample.columns]
        
        df_sample[cols_to_export].to_csv(output_file, index=False, encoding="utf-8")
        print(f" ✅ Đã lưu kết quả Baseline tại: {output_file}")

if __name__ == "__main__":
    main()