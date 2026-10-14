import os
import json
import argparse
import pandas as pd
import numpy as np
import google.generativeai as genai
import yaml
from typing import Dict, List, Any

def load_cache(cache_path: str) -> Dict[str, Dict[str, Any]]:
    """Tải bộ nhớ đệm đã phân tích từ trước để tránh tốn tiền API."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        cache[str(item['review_id'])] = item['analysis']
                    except Exception:
                        continue
    return cache

def save_to_cache(cache_path: str, review_id: str, analysis: Dict[str, Any]):
    """Ghi thêm một bản ghi mới vào tệp cache theo định dạng JSONL."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'review_id': str(review_id), 'analysis': analysis}, ensure_ascii=False) + '\n')

def setup_gemini_client() -> genai.GenerativeModel:
    """Khởi tạo và cấu hình client Gemini sử dụng biến môi trường."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ Không tìm thấy GEMINI_API_KEY trong biến môi trường. Vui lòng cấu hình biến môi trường trước khi chạy.")
    
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0.1,
        "response_mime_type": "application/json"
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-3.5-flash",
        generation_config=generation_config,
        system_instruction=(
            "Bạn là chuyên gia phân tích dữ liệu du lịch. Nhiệm vụ của bạn là đọc một bài đánh giá phòng Airbnb "
            "và trả về kết quả dưới dạng JSON duy nhất. JSON phải có cấu trúc chính xác như sau:\n"
            "{\n"
            '  "sentiment": "positive" hoặc "mixed" hoặc "negative",\n'
            '  "language": "Tên ngôn ngữ hoặc mã ngôn ngữ của bài viết",\n'
            '  "praised_aspects": ["danh sách khía cạnh được khen"],\n'
            '  "criticized_aspects": ["danh sách khía cạnh bị chê"]\n'
            "}"
        )
    )
    return model

def analyze_review_with_llm(model: genai.GenerativeModel, text: str, aspects: List[str]) -> Dict[str, Any]:
    """Gửi nội dung review tới Gemini API và ép cấu trúc đầu ra JSON."""
    prompt = f"""
    Hãy phân tích bài đánh giá phòng Airbnb sau đây:
    \"\"\"{text}\"\"\"

    Lưu ý cho thuộc tính 'praised_aspects' và 'criticized_aspects': Bạn chỉ được phép chọn các khía cạnh nằm trong danh sách sau: {aspects}. Nếu không có khía cạnh nào được nhắc tới, hãy để mảng rỗng [].
    """
    try:
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        return result
    except Exception as e:
        return {
            "sentiment": "mixed",
            "language": "unknown",
            "praised_aspects": [],
            "criticized_aspects": [],
            "error": str(e)
        }

def run_llm_pipeline(config_path: str, snapshot: str, skip_llm: bool = False):
    """Quy trình tổng điều phối bốc mẫu, gọi API xử lý dữ liệu cho một snapshot cụ thể."""
    # Đọc file cấu hình YAML
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    city = config.get('city', 'unknown')
    
    # 1. Định cấu hình động đường dẫn file thô và file đầu ra dựa trên file cấu hình truyền vào
    raw_reviews_path = f"data/raw/{city}/{snapshot}/reviews.csv.gz"
    output_path = f"data/processed/reviews_llm_{snapshot}.csv"
    
    # Kiểm tra xem file dữ liệu thô của snapshot này có tồn tại không
    if not os.path.exists(raw_reviews_path):
        print(f"⚠️ Cảnh báo: Không tìm thấy file dữ liệu thô tại {raw_reviews_path}. Bỏ qua snapshot {snapshot}.")
        return

    print(f"\n📅 --- TIẾN HÀNH XỬ LÝ SNAPSHOT: {snapshot} (Thành phố: {city}) ---")
    
    llm_cfg = config.get('llm_config', {})
    sample_size = llm_cfg.get('sample_size', 1000)
    cache_path = llm_cfg.get('cache_path', 'data/processed/llm_cache.jsonl')
    allowed_aspects = llm_cfg.get('aspects', ["vị trí", "sạch sẽ", "chủ nhà", "giá trị", "tiện tiện nghi", "độ ồn", "an toàn"])
    
    if skip_llm:
        print(f"⏩ Nhận cờ --skip-llm: Bỏ qua bước gọi API cho mốc {snapshot}.")
        return

    print(f"📦 Đang đọc dữ liệu thô từ: {raw_reviews_path}")
    df_reviews = pd.read_csv(raw_reviews_path, compression='gzip')
    df_reviews = df_reviews.dropna(subset=['comments'])
    
    print(f"🎲 Tiến hành lấy ngẫu nhiên {sample_size} mẫu đánh giá (Seed=42)...")
    df_sample = df_reviews.sample(n=min(sample_size, len(df_reviews)), random_state=42).copy()
    
    # Sử dụng cache chung để tối ưu chi phí cho toàn dự án
    cache = load_cache(cache_path)
    print(f"💾 Đã tải bộ nhớ đệm: Tìm thấy {len(cache)} mẫu đã phân tích trước đó.")
    
    model = setup_gemini_client()
    
    processed_records = []
    counter = 0
    
    for idx, row in df_sample.iterrows():
        r_id = str(row['id'])
        review_text = row['comments']
        
        if r_id in cache:
            analysis = cache[r_id]
        else:
            analysis = analyze_review_with_llm(model, review_text, allowed_aspects)
            save_to_cache(cache_path, r_id, analysis)
            counter += 1
            if counter % 50 == 0:
                print(f"   -> Đã gọi API xử lý thành công {counter} dòng đánh giá mới...")
        
        record = {
            'review_id': r_id,
            'listing_id': row['listing_id'],
            'date': row['date'],
            'reviewer_name': row.get('reviewer_name', ''),
            'comments': review_text,
            'llm_sentiment': analysis.get('sentiment', 'mixed'),
            'llm_language': analysis.get('language', 'unknown'),
            'llm_praised_aspects': ",".join(analysis.get('praised_aspects', [])) if isinstance(analysis.get('praised_aspects'), list) else "",
            'llm_criticized_aspects': ",".join(analysis.get('criticized_aspects', [])) if isinstance(analysis.get('criticized_aspects'), list) else ""
        }
        processed_records.append(record)
        
    df_output = pd.DataFrame(processed_records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_output.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ THÀNH CÔNG: Đã xuất dữ liệu có cấu trúc ra file: {output_path}")

def main():
    # Sử dụng argparse để nhận file config động theo đúng yêu cầu bài tập lớn
    parser = argparse.ArgumentParser(description="Quy trình phân tích LLM cho dữ liệu Airbnb.")
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn tới tệp cấu hình .yaml của thành phố")
    parser.add_argument("--skip-llm", action="store_true", help="Bỏ qua bước gọi API Gemini")
    args = parser.parse_args()
    
    print("====== 🚀 BẮT ĐẦU CHẠY MODULE LLM ANALYSIS  ======")
    
    # Kiểm tra tệp cấu hình tồn tại
    if not os.path.exists(args.config):
        print(f"❌ Sai sót: Không tìm thấy tệp cấu hình tại đường dẫn: {args.config}")
        return
        
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    all_snapshots = config.get('snapshots', [])
    print(f"📋 Tìm thấy tổng cộng {len(all_snapshots)} mốc snapshot từ file config.")
    
    # Quét qua toàn bộ các mốc snapshot được định nghĩa trong file cấu hình truyền vào
    for snapshot in all_snapshots:
        run_llm_pipeline(config_path=args.config, snapshot=str(snapshot), skip_llm=args.skip_llm)
        
    print("\n🎉 HOÀN THÀNH TOÀN BỘ MỐC THỜI GIAN CÓ TRONG CONFIG!")

if __name__ == "__main__":
    main()
