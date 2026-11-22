import os
import json
import argparse
import time
import pandas as pd
import numpy as np
import google.generativeai as genai
import yaml
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def load_cache(cache_path: str) -> Dict[str, Dict[str, Any]]:
    """Tải bộ nhớ đệm đã phân tích từ trước để tránh tốn tiền API."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        cache[str(item['review_id']).strip()] = item['analysis']
                    except Exception:
                        continue
    return cache

def save_batch_to_cache(cache_path: str, batch_results: List[Dict[str, Any]]):
    """Ghi thêm một danh sách kết quả lô vào tệp cache định dạng JSONL."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'a', encoding='utf-8') as f:
        for item in batch_results:
            f.write(json.dumps({'review_id': str(item['review_id']).strip(), 'analysis': item['analysis']}, ensure_ascii=False) + '\n')

def setup_gemini_client() -> genai.GenerativeModel:
    """Khởi tạo và cấu hình client Gemini sử dụng biến môi trường."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Không tìm thấy GEMINI_API_KEY trong biến môi trường. Vui lòng cấu hình trước khi chạy.")
    
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0.1,
        "response_mime_type": "application/json"
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-3.5-flash-lite",
        generation_config=generation_config,
        system_instruction=(
            "Bạn là chuyên gia phân tích dữ liệu du lịch. Nhiệm vụ của bạn là nhận vào một mảng JSON chứa các đánh giá phòng Airbnb "
            "và phân tích toàn bộ chúng, trả về một mảng JSON duy nhất chứa kết quả cấu trúc của từng bài đánh giá. "
            "Cấu trúc JSON đầu ra BẮT BUỘC phải là một mảng có định dạng chính xác như sau:\n"
            "[\n"
            "  {\n"
            '    "id": "chuỗi ID nhận vào từ danh sách",\n'
            '    "sentiment": "positive" hoặc "mixed" hoặc "negative",\n'
            '    "language": "Tên ngôn ngữ hoặc mã ngôn ngữ của bài viết",\n'
            '    "praised_aspects": ["danh sách khía cạnh được khen"],\n'
            '    "criticized_aspects": ["danh sách khía cạnh bị chê"]\n'
            "  }\n"
            "]"
        )
    )
    return model

def calculate_token_cost(prompt_text: str, response_text: str = None) -> float:
    """Tính toán chi phí dựa trên số lượng từ quy đổi xấp xỉ sang mã tokens (1 từ ~ 1.5 tokens)."""
    input_tokens = len(prompt_text.split()) * 1.5
    cost_input = (input_tokens / 1_000_000) * 0.30
    
    cost_output = 0.0
    if response_text:
        output_tokens = len(response_text.split()) * 1.5
        cost_output = (output_tokens / 1_000_000) * 2.50
        
    return cost_input + cost_output

def analyze_batch_with_retry(model: genai.GenerativeModel, batch_items: List[Dict[str, str]], aspects: List[str], max_retries: int = 5) -> tuple:
    """Gửi lô đánh giá lên Gemini API kèm cơ chế Exponential Backoff để chống lỗi hạn mức 429."""
    input_data = [{"id": item["id"], "text": item["text"]} for item in batch_items]
    
    prompt = f"""
    Hãy phân tích danh sách các bài đánh giá phòng Airbnb dưới đây:
    {json.dumps(input_data, ensure_ascii=False)}

    Lưu ý cho thuộc tính 'praised_aspects' và 'criticized_aspects': Bạn chỉ được phép chọn các khía cạnh nằm trong danh sách sau: {aspects}. Nếu không có khía cạnh nào được nhắc tới, hãy để mảng rỗng [].
    """
    
    delay = 4  
    initial_cost = calculate_token_cost(prompt, response_text=None)
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            results_list = json.loads(response.text)
            exact_cost = calculate_token_cost(prompt, response.text)
            return {str(res["id"]).strip(): res for res in results_list if "id" in res}, exact_cost
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "quota" in err_msg.lower():
                print(f"    Lỗi 429 (Hết hạn mức). Đang nghỉ tự động {delay} giây trước khi thử lại lần {attempt + 1}/{max_retries}...")
                time.sleep(delay)
                delay *= 2  
            else:
                print(f"    Gặp lỗi API: {err_msg}. Đang thử lại sau 3 giây...")
                time.sleep(3)
                
    return {}, initial_cost

def run_llm_pipeline(config_path: str, snapshot: str, skip_llm: bool = False) -> float:
    """Quy trình trích xuất thông tin dựa theo danh sách mẫu review_id.csv cố định."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Khai báo đường dẫn theo đúng cấu trúc của nhóm
    base_dir = Path(__file__).resolve().parents[1]
    city_name = config.get('city', 'florence')
    
    review_id_map_path = base_dir / "data" / "id" / "review_id.csv"
    raw_reviews_path = base_dir / "data" / "raw" / city_name / snapshot / "reviews.csv.gz"
    output_path = base_dir / "data" / "processed" / f"reviews_llm_{snapshot}.csv"
    
    if not review_id_map_path.exists():
        print(f"❌ Lỗi: Không tìm thấy tệp ID mục tiêu tại {review_id_map_path}")
        return 0.0
    if not raw_reviews_path.exists():
        print(f"⚠️ Cảnh báo: Không tìm thấy dữ liệu thô tại {raw_reviews_path}. Bỏ qua mốc {snapshot}.")
        return 0.0

    print(f"\n --- TIẾN HÀNH XỬ LÝ SNAPSHOT: {snapshot} ---")
    
    llm_cfg = config.get('llm', config.get('llm_config', {}))
    cache_path = llm_cfg.get('cache', llm_cfg.get('cache_path', 'data/processed/llm_cache.jsonl'))
    allowed_aspects = llm_cfg.get('aspects', ["vị trí", "sạch sẽ", "chủ nhà", "giá trị", "tiện nghi", "độ ồn", "an toàn"])

    # 1. Lọc các review_id được đánh dấu 'x' tại cột 'Mẫu LLM'
    print(f"🔍 Đang lọc review_id từ tệp chỉ định: {review_id_map_path}")
    df_map = pd.read_csv(review_id_map_path)
    
    # Ép kiểu và loại bỏ khoảng trắng để lọc chính xác
    df_map['Mẫu LLM'] = df_map['Mẫu LLM'].astype(str).str.strip().str.lower()
    df_target_ids = df_map[df_map['Mẫu LLM'] == 'x']['review_id'].astype(str).str.strip().unique()
    target_ids_set = set(df_target_ids)
    
    print(f" Tìm thấy tổng cộng {len(target_ids_set)} ID được chọn làm Mẫu LLM cho toàn dự án.")

    # 2. Đọc tệp reviews thô và lọc ra các dòng khớp với tập ID mục tiêu
    print(f" Đang tải và trích lọc dữ liệu thô từ: {raw_reviews_path}")
    df_raw = pd.read_csv(raw_reviews_path, compression='gzip')
    df_raw = df_raw.dropna(subset=['comments'])
    df_raw['id'] = df_raw['id'].astype(str).str.strip()
    
    # Giữ lại các dòng thuộc snapshot này nằm trong danh sách mẫu LLM
    df_sample = df_raw[df_raw['id'].isin(target_ids_set)].copy()
    print(f" Tìm thấy {len(df_sample)} dòng mẫu khớp trong mốc {snapshot}.")
    
    if len(df_sample) == 0:
        print(f" Không có ID mẫu nào thuộc mốc {snapshot}. Chuyển mốc tiếp theo.")
        return 0.0

    cache = load_cache(str(base_dir / cache_path))
    print(f" Đã tải bộ nhớ đệm: Tìm thấy {len(cache)} mẫu đã phân tích trước đó.")
    
    processed_records = []
    missing_items = []
    snapshot_api_cost = 0.0

    # Phân loại: Dòng nào có trong cache thì lấy luôn, dòng nào thiếu thì gom lô xử lý mới
    for idx, row in df_sample.iterrows():
        r_id = str(row['id']).strip()
        if r_id in cache:
            analysis = cache[r_id]
            processed_records.append({
                'review_id': r_id, 'listing_id': row['listing_id'], 'date': row['date'],
                'reviewer_name': row.get('reviewer_name', ''), 'comments': row['comments'],
                'llm_sentiment': analysis.get('sentiment', 'mixed'), 'llm_language': analysis.get('language', 'unknown'),
                'llm_praised_aspects': ",".join(analysis.get('praised_aspects', [])) if isinstance(analysis.get('praised_aspects'), list) else "",
                'llm_criticized_aspects': ",".join(analysis.get('criticized_aspects', [])) if isinstance(analysis.get('criticized_aspects'), list) else ""
            })
        else:
            missing_items.append({'id': r_id, 'text': row['comments'], 'row_data': row})

    # Đồng bộ kích thước lô 15 comments/request
    batch_size = 15
    total_missing = len(missing_items)
    
    if total_missing > 0:
        if skip_llm:
            print(f" Nhận cờ --skip-llm: Bỏ qua bước gọi API cho mốc {snapshot}.")
        else:
            model = setup_gemini_client()
            print(f" Đang gửi xử lý dữ liệu mới ({total_missing} dòng) theo lô Batch {batch_size}...")
            
            for i in range(0, total_missing, batch_size):
                current_batch = missing_items[i : i + batch_size]
                print(f" Đang xử lý lô từ dòng {i+1} đến {min(i + batch_size, total_missing)} / {total_missing}...")
                
                # Gửi cụm lô lên API kèm cơ chế retry thông minh (Exponential Backoff)
                batch_analyses, cost = analyze_batch_with_retry(model, current_batch, allowed_aspects)
                snapshot_api_cost += cost
                
                cache_to_save = []
                for item in current_batch:
                    r_id = item['id']
                    row = item['row_data']
                    
                    # Nếu API lỗi hoặc thiếu ID, gán giá trị mặc định tránh sập luồng
                    analysis = batch_analyses.get(r_id, {
                        "sentiment": "mixed", "language": "unknown", "praised_aspects": [], "criticized_aspects": []
                    })
                    
                    cache_to_save.append({'review_id': r_id, 'analysis': analysis})
                    
                    processed_records.append({
                        'review_id': r_id,
                        'listing_id': row['listing_id'],
                        'date': row['date'],
                        'reviewer_name': row.get('reviewer_name', ''),
                        'comments': item['text'],
                        'llm_sentiment': analysis.get('sentiment', 'mixed'),
                        'llm_language': analysis.get('language', 'unknown'),
                        'llm_praised_aspects': ",".join(analysis.get('praised_aspects', [])) if isinstance(analysis.get('praised_aspects'), list) else "",
                        'llm_criticized_aspects': ",".join(analysis.get('criticized_aspects', [])) if isinstance(analysis.get('criticized_aspects'), list) else ""
                    })
                
                # Lưu lưu trữ lô này ngay lập tức vào bộ nhớ đệm (Cache) để bảo toàn tiến độ
                save_batch_to_cache(str(base_dir / cache_path), cache_to_save)
                
                # Nghỉ 12 giây giữa các đợt gửi batch để kiểm soát tốt Rate Limit (RPM / TPM)
                if i + batch_size < total_missing:
                    time.sleep(12)
                
    # Xuất file kết quả sau phân tích ra thư mục dữ liệu đã xử lý
    df_output = pd.DataFrame(processed_records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_output.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ THÀNH CÔNG: Đã xuất kết quả cấu trúc ra: {output_path}")
    if total_missing > 0 and not skip_llm:
        print(f" Chi phí gọi API thật cho đợt bốc mẫu này: ${snapshot_api_cost:.5f} USD")
        
    return snapshot_api_cost

def main():
    parser = argparse.ArgumentParser(description="Quy trình phân tích LLM dựa trên review_id.csv mẫu.")
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn tới tệp cấu hình .yaml")
    parser.add_argument("--skip-llm", action="store_true", help="Bỏ qua bước gọi API Gemini")
    args = parser.parse_args()
    
    print("======  BẮT ĐẦU CHẠY MODULE LLM ANALYSIS  ======")
    
    if not os.path.exists(args.config):
        print(f" Lỗi: Không tìm thấy tệp cấu hình tại: {args.config}")
        return
        
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    all_snapshots = config.get('snapshots', [])
    print(f" Tìm thấy {len(all_snapshots)} mốc snapshot từ file config.")
    
    total_sample_run_cost = 0.0
    total_city_raw_rows = 0
    
    base_dir = Path(__file__).resolve().parent.parent
    city_name = config.get('city', 'florence')
    
    for snapshot in all_snapshots:
        cost = run_llm_pipeline(config_path=args.config, snapshot=str(snapshot), skip_llm=args.skip_llm)
        total_sample_run_cost += cost
        
        # Đếm số lượng dòng thô thực tế để tính toán quy đổi chi phí ngoại suy
        raw_reviews_path = base_dir / "data" / "raw" / city_name / str(snapshot) / "reviews.csv.gz"
        if raw_reviews_path.exists():
            try:
                df_count = pd.read_csv(raw_reviews_path, compression='gzip', usecols=['id'])
                total_city_raw_rows += len(df_count)
            except Exception:
                continue

    # Thuật toán tính toán ngoại suy chi phí tổng thể (Extrapolation) cho toàn bộ thành phố
    # Giả định trung bình 1 dòng ~ 1.5 token input + output quy đổi ra đơn giá Flash
    AVERAGE_UNIT_COST_USD = 0.0004
    estimated_total_city_cost = total_city_raw_rows * AVERAGE_UNIT_COST_USD

    print("\n====================  TỔNG KẾT CHI PHÍ LLM ====================")
    print(f" Chi phí API thực tế đã trả cho đợt chạy mẫu: ${total_sample_run_cost:.5f} USD")
    print(f" Tổng số lượng review gốc toàn thành phố {city_name.upper()} (4 mốc): {total_city_raw_rows:,} dòng")
    print(f" ƯỚC TÍNH CHI PHÍ NẾU XỬ LÝ TOÀN BỘ THÀNH PHỐ: ${estimated_total_city_cost:.2f} USD")
    print("=========================================================================")

if __name__ == "__main__":
    main()