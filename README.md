# 2627_LTXLDL_Airbnb_Nhom10
Project Lập trình xử lý dữ liệu - Inside Airbnb - Nhóm 10 - UET

# Installation
Đầu tiên hãy tải pip qua terminal nếu không có. Sau đấy chuyển directory của terminal qua folder của project. 

Ví dụ:

```shell
cd C:/Documents/<Project_name>
```

Sau đấy hãy chạy:

```shell
pip install -r requirements.txt
```

Nó sẽ xóa và tải lại những packages trên máy bạn mà cũng nằm trong requirements.txt

# How to run
Chạy code bên dưới trong terminal của bạn

```shell
python run_pipeline.py --config configs/<city_name>.yaml
```

Thay <city_name> bằng tên thành phố mà bạn muốn xem

# API Key's variable's name

Nếu dùng, api key sẽ được gọi với tên biến dưới đây

```shell
GEMINI_API_KEY
```

Lưu api key của bạn vào trong file .env với tên biến như trên

```shell
GEMINI_API_KEY = <API KEY>
```

# Where to change city, snapshots

Hãy đặt file configs của bạn vào folder configs với tên <city_name>.yaml
Thay <city_name> bằng tên thành phố tương ứng

# Required properties in config
Download

```shell
city: <city_name>
country: <country_name>
region: <region_name>
currency: <curency_name>
data_root: <country>/<region>/<city>
snapshots: [] #Điền ngày tháng năm của từng snapshots vào trong list bằng str với dạng "YYYY/MM/DD"
url_template: "https://data.insideairbnb.com/{country}/{region}/{city}/{snapshot}/data/{filename}"
files:
  - listings.csv.gz
  - calendar.csv.gz
  - reviews.csv.gz
```
Ngoài các keys có placeholder như city, country hoặc dạng list như snapshots, những mục còn lại không cần thay đổi

QA

```shell
neccesary_values:
  #Các cột nằm trong các list sẽ được gắn cờ tương ứng trong file flags
  kpi:  {"listings": ["price", "estimated_revenue_l365d"], "calendar": [], "reviews": []} 
  llm: {"listings": [], "calendar": [], "reviews": []}
force_outlier_threshold:
  minimum_nights: [0, 365]
```
Tương tự như bên trên, các phần list của key neccesary_values có thể thay đổi và thêm tên cột bạn muốn gắn cờ vào

LLM

```shell
llm_config:
  sample_size: 1000                    # Số lượng mẫu review bốc ngẫu nhiên để phân tích
  cache_path: "data/processed/llm_cache.jsonl"
  output_path: "data/processed/reviews_llm.csv"
  aspects: ["vị trí", "sạch sẽ", "chủ nhà", "giá trị", "tiện nghi", "độ ồn", "an toàn"]
```
Những keys ở phần này không cần thay đổi
