# AI_USAGE.md

## 1. Công cụ đã dùng

AI: GitHub Copilot
Usage: Gợi ý code và hỗ trợ fix bug trực tiếp trên VS Code khi làm các file qa.py, kpi.py, baseline.py.
Detail: Dùng để hoàn thiện nhanh các đoạn xử lý bảng bằng Pandas, tối ưu vòng lặp tìm từ khóa và tìm nguyên nhân báo lỗi kiểu dữ liệu.

AI: Gemini API (gemini-3.5-flash-lite)
Usage: Tích hợp vào file llm.py để chạy phân tích tự động tập mẫu dữ liệu đánh giá (reviews).
Detail: Gửi các bình luận lên API theo từng lô để phân loại sắc thái (positive, mixed, negative), nhận diện ngôn ngữ và lọc ra các khía cạnh khen/chê.

AI: ChatGPT / Gemini Web
Usage: Hỏi đáp về cách chia cấu trúc thư mục, tối ưu prompt, viết code vẽ biểu đồ (viz.py) và đọc hiểu code của các thành viên khác.
Detail: Nhờ AI gợi ý cách lưu cache dạng JSONL để không bị mất dữ liệu khi rớt mạng hoặc hết quota, cũng như cách vẽ khung tọa độ các vùng lên bản đồ.

---

## 2. Prompt then chốt (5-10): bước nào, vào tệp nào

Prompt 1: Bước làm sạch dữ liệu -> Tệp src/qa.py
Nội dung: Viết các hàm bằng Pandas để chuyển cột giá tiền dạng chuỗi có dấu $ sang số thực, lọc ngày tháng bị sai định dạng và gắn cờ các dòng bị thiếu dữ liệu hoặc ngoại lai theo phương pháp IQR thay vì xóa bỏ.

Prompt 2: Bước tính toán KPI -> Tệp src/kpi.py
Nội dung: Tính tỷ lệ lấp đầy, giá trung bình ngày (ADR) và doanh thu trên mỗi phòng (RevPAR) từ bảng calendar và listings, sau đó gộp thêm tỷ lệ review tích cực từ kết quả LLM và in bảng so sánh trước và sau khi lọc ngoại lai.

Prompt 3: Bước gọi LLM phân tích review -> Tệp src/llm.py
Nội dung: Đóng vai người phân tích dữ liệu du lịch, đọc danh sách các bình luận Airbnb truyền vào và trả về định dạng JSON gồm id, sentiment (chỉ chọn positive, mixed hoặc negative), language, praised_aspects và criticized_aspects nằm trong 7 khía cạnh nhóm đã quy định.

Prompt 4: Bước tối ưu gọi API và lưu cache -> Tệp src/llm.py
Nội dung: Viết đoạn code chia danh sách review thành từng lô 15 câu để gọi Gemini API, nếu gặp lỗi 429 giới hạn lượt gọi thì tự động tạm dừng tăng dần thời gian rồi thử lại, đồng thời lưu nối tiếp kết quả vào file llm_cache.jsonl ngay sau mỗi lô.

Prompt 5: Bước làm phương pháp đối chứng -> Tệp src/baseline.py
Nội dung: Viết hàm đếm từ khóa khen/chê không dùng AI để làm baseline, có kiểm tra từ phủ định (như not, never) đứng trước tối đa 2 từ và tách câu theo dấu câu hoặc chữ but để bắt đúng khía cạnh.

Prompt 6: Bước vẽ biểu đồ -> Tệp src/viz.py
Nội dung: Dùng matplotlib vẽ bản đồ phân bố tọa độ các phòng thuê ở mốc cuối cùng, kèm theo khung chữ nhật nét đứt đánh dấu ranh giới 8 vùng phân tích lấy tọa độ từ file region_coordinate.csv.

---

## 3. AI sai ở đâu (ví dụ thật)

AI: GitHub Copilot
Usage: Fix bug hàm findData() trong tệp src/qa.py.
Detail: AI gợi ý dùng .loc để gán cả một bảng DataFrame con vào từng ô của bảng needed_data khiến code báo lỗi và các ô bị trống dữ liệu. Phát hiện bằng cách chạy thử hàm và in bảng ra kiểm tra thấy dữ liệu không vào. Nhóm sửa lại bằng cách đổi từ .loc sang .at để ép gán trọn vẹn bảng con vào đúng vị trí ô.

AI: ChatGPT / Gemini Web
Usage: Viết hàm vẽ biểu đồ đường xu hướng giá trong tệp src/viz.py.
Detail: AI tự ý viết code đọc cột tên là avg_adr từ file thống kê, trong khi file kpi.py của nhóm xuất ra hai cột tên là avg_adr_raw và avg_adr_clean. Phát hiện lúc chạy thử viz.py thì terminal báo cảnh báo không tìm thấy cột dữ liệu hợp lệ nên không xuất được ảnh. Nhóm mở lại file kpi_summary để xem đúng tên cột rồi sửa lại trong viz.py.

AI: Gemini API (gemini-3.5-flash-lite)
Usage: Phân loại cảm xúc và khía cạnh trong tệp src/llm.py.
Detail: Với những bình luận viết bằng tiếng Ý hoặc các câu vừa khen vị trí đẹp nhưng lại chê phòng ồn, mô hình hay đoán nhầm thành positive thay vì mixed, hoặc tự dịch tên khía cạnh sang tiếng Anh. Phát hiện khi đem đối chiếu kết quả của AI với 100 câu nhóm tự đọc và gán nhãn bằng tay. Nhóm khắc phục bằng cách ghi rõ ràng điều kiện phân loại mixed trong prompt, liệt kê cứng danh sách 7 khía cạnh tiếng Việt được phép chọn và chỉnh temperature về 0.1.

---

## 4. Nhóm kiểm chứng thế nào

AI: Toàn bộ các công cụ AI đã dùng
Usage: Kiểm tra lại code, đối chiếu số liệu và chạy thử quy trình.
Detail: Nhóm không copy nguyên code AI sinh ra mà chạy thử từng file nhỏ trên tập dữ liệu mẫu trước để xem đầu ra có đúng ý không, sau đó giải thích lại cho nhau hiểu toàn bộ logic để chuẩn bị vấn đáp. Với phần phân tích cảm xúc, cả nhóm chia nhau tự đọc và gán nhãn tay 100 bình luận thực tế làm chuẩn, rồi chạy file evaluate.py để đo độ chính xác thực tế của LLM và Baseline. Trước khi nộp bài, nhóm clone lại repo sang thư mục mới, cài lại thư viện từ đầu và chạy một lệnh duy nhất ở chế độ không dùng API key để chắc chắn toàn bộ số liệu và biểu đồ đều chạy ra đúng.
