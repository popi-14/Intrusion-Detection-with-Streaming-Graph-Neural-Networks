# Intrusion Detection with Streaming GNN

Demo phát hiện xâm nhập mạng bằng Temporal GraphSAGE trên các file CSV CIC-IDS2017. Mô hình phân loại riêng các nhãn tấn công trong cột `Label`; `BENIGN` được chuẩn hóa thành `Normal`. Các lớp mẫu trong dataset gồm `DDoS`, `PortScan`, từng biến thể DoS, `FTP-Patator`, `SSH-Patator`, `Bot`, các loại Web Attack, `Infiltration` và `Heartbleed`.

> **Giới hạn dữ liệu:** Các CSV hiện tại không có `Source Port`, `Protocol` hoặc timestamp. Đồ thị dùng nút `Unknown Source` nối tới dịch vụ theo `Destination Port`; vì vậy không thể đại diện host nguồn, giao thức hay thứ tự thời gian thực gốc. Đây là demo nghiên cứu, không phải IDS đã được đánh giá triển khai.

## Tính năng

- Đọc một hoặc nhiều CSV bằng DuckDB; tự nhận dạng schema và dùng `Attack` hoặc `Label` làm nhãn đa lớp.
- Huấn luyện GraphSAGE có giám sát, cân bằng lớp bằng trọng số căn bậc hai và xuất confidence từng lớp.
- Chia train/replay ổn định theo từng nhãn để vẫn có mẫu lớp hiếm như `Heartbleed` ở cả hai phần khi số lượng cho phép.
- Replay flow holdout và cập nhật ngữ cảnh đồ thị; dashboard hiển thị nhãn dự đoán/nhãn thật, confidence, biểu đồ phân bố, confusion matrix metrics và trạng thái replay.
- Flask JSON API, Server-Sent Events, điều khiển tạm dừng/tiếp tục và health check.
- Docker Compose mount thư mục `DataSet` ở chế độ chỉ đọc; image không chứa dữ liệu.

## Cấu trúc project

```text
.
├── app.py                  # Flask API, SSE và điều khiển replay
├── gnn_detector.py         # CSV/Parquet reader, GraphSAGE train và inference
├── requirements.txt        # Flask, DuckDB, NumPy và PyTorch CPU
├── Dockerfile
├── docker-compose.yml
├── README.md
└── templates/
    └── index.html          # Dashboard Tailwind CSS, Chart.js và JavaScript
```

Đặt các file CSV vào `..\DataSet` (thư mục `DataSet` nằm cạnh thư mục project). Docker Compose mount thư mục này vào `/data`; backend đọc toàn bộ file CSV theo tên đã sắp xếp. Có thể đặt một file CSV riêng hoặc các file Parquet vào thư mục thay thế; detector tự nhận dạng định dạng và các cột flow.

Dataset CIC-IDS2017 hiện tại có khoảng 2.8 triệu flow và 15 nhãn. Nhãn `BENIGN` hiển thị là `Normal`; các nhãn tấn công còn lại được giữ riêng, không gom thành lớp “Attack” chung. Nếu có từ hai mẫu trở lên, mỗi nhãn được chia 80% train và 20% replay bằng thứ tự xuất hiện của nhãn trong dữ liệu; các mẫu holdout được phát lại theo thứ tự stream CSV. Lớp chỉ có một mẫu được giữ lại cho train.

## Chạy bằng Docker

Yêu cầu Docker Desktop (Windows/macOS) hoặc Docker Engine và Docker Compose plugin.

```powershell
docker compose up --build
```

Mở dashboard tại [http://localhost:5000](http://localhost:5000). Dừng ứng dụng bằng `Ctrl+C`; để chạy nền, dùng:

```powershell
docker compose up --build -d
```

Xem log hoặc dừng container:

```powershell
docker compose logs -f
docker compose down
```

Trên PowerShell, các lệnh `docker compose` trên chạy tương tự. Tailwind CSS, Chart.js và Lucide được tải qua CDN, do đó trình duyệt cần Internet để hiển thị đầy đủ giao diện và biểu đồ.

Lần đầu mở dashboard, backend đọc dataset và huấn luyện GNN trước khi bắt đầu replay. Trạng thái tiến trình hiển thị trên trang; log huấn luyện có thể xem bằng `docker compose logs -f`. Điều chỉnh `TRAINING_FRACTION`, `REPLAY_BATCH_SIZE` và `REPLAY_INTERVAL_SECONDS` trong `docker-compose.yml` nếu cần.

## Chạy cục bộ không dùng Docker

Yêu cầu Python 3.10 trở lên.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:DATASET_PATH = 'C:\Data\HK1_2026_2027\BigData\DataSet'
python app.py
```

Sau đó mở `http://localhost:5000`. Nếu PowerShell chặn kích hoạt script, có thể dùng `.\.venv\Scripts\python.exe` thay cho `python`.

## Kiểm thử

Sau khi build Docker image, chạy bộ kiểm thử detector bằng PowerShell:

```powershell
$project = (Get-Location).Path
docker compose run --rm --no-deps --volume "${project}:/app:ro" --entrypoint python streaming-gnn-ids -m unittest discover -s tests -v
```

Kiểm thử dùng dataset CSV nhỏ tạm thời để xác nhận chuẩn hóa nhãn, chia train/replay và xử lý giá trị số không hữu hạn.

## API

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/` | Trang dashboard |
| `GET` | `/healthz` | Health check cho Docker |
| `GET` | `/api/state` | Snapshot JSON gồm class labels, metrics, graph, event log và series |
| `GET` | `/api/stream` | SSE stream; gửi snapshot mới khi state thay đổi |
| `POST` | `/api/control` | Tạm dừng hoặc tiếp tục dataset replay |

## Mô hình dữ liệu và luồng xử lý

Detector chọn `Attack` nếu có, nếu không dùng `Label`. Mỗi nhãn được chuẩn hóa thành một lớp riêng; nhãn `BENIGN` thành `Normal`, và ký tự hỏng trong tên Web Attack được chuẩn hóa thành dấu gạch ngang. Các nhãn duy nhất được phát hiện trực tiếp từ dataset để xác định kích thước đầu ra GNN và legend trên biểu đồ.

Đồ thị dùng nút `(port, protocol)` nếu các cột này tồn tại. Với các CSV CIC-IDS2017 hiện tại, chỉ có `Destination Port`, nên mỗi flow nối nút nguồn tổng hợp `Unknown Source` tới nút dịch vụ đích. GraphSAGE kết hợp embedding hai đầu cạnh với 24 đặc trưng flow có trong dataset: bytes/packets, duration, tốc độ, khoảng thời gian giữa packet, cờ TCP, và thống kê độ dài packet. Giá trị số được biến đổi log trước khi đưa vào mô hình. Replay đo accuracy, precision, recall, F1 và confusion matrix theo từng lớp.

## Giới hạn và lưu ý

- Không có timestamp nên replay theo thứ tự scan file, không thể xem là thời gian sự kiện thực.
- Không có Source IP/Source Port/Protocol trong CSV hiện tại; đồ thị không biểu diễn host hoặc topology mạng thật.
- Chia holdout theo lớp giúp lớp hiếm có mẫu kiểm tra, nhưng các flow trong một lớp được chọn theo thứ tự file, không phải chia ngẫu nhiên.
- Lớp chỉ xuất hiện một lần không có mẫu đánh giá replay.
- Độ chính xác cần được diễn giải cùng precision/recall/F1 và phân bố mất cân bằng giữa các lớp.
- Trước khi dùng trong môi trường thực tế, cần đánh giá false-positive/false-negative, kiểm soát truy cập, giám sát, bảo vệ dữ liệu và kiểm thử tải.

## Công nghệ

Python, PyTorch, DuckDB, NumPy, Flask, Gunicorn, Server-Sent Events, HTML/CSS, Tailwind CSS CDN, Chart.js, Lucide Icons và Docker Compose.
