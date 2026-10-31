# Intrusion Detection with Streaming GNN

Demo phát hiện flow bất thường bằng Temporal GraphSAGE trên dataset NF-UNSW-NB15-V2. Mỗi nút là cặp `(port, protocol)`; mỗi flow tạo một cạnh có hướng từ cổng nguồn đến cổng đích. Backend fit trên 70% đầu file, hiệu chỉnh ngưỡng trên 10% tiếp theo, rồi replay 20% cuối theo thứ tự dòng qua Server-Sent Events (SSE).

> **Giới hạn dữ liệu:** Parquet hiện tại không có IP nguồn/đích hoặc timestamp. Vì vậy `(port, protocol)` là node dịch vụ, không phải host; thứ tự dòng chỉ dùng làm thứ tự replay, không khôi phục thời gian thực gốc. Mô hình là demo nghiên cứu, không phải IDS đã được đánh giá triển khai.

## Tính năng

- Đọc trực tiếp Parquet, tạo đồ thị kề có hướng thưa từ `L4_SRC_PORT`, `L4_DST_PORT`, `PROTOCOL`.
- Huấn luyện GraphSAGE có giám sát bằng `Label`; chuẩn hóa đặc trưng byte, packet và duration.
- Replay phần holdout theo thứ tự file; mỗi flow được chấm điểm trước khi cập nhật ngữ cảnh hàng xóm.
- Dashboard hiển thị đồ thị cổng/giao thức, dự đoán, nhãn thật, số liệu đánh giá và trạng thái train/replay.
- API JSON, SSE, điều khiển tạm dừng/tiếp tục và health check.
- Docker Compose mount dataset ở chế độ chỉ đọc; image không chứa bản sao dữ liệu.

## Cấu trúc project

```text
.
├── app.py                  # Flask API, SSE và điều khiển replay
├── gnn_detector.py         # Sparse temporal GraphSAGE, train và inference
├── requirements.txt        # Flask, DuckDB, NumPy và PyTorch CPU
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── README.md
└── templates/
    └── index.html          # Dashboard Tailwind CSS, Chart.js và JavaScript
```

Tải file Parquet từ [NF-UNSW-NB15-V2](https://www.kaggle.com/datasets/dhoogla/nfunswnb15v2) và đặt tại `..\DataSet\NF-UNSW-NB15-V2.parquet` (thư mục `DataSet` nằm cạnh thư mục project). Trong Compose, thư mục này được mount read-only vào `/data`. Dataset không được lưu trong repository hoặc Docker image.

## Chạy bằng Docker

Yêu cầu Docker Desktop (Windows/macOS) hoặc Docker Engine và Docker Compose plugin.

```bash
docker compose up --build
```

Mở dashboard tại [http://localhost:5000](http://localhost:5000). Dừng ứng dụng bằng `Ctrl+C`; để chạy nền, dùng:

```bash
docker compose up --build -d
```

Xem log hoặc dừng container:

```bash
docker compose logs -f
docker compose down
```

Trên PowerShell, các lệnh `docker compose` trên chạy tương tự. Tailwind CSS, Chart.js và Lucide được tải qua CDN, do đó trình duyệt cần Internet để hiển thị đầy đủ giao diện và biểu đồ.

Lần đầu mở dashboard, backend đọc dataset và huấn luyện GNN trước khi bắt đầu replay. Trạng thái tiến trình hiển thị trên trang; log huấn luyện có thể xem bằng `docker compose logs -f`. Điều chỉnh `TRAINING_FRACTION`, `REPLAY_BATCH_SIZE` và `REPLAY_INTERVAL_SECONDS` trong `docker-compose.yml` nếu cần.

## Chạy cục bộ không dùng Docker

Yêu cầu Python 3.10 trở lên.

```bash
python -m venv .venv
```

Kích hoạt môi trường:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
source .venv/bin/activate
```

Cài dependencies và khởi động Flask:

```bash
pip install -r requirements.txt
```

PowerShell:

```powershell
$env:DATASET_PATH = 'C:\Data\HK1_2026_2027\BigData\DataSet\NF-UNSW-NB15-V2.parquet'
python app.py
```

Linux/macOS:

```bash
export DATASET_PATH="../DataSet/NF-UNSW-NB15-V2.parquet"
python app.py
```

Sau đó mở `http://localhost:5000`. Nếu PowerShell chặn kích hoạt script, có thể dùng `.\.venv\Scripts\python.exe` thay cho `python`.

## API

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/` | Trang dashboard |
| `GET` | `/healthz` | Health check cho Docker |
| `GET` | `/api/state` | Snapshot JSON gồm metrics, graph, event log và series |
| `GET` | `/api/stream` | SSE stream; gửi snapshot mới khi state thay đổi |
| `POST` | `/api/control` | Tạm dừng hoặc tiếp tục dataset replay |

Ví dụ điều khiển:

```bash
curl -X POST http://localhost:5000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action":"pause"}'

curl -X POST http://localhost:5000/api/control \
  -H "Content-Type: application/json" \
  -d '{"action":"resume"}'
```

Trên PowerShell có thể dùng `Invoke-RestMethod`:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:5000/api/control `
  -ContentType 'application/json' -Body '{"action":"pause"}'
```

## Mô hình dữ liệu và luồng xử lý

Flow dùng các cột:

```text
(L4_SRC_PORT, L4_DST_PORT, PROTOCOL, IN/OUT_BYTES, IN/OUT_PKTS,
 FLOW_DURATION_MILLISECONDS, Label, Attack)
```

Nút là `(port, protocol)` và cạnh có hướng `source node → destination node`. Ma trận kề được lưu thưa qua tập cạnh và phép cộng đặc trưng hàng xóm đầu vào, tránh tạo ma trận dày cỡ `N × N`. GraphSAGE tổng hợp đặc trưng node/hàng xóm cùng đặc trưng flow để phân loại nhị phân. Model fit trên 70% đầu file; 10% tiếp theo dùng chọn ngưỡng F1; 20% cuối được replay theo thứ tự Parquet. `Attack` chỉ dùng làm ground truth để tính TP/TN/FP/FN trong replay.

## Giới hạn và lưu ý

- Không có timestamp nên thứ tự file không thể được xem là thời gian sự kiện; replay chỉ mô phỏng thứ tự tiếp nhận.
- Không có địa chỉ IP nên đồ thị cổng/giao thức không biểu diễn host hoặc topology mạng thật.
- Split theo thứ tự file có thể gây lệch phân phối; cần đánh giá thêm theo phương pháp chia dữ liệu phù hợp trước khi báo cáo kết quả khoa học.
- Trạng thái/replay lưu trong RAM và được khởi tạo lại khi ứng dụng khởi động lại; chỉ dùng một tiến trình.
- Gunicorn được cấu hình một worker để các kết nối SSE cùng quan sát một trạng thái dùng chung trong bộ nhớ.
- CDN yêu cầu Internet. Khi cần chạy hoàn toàn offline, tải các thư viện frontend về static assets và cập nhật template.
- Trước khi dùng trong môi trường thực tế, cần đánh giá false-positive/false-negative, kiểm soát truy cập, lưu trữ bền vững, giám sát, bảo vệ dữ liệu và kiểm thử tải.

## Công nghệ

Python, PyTorch, DuckDB, NumPy, Flask, Gunicorn, Server-Sent Events, HTML/CSS, Tailwind CSS CDN, Chart.js, Lucide Icons và Docker Compose.
