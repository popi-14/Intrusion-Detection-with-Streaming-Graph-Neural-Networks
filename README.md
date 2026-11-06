# Intrusion Detection with Streaming GNN

Demo phát hiện xâm nhập mạng đa lớp trên các file CSV CIC-IDS2017. Ứng dụng chính huấn luyện GraphSAGE trên flow, sau đó phát lại holdout theo lô và cập nhật context đồ thị; `BENIGN` được chuẩn hóa thành `Normal`. Các lớp mẫu gồm `DDoS`, `PortScan`, từng biến thể DoS, `FTP-Patator`, `SSH-Patator`, `Bot`, các loại Web Attack, `Infiltration` và `Heartbleed`.

> **Giới hạn dữ liệu:** Các CSV CIC hiện tại không có IP nguồn/đích, `Source Port`, `Protocol` hoặc timestamp. Dashboard hiện tại dùng nút `Unknown Source` nối tới dịch vụ theo `Destination Port`, nên replay không phải stream thời gian thực. Pipeline temporal mới ở `temporal_graph.py` yêu cầu dataset khác có IP nguồn/đích và timestamp; không thể khôi phục những trường CIC không cung cấp. Đây là demo nghiên cứu, không phải IDS đã được đánh giá triển khai.

## Tính năng

- Đọc một hoặc nhiều CSV bằng DuckDB; tự nhận dạng schema và dùng `Attack` hoặc `Label` làm nhãn đa lớp.
- Huấn luyện GraphSAGE có giám sát, cân bằng lớp bằng trọng số căn bậc hai và xuất confidence từng lớp.
- Chia train/replay ổn định theo từng nhãn để vẫn có mẫu lớp hiếm như `Heartbleed` ở cả hai phần khi số lượng cho phép.
- Replay flow holdout và cập nhật ngữ cảnh đồ thị; dashboard hiển thị nhãn dự đoán/nhãn thật, confidence, biểu đồ phân bố, confusion matrix metrics và trạng thái replay.
- Flask JSON API, Server-Sent Events, điều khiển tạm dừng/tiếp tục và health check.
- Throughput/latency rolling, ROC-AUC one-vs-rest xấp xỉ từ histogram confidence, lọc sự kiện theo lớp, điều chỉnh tốc độ replay và truy vấn lân cận nhiều bước trên tối đa 100 flow gần nhất.
- Pipeline DuckDB độc lập để chuyển flow CSV/Parquet có timestamp và endpoint thành graph snapshots; temporal classifier dùng GRU memory để cập nhật node representation theo từng cửa sổ.
- Docker Compose mount thư mục `DataSet` ở chế độ chỉ đọc; image không chứa dữ liệu.

## Cấu trúc project

```text
.
├── app.py                  # Flask API, SSE và điều khiển replay
├── gnn_detector.py         # CSV/Parquet reader, GraphSAGE train và inference
├── temporal_graph.py       # DuckDB flow-to-temporal-graph pipeline
├── temporal_model.py       # GRU memory + temporal graph edge classifier
├── scripts/
│   └── build_temporal_snapshots.py
├── requirements.txt        # Flask, DuckDB, NumPy và PyTorch CPU
├── Dockerfile
├── docker-compose.yml
├── README.md
└── templates/
    └── index.html          # Dashboard Tailwind CSS, Chart.js và JavaScript
```

Đặt các file CSV vào `..\DataSet` (thư mục `DataSet` nằm cạnh thư mục project). Docker Compose mount thư mục này vào `/data`; backend đọc toàn bộ file CSV theo tên đã sắp xếp. Có thể đặt một file CSV riêng hoặc các file Parquet vào thư mục thay thế; detector tự nhận dạng định dạng và các cột flow.

### Tạo temporal graph snapshots từ NF-UNSW-NB15 hoặc dataset tương đương

Pipeline temporal nhận các biến thể tên cột phổ biến như `IPV4_SRC_ADDR`/`IPV4_DST_ADDR`, `L4_SRC_PORT`/`L4_DST_PORT`, `FLOW_START_MILLISECONDS`, `Attack` hoặc `Label`. Dataset phải có **timestamp, source IP, destination IP, nhãn và ít nhất một cột số**. Port được thêm vào định danh node nếu có. Dòng thiếu timestamp/endpoint/nhãn bị loại khỏi snapshot; CIC-IDS2017 hiện tại sẽ báo lỗi rõ ràng vì thiếu endpoint và timestamp.

```powershell
python -m scripts.build_temporal_snapshots `
  --input 'D:\datasets\NF-UNSW-NB15.parquet' `
  --output '.\artifacts\nf-unsw-snapshots' `
  --window-seconds 10
```

Mỗi cửa sổ được xuất thành `.npz` (`nodes`, `edge_index`, `edge_features`, `labels`); `manifest.json` ghi thời gian cửa sổ, số node/cạnh và feature names. Loader xử lý theo từng cửa sổ, không nạp toàn bộ dataset vào RAM. Dùng `DuckDBTemporalGraphLoader.iter_snapshots()` trực tiếp nếu muốn xử lý online thay vì export.

`TemporalGraphClassifier` (`temporal_model.py`) là module temporal-memory khởi tạo được từ `node_feature_dim`, `edge_feature_dim` và `class_count`; gọi một lần cho mỗi snapshot theo thứ tự thời gian, giữ memory qua các cửa sổ, và gọi `reset_memory()` giữa các stream độc lập. Đây là khối model tham chiếu để tích hợp/thử nghiệm; dashboard hiện tại vẫn dùng pipeline CIC GraphSAGE (`gnn_detector.py`), không tự chuyển qua dataset temporal mới.

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
| `POST` | `/api/control` | Tạm dừng/tiếp tục replay hoặc cập nhật `{"action":"speed","interval_seconds":0.1}` (0.01–10 giây) |
| `GET` | `/api/neighbors?node=...&hops=2` | Lân cận nhiều bước từ tối đa 100 flow replay gần nhất (hops 1–3) |

## Mô hình dữ liệu và luồng xử lý

Detector chọn `Attack` nếu có, nếu không dùng `Label`. Mỗi nhãn được chuẩn hóa thành một lớp riêng; nhãn `BENIGN` thành `Normal`, và ký tự hỏng trong tên Web Attack được chuẩn hóa thành dấu gạch ngang. Các nhãn duy nhất được phát hiện trực tiếp từ dataset để xác định kích thước đầu ra GNN và legend trên biểu đồ. Phần replay tính confusion-matrix metrics và ước lượng ROC-AUC one-vs-rest bằng histogram 100 bins (bounded memory); AUC chỉ có ý nghĩa khi lớp có cả positive và negative trong holdout.

Đồ thị dùng nút `(port, protocol)` nếu các cột này tồn tại. Với các CSV CIC-IDS2017 hiện tại, chỉ có `Destination Port`, nên mỗi flow nối nút nguồn tổng hợp `Unknown Source` tới nút dịch vụ đích. GraphSAGE kết hợp embedding hai đầu cạnh với 24 đặc trưng flow có trong dataset: bytes/packets, duration, tốc độ, khoảng thời gian giữa packet, cờ TCP, và thống kê độ dài packet. Giá trị số được biến đổi log trước khi đưa vào mô hình. Replay đo accuracy, precision, recall, F1 và confusion matrix theo từng lớp.

`temporal_graph.py` là đường xử lý tách biệt: DuckDB đọc CSV/Parquet, chuẩn hóa timestamp, chia cửa sổ, tạo node theo IP/port và directed edge theo flow. `temporal_model.py` cung cấp message aggregation + GRU memory cập nhật theo snapshot; cần huấn luyện/đánh giá trên tập temporal phù hợp trước khi thay model đang phục vụ dashboard. PyG và pandas không bắt buộc: Torch thuần giữ cài đặt CPU/Docker đơn giản, tránh phụ thuộc extension nhị phân theo nền tảng.

## Giới hạn và lưu ý

- Không có timestamp nên replay theo thứ tự scan file, không thể xem là thời gian sự kiện thực.
- Không có Source IP/Source Port/Protocol trong CSV hiện tại; đồ thị không biểu diễn host hoặc topology mạng thật.
- Chia holdout theo lớp giúp lớp hiếm có mẫu kiểm tra, nhưng các flow trong một lớp được chọn theo thứ tự file, không phải chia ngẫu nhiên.
- Lớp chỉ xuất hiện một lần không có mẫu đánh giá replay.
- Độ chính xác cần được diễn giải cùng precision/recall/F1 và phân bố mất cân bằng giữa các lớp.
- Throughput/latency hiện được đo trên cửa sổ 30 batch replay của demo, không thay thế benchmark tải production. ROC-AUC là xấp xỉ từ histogram 100 bins, không phải ROC chính xác trên từng score.
- Temporal GNN mới là pipeline/module tham chiếu, chưa được huấn luyện trên NF-UNSW trong repository và chưa được wire làm backend inference của dashboard CIC.
- Trước khi dùng trong môi trường thực tế, cần đánh giá false-positive/false-negative, kiểm soát truy cập, giám sát, bảo vệ dữ liệu và kiểm thử tải.

## Công nghệ

Python, PyTorch, DuckDB, NumPy, Flask, Gunicorn, Server-Sent Events, HTML/CSS, Tailwind CSS CDN, Chart.js, Lucide Icons và Docker Compose.
