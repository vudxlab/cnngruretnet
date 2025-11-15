# 🚀 QUICK START GUIDE

## Chuẩn bị

Đảm bảo đã cài đặt môi trường conda `tf`:

```bash
conda activate tf
pip install -r requirements.txt
```

## Cách 1: Chạy trực tiếp (Khuyên dùng)

### Windows CMD/PowerShell:

```bash
# Activate environment
conda activate tf

# Train một model
python main.py --models conv1d_gru

# Train nhiều models
python main.py --models conv1d_gru gru conv1d

# Train với tùy chỉnh
python main.py --models conv1d_gru --epochs 500 --batch_size 32
```

### Linux/Mac:

```bash
# Activate environment
conda activate tf

# Train models
python main.py --models conv1d_gru gru conv1d
```

## Cách 2: Dùng Batch Script (Chỉ Windows)

```bash
# Chạy với default (conv1d_gru, 1000 epochs)
run.bat

# Chạy với model và epochs tùy chỉnh
run.bat conv1d_gru 500

# Chạy nhiều models (cần quotes)
run.bat "conv1d_gru gru conv1d" 1000
```

## Cách 3: Train tất cả models

```bash
conda activate tf
python train_all_models.py
```

## Test nhanh (2 epochs)

```bash
conda activate tf
python main.py --models conv1d_gru --epochs 2
```

## Các lệnh hữu ích

### Xem tất cả options:
```bash
python main.py --help
```

### Train Deep Learning models:
```bash
python main.py --models conv1d_gru gru conv1d --epochs 1000
```

### Train Baseline models:
```bash
python main.py --models linear xgboost lightgbm
```

### Train TẤT CẢ 6 models:
```bash
python main.py --models conv1d_gru gru conv1d linear xgboost lightgbm
```

### Không có noise augmentation:
```bash
python main.py --models conv1d_gru --no_noise
```

### Đổi output directory:
```bash
python main.py --models conv1d_gru --output_dir my_experiments
```

### Train với sensor khác:
```bash
python main.py --models conv1d_gru --sensor_idx 1
```

### Thay đổi số timesteps dự đoán (output_steps):
```bash
# Dự đoán 10 timesteps thay vì 5 (default)
python main.py --models conv1d_gru --output_steps 10

# Dự đoán 20 timesteps
python main.py --models conv1d_gru --output_steps 20

# Lựa chọn: 5 (default), 10, 15, 20, 30, 40
```

## Kết quả

Mỗi model lưu vào: `results/{model_name}/`

Files output:
- `model_saved.keras` - Model weights
- `history_saved.pkl` - Training history
- `scaler_values.npy` - Scaler parameters
- `metrics.csv` - Evaluation metrics
- `train_time_log.csv` - Training time
- `loss_plot.png` - Loss curves
- `mae_plot.png` - MAE curves
- `predictions/` - Prediction plots

## Troubleshooting

**Lỗi: ModuleNotFoundError: No module named 'tensorflow'**
```bash
conda activate tf
pip install tensorflow>=2.13.0
```

**Lỗi: FileNotFoundError: TH2_SETUP1.mat**
- Đảm bảo chạy từ thư mục `4_Code/`
- Kiểm tra file `Data/TH2_SETUP1.mat` tồn tại

**Delay 15-20s khi khởi động?**
- Đây là bình thường (TensorFlow import)
- Xem `PERFORMANCE_ANALYSIS.md` để hiểu rõ hơn

## Ví dụ hoàn chỉnh

```bash
# 1. Activate environment
conda activate tf

# 2. Di chuyển vào thư mục code
cd D:\Code\cnngruretnet\4_Code

# 3. Train models
python main.py --models conv1d_gru gru conv1d --epochs 1000

# 4. Kết quả lưu tại results/
dir results
```

---

**Happy Training! 🎯**
