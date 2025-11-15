# HƯỚNG DẪN SỬ DỤNG NHANH

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cách sử dụng

### ✅ CÁCH 1: Train một hoặc nhiều models (Khuyên dùng)

```bash
# Train MỘT model
python main.py --models conv1d_gru

# Train NHIỀU models cùng lúc
python main.py --models conv1d_gru gru conv1d

# Train TẤT CẢ Deep Learning models
python main.py --models conv1d_gru gru conv1d

# Train TẤT CẢ Baseline models
python main.py --models linear xgboost lightgbm

# Train TẤT CẢ 6 models
python main.py --models conv1d_gru gru conv1d linear xgboost lightgbm
```

### ✅ CÁCH 2: Dùng script tự động

```bash
python train_all_models.py
```

## Các model types

| Model Type | Mô tả | Performance |
|-----------|-------|-------------|
| `conv1d_gru` | **Best** - Hybrid CNN-RNN + ResNet | R² ~ 0.976 |
| `gru` | Pure RNN với 3 GRU layers | R² ~ 0.963 |
| `conv1d` | Pure CNN cho time series | R² ~ 0.867 |
| `linear` | Linear Regression baseline | R² ~ 0.867 |
| `xgboost` | XGBoost baseline | R² ~ 0.904 |
| `lightgbm` | LightGBM baseline | R² ~ 0.894 |

## Tùy chỉnh tham số

```bash
# Thay đổi epochs và batch size
python main.py --models conv1d_gru --epochs 500 --batch_size 32

# Train không có noise augmentation
python main.py --models conv1d_gru --no_noise

# Đổi output directory
python main.py --models conv1d_gru --output_dir my_results

# Train với sensor khác (0-7)
python main.py --models conv1d_gru --sensor_idx 1

# Thay đổi số timesteps dự đoán (output_steps)
python main.py --models conv1d_gru --output_steps 10
# Lựa chọn: 5 (mặc định), 10, 15, 20, 30, 40

# Thay đổi patience cho early stopping
python main.py --models conv1d_gru --patience 20

# Kết hợp nhiều tham số
python main.py --models conv1d_gru gru \
    --epochs 500 \
    --batch_size 32 \
    --patience 15 \
    --output_steps 20 \
    --output_dir custom_results
```

## Kết quả output

Mỗi model sẽ tự động lưu vào: `results/{model_name}/`

Ví dụ:
```
results/
├── conv1d_gru/
│   ├── model_saved.keras       # Model weights
│   ├── history_saved.pkl       # Training history
│   ├── scaler_values.npy       # Scaler parameters
│   ├── metrics.csv             # Evaluation metrics
│   ├── train_time_log.csv      # Training time
│   ├── loss_plot.png           # Loss curves
│   ├── mae_plot.png            # MAE curves
│   ├── metrics_comparison.png  # Metrics comparison
│   └── predictions/            # Prediction plots
├── gru/
│   └── ...
└── conv1d/
    └── ...
```

## Xem tất cả options

```bash
python main.py --help
```

## Ví dụ thực tế

### Ví dụ 1: Quick test với ít epochs

```bash
python main.py --models conv1d_gru --epochs 10 --batch_size 64
```

### Ví dụ 2: Train các Deep Learning models với full epochs

```bash
python main.py --models conv1d_gru gru conv1d --epochs 1000
```

### Ví dụ 3: So sánh tất cả models

```bash
python main.py --models conv1d_gru gru conv1d linear xgboost lightgbm
```

### Ví dụ 4: Dự đoán xa hơn với output_steps=20

```bash
python main.py \
    --models conv1d_gru \
    --output_steps 20 \
    --epochs 1000 \
    --output_dir results/predict_20steps
```

### Ví dụ 5: Train với cấu hình tùy chỉnh đầy đủ

```bash
python main.py \
    --models conv1d_gru \
    --epochs 2000 \
    --batch_size 128 \
    --patience 20 \
    --output_steps 15 \
    --no_noise \
    --sensor_idx 0 \
    --output_dir experiments/exp001
```

## Lưu ý

- ⚠️ Training Deep Learning models (conv1d_gru, gru, conv1d) mất nhiều thời gian hơn
- ⚠️ Baseline models (linear, xgboost, lightgbm) train nhanh hơn
- ✅ Mỗi model tự động lưu vào thư mục riêng
- ✅ Data chỉ load 1 lần, dù train nhiều models
- ✅ Random seed được set để reproducibility

## Troubleshooting

**Lỗi: "No module named 'tensorflow'"**
```bash
pip install tensorflow>=2.13.0
```

**Lỗi: "No module named 'xgboost'"**
```bash
pip install xgboost lightgbm
```

**Lỗi: "FileNotFoundError: TH2_SETUP1.mat"**
- Kiểm tra file `Data/TH2_SETUP1.mat` có tồn tại không
- Chạy từ thư mục `4_Code/`

**Lỗi encoding UTF-8**
- Đã được fix trong utils.py và evaluator.py
- Nếu vẫn lỗi, báo lại để kiểm tra
