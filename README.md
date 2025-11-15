# Time Series Forecasting - Vibration Data Prediction

Project dự đoán dữ liệu rung động từ cảm biến công nghiệp sử dụng Deep Learning.

## Kiến trúc Models

### Deep Learning Models
1. **Conv1D**: CNN thuần cho time series
2. **GRU**: RNN với Gated Recurrent Unit
3. **Conv1D-GRU**: Hybrid model với Residual Network (Skip Connection) - **Best Performance (R² ~ 0.976)**

### Baseline Models
4. **Linear Regression**: Baseline đơn giản
5. **XGBoost**: Tree-based model
6. **LightGBM**: Gradient boosting model

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc thư mục

```
4_Code/
├── Data/                    # Dữ liệu .mat
│   └── TH2_SETUP1.mat      # File dữ liệu chính
├── config.py               # Cấu hình (hyperparameters, paths)
├── data_loader.py          # Load dữ liệu từ .mat file
├── data_preprocessing.py   # Preprocessing & augmentation
├── model.py                # Định nghĩa models
├── baseline_models.py      # Baseline models (LR, XGB, LGBM)
├── trainer.py              # Training logic
├── evaluator.py            # Evaluation metrics
├── visualization.py        # Vẽ biểu đồ
├── utils.py                # Utilities
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```

## Sử dụng

### 1. Train một hoặc nhiều models (Cách mới - Khuyên dùng)

```bash
# Train một model
python main.py --models conv1d_gru

# Train nhiều models cùng lúc
python main.py --models conv1d_gru gru conv1d

# Train tất cả Deep Learning models
python main.py --models conv1d_gru gru conv1d --epochs 500

# Train tất cả Baseline models
python main.py --models linear xgboost lightgbm

# Train TẤT CẢ models (6 models)
python main.py --models conv1d_gru gru conv1d linear xgboost lightgbm
```

**Lưu ý:** Mỗi model sẽ tự động lưu vào thư mục riêng: `results/{model_name}/`

**Các model types có sẵn:**
- `conv1d_gru` - Best model (Hybrid CNN-RNN với ResNet)
- `gru` - Pure RNN với 3 GRU layers
- `conv1d` - Pure CNN
- `linear` - Linear Regression baseline
- `xgboost` - XGBoost baseline
- `lightgbm` - LightGBM baseline

### 2. Train TẤT CẢ models bằng script tiện lợi

```bash
python train_all_models.py
```

Script này sẽ train tất cả 6 models tuần tự và hiển thị summary chi tiết.

### 3. Tùy chỉnh tham số

```bash
# Tùy chỉnh epochs, batch size
python main.py --models conv1d_gru gru --epochs 500 --batch_size 32

# Train không có noise
python main.py --models conv1d_gru --no_noise

# Thay đổi output directory
python main.py --models conv1d_gru --output_dir my_results

# Train với sensor khác
python main.py --models conv1d_gru --sensor_idx 1
```

### 4. Xem tất cả options

```bash
python main.py --help
```

## Cấu hình

Chỉnh sửa `config.py` để thay đổi:
- Đường dẫn data
- Hyperparameters (learning rate, batch size, epochs...)
- Kiến trúc model (số layers, units...)
- Data split ratios

## Kết quả

Model sẽ lưu vào `results/` (hoặc folder bạn chỉ định):
- `model_saved.keras`: Model weights
- `history_saved.pkl`: Training history
- `scaler_values.npy`: Scaler parameters
- `metrics.csv`: Evaluation metrics
- `loss_plot.png`, `mae_plot.png`: Training plots

## Performance

| Model | R² (Test) | RMSE | MAE |
|-------|-----------|------|-----|
| **Conv1D-GRU** | **0.976** | 0.0010 | 0.0007 |
| GRU | 0.963 | 0.0013 | 0.0008 |
| XGBoost | 0.904 | 0.0019 | 0.0012 |
| LightGBM | 0.894 | 0.0021 | 0.0013 |
| Conv1D | 0.867 | 0.0023 | 0.0016 |
| Linear Regression | 0.867 | 0.0024 | 0.0017 |

## Đặc điểm kỹ thuật

### Residual Network (Skip Connection)
Conv1D-GRU model sử dụng skip connection giữa input và Conv1D output:
```python
conv_out = Conv1D(64, kernel_size=3, activation='relu')(input_layer)
input_resized = Conv1D(64, kernel_size=1, activation='linear')(input_layer)
conv_out = Add()([conv_out, input_resized])  # Residual connection
```

### Data Leakage Prevention
- Scaler fit **chỉ trên train data**
- Data split **theo thời gian** (không shuffle)
- Sequences tạo **sau khi split** data

## Tác giả

Project nghiên cứu về Time Series Forecasting cho dữ liệu rung động công nghiệp.
