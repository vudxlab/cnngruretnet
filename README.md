# Time Series Forecasting - Vibration Data Prediction

Project dự đoán dữ liệu rung động từ cảm biến công nghiệp sử dụng Deep Learning.

## Kiến trúc Models

### Main Models
1. **CNN+ResNet+GRU** (`cnn_resnet_gru`): Hybrid model với CNN, Residual Network và GRU - **Best Performance (R² ~ 0.976)**
2. **CNN** (`cnn`): CNN thuần cho time series
3. **GRU** (`gru`): RNN với Gated Recurrent Unit (3 layers)

### Ablation Study Models
4. **CNN+GRU** (`cnn_gru`): CNN + GRU **KHÔNG CÓ** Residual Connection (để test tác động của ResNet)
5. **CNN+ResNet** (`cnn_resnet`): CNN + ResNet **KHÔNG CÓ** GRU layers (để test tác động của recurrent layers)
6. **CNN+ResNet+GRU+BN** (`cnn_resnet_gru_bn`): Full model **VỚI** BatchNorm/Dropout (để test tác động của regularization)
7. **CNN+ResNet+GRU (1L/2L/4L)** (`cnn_resnet_gru_var`): Variable depth models với 1, 2, hoặc 4 GRU layers (để test tác động của model depth)

### Baseline Models
8. **Linear Regression** (`linear`): Baseline đơn giản
9. **XGBoost** (`xgboost`): Tree-based model
10. **LightGBM** (`lightgbm`): Gradient boosting model

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
python main.py --models cnn_resnet_gru

# Train nhiều models cùng lúc
python main.py --models cnn_resnet_gru gru cnn

# Train tất cả Deep Learning models
python main.py --models cnn_resnet_gru gru cnn --epochs 500

# Train tất cả Baseline models
python main.py --models linear xgboost lightgbm

# Train TẤT CẢ main models
python main.py --models cnn_resnet_gru gru cnn linear xgboost lightgbm
```

**Lưu ý:** Mỗi model sẽ tự động lưu vào thư mục riêng: `results/{model_name}/`

**Các model types có sẵn:**

**Main Models:**
- `cnn_resnet_gru` - Best model (CNN+ResNet+GRU: Hybrid CNN-RNN với ResNet)
- `gru` - Pure RNN với 3 GRU layers
- `cnn` - Pure CNN

**Ablation Study Models:**
- `cnn_gru` - CNN+GRU (không có residual connection)
- `cnn_resnet` - CNN+ResNet (không có GRU layers)
- `cnn_resnet_gru_bn` - CNN+ResNet+GRU với BatchNorm/Dropout
- `cnn_resnet_gru_var` - CNN+ResNet+GRU với số GRU layers tùy chỉnh (dùng `--num_gru_layers`)

**Baseline Models:**
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
python main.py --models cnn_resnet_gru gru --epochs 500 --batch_size 32

# Thay đổi số timesteps dự đoán (output_steps)
python main.py --models cnn_resnet_gru --output_steps 10
# Choices: 5 (mặc định), 10, 15, 20, 30, 40

# Train không có noise
python main.py --models cnn_resnet_gru --no_noise

# Thay đổi output directory
python main.py --models cnn_resnet_gru --output_dir my_results

# Train với sensor khác
python main.py --models cnn_resnet_gru --sensor_idx 1

# Kết hợp nhiều tham số
python main.py --models cnn_resnet_gru --output_steps 20 --epochs 1000 --batch_size 128
```

### 4. Ablation Study - So sánh Model Variants

Train các model variants để phân tích contribution của từng component:

```bash
# Test 1: Loại bỏ Residual Connection
python main.py --models cnn_gru --epochs 500 --output_steps 10

# Test 2: Loại bỏ GRU layers
python main.py --models cnn_resnet --epochs 500 --output_steps 10

# Test 3: Thêm BatchNorm và Dropout
python main.py --models cnn_resnet_gru_bn --epochs 500 --output_steps 10

# Test 4: Số lượng GRU layers khác nhau
python main.py --models cnn_resnet_gru_var --num_gru_layers 1 --epochs 500 --output_steps 10
python main.py --models cnn_resnet_gru_var --num_gru_layers 2 --epochs 500 --output_steps 10
python main.py --models cnn_resnet_gru_var --num_gru_layers 4 --epochs 500 --output_steps 10

# Train TẤT CẢ ablation variants cùng lúc
python main.py --models cnn_resnet_gru cnn_gru cnn_resnet cnn_resnet_gru_bn --epochs 500 --output_steps 10
```

**Mục đích Ablation Study:**
- ❌ **CNN+GRU** (no residual): Đánh giá tác động của **residual connection** → ΔRMSE = ?
- ❌ **CNN+ResNet** (no GRU): Đánh giá tác động của **GRU layers** → ΔRMSE = ?
- ➕ **CNN+ResNet+GRU+BN**: Đánh giá tác động của **regularization** → ΔRMSE = ?
- 🔢 **Variable Depth (1L, 2L, 4L)**: Đánh giá tác động của **model depth** → Best depth = ?

Sau khi train, phân tích kết quả:
```bash
python analyze_existing_results.py --results_dir results --plot_predictions
```

### 5. Sử dụng Cache (Tiết kiệm thời gian)

**Cache được bật mặc định** - Preprocessed data sẽ được lưu lại và tái sử dụng!

```bash
# Lần đầu: Preprocess và lưu cache (~30s)
python main.py --models cnn_resnet_gru

# Lần sau: Load từ cache (~1s) - NHANH HƠN 30 LẦN!
python main.py --models gru

# Tắt cache (preprocess lại từ đầu)
python main.py --models cnn_resnet_gru --no_cache

# Xóa tất cả cache trước khi chạy
python main.py --models cnn_resnet_gru --clear_cache
```

**Lưu ý:** Cache dựa trên `sensor_idx`, `output_steps`, `add_noise`, `input_steps`. Thay đổi bất kỳ tham số nào sẽ tạo cache mới.

### 6. Xem tất cả options

```bash
python main.py --help
```

## Cấu hình

Chỉnh sửa `config.py` để thay đổi:
- Đường dẫn data
- Hyperparameters (learning rate, batch size, epochs...)
- Kiến trúc model (số layers, units...)
- Data split ratios

## Data Augmentation Strategies (NEW)

**Response to Reviewer Feedback**: Để test robustness của model trên nhiều kịch bản nhiễu khác nhau, project đã được mở rộng với multiple augmentation strategies.

### Available Strategies

#### 1. **Multiple Noise Levels** (Enhanced)
Test với nhiều mức độ nhiễu khác nhau thay vì chỉ 1 mức (σ = 0.1 × std):

```bash
# Test với nhiều noise levels: [0.05, 0.1, 0.15, 0.2]
python main.py --models cnn_resnet_gru \
    --use_multiple_noise_levels \
    --noise_factors 0.05 0.1 0.15 0.2 \
    --output_dir results/multi_noise_test
```

#### 2. **Random Dropout of Segments** (NEW)
Simulate missing data segments do transmission errors:

```bash
# Sử dụng dropout augmentation
python main.py --models cnn_resnet_gru \
    --augmentation_strategies noise dropout \
    --dropout_prob 0.1 \
    --output_dir results/dropout_test
```

#### 3. **Block Missingness** (NEW)
Simulate sensor failures với large missing blocks:

```bash
# Sử dụng block missingness
python main.py --models cnn_resnet_gru \
    --augmentation_strategies noise block_missingness \
    --block_miss_prob 0.05 \
    --block_miss_fill_method interpolate \
    --output_dir results/block_miss_test
```

#### 4. **Combined Strategies** (Comprehensive Test)
Test với tất cả strategies để đánh giá robustness toàn diện:

```bash
# Comprehensive robustness test
python main.py --models cnn_resnet_gru \
    --augmentation_strategies noise dropout block_missingness \
    --use_multiple_noise_levels \
    --noise_factors 0.05 0.1 0.15 0.2 \
    --output_dir results/robustness_test
```

### Test Augmentation Strategies

Chạy demo script để visualize các augmentation strategies:

```bash
python test_augmentations.py
```

**Output**: 5 PNG files minh họa từng strategy

### Configuration Parameters

Trong `config.py`:

```python
# Multiple noise levels
USE_MULTIPLE_NOISE_LEVELS = False  # Bật để test nhiều mức độ
NOISE_FACTORS = [0.05, 0.1, 0.15, 0.2]

# Augmentation strategies
AUGMENTATION_STRATEGIES = ['noise']  # Options: 'noise', 'dropout', 'block_missingness'

# Random dropout
DROPOUT_PROB = 0.1
DROPOUT_MIN_LENGTH = 1
DROPOUT_MAX_LENGTH = 5

# Block missingness
BLOCK_MISS_PROB = 0.05
BLOCK_MISS_MIN_LENGTH = 3
BLOCK_MISS_MAX_LENGTH = 10
BLOCK_MISS_FILL_METHOD = 'interpolate'  # Options: 'zero', 'mean', 'interpolate'
```

**📖 Detailed Guide**: Xem `AUGMENTATION_GUIDE.md` cho chi tiết và best practices

## Kết quả

Model sẽ lưu vào `results/` (hoặc folder bạn chỉ định):
- `model_saved.keras`: Model weights
- `history_saved.pkl`: Training history
- `scaler_values.npy`: Scaler parameters
- `metrics.csv`: Evaluation metrics
- `loss_plot.png`, `mae_plot.png`: Training plots

## Performance

### Main Models Performance (output_steps=5)

| Model | R² (Test) | RMSE | MAE |
|-------|-----------|------|-----|
| **CNN+ResNet+GRU** | **0.976** | 0.0010 | 0.0007 |
| GRU | 0.963 | 0.0013 | 0.0008 |
| XGBoost | 0.904 | 0.0019 | 0.0012 |
| LightGBM | 0.894 | 0.0021 | 0.0013 |
| CNN | 0.867 | 0.0023 | 0.0016 |
| Linear Regression | 0.867 | 0.0024 | 0.0017 |

### Ablation Study Results

Sau khi train các ablation models, bạn có thể so sánh để xem:
- **Impact của Residual Connection**: So sánh CNN+ResNet+GRU vs CNN+GRU
- **Impact của GRU Layers**: So sánh CNN+ResNet+GRU vs CNN+ResNet
- **Impact của BatchNorm/Dropout**: So sánh CNN+ResNet+GRU vs CNN+ResNet+GRU+BN
- **Optimal Model Depth**: So sánh các variants 1L, 2L, 3L, 4L

Phân tích bằng:
```bash
python analyze_existing_results.py --results_dir results
```

## Đặc điểm kỹ thuật

### Residual Network (Skip Connection)
CNN+ResNet+GRU model sử dụng skip connection giữa input và Conv1D output:
```python
conv_out = Conv1D(64, kernel_size=3, activation='relu')(input_layer)
input_resized = Conv1D(64, kernel_size=1, activation='linear')(input_layer)
conv_out = Add()([conv_out, input_resized])  # Residual connection
```

### Model Architecture Comparison

| Component | CNN+ResNet+GRU | CNN+GRU | CNN+ResNet | CNN | GRU |
|-----------|----------------|---------|------------|-----|-----|
| **Conv1D Layer** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Residual Connection** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **GRU Layers (3)** | ✅ | ✅ | ❌ | ❌ | ✅ |
| **BatchNorm/Dropout** | ❌* | ❌ | ❌ | ✅ | ❌ |

*Sử dụng `cnn_resnet_gru_bn` để thêm BatchNorm/Dropout vào full model.

### Data Leakage Prevention

Dự án này **ĐÃ SỬA** các vấn đề data leakage phổ biến:

**✅ Flow ĐÚNG (Tránh Data Leakage):**
```
1. Split data theo thời gian (train/val/test) - TRƯỚC
2. Augmentation (add noise) CHỈ trên TRAIN data - SAU
3. Val/Test data GIỮ NGUYÊN (không augment)
4. Scaler fit CHỈ trên train sequences
5. Tạo sequences SAU khi split
```

**❌ Flow SAI (Có Data Leakage):**
```
1. Augmentation trên TOÀN BỘ data
2. Split data (train/val/test)
→ Kết quả: Cùng 1 mẫu xuất hiện ở cả train và test (bản gốc + bản noisy)
```

**Chi tiết:**
- **Temporal Split**: Data split theo thời gian (60/20/20), **KHÔNG shuffle**
- **Augmentation**: CHỈ áp dụng cho train data (val/test giữ nguyên)
- **Scaler Fitting**: Fit CHỈ trên train sequences, apply lên val/test
- **Sequence Creation**: Tạo sau khi split data (không tạo trước)

## So sánh Models và Output Steps

### Quick Analysis (30 giây)

```bash
# Phân tích và so sánh tất cả models với các output_steps
python analyze_existing_results.py
```

Tạo ra:
- `comparison_table.csv` - Bảng so sánh đầy đủ
- `metrics_vs_output_steps.png` - Line charts (R², RMSE, MAE)
- `heatmaps.png` - Heatmaps cho visual comparison
- `best_configurations.csv` - Best configs cho từng metric
- `summary_report.txt` - Báo cáo chi tiết

### Full Analysis với Prediction Visualizations (3-5 phút)

```bash
# Phân tích metrics + vẽ prediction comparisons
python analyze_existing_results.py --plot_predictions

# Tùy chỉnh số samples
python analyze_existing_results.py --plot_predictions --num_samples 3
```

Tạo thêm:
- ⭐ `predictions_comparison/overlay_out*.png` - **Overlay 3 models (KHUYÊN XEM)**
- `predictions_comparison/comparison_out*.png` - So sánh models (3 subplots)
- `predictions_comparison/comparison_*.png` - So sánh output_steps theo model
- `predictions_comparison/grid_sample*.png` - Grid tổng quan

**Overlay plots:** Format giống `prediction_sample_1.png` với Past Data + Actual + CẢ 3 predictions overlay!

**Xem chi tiết:**
- `QUICK_COMPARISON.md` - Hướng dẫn nhanh
- `PREDICTION_COMPARISON_GUIDE.md` - Hướng dẫn chi tiết predictions
- `COMPARISON_GUIDE.md` - Hướng dẫn train từ đầu

## Tên Model Thống Nhất

Trong các biểu đồ và báo cáo, tên model được chuẩn hóa như sau:

| Model Type (Code) | Tên Hiển Thị (Charts/Reports) |
|-------------------|-------------------------------|
| `cnn_resnet_gru` | **CNN+ResNet+GRU** |
| `cnn_gru` | **CNN+GRU** |
| `cnn_resnet` | **CNN+ResNet** |
| `cnn_resnet_gru_bn` | **CNN+ResNet+GRU+BN** |
| `cnn_resnet_gru_var` | **CNN+ResNet+GRU (XL)** (X = số layers) |
| `cnn` | **CNN** |
| `gru` | **GRU** |
| `linear` | **Linear Regression** |
| `xgboost` | **XGBoost** |
| `lightgbm` | **LightGBM** |

Tên này được sử dụng nhất quán trong:
- ✅ Line charts (metrics vs output_steps)
- ✅ Heatmaps
- ✅ Comparison tables
- ✅ Summary reports
- ✅ Prediction plots

## Tác giả

Project nghiên cứu về Time Series Forecasting cho dữ liệu rung động công nghiệp.
