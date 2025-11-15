# 📊 HƯỚNG DẪN SO SÁNH PREDICTIONS

Hướng dẫn vẽ biểu đồ so sánh predictions giữa các models và output_steps.

## 🎯 Mục đích

Visualization giúp bạn:
1. **So sánh predictions** của các models (Conv1D-GRU, GRU, Conv1D)
2. **So sánh predictions** với các output_steps khác nhau (5, 10, 15, 20, 30, 40)
3. **Xem chất lượng dự đoán** trực quan qua biểu đồ
4. **Tìm best model** dựa trên visual inspection

## 🚀 Sử dụng

### Cách 1: Tích hợp với analyze_existing_results.py (Khuyên dùng!)

```bash
conda activate tf

# Chạy phân tích metrics + predictions cùng lúc
python analyze_existing_results.py --plot_predictions

# Tùy chỉnh số samples
python analyze_existing_results.py --plot_predictions --num_samples 3
```

### Cách 2: Chạy script riêng

```bash
conda activate tf

# Basic usage
python plot_prediction_comparison.py

# Tùy chỉnh
python plot_prediction_comparison.py \
    --results_dir results \
    --output_dir analysis \
    --num_samples 5
```

### Arguments

- `--results_dir`: Folder chứa kết quả (default: `results/`)
- `--output_dir`: Folder lưu biểu đồ (default: `analysis/`)
- `--num_samples`: Số samples để vẽ (default: 5)

## 📊 Các biểu đồ được tạo

Script tạo 3 loại visualization trong folder `analysis/predictions_comparison/`:

### 1. Comparison by Output Step

**Files:** `comparison_out5.png`, `comparison_out10.png`, ..., `comparison_out40.png`

**Mục đích:** So sánh 3 models cho cùng output_step

**Layout:**
```
        Conv1D-GRU          GRU             Conv1D
Sample 1  [Plot]          [Plot]          [Plot]
Sample 2  [Plot]          [Plot]          [Plot]
Sample 3  [Plot]          [Plot]          [Plot]
```

**Insight:**
- Model nào predict gần ground truth nhất?
- Model nào có MAE thấp nhất?
- Đường predicted (màu) có fit đường true (đen) không?

**Ví dụ:**
```
Output Steps = 5
- Conv1D-GRU: MAE = 0.000745 (tốt nhất!)
- GRU: MAE = 0.000823
- Conv1D: MAE = 0.001634 (kém nhất)
```

### 2. Comparison by Model

**Files:** `comparison_conv1d_gru.png`, `comparison_gru.png`, `comparison_conv1d.png`

**Mục đích:** So sánh các output_steps khác nhau cho cùng model

**Layout:**
```
          out=5    out=10   out=15   out=20   out=30   out=40
Sample 1  [Plot]   [Plot]   [Plot]   [Plot]   [Plot]   [Plot]
Sample 2  [Plot]   [Plot]   [Plot]   [Plot]   [Plot]   [Plot]
Sample 3  [Plot]   [Plot]   [Plot]   [Plot]   [Plot]   [Plot]
```

**Insight:**
- Khi output_step tăng, prediction quality có giảm không?
- Output_step nào vẫn maintain good quality?
- Trade-off giữa horizon và accuracy là gì?

**Ví dụ:**
```
Conv1D-GRU:
- out=5:  MAE = 0.000745 (tốt)
- out=10: MAE = 0.000892 (vẫn ok)
- out=20: MAE = 0.001123 (acceptable)
- out=40: MAE = 0.001567 (degraded nhưng vẫn dùng được)
```

### 3. Grid Overview

**Files:** `grid_sample0.png`, `grid_sample1.png`, `grid_sample2.png`

**Mục đích:** Xem tổng quan TẤT CẢ combinations trong một màn hình

**Layout:**
```
           out=5    out=10   out=15   out=20   out=30   out=40
Conv1D-GRU [Plot]   [Plot]   [Plot]   [Plot]   [Plot]   [Plot]
GRU        [Plot]   [Plot]   [Plot]   [Plot]   [Plot]   [Plot]
Conv1D     [Plot]   [Plot]   [Plot]   [Plot]   [Plot]   [Plot]
```

**Insight:**
- Nhìn toàn cảnh 18 combinations (3 models × 6 output_steps)
- Dễ dàng spot best/worst combinations
- Compare nhanh visual quality

## 🎨 Cách đọc biểu đồ

### Màu sắc

- **Đường đen (●—●)**: Ground truth (giá trị thật)
- **Đường màu (■—■)**: Predictions
  - Xanh lá: Conv1D-GRU
  - Xanh dương: GRU
  - Đỏ: Conv1D

### Chất lượng predictions

✅ **Good prediction:**
- Đường màu overlap đường đen
- MAE nhỏ (< 0.001)
- Smooth, không có jumps

❌ **Poor prediction:**
- Đường màu xa đường đen
- MAE lớn (> 0.002)
- Có spikes/jumps

## 💡 Use Cases

### Use Case 1: Full analysis (metrics + predictions)

```bash
# Một lệnh duy nhất cho tất cả
python analyze_existing_results.py --plot_predictions

# Vừa có metrics vừa có predictions!
```

### Use Case 2: Tìm best model

```bash
# Vẽ comparison
python analyze_existing_results.py --plot_predictions

# Xem file comparison_out5.png
# Model nào có predictions fit nhất? → Chọn model đó
```

### Use Case 3: Validate metrics

```bash
# Đã xem metrics trong analysis/comparison_table.csv
# Giờ validate bằng visual

# Xem prediction plots
open analysis/predictions_comparison/comparison_out5.png

# Metrics có match với visual quality không?
```

### Use Case 4: Chọn output_step phù hợp

```bash
# Xem comparison_{model}.png
# Ví dụ: comparison_conv1d_gru.png

# Output_step nào vẫn maintain good quality?
# → Chọn largest output_step mà vẫn acceptable quality
```

### Use Case 5: Present results

```bash
# Grid overview rất tốt cho presentations!
open analysis/predictions_comparison/grid_sample0.png

# Một hình duy nhất show tất cả 18 combinations
```

## 📈 Ví dụ kết quả

### Comparison out=5 (Best case)

```
Conv1D-GRU          GRU             Conv1D
●●●●●              ●●●●●           ●●●●●
█████ (fit)        ████▌ (close)   ███░░ (off)
MAE=0.000745       MAE=0.000823    MAE=0.001634
```

**Insight:** Conv1D-GRU có predictions tốt nhất

### Comparison Conv1D-GRU

```
out=5   out=10  out=15  out=20  out=30  out=40
●●●●●   ●●●●●●  ●●●●●●  ●●●●●●  ●●●●●●  ●●●●●●
█████   ██████  ██████  █████▌  ████░░  ███░░░
0.0007  0.0009  0.0010  0.0011  0.0013  0.0016
```

**Insight:** Quality degradation nhẹ khi output_step tăng

### Grid Overview

```
        5       10      15      20      30      40
GRU    ✅ fit  ✅ fit  ✅ fit  ✅ ok   ⚠️ fair ⚠️ fair
Conv   ⚠️ ok   ⚠️ ok   ⚠️ fair ❌ poor ❌ poor ❌ poor
```

**Insight:** Conv1D-GRU và GRU tốt, Conv1D kém

## 🔧 Troubleshooting

### Lỗi: "No Data" trong plots

**Nguyên nhân:** Không tìm thấy model hoặc cache

**Giải pháp:**
```bash
# Check folder structure
ls results/5/
# Phải có: conv1d_gru/, gru/, conv1d/

# Check cache exists
ls cache/
# Phải có: data_sensor0_in50_out5_*.pkl
```

### Lỗi: TensorFlow import error

**Giải pháp:**
```bash
conda activate tf
pip install tensorflow>=2.13.0
```

### Script chạy chậm

**Nguyên nhân:** Phải load models và regenerate predictions

**Giải pháp:**
- Giảm `--num_samples` (default 5 → 3)
- Hoặc chờ (~2-5 phút cho tất cả plots)

## 📊 Performance

| Operation | Time |
|-----------|------|
| Load 1 model | ~2-3s |
| Generate 1 plot | ~5-10s |
| **Total** | **~3-5 phút** (tất cả plots) |

## 🎯 Best Practices

### ✅ DO

1. **Chạy sau khi phân tích metrics:**
   ```bash
   python analyze_existing_results.py
   python plot_prediction_comparison.py
   ```

2. **Xem grid overview trước:** Quick scan tất cả combinations

3. **Deep dive vào specific plots:** Xem chi tiết models/output_steps quan tâm

4. **So sánh metrics vs visual:** Validate metrics bằng predictions

### ❌ DON'T

1. **Không chạy nếu chưa có cache:** Cần cache data trước

2. **Không set num_samples quá lớn:** Tốn thời gian và không cần thiết

3. **Không chỉ nhìn một sample:** Xem ít nhất 3-5 samples để representative

## 📁 Output Structure

```
analysis/
└── predictions_comparison/
    ├── comparison_out5.png          # Models comparison for out=5
    ├── comparison_out10.png          # Models comparison for out=10
    ├── ...
    ├── comparison_out40.png
    ├── comparison_conv1d_gru.png     # Output_steps comparison for Conv1D-GRU
    ├── comparison_gru.png            # Output_steps comparison for GRU
    ├── comparison_conv1d.png         # Output_steps comparison for Conv1D
    ├── grid_sample0.png              # Grid overview sample 0
    ├── grid_sample1.png              # Grid overview sample 1
    └── grid_sample2.png              # Grid overview sample 2
```

## 🚀 Quick Start

```bash
# Cách 1: Tích hợp (Khuyên dùng!)
conda activate tf && python analyze_existing_results.py --plot_predictions

# Cách 2: Standalone
conda activate tf && python plot_prediction_comparison.py

# Xem kết quả
open analysis/predictions_comparison/grid_sample0.png
open analysis/predictions_comparison/comparison_out5.png
open analysis/predictions_comparison/comparison_conv1d_gru.png
```

**Thời gian:**
- Metrics only: ~30 giây
- Metrics + predictions: ~3-5 phút

---

**Happy Visualizing! 📊**
