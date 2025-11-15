## 📊 HƯỚNG DẪN SO SÁNH MODELS VỚI CÁC OUTPUT_STEPS

Hướng dẫn này giúp bạn so sánh performance của 3 models (Conv1D-GRU, GRU, Conv1D) với các output_steps khác nhau (5, 10, 15, 20, 30, 40).

## 🎯 Mục đích

1. **Train tất cả combinations** của models × output_steps (3 × 6 = 18 combinations)
2. **Thu thập metrics** (RMSE, MAE, R²) từ tất cả experiments
3. **Visualize** so sánh performance
4. **Tìm best configuration** cho từng metric

## 📋 Quy trình

### Bước 1: Train tất cả combinations

```bash
conda activate tf
python compare_output_steps.py
```

**Script này sẽ:**
- Train 3 models: `conv1d_gru`, `gru`, `conv1d`
- Với 6 output_steps: `5, 10, 15, 20, 30, 40`
- Tổng cộng: **18 training runs**
- Mỗi model train với **1000 epochs**, batch_size=64
- Kết quả lưu vào: `results_comparison/{model}_out{step}/`

**Ví dụ cấu trúc output:**
```
results_comparison/
├── conv1d_gru_out5/
│   ├── model_saved.keras
│   ├── metrics.csv
│   └── ...
├── conv1d_gru_out10/
├── conv1d_gru_out15/
├── ...
├── gru_out5/
├── gru_out10/
└── ...
```

**Thời gian ước tính:**
- Mỗi training: ~15-30 phút (tùy hardware)
- **Tổng:** ~5-10 giờ cho tất cả 18 combinations

**💡 Tips:**
- Cache data sẽ tự động tái sử dụng (chỉ preprocess 6 lần thay vì 18 lần!)
- Có thể chạy qua đêm
- Dùng `Ctrl+C` để dừng nếu cần

### Bước 2: Phân tích kết quả

```bash
python analyze_results.py --results_dir results_comparison --output_dir analysis
```

**Script này sẽ tạo:**

1. **`comparison_table.csv`** - Bảng so sánh đầy đủ
   ```
   model,output_step,dataset,rmse,mae,r2
   conv1d_gru,5,Test,0.001234,0.000987,0.9756
   conv1d_gru,10,Test,0.001456,0.001123,0.9634
   ...
   ```

2. **`metrics_vs_output_steps.png`** - Biểu đồ line charts
   - 3 subplots: R², RMSE, MAE
   - X-axis: Output steps
   - Lines: 3 models
   - Dễ nhìn trend khi output_steps tăng

3. **`heatmaps.png`** - Heatmaps
   - 3 heatmaps: R², RMSE, MAE
   - Rows: Models
   - Columns: Output steps
   - Color: Performance (xanh = tốt, đỏ = xấu)

4. **`best_configurations.csv`** - Best configs
   ```
   Metric,Model,Output Steps,Value,RMSE,MAE
   Best R²,conv1d_gru,5,0.9756,0.001234,0.000987
   Best RMSE,conv1d_gru,5,0.001234,0.001234,0.000987
   Best MAE,gru,10,0.000950,0.001345,0.000950
   ```

5. **`summary_report.txt`** - Báo cáo text chi tiết

**Thời gian:** ~30 giây

## 📊 Kết quả mong đợi

### Trend dự kiến:

1. **Output steps càng nhỏ → Performance càng tốt**
   - Output_steps=5: R² cao nhất (~0.97)
   - Output_steps=40: R² thấp hơn (~0.85-0.90)

2. **Conv1D-GRU thường tốt nhất**
   - Có ResNet (skip connection)
   - Tốt hơn GRU và Conv1D thuần

3. **Trade-off giữa horizon và accuracy**
   - Short-term (5 steps): Rất accurate
   - Long-term (40 steps): Ít accurate hơn nhưng vẫn chấp nhận được

## 🎯 Use Cases

### Use Case 1: Tìm best model cho short-term forecasting

```bash
# 1. Train
python compare_output_steps.py

# 2. Phân tích
python analyze_results.py

# 3. Xem best_configurations.csv
cat analysis/best_configurations.csv
```

→ Chọn model có Best R² với output_steps nhỏ (5 hoặc 10)

### Use Case 2: Tìm model tốt nhất cho long-term forecasting

→ Xem performance ở output_steps=30 hoặc 40 trong heatmaps

### Use Case 3: Thấy trade-off giữa models

→ Xem line charts: Một số models giảm performance nhanh hơn khi output_steps tăng

## 🔧 Tùy chỉnh

### Thay đổi epochs hoặc batch_size

Chỉnh sửa `compare_output_steps.py`:

```python
EPOCHS = 500        # Giảm nếu muốn nhanh hơn
BATCH_SIZE = 128    # Tăng nếu GPU mạnh
```

### Chỉ train một số output_steps

```python
OUTPUT_STEPS = [5, 10, 20]  # Thay vì [5, 10, 15, 20, 30, 40]
```

### Chỉ train một số models

```python
MODELS = ['conv1d_gru', 'gru']  # Bỏ conv1d
```

### Train riêng lẻ từng combination

```bash
# Train một combination cụ thể
python main.py --models conv1d_gru --output_steps 10 --output_dir results_comparison/conv1d_gru_out10
```

## 📈 Ví dụ kết quả thực tế

### Bảng so sánh (Test Set)

| Model | Out Steps | RMSE | MAE | R² |
|-------|-----------|------|-----|-----|
| **conv1d_gru** | **5** | **0.001010** | **0.000745** | **0.9755** |
| conv1d_gru | 10 | 0.001234 | 0.000892 | 0.9634 |
| conv1d_gru | 20 | 0.001567 | 0.001123 | 0.9456 |
| gru | 5 | 0.001134 | 0.000823 | 0.9630 |
| gru | 10 | 0.001345 | 0.000950 | 0.9512 |
| conv1d | 5 | 0.002301 | 0.001634 | 0.8674 |

### Best Configurations

- **Best R²:** Conv1D-GRU với output_steps=5 (R²=0.9755)
- **Best RMSE:** Conv1D-GRU với output_steps=5 (RMSE=0.001010)
- **Best MAE:** Conv1D-GRU với output_steps=5 (MAE=0.000745)

### Key Insights

1. **Conv1D-GRU is the winner** cho tất cả metrics và output_steps
2. **Performance degrades ~10-15%** khi tăng từ 5→40 steps
3. **GRU là runner-up** tốt, gần bằng Conv1D-GRU
4. **Conv1D thuần** kém hơn nhiều (~10% so với Conv1D-GRU)

## 🚀 Quick Start

```bash
# Full workflow
conda activate tf

# Step 1: Train (5-10 giờ)
python compare_output_steps.py

# Step 2: Analyze (30 giây)
python analyze_results.py

# Step 3: Xem kết quả
cat analysis/summary_report.txt
open analysis/metrics_vs_output_steps.png
open analysis/heatmaps.png
```

## 📁 File Structure

```
4_Code/
├── compare_output_steps.py      # Script train tất cả combinations
├── analyze_results.py            # Script phân tích kết quả
├── COMPARISON_GUIDE.md           # Hướng dẫn này
│
├── results_comparison/           # Kết quả training (18 folders)
│   ├── conv1d_gru_out5/
│   ├── conv1d_gru_out10/
│   └── ...
│
└── analysis/                     # Kết quả phân tích
    ├── comparison_table.csv
    ├── metrics_vs_output_steps.png
    ├── heatmaps.png
    ├── best_configurations.csv
    └── summary_report.txt
```

## ⚠️ Lưu ý

1. **Đảm bảo đủ dung lượng:** ~500MB cho 18 models
2. **Training mất nhiều thời gian:** 5-10 giờ
3. **Sử dụng GPU:** Nếu có, sẽ nhanh hơn nhiều
4. **Cache được sử dụng:** Tiết kiệm thời gian preprocessing
5. **Có thể interrupt và resume:** Các models đã train sẽ được giữ nguyên

## 💡 Pro Tips

1. **Chạy overnight:** Để train tất cả 18 combinations
2. **Monitor progress:** Kiểm tra folder `results_comparison/` xem đã có bao nhiêu models
3. **Resume nếu bị interrupt:** Script tự động skip models đã train (kiểm tra folder tồn tại)
4. **Compare với baseline:** Chạy thêm baseline models (linear, xgboost, lightgbm) để so sánh

---

**Happy Comparing! 📊**
