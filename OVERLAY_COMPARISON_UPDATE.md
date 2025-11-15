# 🎉 CẬP NHẬT MỚI: OVERLAY COMPARISON PLOTS

## ⭐ Tính năng mới

Bây giờ bạn có thể xem **CẢ 3 models trên cùng một biểu đồ** để so sánh trực tiếp!

### Format giống `prediction_sample_1.png`

```
┌─────────────────────────────────────────────────────────────────┐
│  Past Data (Input)          │   Future (Predictions)             │
│  [50 timesteps]              │   [5/10/15/... timesteps]          │
│  ══════════════════          │   ● Actual (xanh dương)            │
│  Màu xanh lá                 │   ■ Conv1D-GRU (xanh lá)           │
│                               │   ▲ GRU (xanh dương)               │
│                               │   ◆ Conv1D (đỏ)                    │
└─────────────────────────────────────────────────────────────────┘
```

### Ưu điểm

✅ **Dễ so sánh trực tiếp** - Cả 3 models overlay trên cùng subplot
✅ **Format quen thuộc** - Giống `prediction_sample_1.png` bạn đã thấy
✅ **Nhìn ngay được model nào tốt nhất** - Đường nào gần Actual nhất
✅ **Có Past Data** - Hiểu context trước khi predict

## 🚀 Cách sử dụng

### Cách 1: Tích hợp với analyze_existing_results.py (Khuyên dùng)

```bash
# Chạy phân tích đầy đủ
python analyze_existing_results.py --plot_predictions

# Files overlay được tạo:
# analysis/predictions_comparison/overlay_out5.png
# analysis/predictions_comparison/overlay_out10.png
# ...
# analysis/predictions_comparison/overlay_out40.png
```

### Cách 2: Standalone

```bash
# Chạy riêng script
python plot_prediction_comparison.py

# Cũng tạo overlay plots!
```

## 📊 Output files

Bây giờ có **4 loại** visualization:

1. ⭐ **`overlay_out*.png`** - Overlay 3 models (KHUYÊN XEM TRƯỚC)
2. `comparison_out*.png` - So sánh models (3 subplots riêng biệt)
3. `comparison_{model}.png` - So sánh output_steps cho mỗi model
4. `grid_sample*.png` - Grid overview tất cả combinations

## 🎯 Ví dụ sử dụng

### Tìm best model nhanh nhất

```bash
# 1. Chạy analysis
python analyze_existing_results.py --plot_predictions

# 2. Mở overlay_out5.png
open analysis/predictions_comparison/overlay_out5.png

# 3. Nhìn thấy ngay:
#    - Past Data: 50 points màu xanh lá
#    - Actual Future: 5 points màu xanh dương đậm
#    - Conv1D-GRU pred: Gần Actual nhất ✅
#    - GRU pred: Gần Actual
#    - Conv1D pred: Xa Actual ❌
#
# → Conv1D-GRU is the winner!
```

### So sánh với các output_steps khác

```bash
# Xem overlay cho các output_steps
open analysis/predictions_comparison/overlay_out5.png   # Short-term
open analysis/predictions_comparison/overlay_out20.png  # Medium-term
open analysis/predictions_comparison/overlay_out40.png  # Long-term

# Thấy được:
# - Short-term: Cả 3 models fit tốt
# - Medium-term: Conv1D-GRU vẫn tốt, Conv1D bắt đầu sai
# - Long-term: Chỉ Conv1D-GRU và GRU còn acceptable
```

## 📖 Chi tiết kỹ thuật

### Plot elements

- **Past Data (Input)**: 50 timesteps, màu xanh lá, marker 'o'
- **Actual Future**: Output_steps timesteps, màu xanh dương đậm, marker 'o', linewidth=2.5
- **Conv1D-GRU Prediction**: Màu xanh lá, marker 's', dashed line
- **GRU Prediction**: Màu xanh dương, marker 's', dashed line
- **Conv1D Prediction**: Màu đỏ, marker 's', dashed line

### Layout

- Mỗi file có **num_samples** subplots (mặc định 5)
- Mỗi subplot = 1 sample từ test set
- Tất cả plots có:
  - Grid background (alpha=0.3)
  - Zero line (horizontal dashed)
  - Legend (top right)
  - Title với sample number và output_step

## 🔧 Tùy chỉnh

### Thay đổi số samples

```bash
# Chỉ vẽ 3 samples thay vì 5 (nhanh hơn)
python analyze_existing_results.py --plot_predictions --num_samples 3
```

### Chỉ chạy overlay (không cần tất cả plots)

Hiện tại chưa có flag riêng, nhưng bạn có thể edit `plot_prediction_comparison.py`:

```python
# Comment out các plots không cần:
# plot_comparison_by_output_step(...)  # Bỏ
# plot_comparison_by_model(...)         # Bỏ
# plot_all_combinations_grid(...)       # Bỏ

# Chỉ giữ:
plot_overlay_comparison(...)  # Giữ
```

## 💡 Tips

### ✅ DO

1. **Xem overlay trước tiên** - Dễ so sánh nhất
2. **So sánh nhiều output_steps** - Xem degradation
3. **Kiểm tra ít nhất 3-5 samples** - Đảm bảo representative

### ❌ DON'T

1. **Không chỉ nhìn 1 sample** - Có thể là outlier
2. **Không bỏ qua Past Data** - Hiểu context quan trọng
3. **Không chỉ tin metrics** - Visual validation luôn cần thiết

## 🎨 Màu sắc

| Model | Màu | Ý nghĩa |
|-------|-----|---------|
| Conv1D-GRU | 🟢 Xanh lá (#2ecc71) | Best model |
| GRU | 🔵 Xanh dương (#3498db) | Runner-up |
| Conv1D | 🔴 Đỏ (#e74c3c) | Baseline |
| Actual | 🔵 Xanh dương đậm | Ground truth |
| Past Data | 🟢 Xanh lá | Historical input |

## 📚 Tài liệu liên quan

- `QUICK_COMPARISON.md` - Workflow nhanh
- `PREDICTION_COMPARISON_GUIDE.md` - Hướng dẫn chi tiết
- `README.md` - Overview project

---

**Enjoy the new overlay comparison plots! 🎉**
