# 💾 DATA CACHE GUIDE

## Tại sao cần Cache?

Preprocessing data (load .mat, split, scale, create sequences) mất **~20-30 giây** mỗi lần chạy.

Với cache, chỉ cần preprocess **1 lần duy nhất**, các lần sau load từ cache trong **~1 giây**!

**Tiết kiệm:** ~30 lần nhanh hơn! ⚡

## Cache hoạt động như thế nào?

1. **Lần đầu tiên chạy:** Preprocess data và lưu vào `cache/` folder
2. **Lần sau:** Tự động phát hiện cached data và load ngay lập tức
3. **Tự động invalidate:** Thay đổi tham số (sensor_idx, output_steps, etc.) → Tạo cache mới

### Cache Key

Cache key được tạo từ:
- `sensor_idx` - Sensor nào (0-7)
- `input_steps` - Số timesteps input (default: 50)
- `output_steps` - Số timesteps output (5, 10, 15, 20, 30, 40)
- `add_noise` - Có noise augmentation không (True/False)

**Ví dụ cache key:** `data_sensor0_in50_out5_noiseTrue_a1b2c3d4e5f6`

## Sử dụng Cache

### 1. Mặc định (Cache enabled)

```bash
# Lần đầu: Preprocess và save cache (~30s)
python main.py --models conv1d_gru

# Output:
# ⚠️  Không tìm thấy cache, đang preprocess từ đầu...
# STEP 1: LOAD DATA
# ...
# STEP 2: PREPROCESS DATA
# ...
# 💾 Đang lưu preprocessed data vào cache...
# ✓ Đã lưu cache: cache/data_sensor0_in50_out5_noiseTrue_xxx.pkl
# ✓ Kích thước: 15.23 MB
```

```bash
# Lần sau: Load từ cache (~1s) - NHANH!
python main.py --models gru

# Output:
# 🚀 Tìm thấy cached data! Loading từ cache...
# 📂 Đang load preprocessed data từ cache...
# ✓ Đã load cache từ: cache/data_sensor0_in50_out5_noiseTrue_xxx.pkl
# ✓ Tạo lúc: 2025-11-15T11:30:00
# ✅ Tiết kiệm thời gian preprocessing!
```

### 2. Tắt cache (Preprocess lại từ đầu)

```bash
python main.py --models conv1d_gru --no_cache
```

Sử dụng khi:
- Muốn đảm bảo data được preprocess mới nhất
- Debug preprocessing logic
- Data source (.mat file) đã thay đổi

### 3. Xóa cache

```bash
# Xóa cache trước khi chạy
python main.py --models conv1d_gru --clear_cache

# Kết quả:
# 🗑️  Đang xóa tất cả cached data...
# ✓ Đã xóa 3 cache file(s)
```

## Ví dụ thực tế

### Scenario 1: Train nhiều models với cùng config

```bash
# Lần 1: Train Conv1D-GRU (preprocess + cache: ~30s)
python main.py --models conv1d_gru --epochs 1000

# Lần 2: Train GRU (load cache: ~1s) - Tiết kiệm 30s!
python main.py --models gru --epochs 1000

# Lần 3: Train Conv1D (load cache: ~1s) - Tiết kiệm 30s!
python main.py --models conv1d --epochs 1000
```

**Tổng thời gian tiết kiệm:** ~60 giây cho 3 models!

### Scenario 2: Thử nghiệm nhiều output_steps

```bash
# output_steps=5 (cache mới)
python main.py --models conv1d_gru --output_steps 5

# output_steps=10 (cache mới vì khác output_steps)
python main.py --models conv1d_gru --output_steps 10

# output_steps=20 (cache mới)
python main.py --models conv1d_gru --output_steps 20

# Quay lại output_steps=5 (dùng cache cũ!) - Nhanh!
python main.py --models conv1d_gru --output_steps 5
```

### Scenario 3: Debug - Disable cache

```bash
# Khi cần debug preprocessing
python main.py --models conv1d_gru --no_cache
```

## Cache Management

### Xem cache files

```bash
# Windows
dir cache

# Linux/Mac
ls -lh cache/
```

Output:
```
data_sensor0_in50_out5_noiseTrue_a1b2.pkl   (15.2 MB)
data_sensor0_in50_out10_noiseTrue_c3d4.pkl  (18.5 MB)
data_sensor0_in50_out20_noiseTrue_e5f6.pkl  (25.1 MB)
```

### Xóa cache thủ công

```bash
# Windows
rmdir /s cache

# Linux/Mac
rm -rf cache/
```

### Kiểm tra dung lượng cache

```bash
# Windows
dir cache | find "File(s)"

# Linux/Mac
du -sh cache/
```

## Cache Location

- **Default:** `cache/` trong thư mục `4_Code/`
- **Gitignored:** Cache folder được ignore trong git (không commit)

## Performance Comparison

| Lần chạy | Không có cache | Có cache | Tiết kiệm |
|----------|---------------|----------|-----------|
| Lần 1 | 30s | 30s + save (1s) | - |
| Lần 2 | 30s | 1s | **29s (96%)** |
| Lần 3 | 30s | 1s | **29s (96%)** |
| Lần 4 | 30s | 1s | **29s (96%)** |

**Train 10 models:** Tiết kiệm **~270 giây (4.5 phút)**!

## Best Practices

### ✅ DO

1. **Để cache enabled (default)** - Tiết kiệm thời gian
2. **Clear cache khi:**
   - Thay đổi source data (.mat file)
   - Update preprocessing logic
   - Nghi ngờ cache bị corrupt
3. **Train nhiều models cùng lúc:**
   ```bash
   python main.py --models conv1d_gru gru conv1d
   ```
   → Chỉ preprocess 1 lần, train cả 3 models

### ❌ DON'T

1. **Commit cache vào git** - Đã gitignored rồi
2. **Quên xóa cache khi data thay đổi** - Có thể dùng `--clear_cache`
3. **Disable cache không cần thiết** - Lãng phí thời gian

## Troubleshooting

### Lỗi: "Cache corrupt" hoặc "Pickle error"

**Giải pháp:** Xóa cache và chạy lại
```bash
python main.py --models conv1d_gru --clear_cache
```

### Cache chiếm nhiều dung lượng

**Giải pháp:** Xóa các cache không dùng
```bash
# Xóa tất cả
rm -rf cache/

# Hoặc xóa từng file cụ thể
rm cache/data_sensor0_in50_out40_*.pkl
```

### Load cache nhưng shape không đúng

**Nguyên nhân:** Output_steps khác với lúc train

**Giải pháp:** Cache tự động invalidate, sẽ tạo cache mới với shape đúng

## Technical Details

### Cache File Format

- **Format:** Python Pickle (.pkl)
- **Contents:**
  - `X_train`, `y_train` - Training data
  - `X_val`, `y_val` - Validation data
  - `X_test`, `y_test` - Test data
  - `preprocessor` - Scaler object
  - `metadata` - Timestamp, shapes info

### Cache Key Generation

```python
# Tạo params string
params_str = f"sensor{sensor_idx}_in{input_steps}_out{output_steps}_noise{add_noise}"

# Hash MD5 (12 ký tự đầu)
cache_key = hashlib.md5(params_str.encode()).hexdigest()[:12]

# Final filename
filename = f"data_{params_str}_{cache_key}.pkl"
```

### Cache Invalidation

Cache tự động invalidate (tạo mới) khi:
1. `sensor_idx` thay đổi
2. `output_steps` thay đổi
3. `input_steps` thay đổi
4. `add_noise` thay đổi

---

**Kết luận:** Cache giúp tiết kiệm ~30 giây mỗi lần chạy. Với 10 models, tiết kiệm **4.5 phút**! 🚀
