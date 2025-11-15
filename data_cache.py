"""
Data Cache Module
Quản lý việc cache preprocessed data để tái sử dụng, tiết kiệm thời gian
"""

import os
import pickle
import hashlib
import numpy as np
from datetime import datetime


class DataCache:
    """
    Class quản lý cache cho preprocessed data
    """

    def __init__(self, cache_dir="cache"):
        """
        Khởi tạo DataCache

        Args:
            cache_dir: Thư mục chứa cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_key(self, sensor_idx, output_steps, add_noise, input_steps=50):
        """
        Tạo cache key dựa trên các tham số preprocessing

        Args:
            sensor_idx: Index của sensor
            output_steps: Số timesteps output
            add_noise: Có thêm noise không
            input_steps: Số timesteps input

        Returns:
            str: Cache key (hash)
        """
        # Tạo string từ các params
        params_str = f"sensor{sensor_idx}_in{input_steps}_out{output_steps}_noise{add_noise}"

        # Hash để tạo cache key ngắn gọn
        cache_key = hashlib.md5(params_str.encode()).hexdigest()[:12]

        return f"data_{params_str}_{cache_key}"

    def get_cache_path(self, cache_key):
        """
        Lấy đường dẫn đầy đủ đến cache file

        Args:
            cache_key: Cache key

        Returns:
            str: Đường dẫn cache file
        """
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def cache_exists(self, cache_key):
        """
        Kiểm tra cache có tồn tại không

        Args:
            cache_key: Cache key

        Returns:
            bool: True nếu cache tồn tại
        """
        cache_path = self.get_cache_path(cache_key)
        return os.path.exists(cache_path)

    def save_cache(self, data_dict, cache_key):
        """
        Lưu preprocessed data vào cache

        Args:
            data_dict: Dictionary chứa X_train, y_train, X_val, y_val, X_test, y_test, preprocessor
            cache_key: Cache key
        """
        cache_path = self.get_cache_path(cache_key)

        print(f"\n💾 Đang lưu preprocessed data vào cache...")
        print(f"   Cache key: {cache_key}")

        # Tạo cache data
        cache_data = {
            'X_train': data_dict['X_train'],
            'y_train': data_dict['y_train'],
            'X_val': data_dict['X_val'],
            'y_val': data_dict['y_val'],
            'X_test': data_dict['X_test'],
            'y_test': data_dict['y_test'],
            'preprocessor': data_dict['preprocessor'],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'shapes': {
                    'X_train': data_dict['X_train'].shape,
                    'y_train': data_dict['y_train'].shape,
                    'X_val': data_dict['X_val'].shape,
                    'y_val': data_dict['y_val'].shape,
                    'X_test': data_dict['X_test'].shape,
                    'y_test': data_dict['y_test'].shape,
                }
            }
        }

        # Lưu vào file
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        # Tính kích thước file
        file_size = os.path.getsize(cache_path) / (1024 * 1024)  # MB

        print(f"   ✓ Đã lưu cache: {cache_path}")
        print(f"   ✓ Kích thước: {file_size:.2f} MB")

    def load_cache(self, cache_key):
        """
        Load preprocessed data từ cache

        Args:
            cache_key: Cache key

        Returns:
            dict: data_dict chứa preprocessed data
        """
        cache_path = self.get_cache_path(cache_key)

        if not self.cache_exists(cache_key):
            raise FileNotFoundError(f"Cache không tồn tại: {cache_path}")

        print(f"\n📂 Đang load preprocessed data từ cache...")
        print(f"   Cache key: {cache_key}")

        # Load từ file
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        # Hiển thị thông tin
        metadata = cache_data.get('metadata', {})
        created_at = metadata.get('created_at', 'Unknown')
        shapes = metadata.get('shapes', {})

        print(f"   ✓ Đã load cache từ: {cache_path}")
        print(f"   ✓ Tạo lúc: {created_at}")
        print(f"   ✓ Shapes:")
        for key, shape in shapes.items():
            print(f"      - {key}: {shape}")

        # Tạo data_dict
        data_dict = {
            'X_train': cache_data['X_train'],
            'y_train': cache_data['y_train'],
            'X_val': cache_data['X_val'],
            'y_val': cache_data['y_val'],
            'X_test': cache_data['X_test'],
            'y_test': cache_data['y_test'],
            'preprocessor': cache_data['preprocessor']
        }

        return data_dict

    def clear_cache(self, cache_key=None):
        """
        Xóa cache

        Args:
            cache_key: Cache key cụ thể (nếu None, xóa tất cả)
        """
        if cache_key:
            # Xóa một cache cụ thể
            cache_path = self.get_cache_path(cache_key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"✓ Đã xóa cache: {cache_path}")
            else:
                print(f"⚠️  Cache không tồn tại: {cache_path}")
        else:
            # Xóa tất cả cache
            count = 0
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
                    count += 1
            print(f"✓ Đã xóa {count} cache file(s)")

    def list_caches(self):
        """
        Liệt kê tất cả cache files

        Returns:
            list: Danh sách cache files
        """
        caches = []

        if not os.path.exists(self.cache_dir):
            return caches

        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.cache_dir, filename)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)

                caches.append({
                    'filename': filename,
                    'path': filepath,
                    'size_mb': size_mb
                })

        return caches


def preprocess_with_cache(data_sensor, sensor_idx, output_steps, add_noise,
                          input_steps=50, use_cache=True, cache_dir="cache"):
    """
    Preprocess data với caching

    Args:
        data_sensor: Raw sensor data
        sensor_idx: Index của sensor
        output_steps: Số timesteps output
        add_noise: Có thêm noise không
        input_steps: Số timesteps input
        use_cache: Có sử dụng cache không
        cache_dir: Thư mục cache

    Returns:
        dict: data_dict chứa preprocessed data
    """
    from data_preprocessing import preprocess_data

    cache = DataCache(cache_dir)
    cache_key = cache.get_cache_key(sensor_idx, output_steps, add_noise, input_steps)

    # Kiểm tra cache
    if use_cache and cache.cache_exists(cache_key):
        print(f"\n🚀 Tìm thấy cached data! Loading từ cache...")
        data_dict = cache.load_cache(cache_key)
        print(f"✅ Tiết kiệm thời gian preprocessing!")
        return data_dict

    # Nếu không có cache, preprocess từ đầu
    print(f"\n⚙️  Không tìm thấy cache, đang preprocess data...")
    data_dict = preprocess_data(data_sensor, add_noise=add_noise)

    # Lưu cache
    if use_cache:
        cache.save_cache(data_dict, cache_key)

    return data_dict


if __name__ == "__main__":
    # Test cache
    print("Testing DataCache...")

    cache = DataCache("test_cache")

    # Test cache key
    key1 = cache.get_cache_key(sensor_idx=0, output_steps=5, add_noise=True)
    key2 = cache.get_cache_key(sensor_idx=0, output_steps=10, add_noise=True)
    key3 = cache.get_cache_key(sensor_idx=0, output_steps=5, add_noise=True)

    print(f"Key 1: {key1}")
    print(f"Key 2: {key2}")
    print(f"Key 3: {key3}")
    print(f"Key 1 == Key 3: {key1 == key3}")
    print(f"Key 1 != Key 2: {key1 != key2}")

    print("\n✓ DataCache test completed!")
