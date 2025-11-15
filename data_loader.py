"""
Data Loader Module
Chịu trách nhiệm load và trích xuất dữ liệu từ file .mat
"""

import numpy as np
import scipy.io
from config import Config


class VibrationDataLoader:
    """
    Class để load và xử lý dữ liệu rung động từ file .mat
    """

    def __init__(self, mat_file_path=None):
        """
        Khởi tạo DataLoader

        Args:
            mat_file_path: Đường dẫn đến file .mat (nếu None, dùng từ Config)
        """
        self.mat_file_path = mat_file_path or Config.get_mat_file_path()
        self.raw_data = None
        self.data_dict = None

    def load_mat_file(self):
        """
        Đọc dữ liệu từ file .mat

        Returns:
            data: numpy array shape (timesteps, sensors)
        """
        print(f"Đang đọc dữ liệu từ {self.mat_file_path}...")

        try:
            self.data_dict = scipy.io.loadmat(self.mat_file_path)
            self.raw_data = self.data_dict.get('TH2_SETUP1')

            if self.raw_data is None:
                raise ValueError("Không tìm thấy key 'TH2_SETUP1' trong file .mat")

            print(f"✓ Đã load thành công!")
            print(f"  Shape: {self.raw_data.shape}")
            print(f"  Dtype: {self.raw_data.dtype}")

            return self.raw_data

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Không tìm thấy file: {self.mat_file_path}\n"
                f"Vui lòng kiểm tra đường dẫn hoặc cập nhật Config.MAT_FILE"
            )
        except Exception as e:
            raise Exception(f"Lỗi khi đọc file .mat: {str(e)}")

    def extract_sensor_data(self, sensor_idx=None,
                           start_col=None, end_col=None,
                           start_slice=None, end_slice=None):
        """
        Trích xuất dữ liệu từ một cảm biến cụ thể

        Args:
            sensor_idx: Index của sensor (0-7), nếu None lấy từ Config
            start_col, end_col: Khoảng cột để lấy
            start_slice, end_slice: Khoảng để slice tiếp

        Returns:
            data_sensor: numpy array 1D chứa dữ liệu từ sensor
        """
        if self.raw_data is None:
            raise ValueError("Chưa load dữ liệu! Gọi load_mat_file() trước.")

        # Lấy parameters từ Config nếu không được cung cấp
        sensor_idx = sensor_idx if sensor_idx is not None else Config.SENSOR_IDX
        start_col = start_col if start_col is not None else Config.START_COL
        end_col = end_col if end_col is not None else Config.END_COL
        start_slice = start_slice if start_slice is not None else Config.START_SLICE
        end_slice = end_slice if end_slice is not None else Config.END_SLICE

        print(f"\nTrích xuất dữ liệu từ sensor {sensor_idx + 1}...")

        # Transpose: (timesteps, sensors) -> (sensors, timesteps)
        data = np.transpose(self.raw_data)
        print(f"  Sau transpose: {data.shape}")

        # Slice 1: Lấy khoảng cột
        data = data[:, start_col:end_col]
        print(f"  Sau slice 1 [{start_col}:{end_col}]: {data.shape}")

        # Slice 2: Tinh chỉnh thêm
        data = data[:, start_slice:end_slice]
        print(f"  Sau slice 2 [{start_slice}:{end_slice}]: {data.shape}")

        # Lấy dữ liệu của sensor
        data_sensor = data[sensor_idx, :]
        print(f"  Dữ liệu sensor {sensor_idx + 1}: {data_sensor.shape}")
        print(f"  Min: {data_sensor.min():.6f}, Max: {data_sensor.max():.6f}")
        print(f"  Mean: {data_sensor.mean():.6f}, Std: {data_sensor.std():.6f}")

        return data_sensor

    def get_data_info(self):
        """
        Lấy thông tin về dữ liệu đã load

        Returns:
            dict: Dictionary chứa thông tin về dữ liệu
        """
        if self.raw_data is None:
            return {"status": "No data loaded"}

        return {
            "shape": self.raw_data.shape,
            "dtype": str(self.raw_data.dtype),
            "num_timesteps": self.raw_data.shape[0],
            "num_sensors": self.raw_data.shape[1],
            "min_value": float(self.raw_data.min()),
            "max_value": float(self.raw_data.max()),
            "mean_value": float(self.raw_data.mean()),
            "std_value": float(self.raw_data.std())
        }

    def load_and_extract(self, sensor_idx=None):
        """
        Wrapper function: Load file và extract dữ liệu sensor trong một bước

        Args:
            sensor_idx: Index của sensor (0-7)

        Returns:
            data_sensor: numpy array 1D
        """
        self.load_mat_file()
        return self.extract_sensor_data(sensor_idx)


def load_vibration_data(mat_file_path=None, sensor_idx=None):
    """
    Helper function để load dữ liệu nhanh

    Args:
        mat_file_path: Đường dẫn file .mat
        sensor_idx: Index sensor

    Returns:
        data_sensor: numpy array 1D
    """
    loader = VibrationDataLoader(mat_file_path)
    return loader.load_and_extract(sensor_idx)


if __name__ == "__main__":
    # Test DataLoader
    print("=" * 60)
    print("TESTING DATA LOADER")
    print("=" * 60)

    # Test với config mặc định
    loader = VibrationDataLoader()

    # Load data
    raw_data = loader.load_mat_file()

    # Get info
    info = loader.get_data_info()
    print("\nData Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Extract sensor data
    sensor_data = loader.extract_sensor_data(sensor_idx=0)

    print(f"\nExtracted sensor data shape: {sensor_data.shape}")
    print(f"First 10 values: {sensor_data[:10]}")

    # Test helper function
    print("\n" + "=" * 60)
    print("Testing helper function")
    data = load_vibration_data(sensor_idx=1)
    print(f"Sensor 2 data shape: {data.shape}")
