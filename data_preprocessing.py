"""
Data Preprocessing Module
Chịu trách nhiệm:
- Data augmentation (thêm noise)
- Tạo sequences (sliding window)
- Chia dữ liệu theo thời gian
- Scaling (Min-Max normalization)
"""

import numpy as np
from config import Config


class DataPreprocessor:
    """
    Class xử lý tiền xử lý dữ liệu cho time series forecasting
    """

    def __init__(self):
        self.min_scaler = None
        self.max_scaler = None

    # ==================== DATA AUGMENTATION ====================

    @staticmethod
    def add_noise(data, noise_level_factor=None):
        """
        Thêm Gaussian noise vào dữ liệu (Data Augmentation)

        Args:
            data: numpy array 1D
            noise_level_factor: Hệ số nhân với std_dev (nếu None, lấy từ Config)

        Returns:
            noisy_data: numpy array 1D với noise đã thêm
        """
        noise_level_factor = noise_level_factor or Config.NOISE_FACTOR

        std_dev = np.std(data)
        noise_level = noise_level_factor * std_dev
        noise = np.random.normal(0, noise_level, data.shape)

        print(f"Thêm noise vào dữ liệu:")
        print(f"  Standard Deviation: {std_dev:.6f}")
        print(f"  Noise level (factor={noise_level_factor}): {noise_level:.6f}")

        return data + noise

    # ==================== SEQUENCE CREATION ====================

    @staticmethod
    def create_sequences(data, input_steps=None, output_steps=None):
        """
        Tạo sequences bằng sliding window

        Args:
            data: numpy array 1D
            input_steps: Số timesteps cho input (X)
            output_steps: Số timesteps cho output (y)

        Returns:
            X: numpy array shape (num_samples, input_steps)
            y: numpy array shape (num_samples, output_steps)
        """
        input_steps = input_steps or Config.INPUT_STEPS
        output_steps = output_steps or Config.OUTPUT_STEPS

        X, y = [], []
        for i in range(len(data) - input_steps - output_steps + 1):
            X.append(data[i : i + input_steps])
            y.append(data[i + input_steps : i + input_steps + output_steps])

        X = np.array(X)
        y = np.array(y)

        print(f"Đã tạo sequences:")
        print(f"  Input steps: {input_steps}, Output steps: {output_steps}")
        print(f"  X shape: {X.shape}, y shape: {y.shape}")

        return X, y

    # ==================== DATA SPLITTING ====================

    @staticmethod
    def split_data_temporally(data, train_ratio=None, val_ratio=None):
        """
        Chia dữ liệu theo thời gian (KHÔNG shuffle)
        Đây là cách đúng để tránh data leakage trong time series

        Args:
            data: numpy array 1D
            train_ratio: Tỷ lệ tập train
            val_ratio: Tỷ lệ tập validation

        Returns:
            train_data, val_data, test_data: numpy arrays
        """
        train_ratio = train_ratio or Config.TRAIN_RATIO
        val_ratio = val_ratio or Config.VAL_RATIO

        n = len(data)
        train_size = int(train_ratio * n)
        val_size = int(val_ratio * n)

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        print(f"Chia dữ liệu theo thời gian (train={train_ratio}, val={val_ratio}):")
        print(f"  Total: {n} timesteps")
        print(f"  Train: {len(train_data)} timesteps ({len(train_data)/n*100:.1f}%)")
        print(f"  Val:   {len(val_data)} timesteps ({len(val_data)/n*100:.1f}%)")
        print(f"  Test:  {len(test_data)} timesteps ({len(test_data)/n*100:.1f}%)")

        return train_data, val_data, test_data

    # ==================== SCALING ====================

    def fit_scaler(self, X_train, y_train):
        """
        Fit scaler CHỈ trên tập train (quan trọng để tránh data leakage)

        Args:
            X_train: Training input data
            y_train: Training output data
        """
        all_train_data = np.concatenate([X_train.flatten(), y_train.flatten()])
        self.max_scaler = all_train_data.max()
        self.min_scaler = all_train_data.min()

        print(f"Scaler fitted trên TRAIN data:")
        print(f"  min_scaler = {self.min_scaler:.6f}")
        print(f"  max_scaler = {self.max_scaler:.6f}")

    def transform(self, data):
        """
        Transform dữ liệu bằng scaler đã fit

        Args:
            data: numpy array cần transform

        Returns:
            data_scaled: numpy array đã chuẩn hóa [0, 1]
        """
        if self.min_scaler is None or self.max_scaler is None:
            raise ValueError("Scaler chưa được fit! Gọi fit_scaler() trước.")

        denom = self.max_scaler - self.min_scaler
        if denom == 0:
            denom = 1  # Tránh chia cho 0

        return (data - self.min_scaler) / denom

    def inverse_transform(self, data_scaled):
        """
        Inverse transform để lấy lại giá trị gốc

        Args:
            data_scaled: numpy array đã chuẩn hóa

        Returns:
            data: numpy array giá trị gốc
        """
        if self.min_scaler is None or self.max_scaler is None:
            raise ValueError("Scaler chưa được fit!")

        return data_scaled * (self.max_scaler - self.min_scaler) + self.min_scaler

    def scale_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Fit scaler trên train và transform tất cả các tập

        Args:
            X_train, y_train, X_val, y_val, X_test, y_test: Dữ liệu chưa scale

        Returns:
            Tuple: (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                   X_test_scaled, y_test_scaled)
        """
        # Fit CHỈ trên train
        self.fit_scaler(X_train, y_train)

        # Transform tất cả
        X_train_scaled = self.transform(X_train)
        y_train_scaled = self.transform(y_train)

        X_val_scaled = self.transform(X_val)
        y_val_scaled = self.transform(y_val)

        X_test_scaled = self.transform(X_test)
        y_test_scaled = self.transform(y_test)

        print("✓ Đã scale tất cả các tập dữ liệu")

        return (X_train_scaled, y_train_scaled,
                X_val_scaled, y_val_scaled,
                X_test_scaled, y_test_scaled)

    def save_scaler(self, filepath):
        """
        Lưu scaler parameters vào file

        Args:
            filepath: Đường dẫn file để lưu
        """
        if self.min_scaler is None or self.max_scaler is None:
            raise ValueError("Scaler chưa được fit!")

        np.save(filepath, np.array([self.min_scaler, self.max_scaler]))
        print(f"✓ Đã lưu scaler tại: {filepath}")

    def load_scaler(self, filepath):
        """
        Load scaler parameters từ file

        Args:
            filepath: Đường dẫn file
        """
        scaler_values = np.load(filepath)
        self.min_scaler, self.max_scaler = scaler_values
        print(f"✓ Đã load scaler từ: {filepath}")
        print(f"  min_scaler = {self.min_scaler:.6f}")
        print(f"  max_scaler = {self.max_scaler:.6f}")

    # ==================== RESHAPE FOR DEEP LEARNING ====================

    @staticmethod
    def reshape_for_model(X, n_features=None):
        """
        Reshape dữ liệu thành 3D tensor cho RNN/CNN
        From: (samples, timesteps)
        To: (samples, timesteps, features)

        Args:
            X: numpy array shape (samples, timesteps)
            n_features: Số features (default=1)

        Returns:
            X_reshaped: numpy array shape (samples, timesteps, features)
        """
        n_features = n_features or Config.N_FEATURES

        if len(X.shape) == 2:
            X_reshaped = X.reshape((X.shape[0], X.shape[1], n_features))
            print(f"Reshaped: {X.shape} -> {X_reshaped.shape}")
            return X_reshaped
        elif len(X.shape) == 3:
            print(f"Dữ liệu đã ở dạng 3D: {X.shape}")
            return X
        else:
            raise ValueError(f"Unexpected shape: {X.shape}")


# ==================== PIPELINE FUNCTION ====================

def preprocess_data(data_sensor, add_noise=None):
    """
    Pipeline hoàn chỉnh để preprocess dữ liệu
    Đây là function chính được gọi từ main

    Args:
        data_sensor: numpy array 1D từ sensor
        add_noise: Có thêm noise không (nếu None, lấy từ Config)

    Returns:
        dict chứa tất cả dữ liệu đã xử lý và preprocessor instance
    """
    add_noise = add_noise if add_noise is not None else Config.ADD_NOISE

    preprocessor = DataPreprocessor()

    print("\n" + "=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Data Augmentation (nếu có)
    if add_noise:
        print("\n[1/6] Data Augmentation...")
        noisy_data = preprocessor.add_noise(data_sensor)
        combined_data = np.concatenate([data_sensor, noisy_data])
        print(f"  Kết hợp: {data_sensor.shape} + {noisy_data.shape} = {combined_data.shape}")
    else:
        print("\n[1/6] Bỏ qua Data Augmentation")
        combined_data = data_sensor

    # 2. Chia dữ liệu theo thời gian
    print("\n[2/6] Chia dữ liệu theo thời gian...")
    train_data, val_data, test_data = preprocessor.split_data_temporally(combined_data)

    # 3. Tạo sequences
    print("\n[3/6] Tạo sequences cho từng tập...")
    X_train, y_train = preprocessor.create_sequences(train_data)
    X_val, y_val = preprocessor.create_sequences(val_data)
    X_test, y_test = preprocessor.create_sequences(test_data)

    # 4. Scaling
    print("\n[4/6] Chuẩn hóa dữ liệu (Min-Max Scaling)...")
    (X_train_scaled, y_train_scaled,
     X_val_scaled, y_val_scaled,
     X_test_scaled, y_test_scaled) = preprocessor.scale_data(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # 5. Reshape cho model
    print("\n[5/6] Reshape cho Deep Learning model...")
    X_train_scaled = preprocessor.reshape_for_model(X_train_scaled)
    X_val_scaled = preprocessor.reshape_for_model(X_val_scaled)
    X_test_scaled = preprocessor.reshape_for_model(X_test_scaled)

    # 6. Tổng kết
    print("\n[6/6] Tổng kết:")
    print(f"  X_train: {X_train_scaled.shape}, y_train: {y_train_scaled.shape}")
    print(f"  X_val:   {X_val_scaled.shape}, y_val:   {y_val_scaled.shape}")
    print(f"  X_test:  {X_test_scaled.shape}, y_test:  {y_test_scaled.shape}")

    print("=" * 60)

    return {
        'X_train': X_train_scaled,
        'y_train': y_train_scaled,
        'X_val': X_val_scaled,
        'y_val': y_val_scaled,
        'X_test': X_test_scaled,
        'y_test': y_test_scaled,
        'preprocessor': preprocessor
    }


if __name__ == "__main__":
    # Test preprocessing
    print("Testing Data Preprocessing...")

    # Tạo dummy data
    dummy_data = np.random.randn(10000)

    # Test pipeline
    result = preprocess_data(dummy_data, add_noise=True)

    print("\nResult keys:", result.keys())
    print("Preprocessor:", result['preprocessor'])
