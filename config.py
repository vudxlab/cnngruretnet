"""
Configuration file cho Time Series Forecasting Project
Quản lý tất cả các constants và hyperparameters
"""

import os


class Config:
    """
    Class chứa tất cả cấu hình cho project
    """

    # ==================== DATA PATHS ====================
    DATA_DIR = "Data"
    MAT_FILE = "TH2_SETUP1.mat"
    OUTPUT_DIR = "results/10"

    # ==================== DATA EXTRACTION ====================
    # Slice parameters cho dữ liệu gốc
    START_COL = 300000
    END_COL = 600000
    START_SLICE = 23000
    END_SLICE = 33000

    # Sensor selection (0-7)
    SENSOR_IDX = 0

    # ==================== DATA AUGMENTATION ====================
    # Có thêm noise vào dữ liệu không
    ADD_NOISE = True
    # Hệ số noise (nhân với standard deviation)
    NOISE_FACTOR = 0.03

    # Multiple noise levels for robustness testing
    # Để test nhiều mức độ nhiễu khác nhau (theo đề xuất reviewer)
    NOISE_FACTORS = [0.05, 0.1, 0.15, 0.2]  # Danh sách các mức độ nhiễu
    USE_MULTIPLE_NOISE_LEVELS = False  # Bật để test với nhiều mức độ

    # Augmentation strategies
    AUGMENTATION_STRATEGIES = ['noise']  # Mặc định chỉ dùng noise
    # Các option khả dụng: 'noise', 'dropout', 'block_missingness'

    # Random dropout of data segments
    DROPOUT_PROB = 0.1  # Xác suất dropout cho mỗi timestep
    DROPOUT_MIN_LENGTH = 1  # Độ dài tối thiểu của segment dropout
    DROPOUT_MAX_LENGTH = 5  # Độ dài tối đa của segment dropout

    # Block missingness (simulate sensor failures)
    BLOCK_MISS_PROB = 0.05  # Xác suất xuất hiện block missingness
    BLOCK_MISS_MIN_LENGTH = 3  # Độ dài tối thiểu của block
    BLOCK_MISS_MAX_LENGTH = 10  # Độ dài tối đa của block
    BLOCK_MISS_FILL_METHOD = 'interpolate'  # Phương pháp fill: 'zero', 'mean', 'interpolate'

    # Augmentation multiplier (số lần nhân dữ liệu)
    # Với multiple strategies, mỗi strategy tạo ra 1 bản sao
    AUGMENTATION_MULTIPLIER = 1  # Số lần augment (ngoài noise)

    # ==================== SEQUENCE PARAMETERS ====================
    # Số timesteps cho input (X)
    INPUT_STEPS = 50
    # Số timesteps cho output (y)
    OUTPUT_STEPS = 5
    # Số features (univariate = 1)
    N_FEATURES = 1

    # ==================== DATA SPLIT ====================
    # Tỷ lệ chia dữ liệu (theo thời gian, KHÔNG shuffle)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15  # Tự động = 1 - train - val

    # ==================== MODEL ARCHITECTURE ====================
    # Conv1D parameters
    CONV_FILTERS = 256
    CONV_KERNEL_SIZE = 5
    CONV_ACTIVATION = 'relu'
    CONV_PADDING = 'same'

    # GRU parameters
    GRU_UNITS_1 = 256
    GRU_UNITS_2 = 128
    GRU_UNITS_3 = 64
    GRU_ACTIVATION = 'tanh'
    GRU_RECURRENT_ACTIVATION = 'sigmoid'

    # ==================== TRAINING PARAMETERS ====================
    # Optimizer
    OPTIMIZER = 'adam'
    # Loss function
    LOSS = 'mse'
    # Metrics
    METRICS = ['mae']

    # Training parameters
    EPOCHS = 1000
    BATCH_SIZE = 64
    VALIDATION_SPLIT = 0.0  # Không dùng vì đã chia riêng val set

    # Early stopping
    EARLY_STOPPING_MONITOR = 'val_loss'
    EARLY_STOPPING_PATIENCE = 40
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS = True

    # ==================== PREDICTION PARAMETERS ====================
    # Số samples để predict khi test
    NUM_PREDICTION_SAMPLES = 20
    # Kích thước window khi lấy sample (input + output + buffer)
    PREDICTION_WINDOW_SIZE = 105
    # Số biểu đồ cần vẽ
    NUM_PLOTS = 10

    # ==================== RANDOM SEED ====================
    SEED = 42

    # ==================== OUTPUT FILES ====================
    MODEL_FILE = "model_saved.keras"
    HISTORY_FILE = "history_saved.pkl"
    SCALER_FILE = "scaler_values.npy"
    METRICS_FILE = "metrics.csv"
    TIME_LOG_FILE = "train_time_log.csv"
    LOSS_PLOT_FILE = "loss_plot.png"
    MAE_PLOT_FILE = "mae_plot.png"

    # ==================== LOGGING ====================
    VERBOSE = 1  # 0: silent, 1: progress bar, 2: one line per epoch

    @classmethod
    def get_mat_file_path(cls):
        """Trả về đường dẫn đầy đủ đến file .mat"""
        return os.path.join(cls.DATA_DIR, cls.MAT_FILE)

    @classmethod
    def get_output_path(cls, filename):
        """Trả về đường dẫn đầy đủ đến file output"""
        return os.path.join(cls.OUTPUT_DIR, filename)

    @classmethod
    def create_output_dir(cls):
        """Tạo thư mục output nếu chưa có"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

    @classmethod
    def update_from_args(cls, args):
        """
        Cập nhật config từ command line arguments

        Args:
            args: argparse.Namespace object
        """
        for key, value in vars(args).items():
            key_upper = key.upper()
            if hasattr(cls, key_upper):
                setattr(cls, key_upper, value)

    @classmethod
    def print_config(cls):
        """In ra tất cả cấu hình hiện tại"""
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)

        sections = [
            ("DATA PATHS", ["DATA_DIR", "MAT_FILE", "OUTPUT_DIR"]),
            ("DATA EXTRACTION", ["START_COL", "END_COL", "START_SLICE",
                               "END_SLICE", "SENSOR_IDX"]),
            ("DATA AUGMENTATION", ["ADD_NOISE", "NOISE_FACTOR", "USE_MULTIPLE_NOISE_LEVELS",
                                  "NOISE_FACTORS", "AUGMENTATION_STRATEGIES",
                                  "DROPOUT_PROB", "BLOCK_MISS_PROB"]),
            ("SEQUENCE PARAMETERS", ["INPUT_STEPS", "OUTPUT_STEPS", "N_FEATURES"]),
            ("DATA SPLIT", ["TRAIN_RATIO", "VAL_RATIO", "TEST_RATIO"]),
            ("MODEL ARCHITECTURE", ["CONV_FILTERS", "CONV_KERNEL_SIZE",
                                   "GRU_UNITS_1", "GRU_UNITS_2", "GRU_UNITS_3"]),
            ("TRAINING PARAMETERS", ["OPTIMIZER", "LOSS", "EPOCHS",
                                   "BATCH_SIZE", "EARLY_STOPPING_PATIENCE"]),
            ("RANDOM SEED", ["SEED"])
        ]

        for section_name, keys in sections:
            print(f"\n{section_name}:")
            for key in keys:
                value = getattr(cls, key)
                print(f"  {key}: {value}")

        print("=" * 60)


# Tạo instance để dễ sử dụng
config = Config()


if __name__ == "__main__":
    # Test config
    Config.print_config()

    # Test path methods
    print(f"\nMAT file path: {Config.get_mat_file_path()}")
    print(f"Model output path: {Config.get_output_path(Config.MODEL_FILE)}")
