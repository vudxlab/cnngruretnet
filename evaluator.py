"""
Evaluator Module
Chịu trách nhiệm đánh giá model với các metrics (RMSE, MAE, R²)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import Config


class ModelEvaluator:
    """
    Class đánh giá hiệu suất model
    """

    def __init__(self, model, preprocessor):
        """
        Khởi tạo Evaluator

        Args:
            model: Keras model đã train
            preprocessor: DataPreprocessor instance (có scaler)
        """
        self.model = model
        self.preprocessor = preprocessor
        self.metrics = {}

    def predict(self, X):
        """
        Dự đoán với model

        Args:
            X: Input data (đã scaled)

        Returns:
            y_pred: Predictions (đã scaled)
        """
        return self.model.predict(X, verbose=0)

    def evaluate_dataset(self, X, y, dataset_name='test'):
        """
        Đánh giá trên một dataset

        Args:
            X, y: Dữ liệu đã scaled
            dataset_name: Tên dataset ('train', 'val', 'test')

        Returns:
            dict: Metrics (RMSE, MAE, R²)
        """
        # Predict
        y_pred_scaled = self.predict(X)

        # Denormalize
        y_real = self.preprocessor.inverse_transform(y)
        y_pred = self.preprocessor.inverse_transform(y_pred_scaled)

        # Tính metrics
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        mae = mean_absolute_error(y_real, y_pred)
        r2 = r2_score(y_real, y_pred)

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

        # Lưu vào self.metrics
        self.metrics[dataset_name] = metrics

        return metrics

    def evaluate_all(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Đánh giá trên tất cả datasets

        Args:
            X_train, y_train, X_val, y_val, X_test, y_test: Dữ liệu đã scaled

        Returns:
            dict: Metrics cho cả 3 datasets
        """
        print("\n" + "=" * 60)
        print("ĐÁNH GIÁ MÔ HÌNH")
        print("=" * 60)

        # Evaluate từng dataset
        print("Đánh giá train set...")
        train_metrics = self.evaluate_dataset(X_train, y_train, 'train')

        print("Đánh giá validation set...")
        val_metrics = self.evaluate_dataset(X_val, y_val, 'val')

        print("Đánh giá test set...")
        test_metrics = self.evaluate_dataset(X_test, y_test, 'test')

        # In kết quả
        self.print_metrics()

        return self.metrics

    def print_metrics(self):
        """In bảng metrics"""
        print("\n" + "=" * 60)
        print("KẾT QUẢ ĐÁNH GIÁ")
        print("=" * 60)
        print(f"{'Dataset':<12} {'RMSE':<15} {'MAE':<15} {'R²':<10}")
        print("-" * 60)

        for dataset in ['train', 'val', 'test']:
            if dataset in self.metrics:
                m = self.metrics[dataset]
                dataset_display = dataset.capitalize()
                if dataset == 'val':
                    dataset_display = 'Validation'

                print(f"{dataset_display:<12} {m['rmse']:<15.6f} "
                     f"{m['mae']:<15.6f} {m['r2']:<10.4f}")

        print("=" * 60)

    def save_metrics(self, filepath=None):
        """
        Lưu metrics vào CSV

        Args:
            filepath: Đường dẫn file (nếu None, dùng từ Config)
        """
        if not self.metrics:
            raise ValueError("Chưa có metrics! Gọi evaluate_all() trước.")

        filepath = filepath or Config.get_output_path(Config.METRICS_FILE)

        # Tạo DataFrame
        data = []
        for dataset in ['train', 'val', 'test']:
            if dataset in self.metrics:
                m = self.metrics[dataset]
                dataset_name = {
                    'train': 'Train',
                    'val': 'Validation',
                    'test': 'Test'
                }[dataset]

                data.append({
                    'Dataset': dataset_name,
                    'RMSE': m['rmse'],
                    'MAE': m['mae'],
                    'R2': m['r2']
                })

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')

        print(f"✓ Đã lưu metrics tại: {filepath}")

    def get_metrics(self, dataset=None):
        """
        Lấy metrics

        Args:
            dataset: Tên dataset ('train', 'val', 'test')
                    Nếu None, trả về tất cả

        Returns:
            dict hoặc metrics của dataset
        """
        if dataset is None:
            return self.metrics

        if dataset not in self.metrics:
            raise ValueError(f"Dataset '{dataset}' chưa được evaluate")

        return self.metrics[dataset]


def evaluate_model(model, preprocessor, X_train, y_train,
                  X_val, y_val, X_test, y_test,
                  save_metrics=True):
    """
    Helper function để evaluate model nhanh

    Args:
        model: Keras model
        preprocessor: DataPreprocessor instance
        X_train, y_train, X_val, y_val, X_test, y_test: Dữ liệu
        save_metrics: Có lưu metrics không

    Returns:
        evaluator: ModelEvaluator instance
        metrics: Dict chứa metrics
    """
    evaluator = ModelEvaluator(model, preprocessor)
    metrics = evaluator.evaluate_all(X_train, y_train, X_val, y_val, X_test, y_test)

    if save_metrics:
        evaluator.save_metrics()

    return evaluator, metrics


if __name__ == "__main__":
    # Test evaluator
    print("Testing Evaluator...")

    from model import create_model
    from data_preprocessing import DataPreprocessor
    import numpy as np

    # Tạo dummy model và data
    model = create_model('conv1d_gru')
    preprocessor = DataPreprocessor()
    preprocessor.min_scaler = 0
    preprocessor.max_scaler = 1

    X_test = np.random.rand(100, 50, 1)
    y_test = np.random.rand(100, 5)

    # Evaluate
    evaluator = ModelEvaluator(model, preprocessor)
    metrics = evaluator.evaluate_dataset(X_test, y_test, 'test')

    print("\nMetrics:", metrics)
