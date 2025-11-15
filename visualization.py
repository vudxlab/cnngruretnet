"""
Visualization Module
Chịu trách nhiệm vẽ biểu đồ (loss, MAE, predictions)
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from config import Config


class Visualizer:
    """
    Class vẽ các loại biểu đồ
    """

    @staticmethod
    def plot_training_history(history, save_path=None):
        """
        Vẽ biểu đồ training history (Loss và MAE)

        Args:
            history: Keras training history
            save_path: Thư mục lưu hình (nếu None, chỉ show)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot Loss
        axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epochs', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Plot MAE
        if 'mae' in history.history:
            axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
            axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
            axes[1].set_xlabel('Epochs', fontsize=12)
            axes[1].set_ylabel('MAE', fontsize=12)
            axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            loss_path = Config.get_output_path(Config.LOSS_PLOT_FILE)
            plt.savefig(loss_path, dpi=150, bbox_inches='tight')
            print(f"✓ Đã lưu biểu đồ tại: {loss_path}")

        plt.show()

    @staticmethod
    def plot_predictions(X, y_true, y_pred, num_samples=5, preprocessor=None,
                        save_dir=None):
        """
        Vẽ biểu đồ so sánh predictions vs actual

        Args:
            X: Input data (scaled hoặc real)
            y_true: True values (scaled hoặc real)
            y_pred: Predictions (scaled hoặc real)
            num_samples: Số samples cần vẽ
            preprocessor: DataPreprocessor để denormalize (nếu có)
            save_dir: Thư mục lưu hình
        """
        num_samples = min(num_samples, len(X))

        # Denormalize nếu có preprocessor
        if preprocessor:
            if X.shape[-1] == 1:
                X = X.reshape(X.shape[0], X.shape[1])
            X_plot = preprocessor.inverse_transform(X)
            y_true_plot = preprocessor.inverse_transform(y_true)
            y_pred_plot = preprocessor.inverse_transform(y_pred)
        else:
            X_plot = X if len(X.shape) == 2 else X.reshape(X.shape[0], X.shape[1])
            y_true_plot = y_true
            y_pred_plot = y_pred

        for i in range(num_samples):
            plt.figure(figsize=(16, 4))

            input_steps = X_plot.shape[1]
            output_steps = y_true_plot.shape[1]

            # Vẽ input (past data)
            time_input = np.arange(input_steps)
            plt.plot(time_input, X_plot[i], marker='s',
                    label='Past Data (Input)', color='green', linewidth=2)

            # Vẽ actual future
            time_future = np.arange(input_steps, input_steps + output_steps)
            plt.plot(time_future, y_true_plot[i], marker='o',
                    label='Actual Future', color='blue', linewidth=2)

            # Vẽ predicted future
            plt.plot(time_future, y_pred_plot[i], marker='D',
                    label='Predicted Future', color='red',
                    linestyle='dashed', linewidth=2)

            # Nối điểm cuối past với actual
            plt.plot([input_steps - 1, input_steps],
                    [X_plot[i, -1], y_true_plot[i, 0]],
                    color='blue', linewidth=2)

            # Nối điểm cuối past với predicted
            plt.plot([input_steps - 1, input_steps],
                    [X_plot[i, -1], y_pred_plot[i, 0]],
                    color='red', linestyle='dashed', linewidth=2)

            # Thêm đường y=0
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

            plt.xlabel("Time Step", fontsize=14)
            plt.ylabel("Value", fontsize=14)
            plt.title(f"Time Series Prediction - Sample {i+1}", fontsize=16, fontweight='bold')
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)

            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"prediction_sample_{i+1}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"✓ Saved: {save_path}")

            plt.show()

    @staticmethod
    def plot_data_overview(data, title="Data Overview", max_points=10000):
        """
        Vẽ tổng quan về dữ liệu

        Args:
            data: numpy array 1D hoặc 2D
            title: Tiêu đề
            max_points: Số điểm tối đa để vẽ (tránh quá chậm)
        """
        plt.figure(figsize=(16, 4))

        if len(data) > max_points:
            # Downsample nếu quá nhiều điểm
            indices = np.linspace(0, len(data) - 1, max_points, dtype=int)
            data_plot = data[indices]
            x_plot = indices
        else:
            data_plot = data
            x_plot = np.arange(len(data))

        plt.plot(x_plot, data_plot, linewidth=1)
        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_metrics_comparison(metrics_dict, save_path=None):
        """
        Vẽ biểu đồ so sánh metrics giữa train/val/test

        Args:
            metrics_dict: Dict chứa metrics {'train': {...}, 'val': {...}, 'test': {...}}
            save_path: Đường dẫn lưu hình
        """
        datasets = ['train', 'val', 'test']
        dataset_labels = ['Train', 'Validation', 'Test']

        rmse_values = [metrics_dict[d]['rmse'] for d in datasets if d in metrics_dict]
        mae_values = [metrics_dict[d]['mae'] for d in datasets if d in metrics_dict]
        r2_values = [metrics_dict[d]['r2'] for d in datasets if d in metrics_dict]

        labels = [dataset_labels[i] for i, d in enumerate(datasets) if d in metrics_dict]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # RMSE
        axes[0].bar(labels, rmse_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0].set_ylabel('RMSE', fontsize=12)
        axes[0].set_title('Root Mean Squared Error', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')

        # MAE
        axes[1].bar(labels, mae_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        # R²
        axes[2].bar(labels, r2_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[2].set_ylabel('R²', fontsize=12)
        axes[2].set_title('R² Score', fontsize=13, fontweight='bold')
        axes[2].set_ylim([0, 1])
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Đã lưu biểu đồ metrics tại: {save_path}")

        plt.show()


if __name__ == "__main__":
    # Test visualizer
    print("Testing Visualizer...")

    # Test plot training history
    class MockHistory:
        def __init__(self):
            self.history = {
                'loss': np.random.rand(50) * 0.1,
                'val_loss': np.random.rand(50) * 0.1 + 0.02,
                'mae': np.random.rand(50) * 0.05,
                'val_mae': np.random.rand(50) * 0.05 + 0.01,
            }

    history = MockHistory()
    Visualizer.plot_training_history(history)

    # Test plot predictions
    X = np.random.randn(10, 50)
    y_true = np.random.randn(10, 5)
    y_pred = y_true + np.random.randn(10, 5) * 0.1

    Visualizer.plot_predictions(X, y_true, y_pred, num_samples=2)

    print("✓ Visualization test completed!")
