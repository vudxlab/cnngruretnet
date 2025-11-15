"""
Trainer Module
Chịu trách nhiệm training model với early stopping và tracking
"""

import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from config import Config


class ModelTrainer:
    """
    Class quản lý quá trình training
    """

    def __init__(self, model):
        """
        Khởi tạo Trainer

        Args:
            model: Keras model đã compile
        """
        self.model = model
        self.history = None
        self.training_time = 0

    def create_callbacks(self, monitor=None, patience=None, save_best_path=None):
        """
        Tạo callbacks cho training

        Args:
            monitor: Metric để monitor (mặc định: val_loss)
            patience: Early stopping patience
            save_best_path: Đường dẫn lưu best model (nếu None thì không lưu)

        Returns:
            list: Danh sách callbacks
        """
        monitor = monitor or Config.EARLY_STOPPING_MONITOR
        patience = patience or Config.EARLY_STOPPING_PATIENCE

        callbacks = []

        # Early Stopping
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=Config.EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
            verbose=1
        )
        callbacks.append(early_stopping)

        # Model Checkpoint (nếu có path)
        if save_best_path:
            checkpoint = ModelCheckpoint(
                filepath=save_best_path,
                monitor=monitor,
                save_best_only=True,
                verbose=0
            )
            callbacks.append(checkpoint)

        # Reduce Learning Rate on Plateau
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)

        return callbacks

    def train(self, X_train, y_train, X_val, y_val,
             epochs=None, batch_size=None, verbose=None,
             callbacks=None):
        """
        Training model

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Số epochs (nếu None, lấy từ Config)
            batch_size: Batch size (nếu None, lấy từ Config)
            verbose: Verbose level (nếu None, lấy từ Config)
            callbacks: Custom callbacks (nếu None, tự tạo)

        Returns:
            history: Training history
        """
        epochs = epochs or Config.EPOCHS
        batch_size = batch_size or Config.BATCH_SIZE
        verbose = verbose if verbose is not None else Config.VERBOSE

        # Tạo callbacks nếu chưa có
        if callbacks is None:
            callbacks = self.create_callbacks()

        print("\n" + "=" * 60)
        print("BẮT ĐẦU TRAINING")
        print("=" * 60)
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        print("=" * 60 + "\n")

        # Bắt đầu đo thời gian
        start_time = time.time()

        # Training
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )

        # Kết thúc đo thời gian
        end_time = time.time()
        self.training_time = end_time - start_time

        print("\n" + "=" * 60)
        print("HOÀN THÀNH TRAINING")
        print("=" * 60)
        print(f"Tổng thời gian: {self.training_time:.2f} giây")
        print(f"Thời gian/epoch: {self.training_time / len(self.history.history['loss']):.2f} giây")
        print(f"Số epochs đã train: {len(self.history.history['loss'])}")
        print(f"Best val_loss: {min(self.history.history['val_loss']):.6f}")
        print("=" * 60 + "\n")

        return self.history

    def get_history(self):
        """Trả về training history"""
        return self.history

    def get_training_time(self):
        """Trả về thời gian training"""
        return self.training_time

    def get_best_epoch(self, metric='val_loss', mode='min'):
        """
        Tìm epoch tốt nhất theo metric

        Args:
            metric: Tên metric
            mode: 'min' hoặc 'max'

        Returns:
            int: Epoch tốt nhất (1-indexed)
        """
        if self.history is None:
            raise ValueError("Chưa training! Gọi train() trước.")

        if metric not in self.history.history:
            raise ValueError(f"Metric '{metric}' không tồn tại trong history")

        values = self.history.history[metric]

        if mode == 'min':
            best_epoch = values.index(min(values)) + 1
            best_value = min(values)
        else:
            best_epoch = values.index(max(values)) + 1
            best_value = max(values)

        print(f"Best {metric}: {best_value:.6f} at epoch {best_epoch}")
        return best_epoch


def train_model(model, X_train, y_train, X_val, y_val, **kwargs):
    """
    Helper function để train model nhanh

    Args:
        model: Keras model
        X_train, y_train, X_val, y_val: Dữ liệu
        **kwargs: Các tham số khác cho train()

    Returns:
        trainer: ModelTrainer instance
        history: Training history
    """
    trainer = ModelTrainer(model)
    history = trainer.train(X_train, y_train, X_val, y_val, **kwargs)
    return trainer, history


if __name__ == "__main__":
    # Test trainer
    print("Testing Trainer...")

    # Tạo dummy model
    from model import create_model
    import numpy as np

    model = create_model('conv1d_gru', input_steps=50, output_steps=5)

    # Tạo dummy data
    X_train = np.random.rand(1000, 50, 1)
    y_train = np.random.rand(1000, 5)
    X_val = np.random.rand(200, 50, 1)
    y_val = np.random.rand(200, 5)

    # Train
    trainer = ModelTrainer(model)
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=32)

    print(f"\nTraining time: {trainer.get_training_time():.2f}s")
    print(f"History keys: {history.history.keys()}")
