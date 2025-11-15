"""
Baseline Models Module
Các model baseline để so sánh với Deep Learning models:
- Linear Regression
- XGBoost Regressor
- LightGBM Regressor
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor


class BaselineModel:
    """
    Base class cho các baseline models
    """

    def __init__(self, model_name="Baseline"):
        self.model_name = model_name
        self.model = None
        self.is_fitted = False

    def fit(self, X, y):
        """
        Train baseline model

        Args:
            X: Input data
            y: Target data
        """
        raise NotImplementedError("Subclass must implement fit()")

    def predict(self, X):
        """
        Predict với baseline model

        Args:
            X: Input data

        Returns:
            predictions
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được fit! Gọi fit() trước.")
        return self.model.predict(X)

    def evaluate(self, X, y, preprocessor=None):
        """
        Evaluate baseline model

        Args:
            X: Input data (đã scaled)
            y: True values (đã scaled)
            preprocessor: DataPreprocessor để denormalize

        Returns:
            dict: Metrics (RMSE, MAE, R²)
        """
        # Predict
        y_pred_scaled = self.predict(X)

        # Denormalize nếu có preprocessor
        if preprocessor:
            y_real = preprocessor.inverse_transform(y)
            y_pred = preprocessor.inverse_transform(y_pred_scaled)
        else:
            y_real = y
            y_pred = y_pred_scaled

        # Tính metrics
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        mae = mean_absolute_error(y_real, y_pred)
        r2 = r2_score(y_real, y_pred)

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }


class LinearRegressionBaseline(BaselineModel):
    """
    Linear Regression Baseline
    """

    def __init__(self):
        super().__init__(model_name="Linear Regression")
        self.model = LinearRegression()

    def fit(self, X, y):
        """
        Fit Linear Regression

        Args:
            X: Input data shape (samples, timesteps, features) or (samples, timesteps)
            y: Target data
        """
        # Reshape X nếu là 3D
        if len(X.shape) == 3:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X

        print(f"Training {self.model_name}...")
        print(f"  X shape: {X_reshaped.shape}, y shape: {y.shape}")

        self.model.fit(X_reshaped, y)
        self.is_fitted = True

        print(f"✓ {self.model_name} đã fit xong")

    def predict(self, X):
        """Predict với Linear Regression"""
        if not self.is_fitted:
            raise ValueError("Model chưa được fit!")

        # Reshape X nếu là 3D
        if len(X.shape) == 3:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X

        return self.model.predict(X_reshaped)


class XGBoostBaseline(BaselineModel):
    """
    XGBoost Baseline
    """

    def __init__(self, random_state=42, **kwargs):
        super().__init__(model_name="XGBoost")

        try:
            from xgboost import XGBRegressor
            base_model = XGBRegressor(random_state=random_state, **kwargs)
            self.model = MultiOutputRegressor(base_model)
        except ImportError:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )

    def fit(self, X, y):
        """Fit XGBoost"""
        # Reshape X nếu là 3D
        if len(X.shape) == 3:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X

        print(f"Training {self.model_name}...")
        print(f"  X shape: {X_reshaped.shape}, y shape: {y.shape}")

        self.model.fit(X_reshaped, y)
        self.is_fitted = True

        print(f"✓ {self.model_name} đã fit xong")

    def predict(self, X):
        """Predict với XGBoost"""
        if not self.is_fitted:
            raise ValueError("Model chưa được fit!")

        # Reshape X nếu là 3D
        if len(X.shape) == 3:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X

        return self.model.predict(X_reshaped)


class LightGBMBaseline(BaselineModel):
    """
    LightGBM Baseline
    """

    def __init__(self, random_state=42, **kwargs):
        super().__init__(model_name="LightGBM")

        try:
            from lightgbm import LGBMRegressor
            # Thêm verbose=-1 để giảm log
            kwargs_with_verbose = {'verbose': -1, 'random_state': random_state, **kwargs}
            base_model = LGBMRegressor(**kwargs_with_verbose)
            self.model = MultiOutputRegressor(base_model)
        except ImportError:
            raise ImportError(
                "LightGBM not installed. Install with: pip install lightgbm"
            )

    def fit(self, X, y):
        """Fit LightGBM"""
        # Reshape X nếu là 3D
        if len(X.shape) == 3:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X

        print(f"Training {self.model_name}...")
        print(f"  X shape: {X_reshaped.shape}, y shape: {y.shape}")

        self.model.fit(X_reshaped, y)
        self.is_fitted = True

        print(f"✓ {self.model_name} đã fit xong")

    def predict(self, X):
        """Predict với LightGBM"""
        if not self.is_fitted:
            raise ValueError("Model chưa được fit!")

        # Reshape X nếu là 3D
        if len(X.shape) == 3:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X

        return self.model.predict(X_reshaped)


# ==================== FACTORY FUNCTION ====================

def create_baseline_model(model_type='linear', **kwargs):
    """
    Factory function để tạo baseline model

    Args:
        model_type: 'linear', 'xgboost', 'lightgbm'
        **kwargs: Các tham số cho model

    Returns:
        baseline_model: Instance của baseline model
    """
    models = {
        'linear': LinearRegressionBaseline,
        'xgboost': XGBoostBaseline,
        'lightgbm': LightGBMBaseline,
        'lr': LinearRegressionBaseline,  # Alias
        'xgb': XGBoostBaseline,  # Alias
        'lgbm': LightGBMBaseline  # Alias
    }

    model_type_lower = model_type.lower()

    if model_type_lower not in models:
        raise ValueError(
            f"model_type '{model_type}' không hợp lệ. "
            f"Chọn từ: {list(set(models.keys()))}"
        )

    return models[model_type_lower](**kwargs)


if __name__ == "__main__":
    # Test baseline models
    print("Testing Baseline Models...")

    # Tạo dummy data
    X_train = np.random.rand(1000, 50)
    y_train = np.random.rand(1000, 5)
    X_test = np.random.rand(200, 50)
    y_test = np.random.rand(200, 5)

    # Test Linear Regression
    print("\n1. Testing Linear Regression:")
    lr = create_baseline_model('linear')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print(f"Predictions shape: {y_pred.shape}")

    # Test với 3D input
    print("\n2. Testing with 3D input:")
    X_train_3d = X_train.reshape(1000, 50, 1)
    X_test_3d = X_test.reshape(200, 50, 1)
    lr2 = create_baseline_model('linear')
    lr2.fit(X_train_3d, y_train)
    y_pred2 = lr2.predict(X_test_3d)
    print(f"Predictions shape: {y_pred2.shape}")

    # Test XGBoost (nếu có)
    print("\n3. Testing XGBoost:")
    try:
        xgb = create_baseline_model('xgboost')
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        print(f"XGBoost predictions shape: {y_pred_xgb.shape}")
    except ImportError as e:
        print(f"XGBoost not available: {e}")

    # Test LightGBM (nếu có)
    print("\n4. Testing LightGBM:")
    try:
        lgbm = create_baseline_model('lightgbm')
        lgbm.fit(X_train, y_train)
        y_pred_lgbm = lgbm.predict(X_test)
        print(f"LightGBM predictions shape: {y_pred_lgbm.shape}")
    except ImportError as e:
        print(f"LightGBM not available: {e}")

    print("\n✓ Baseline models test completed!")
