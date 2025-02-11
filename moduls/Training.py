from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas

from moduls import prediction_analysics_visualisation  # Import function for visualization of predictions

# 🔹 Dictionary containing different machine learning models with optimized hyperparameters
models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=4, min_samples_leaf=2,
                                           random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=200, max_depth=20, min_samples_split=4, min_samples_leaf=2,
                                       random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                            random_state=42),
    "CatBoost": CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=8, l2_leaf_reg=3, verbose=100,
                                  random_seed=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8,
                                                   random_state=42),
    "SVR": SVR(kernel='rbf', C=10, gamma='scale'),
    "Bayesian Ridge": BayesianRidge()
}


def Data_separation(df):
    """
    Splits the dataset into training and testing sets.

    This function:
    - Shuffles the dataset.
    - Separates features (X) and target variable (y).
    - Splits the data into 80% training and 20% testing.

    Parameters:
        df (pd.DataFrame): The input dataset.

    Returns:
        X_train, X_test, y_train, y_test: Split training and testing sets.
    """
    df = df.sample(frac=1, random_state=42)  # Shuffle dataset to avoid bias

    # Separate features (X) and target variable (y)
    X = df.drop(columns=["Temperature (C)", "Apparent Temperature (C)"])
    y = df["Temperature (C)"]

    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def evaluate_model(func):
    """
    Decorator function to train a model, evaluate it, and visualize predictions.

    This function:
    - Trains the model.
    - Makes predictions on the test set.
    - Calculates and prints performance metrics (MAE, MSE, R²).
    - Visualizes the model's predictions.

    Parameters:
        func (function): The function that trains the model.

    Returns:
        The trained model.
    """

    def wrapper(X_train, y_train, X_test, y_test, models):
        model = func(X_train, y_train, models)  # Train the model
        y_pred = model.predict(X_test)  # Make predictions

        # Evaluate model performance
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n📌 {func.__name__} Results:")
        print(f"🔹 MAE: {mae:.2f}")
        print(f"🔹 MSE: {mse:.2f}")
        print(f"🔹 R² Score: {r2:.2f}")

        # Visualize predictions
        prediction_analysics_visualisation(y_test, y_pred)

        return model  # Return the trained model

    return wrapper


# 🔹 Model training functions, wrapped with the evaluation decorator

@evaluate_model
def Training_RandomForset(X_train, y_train, models):
    """Trains a Random Forest model."""
    model = models["Random Forest"]
    model.fit(X_train, y_train)
    return model


@evaluate_model
def Extra_Trees(X_train, y_train, models):
    """Trains an Extra Trees model."""
    model = models["Extra Trees"]
    model.fit(X_train, y_train)
    return model


@evaluate_model
def XGBoost(X_train, y_train, models):
    """Trains an XGBoost model."""
    model = models["XGBoost"]
    model.fit(X_train, y_train)
    return model


@evaluate_model
def CatBoost(X_train, y_train, models):
    """Trains a CatBoost model."""
    model = models["CatBoost"]
    model.fit(X_train, y_train)
    return model


@evaluate_model
def Gradient_Boosting(X_train, y_train, models):
    """Trains a Gradient Boosting model."""
    model = models["Gradient Boosting"]
    model.fit(X_train, y_train)
    return model


@evaluate_model
def SVR_model(X_train, y_train, models):
    """Trains a Support Vector Regression (SVR) model."""
    model = models["SVR"]
    model.fit(X_train, y_train)
    return model


@evaluate_model
def Bayesian_Ridge(X_train, y_train, models):
    """Trains a Bayesian Ridge regression model."""
    model = models["Bayesian Ridge"]
    model.fit(X_train, y_train)
    return model
