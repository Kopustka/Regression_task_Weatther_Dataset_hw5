import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


def removing_gaps(df):
    """
    Cleans the dataset by removing unnecessary columns and handling missing values.

    This function:
    - Drops the "Precip Type" column if it exists.
    - Removes rows containing missing values.

    Parameters:
        df (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: The cleaned dataset with no missing values.
    """
    print("Cleaning gaps and missing values...\n")
    df = df.drop(columns=["Precip Type"], errors="ignore")  # Remove "Precip Type" if it exists
    df = df.dropna()  # Remove all rows with NaN values

    return df


def removing_Loud_Cover(df):
    """
    Removes the 'Loud Cover' column from the dataset.

    Parameters:
        df (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: The dataset without the 'Loud Cover' column.
    """
    print("Removing the 'Loud Cover' column...\n")
    df = df.drop(columns=['Loud Cover'], errors="ignore")  # Avoids errors if the column is missing

    return df


def time_preprocessing(df):
    """
    Converts the 'Formatted Date' column to a datetime format and extracts time-based features.

    This function:
    - Converts 'Formatted Date' to datetime format.
    - Extracts 'Year', 'Month', 'Day', and 'Hour' features.
    - Removes the original 'Formatted Date' column.

    Parameters:
        df (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: The dataset with new time-related features.
    """
    print("Formatting time data...\n")
    df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], errors='coerce', utc=True)  # Convert to datetime

    # Check for conversion errors (NaT values)
    print(f"Missing values in 'Formatted Date': {df['Formatted Date'].isna().sum()}")

    # Extract relevant date-time features
    df["Year"] = df["Formatted Date"].dt.year
    df["Month"] = df["Formatted Date"].dt.month
    df["Day"] = df["Formatted Date"].dt.day
    df["Hour"] = df["Formatted Date"].dt.hour

    df = df.drop(columns=["Formatted Date"])  # Remove the original column

    return df


def outliers_deletion(df):
    """
    Removes temperature outliers from the dataset.

    The dataset contains temperature values that should not be below -16.1°C.
    Any row with a temperature lower than -16.1°C will be removed.

    Parameters:
        df (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: The dataset without extreme temperature outliers.
    """
    print("Removing temperature outliers...\n")
    df = df.loc[df['Temperature (C)'] >= -16.1]  # Keep only valid temperature values

    return df


def encoder(df):
    """
    Encodes categorical (text-based) features into numerical values using Label Encoding.

    This function:
    - Identifies categorical columns (dtype == object).
    - Applies Label Encoding to convert them into numerical values.

    Parameters:
        df (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: The dataset with categorical features converted to numeric values.
    """
    print("Encoding categorical text data...\n")
    categorical_features = df.select_dtypes(include=["object"]).columns  # Identify categorical columns

    if len(categorical_features) > 0:
        encoder = LabelEncoder()
        for col in categorical_features:
            df[col] = encoder.fit_transform(df[col])  # Apply label encoding

    return df
