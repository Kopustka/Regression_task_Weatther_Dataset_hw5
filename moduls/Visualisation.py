import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_csv

# 🔹 Global visualization settings
plt.style.use('ggplot')  # Apply ggplot style to all plots
sns.set_theme(style="whitegrid")  # Set Seaborn theme for consistent visualization

# Configure font sizes and plot properties
plt.rcParams.update({
    'figure.figsize': (10, 6),  # Default figure size
    'axes.labelsize': 14,  # Font size for axis labels
    'axes.titlesize': 16,  # Font size for titles
    'xtick.labelsize': 12,  # Font size for X-axis ticks
    'ytick.labelsize': 12,  # Font size for Y-axis ticks
    'legend.fontsize': 12,  # Font size for legends
    'grid.alpha': 0.3  # Grid transparency
})


# 🔹 Function to plot histogram (distribution of values)
def plot_histogram(data, column, bins=30):
    """
    Plots a histogram to show the distribution of values.

    Parameters:
        data (pd.DataFrame): The dataset.
        column (str): The column to visualize.
        bins (int): Number of bins for the histogram.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(data[column], bins=bins, kde=True, color='royalblue')
    plt.title(f'Distribution of {column}', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()


# 🔹 Function to plot scatter plot (relationship between two features)
def plot_scatter(df, x_col, y_col):
    """
    Creates a scatter plot to show the relationship between two features.

    Parameters:
        df (pd.DataFrame): The dataset.
        x_col (str): The column for the X-axis.
        y_col (str): The column for the Y-axis.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=df[x_col], y=df[y_col], alpha=0.7, edgecolor=None)
    plt.title(f'Relationship between {x_col} and {y_col}', fontsize=16)
    plt.xlabel(x_col, fontsize=14)
    plt.ylabel(y_col, fontsize=14)
    plt.grid(True)
    plt.show()


# 🔹 Function to generate a correlation matrix
def plot_correlation_matrix(df):
    """
    Plots the correlation matrix to examine feature relationships.

    Parameters:
        df (pd.DataFrame): The dataset.

    Returns:
        None
    """
    print('Checking for non-linearity...\n')

    plt.figure(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=["number"])  # Select numerical columns
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix of Weather Parameters")
    plt.show()

    print('The target variable does not have significant correlation with any feature.')


# 🔹 Function to detect temperature outliers
def finding_outliers(df):
    """
    Identifies temperature outliers using a boxplot.

    Parameters:
        df (pd.DataFrame): The dataset.

    Returns:
        None
    """
    print('Checking for temperature outliers...\n')
    sns.boxplot(df["Temperature (C)"])
    plt.show()


# 🔹 Function to visualize prediction analysis
def prediction_analysics_visualisation(y_test, y_pred):
    """
    Compares actual vs. predicted values visually.

    Parameters:
        y_test (pd.Series): Actual target values.
        y_pred (np.array): Predicted target values.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values[:100], label="Actual", marker="o")
    plt.plot(y_pred[:100], label="Predicted", marker="x")
    plt.legend()
    plt.title("Comparison of Actual and Predicted Values")
    plt.show()
