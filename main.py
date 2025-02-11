from joblib import PrintTime
from moduls import importing, csv_reader

from moduls import (removing_gaps,
                    time_preprocessing,
                    removing_Loud_Cover,
                    outliers_deletion,
                    encoder)

from moduls import (plot_scatter,
                    plot_histogram,
                    plot_correlation_matrix,
                    finding_outliers,
                    prediction_analysics_visualisation)

from moduls import (Data_separation,
                    Training_RandomForset,
                    XGBoost,
                    CatBoost,
                    Gradient_Boosting,
                    SVR_model,
                    Extra_Trees,
                    Bayesian_Ridge)

from moduls import models

path = 'datas/weatherHistory.csv'

if __name__ == "__main__":
    """ STEP 1: Importing Data """
    df = csv_reader(path)

    # Alternative: Downloading the dataset from Kaggle (currently commented out)
    # importing('muthuj7/weather-dataset', '../datas')

    """ STEP 2: Data Analysis and Visualization """
    print('Analysis...')
    print(df.describe())

    finding_outliers(df)

    print('Visualization...')
    plot_histogram(df, 'Temperature (C)')
    plot_correlation_matrix(df)

    """ STEP 3: Data Preprocessing """
    print("Preprocessing...")
    df = removing_gaps(df)

    print("Checking for missing values")
    print(df.isnull().sum())

    df = time_preprocessing(df)
    df = removing_Loud_Cover(df)
    df = outliers_deletion(df)

    df = encoder(df)
    print("\n🔍 Data types after encoding:\n", df.dtypes)

    """ STEP 4: Model Training """
    print('Training...')
    X_train, X_test, y_train, y_test = Data_separation(df)

    Training_RandomForset(X_train, y_train, X_test, y_test, models)
    # Extra_Trees(X_train, y_train, X_test, y_test, models)
    # XGBoost(X_train, y_train, X_test, y_test, models)
    # CatBoost(X_train, y_train, X_test, y_test, models)
    # Gradient_Boosting(X_train, y_train, X_test, y_test, models)
    # SVR_model(X_train, y_train, X_test, y_test, models)
    # Bayesian_Ridge(X_train, y_train, X_test, y_test, models)
