"""
BikeShare Pipeline
==================
This file is responsible for collecting and executing all preprocessing steps.
This file will be called from the UI files.
"""

# Works both as a package (relative) and when called from a notebook (absolute)
try:
    from .preprocessor import BikeSharePreprocessor
except ImportError:
    from preprocessor import BikeSharePreprocessor


def run_bikeshare_pipeline(file_path):
    """
    This function executes the full preprocessing pipeline and returns the prepared datasets.

    Parameters:
    file_path (str): Path to the CSV data file

    Returns:
    tuple: (df_clean, df_processed)
        - df_clean: Cleaned data used for analysis and visualization (EDA).
        - df_processed: Encoded and scaled data used for Machine Learning (ML).
    """

    # 1. Initialize the preprocessor object
    preprocessor = BikeSharePreprocessor(file_path)

    # 2. Execute preprocessing steps in sequence (Pipeline call)
    (preprocessor.load_data()
                 .clean_data()
                 .engineer_features()
                 .encode_and_scale())

    # 3. Retrieve final datasets
    df_clean = preprocessor.get_data()
    df_processed = preprocessor.get_processed_data()

    print("Pipeline executed successfully!")

    return df_clean, df_processed


if __name__ == "__main__":
    file_path = "C:/Users/Test/Desktop/DEPI-ONL4_AIS2_S2/dataAnalysis/Finalproject/code/data/fordgobike-tripdataFor201902.csv"

    clean_data, ml_data = run_bikeshare_pipeline(file_path)

    print(f"Clean Data Shape: {clean_data.shape}")
    print(f"ML Data Shape:    {ml_data.shape}")