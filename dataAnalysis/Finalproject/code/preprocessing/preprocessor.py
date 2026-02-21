"""
BikeShare Pipeline
==================

This module contains the BikeSharePreprocessor class.

Responsibilities:
- Load data
- Clean data
- Engineer features
- Encode and scale data
- Provide processed dataset
"""

import pandas as pd
import numpy as np


class BikeSharePreprocessor:
    """
    Pipeline class to process BikeShare dataset step by step.
    """

    def __init__(self, file_path):
        """
        Initialize pipeline with dataset path.

        Parameters:
        file_path (str): path to CSV file
        """

        self.file_path = file_path
        self.df = None
        self.df_processed = None


    def load_data(self):
        """
        Load dataset from CSV file.

        Returns:
        self: allows method chaining
        """

        self.df = pd.read_csv(self.file_path, low_memory=False)

        if self.df is None or self.df.empty:
            raise ValueError("Dataset failed to load or is empty.")

        print("Shape:", self.df.shape)

        return self


    def clean_data(self):
        """
        Clean dataset and fix data quality issues.

        Steps:
        - Remove missing station IDs
        - Fill missing gender using mode
        - Remove missing birth year
        - Convert station IDs to integer
        - Create age column
        - Remove unrealistic ages
        - Remove duplicates
        - Remove duration outliers using IQR
        """

        df = self.df.copy()

        df.dropna(
            subset=[
                'start_station_id',
                'end_station_id'
            ],
            inplace=True
        )

        if df['member_gender'].isna().sum() > 0:
            mode_gender = df['member_gender'].mode()[0]
            df['member_gender'] = df['member_gender'].fillna(mode_gender)

        df.dropna(
            subset=['member_birth_year'],
            inplace=True
        )

        df['start_station_id'] = df['start_station_id'].astype(int)
        df['end_station_id'] = df['end_station_id'].astype(int)

        dataset_year = 2019
        df['age'] = dataset_year - df['member_birth_year']

        df = df[
            (df['age'] >= 15) &
            (df['age'] <= 80)
        ]

        df.drop_duplicates(inplace=True)

        Q1 = df['duration_sec'].quantile(0.25)
        Q3 = df['duration_sec'].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[
            (df['duration_sec'] >= lower) &
            (df['duration_sec'] <= upper)
        ]

        self.df = df

        print("Shape after cleaning:", df.shape)

        return self


    def _parse_datetime(self, df):
        """
        Convert start_time column to proper datetime format.

        This function uses real dataset values only.
        No random or synthetic timestamps are generated.
        """

        parsed = pd.to_datetime(
            df['start_time'],
            errors='coerce'
        )

        if parsed.isna().all():
            raise ValueError(
                "start_time could not be parsed. Check dataset format."
            )

        failed_ratio = parsed.isna().mean()

        if failed_ratio > 0:
            print(f"Warning: {failed_ratio:.2%} invalid datetime values found.")

        print("Datetime parsing completed.")

        return parsed


    def engineer_features(self):
        """
        Create useful features for analysis and visualization.

        Features created:
        - duration_min
        - start_time_dt
        - hour
        - weekend_flag
        - age_group
        """

        df = self.df.copy()

        df['duration_min'] = df['duration_sec'] / 60

        df['start_time_dt'] = self._parse_datetime(df)

        df['hour'] = df['start_time_dt'].dt.hour

        df['weekend_flag'] = df['start_time_dt'].dt.dayofweek.apply(
            lambda x: 1 if x >= 5 else 0
        )

        df['age_group'] = pd.cut(
            df['age'],
            bins=[15, 30, 45, 60, 80],
            labels=['15-29', '30-44', '45-59', '60-79'],
            right=False
        )

        self.df = df

        print("Feature engineering completed.")
        print("New columns added: duration_min, start_time_dt, hour, weekend_flag, age_group")

        return self


    def encode_and_scale(self):
        """
        Encode categorical variables and scale numeric features.

        - One-hot encode categorical columns
        - Apply Min-Max scaling to numeric columns
        """

        df = self.df.copy()

        df_encoded = pd.get_dummies(
            df,
            columns=[
                'user_type',
                'member_gender',
                'bike_share_for_all_trip'
            ],
            drop_first=True
        )

        scale_cols = [
            'duration_min',
            'age'
        ]

        for col in scale_cols:

            min_val = df_encoded[col].min()
            max_val = df_encoded[col].max()

            if max_val != min_val:

                df_encoded[col] = (
                    df_encoded[col] - min_val
                ) / (max_val - min_val)

            else:
                df_encoded[col] = 0

        self.df_processed = df_encoded

        print("Encoding and scaling completed.")

        return self


    def get_data(self):
        """
        Return cleaned dataset for EDA and dashboard.
        """

        return self.df


    def get_processed_data(self):
        """
        Return processed dataset for machine learning.
        """

        return self.df_processed
