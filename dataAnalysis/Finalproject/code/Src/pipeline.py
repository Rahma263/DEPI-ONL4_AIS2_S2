"""
BikeShare Pipeline
==================

This module contains the BikeSharePipeline class.

Responsibilities:
- Load data
- Clean data
- Engineer features
- Encode and scale data
- Provide processed dataset

This file is imported and used inside the notebook.
"""

# Import required libraries
import pandas as pd
import numpy as np


class BikeSharePipeline:
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

        # Raw dataset
        self.df = None

        # Processed dataset for ML
        self.df_processed = None


    def load_data(self):
        """
        Load dataset from CSV file.

        Returns:
        self (allows method chaining)
        """

        self.df = pd.read_csv(self.file_path)

        print("Data loaded successfully")
        print("Shape:", self.df.shape)

        return self


    def clean_data(self):
        """
        Clean dataset and fix data quality issues.

        Steps:
        - Remove missing station data
        - Fill missing gender
        - Remove missing birth year
        - Convert IDs to integer
        - Create age feature
        - Remove duplicates
        - Remove duration outliers using IQR
        """

        df = self.df.copy()

        # Remove rows missing station information
        df.dropna(
            subset=[
                'start_station_id',
                'end_station_id'
            ],
            inplace=True
        )


        # Fill missing gender using mode
        mode_gender = df['member_gender'].mode()[0]

        df['member_gender'].fillna(
            mode_gender,
            inplace=True
        )


        # Remove rows missing birth year
        df.dropna(
            subset=['member_birth_year'],
            inplace=True
        )


        # Convert station IDs to integer
        df['start_station_id'] = df['start_station_id'].astype(int)
        df['end_station_id'] = df['end_station_id'].astype(int)


        # Create age column
        df['age'] = 2019 - df['member_birth_year']


        # Remove unrealistic ages
        df = df[
            (df['age'] >= 15) &
            (df['age'] <= 80)
        ]


        # Remove duplicates
        df.drop_duplicates(inplace=True)


        # Remove duration outliers using IQR
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

        print("Data cleaned")
        print("Shape after cleaning:", df.shape)

        return self


    def engineer_features(self):
        """
        Create new useful features.

        Features created:
        - duration_min
        - weekday
        - age_group
        """

        df = self.df.copy()

        # Convert duration from seconds to minutes
        df['duration_min'] = df['duration_sec'] / 60


        # Convert start_time to datetime
        df['start_time_dt'] = pd.to_datetime(
            df['start_time'],
            errors='coerce'
        )


        # Create weekday feature safely
        if df['start_time_dt'].notna().sum() > 0:

            df['weekday'] = df['start_time_dt'].dt.day_name()

            print("Weekday feature created")

        else:

            print("Weekday feature skipped")


        # Create age groups
        df['age_group'] = pd.cut(

            df['age'],

            bins=[18, 30, 45, 60, 100],

            labels=[
                'Young Adult',
                'Adult',
                'Middle Aged',
                'Senior'
            ],

            right=False
        )


        self.df = df

        print("Feature engineering completed")

        return self


    def encode_and_scale(self):
        """
        Encode categorical variables and scale numeric features.
        """

        df = self.df.copy()


        # One-hot encoding
        df_encoded = pd.get_dummies(

            df,

            columns=[
                'user_type',
                'member_gender',
                'bike_share_for_all_trip'
            ],

            drop_first=True
        )


        # Scale numeric features using Min-Max scaling
        scale_cols = ['duration_min', 'age']

        for col in scale_cols:

            min_val = df_encoded[col].min()
            max_val = df_encoded[col].max()

            df_encoded[col] = (

                df_encoded[col] - min_val

            ) / (max_val - min_val)


        self.df_processed = df_encoded

        print("Encoding and scaling completed")

        return self


    def get_data(self):
        """
        Return cleaned dataset.
        """

        return self.df


    def get_processed_data(self):
        """
        Return encoded and scaled dataset.
        """

        return self.df_processed
