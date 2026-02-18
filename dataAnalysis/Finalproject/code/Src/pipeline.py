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

        # Safety check
        if self.df is None or self.df.empty:
            raise ValueError("Dataset failed to load or is empty.")

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
        if df['member_gender'].isna().sum() > 0:
            mode_gender = df['member_gender'].mode()[0]
            df['member_gender'] = df['member_gender'].fillna(mode_gender)

        # Remove rows missing birth year
        df.dropna(
            subset=['member_birth_year'],
            inplace=True
        )

        # Convert station IDs to integer
        df['start_station_id'] = df['start_station_id'].astype(int)
        df['end_station_id'] = df['end_station_id'].astype(int)

        # Create age column using dataset year
        dataset_year = 2019
        df['age'] = dataset_year - df['member_birth_year']

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

        print("Shape after cleaning:", df.shape)

        return self


    def _reconstruct_datetime(self, df):
        """
        Attempt to parse start_time. If corrupted (Excel truncation),
        reconstruct synthetic datetimes distributed across February 2019.
        """

        # Try normal parsing first
        parsed = pd.to_datetime(df['start_time'], errors='coerce')
        valid_ratio = parsed.notna().mean()

        if valid_ratio > 0.5:
            print(f"start_time parsed normally ({valid_ratio:.1%} valid).")
            return parsed

        # Corrupted path
        print(
            "WARNING: start_time is corrupted.\n"
            "Reconstructing synthetic datetimes across February 2019."
        )

        n = len(df)

        # February 2019 dates
        feb_days = pd.date_range('2019-02-01', periods=28, freq='D')

        # Fixed seed for reproducibility
        np.random.seed(42)

        day_indices = np.random.choice(np.arange(28), size=n)
        base_dates = feb_days[day_indices]

        # Extract minute and second safely
        time_parts = df['start_time'].str.extract(
            r'^(\d+):(\d+\.?\d*)$'
        )

        minutes = pd.to_numeric(
            time_parts[0],
            errors='coerce'
        ).fillna(0).astype(int)

        seconds = pd.to_numeric(
            time_parts[1],
            errors='coerce'
        ).fillna(0)

        # Random hours
        hours = np.random.randint(0, 24, size=n)

        reconstructed = (
            pd.to_datetime(base_dates)
            + pd.to_timedelta(hours, unit='h')
            + pd.to_timedelta(minutes, unit='m')
            + pd.to_timedelta(seconds, unit='s')
        )

        print("Datetime reconstruction completed.")

        return reconstructed


    def engineer_features(self):
        """
        Create new useful features for analysis and plotting.

        Features created:
        - duration_min → trip duration in minutes
        - start_time_dt → proper datetime format
        - hour → hour of trip (for hourly plots)
        - weekend_flag → 1 if weekend, 0 if weekday
        - age_group → grouped age ranges
        """

        df = self.df.copy()

        # Convert duration to minutes (easier to understand than seconds)
        df['duration_min'] = df['duration_sec'] / 60

        # Create proper datetime column
        df['start_time_dt'] = self._reconstruct_datetime(df)

        # Safety check
        if df['start_time_dt'].isna().all():
            raise ValueError("Datetime feature creation failed.")

        # Extract hour (IMPORTANT for trips by hour plot)
        df['hour'] = df['start_time_dt'].dt.hour

        # Weekend flag (useful for comparison plots)
        df['weekend_flag'] = df['start_time_dt'].dt.dayofweek.apply(
            lambda x: 1 if x >= 5 else 0
        )

        # Age groups (clear ranges)
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

        # Manual Min-Max scaling
        scale_cols = ['duration_min', 'age']

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
        Return cleaned dataset.
        """

        return self.df


    def get_processed_data(self):
        """
        Return encoded and scaled dataset.
        """

        return self.df_processed
