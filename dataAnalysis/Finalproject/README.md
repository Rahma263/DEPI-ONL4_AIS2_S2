# BikeShare Analysis Project

## Overview

This project analyzes bike sharing data using Python and Pandas.

Dataset: Ford GoBike trip data

Goal:

- Clean dataset
- Create new features
- Perform exploratory data analysis
- Prepare data for machine learning

## Project Structure

Finalproject/

data/
fordgobike-tripdataFor201902.csv

notebook/
analysis.ipynb

src/
pipeline.py

README.md

## Features Created

- duration_min
- age
- age_group
- weekday

## Processing Steps

1. Data Loading
2. Data Cleaning
3. Feature Engineering
4. Encoding and Scaling
5. Exploratory Data Analysis

## Dataset

- **Source**: Ford GoBike System Data – February 2019
- **Rows**: 183,416 raw → 173,121 after cleaning
- **Columns**: 16 raw → 31 after feature engineering

## Key Findings

- **Subscribers** make up >80% of all trips
- **Customers** take significantly longer rides (leisure pattern)
- Clear **commuter peaks** at 8 AM and 5 PM
- **25–34** is the most active age group
- ~80% of trips occur on **weekdays**

## Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Jupyter Notebook

## How to Run

Open notebook:
