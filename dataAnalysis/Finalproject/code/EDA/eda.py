
import sys
import os

# Hardcode the code root path directly — most reliable in Jupyter
code_root = r"C:/Users/Test/Desktop/DEPI-ONL4_AIS2_S2/dataAnalysis/Finalproject/code"
sys.path.insert(0, code_root)

from preprocessing.pipeline import run_bikeshare_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
#  Load Data 
# File path is relative to code_root
file_path = r"C:/Users/Test/Desktop/DEPI-ONL4_AIS2_S2/dataAnalysis/Finalproject/code/data/fordgobike-tripdataFor201902.csv"

df_clean, df_ml = run_bikeshare_pipeline(file_path)
df_clean.head()

#  Trip Duration Distribution 
plt.figure(figsize=(10, 6))
sns.histplot(
    df_clean['duration_min'],   
    bins=50,
    kde=True
)
plt.title("Trip Duration Distribution")
plt.xlabel("Duration (minutes)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#User Type Distribution 
plt.figure(figsize=(8, 6))
df_clean['user_type'].value_counts().plot(   # ← fixed: was df
    kind='pie',
    autopct='%1.1f%%'
)
plt.title("User Type Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()

# Age Group vs User Type 
plt.figure(figsize=(10, 6))
sns.countplot(
    data=df_clean,             
    x='age_group',
    hue='user_type'
)
plt.title("Age Group vs User Type")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#  Cell 6: Trips by Hour of Day
hour_counts = (
    df_clean['hour']            
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(12, 6))
sns.barplot(
    x=hour_counts.index,
    y=hour_counts.values
)
plt.title("Trips by Hour of Day", fontsize=14)
plt.xlabel("Hour of Day", fontsize=12)
plt.ylabel("Number of Trips", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()