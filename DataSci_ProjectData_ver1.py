%%writefile DataSci_ProjectData_ver1.py
##this
!git init
!git add DataSci_ProjectData_ver1.py
!git commit -m "Initial commit"
!git remote add origin https://github.com/a1freed/DataSci_ProjectData_ver1.git
!git remote -v
!git push -u origin master 
!pip install pmdarima
!pip install streamlit


#imports
import kagglehub
import streamlit
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import os
from sklearn.preprocessing import LabelEncoder
import altair as alt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import json
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import zscore
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error

#Files
csv_files = [
    '/root/.cache/kagglehub/datasets/gyejr95/league-of-legendslol-ranked-games-2020-ver1/versions/4/challenger_match.csv',
    '/root/.cache/kagglehub/datasets/gyejr95/league-of-legendslol-ranked-games-2020-ver1/versions/4/match_data_version1.csv',
    '/root/.cache/kagglehub/datasets/gyejr95/league-of-legendslol-ranked-games-2020-ver1/versions/4/match_loser_data_version1.csv',
    '/root/.cache/kagglehub/datasets/gyejr95/league-of-legendslol-ranked-games-2020-ver1/versions/4/match_winner_data_version1.csv'
]

dataframes = [pd.read_csv(file) for file in csv_files]

# Display data
for i, df in enumerate(dataframes):
    print(f"Data # {i + 1}:")
    print(df)
    print("\n")
