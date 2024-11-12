#imports
import kagglehub
import streamlit as st
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
    '/data/challenger_match.csv',
    '/data/match_data_version1.csv',
    '/data/match_loser_data_version1.csv',
    '/data/match_winner_data_version1.csv'
]

dataframes = [pd.read_csv(file) for file in csv_files]

st.title("League of Legends Match Data")
for i, df in enumerate(dataframes):
    st.write(f"Data # {i + 1}:")
    st.dataframe(df)
    st.markdown("\n")

challenger_df = pd.read_csv('/data/challenger_match.csv')

match_data_df = pd.read_csv('/data/match_data_version1.csv')

st.title("League of Legends Match Data Analysis")

st.subheader("Challenger Match Data")
st.write("### First Few Rows:")
st.dataframe(challenger_df.head())
st.write("### Basic Info:")
st.text(challenger_df.info())

st.subheader("Match Data Version 1")
st.write("### First Few Rows:")
st.dataframe(match_data_df.head())
st.write("### Basic Info:")
st.text(match_data_df.info())

match_winner_data = pd.read_csv('/data/match_winner_data_version1.csv')
match_loser_data = pd.read_csv('/data/match_loser_data_version1.csv')

st.title("Match Winner and Loser Data Analysis")

st.subheader("Match Winner Data")
st.write("### First Few Rows:")
st.dataframe(match_winner_data.head())

st.subheader("Match Loser Data")
st.write("### First Few Rows:")
st.dataframe(match_loser_data.head())

challenger_df = pd.read_csv('/data/challenger_match.csv')

missing_values = challenger_df.isnull().sum()
st.write("Missing values per column:")
st.write(missing_values)

descriptive_stats = challenger_df.describe()
st.write("Descriptive statistics for numerical columns:")
st.write(descriptive_stats)

numerical_features = challenger_df.select_dtypes(include=['number']).columns
numerical_df = challenger_df[numerical_features]

plt.figure(figsize=(10, 8))
sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlations")
st.pyplot(plt)

challenger_df = pd.read_csv('/data/challenger_match.csv')

sns.countplot(data=challenger_df, x='role')
plt.xlabel('Role')
plt.ylabel('Count')
plt.title('Distribution of Roles')
st.pyplot(plt)

sns.countplot(data=challenger_df, x='lane')
plt.xlabel('Lane')
plt.ylabel('Count')
plt.title('Distribution of Lanes')
st.pyplot(plt)

sns.countplot(data=challenger_df, x='season')
plt.xlabel('Season')
plt.ylabel('Count')
plt.title('Games per Season')
st.pyplot(plt)

st.write("Unique game IDs: {}".format(challenger_df['gameId'].nunique()))
st.write("Unique account IDs: {}".format(challenger_df['accountId'].nunique()))

duplicates = challenger_df.duplicated().sum()
st.write(f"Number of duplicate rows: {duplicates}")

role_lane_counts = challenger_df.groupby(['lane', 'role']).size().unstack()
role_lane_counts.plot(kind='bar', stacked=True)
plt.xlabel('Lane')
plt.ylabel('Count')
plt.title('Role Frequency by Lane')
plt.legend(title='Role')
st.pyplot(plt)

challenger_df = pd.read_csv(file_path)

st.write(challenger_df.head())

st.write("Duplicated rows:", challenger_df.duplicated().sum())
st.write("DataFrame Info:")
st.write(challenger_df.info())
st.write("Missing values per column:")
st.write(challenger_df.isnull().sum())

challenger_df = challenger_df.dropna()
st.write(challenger_df.head())
st.write(challenger_df.info())

st.write("Unique Roles:", challenger_df['role'].unique())
st.write("Unique Lanes:", challenger_df['lane'].unique())
st.write("Unique Seasons:", challenger_df['season'].unique())

st.write("Columns in challenger_df:", challenger_df.columns)
st.write("Data Types:\n", challenger_df.dtypes)

challenger_df['gameDuration'] = pd.to_numeric(challenger_df['gameDuration'], errors='coerce')
duration_counts = challenger_df['gameDuration'].value_counts()
st.write("Game Duration Counts:\n", duration_counts)

highest_duration_row = challenger_df.loc[challenger_df['gameDuration'].idxmax()]
st.write("Match with the highest duration:")
st.write(highest_duration_row)

plt.figure(figsize=(10, 6))
plt.bar(duration_counts.index, duration_counts.values, color='blue', alpha=0.7, edgecolor='darkorange')
plt.title('Frequency Distribution of Game Duration')
plt.xlabel('Game Duration (in seconds)')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(axis='y')
st.pyplot(plt)

unique_game_modes = challenger_df['gameMode'].unique()
st.write("Unique Game Modes:", unique_game_modes)

game_duration_counts = challenger_df['gameDuration'].value_counts()
st.write("Game Duration Counts:\n", game_duration_counts)

challenger_df['gameDuration_Rounded'] = challenger_df['gameDuration'].round()
st.write("Rounded Game Durations:\n", challenger_df['gameDuration_Rounded'])

unique_game_durations = challenger_df['gameDuration_Rounded'].unique()
st.write("Unique Rounded Game Durations:", unique_game_durations)

feature_columns = ['firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron',
                   'firstDragon', 'firstRiftHerald', 'towerKills',
                   'inhibitorKills', 'baronKills', 'dragonKills']

feature_counts = challenger_df[feature_columns].sum()
custom_colors = ['darkseagreen', 'hotpink', 'mediumvioletred', 'palevioletred',
                 'pink', 'lightpink', 'darkseagreen', 'orchid', 'lightcoral', 'lightblue']

plt.figure(figsize=(12, 8))
plt.bar(feature_counts.index, feature_counts.values, color=custom_colors[:len(feature_counts)])
plt.title('Frequency Distribution of Key Game Events')
plt.xlabel('Game Events')
plt.ylabel('Frequency of Occurrence')
plt.xticks(rotation=45)
st.pyplot(plt)

columns_to_check = ['win', 'firstBlood', 'firstTower', 'firstInhibitor',
                    'firstBaron', 'firstDragon', 'firstRiftHerald',
                    'towerKills', 'inhibitorKills', 'baronKills', 'dragonKills']

for column in columns_to_check:
    st.write(f"Winner dataset - {column} unique values:\n", challenger_df[column].unique())

for column in columns_to_check:
    st.write(f"Loser dataset - {column} unique values:\n", challenger_df[column].unique())

challenger_df[columns_to_check] = challenger_df[columns_to_check].astype(int)

winner_event_counts = challenger_df[columns_to_check].sum()
loser_event_counts = challenger_df[columns_to_check].sum()

st.write("Winner event counts:\n", winner_event_counts)
st.write("Loser event counts:\n", loser_event_counts)

plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.bar(winner_event_counts.index, winner_event_counts .values, color='green')
plt.title('Frequency Distribution of Key Game Events (Winners)')
plt.xlabel('Game Events')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(loser_event_counts.index, loser_event_counts.values, color='red')
plt.title('Frequency Distribution of Key Game Events (Losers)')
plt.xlabel('Game Events')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

plt.tight_layout()
st.pyplot(plt)

