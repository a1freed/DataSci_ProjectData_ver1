#imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.cluster import KMeans
import numpy as np

# Load data from CSV files
csv_files = [
    'https://github.com/a1freed/DataSci_ProjectData_ver1/tree/aaf566404860eb35e89822291becd21e7b577fe3/data/challenger_match.csv',
    'https://github.com/a1freed/DataSci_ProjectData_ver1/tree/aaf566404860eb35e89822291becd21e7b577fe3/data/match_data_version1.csv',
    'https://github.com/a1freed/DataSci_ProjectData_ver1/tree/aaf566404860eb35e89822291becd21e7b577fe3/data/match_loser_data_version1.csv',
    'https://github.com/a1freed/DataSci_ProjectData_ver1/tree/aaf566404860eb35e89822291becd21e7b577fe3/data/match_winner_data_version1.csv'

]

dataframes = [pd.read_csv(file) for file in csv_files]

# Title and Data Display
st.title("League of Legends Match Data")
for i, df in enumerate(dataframes):
    st.write(f"Data # {i + 1}:")
    st.dataframe(df)

# Analysis of Challenger Match Data
challenger_df = dataframes[0]
st.title("League of Legends Match Data Analysis")
st.subheader("Challenger Match Data")
st.write("### First Few Rows:")
st.dataframe(challenger_df.head())
st.write("### Basic Info:")
st.text(challenger_df.info())

# Missing Values and Descriptive Statistics
missing_values = challenger_df.isnull().sum()
st.write("Missing values per column:")
st.write(missing_values)

descriptive_stats = challenger_df.describe()
st.write("Descriptive statistics for numerical columns:")
st.write(descriptive_stats)

# Correlation Heatmap
st.subheader("Feature Correlations")
plt.figure(figsize=(10, 8))
sns.heatmap(challenger_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlations")
st.pyplot(plt)

# Role Distribution
st.subheader("Role Distribution")
sns.countplot(data=challenger_df, x='role')
plt.title('Distribution of Roles')
st.pyplot(plt)

# Lane Distribution
st.subheader("Lane Distribution")
sns.countplot(data=challenger_df, x='lane')
plt.title('Distribution of Lanes')
st.pyplot(plt)

# Unique IDs and Duplicates
st.write("Unique game IDs: {}".format(challenger_df['gameId'].nunique()))
st.write("Unique account IDs: {}".format(challenger_df['accountId'].nunique()))
duplicates = challenger_df.duplicated().sum()
st.write(f"Number of duplicate rows: {duplicates}")

# Game Duration Analysis
challenger_df['gameDuration'] = pd.to_numeric(challenger_df['gameDuration'], errors='coerce')
duration_counts = challenger_df['gameDuration'].value_counts()
st.write("Game Duration Counts:\n", duration_counts)

highest_duration_row = challenger_df.loc[challenger_df['gameDuration'].idxmax()]
st.write("Match with the highest duration:")
st.write(highest_duration_row)

# Frequency Distribution of Game Duration
plt.figure(figsize=(10, 6))
plt.bar(duration_counts.index, duration_counts.values, color='blue', alpha=0.7, edgecolor='darkorange')
plt.title('Frequency Distribution of Game Duration')
plt.xlabel('Game Duration (in seconds)')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(axis='y')
st.pyplot(plt)

# Feature Importance with Decision Tree
features = challenger_df[['season', 'role', 'lane']]
target = challenger_df['gameId']  # Change this if necessary
label_encoder = LabelEncoder()
features['role_encoded'] = label_encoder.fit_transform(features['role'])
features['lane_encoded'] = label_encoder.fit_transform(features['lane'])

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)
model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)

# Feature Importances
importance = model.feature_importances_
feature_importance = pd.Series(importance, index=features.columns)
st.write("Feature Importances:")
st.write(feature_importance.sort_values(ascending=False))

# Outlier Detection
numeric_cols = challenger_df.select_dtypes(include=['float64', 'int64']).columns
z_scores = challenger_df[numeric_cols].apply(lambda x: (x - x.mean ()) / x.std())
outliers = (z_scores.abs() > 3).any(axis=1)
st.write("Outliers Detected (Rows):")
st.write(challenger_df[outliers])

# Time Series Analysis with ARIMA
data = challenger_df[['season', 'role_encoded']].dropna()
data = data.sort_values('season')
data.set_index('season', inplace=True)

arima_model = auto_arima(data['role_encoded'], seasonal=True, m=1, stepwise=True, trace=True)
arima_model_fit = arima_model.fit(data['role_encoded'])

# Forecasting
forecast_seasons = 5
arima_forecast = arima_model_fit.predict(n_periods=forecast_seasons)
forecast_dates = pd.date_range(start=data.index[-1] + 1, periods=forecast_seasons, freq='A')

# Plotting the forecast
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['role_encoded'], label='Historical Role Encoded Data', color='blue')
plt.plot(forecast_dates, arima_forecast, label='ARIMA Forecast', color='orange')
plt.title("Role Encoded Forecast by Season")
plt.xlabel("Season")
plt.ylabel("Role Encoded Value")
plt.legend()
st.pyplot(plt)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
challenger_df['Cluster'] = kmeans.fit_predict(features[['role_encoded', 'lane_encoded']])

# Plotting Clusters
plt.figure(figsize=(10, 6))
plt.scatter(challenger_df['season'], challenger_df['role_encoded'], c=challenger_df['Cluster'], cmap='viridis', alpha=0.6)
plt.title('Clustering: Role Encoded vs Season')
plt.xlabel('Season')
plt.ylabel('Role Encoded Value')
plt.colorbar(label='Cluster')
st.pyplot(plt)

# Elbow Method for Optimal Clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features[['role_encoded', 'lane_encoded']])
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
st.pyplot(plt)

# Final Output
st.write("Analysis Complete. Explore the visualizations and insights above!")
