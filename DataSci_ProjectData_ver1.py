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
    'https://github.com/a1freed/DataSci_ProjectData_ver1/blob/83da999e2c98a1a3bec3ca78a09e99073cdb885d/data/challenger_match.csv',
    'https://github.com/a1freed/DataSci_ProjectData_ver1/blob/83da999e2c98a1a3bec3ca78a09e99073cdb885d/data/match_data_version1.csv',
    'https://github.com/a1freed/DataSci_ProjectData_ver1/blob/83da999e2c98a1a3bec3ca78a09e99073cdb885d/data/match_loser_data_version1.csv',
    'https://github.com/a1freed/DataSci_ProjectData_ver1/blob/83da999e2c98a1a3bec3ca78a09e99073cdb885d/data/match_winner_data_version1.csv'
]

dataframes = [pd.read_csv(file) for file in csv_files]

st.title("League of Legends Match Data")
for i, df in enumerate(dataframes):
    st.write(f"Data # {i + 1}:")
    st.dataframe(df)
    st.markdown("\n")

challenger_df = pd.read_csv('https://github.com/a1freed/DataSci_ProjectData_ver1/blob/83da999e2c98a1a3bec3ca78a09e99073cdb885d/data/challenger_match.csv')

match_data_df = pd.read_csv('https://github.com/a1freed/DataSci_ProjectData_ver1/blob/83da999e2c98a1a3bec3ca78a09e99073cdb885d/data/match_data_version1.csv')

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

match_winner_data = pd.read_csv('https://github.com/a1freed/DataSci_ProjectData_ver1/blob/83da999e2c98a1a3bec3ca78a09e99073cdb885d/data/match_winner_data_version1.csv')
match_loser_data = pd.read_csv('https://github.com/a1freed/DataSci_ProjectData_ver1/blob/83da999e2c98a1a3bec3ca78a09e99073cdb885d/data/match_loser_data_version1.csv')

st.title("Match Winner and Loser Data Analysis")

st.subheader("Match Winner Data")
st.write("### First Few Rows:")
st.dataframe(match_winner_data.head())

st.subheader("Match Loser Data")
st.write("### First Few Rows:")
st.dataframe(match_loser_data.head())

challenger_df = pd.read_csv('https://github.com/a1freed/DataSci_ProjectData_ver1/blob/83da999e2c98a1a3bec3ca78a09e99073cdb885d/data/challenger_match.csv')

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

challenger_df = pd.read_csv('https://github.com/a1freed/DataSci_ProjectData_ver1/blob/83da999e2c98a1a3bec3ca78a09e99073cdb885d/data/challenger_match.csv')

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
# Count the frequency of each role
role_counts = data['role'].value_counts()

# Get the top 10 roles (if you have more than 10 unique roles, otherwise it will return all)
top_roles = role_counts.head(10)

# Display the top roles in Streamlit
st.write("Top 10 Roles by Frequency:")
st.write(top_roles)

# Optionally, you can visualize the top roles using a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=top_roles.index, y=top_roles.values, palette='viridis')
plt.title('Top 10 Roles by Frequency in League of Legends (Challenger Dataset)')
plt.xlabel('Role')
plt.ylabel('Count')
plt.xticks(rotation=45)
st.pyplot(plt)

st.write("Column Names in Dataset:")
st.write(challenger_df.columns)

# Convert 'role' to a categorical numerical value if necessary
challenger_df['role_encoded'] = challenger_df['role'].astype('category').cat.codes

# Use pairplot to uncover relationships (this is just an example, you can change columns)
sns.pairplot(challenger_df[['lane', 'role_encoded']])
st.plt.show()

# Convert 'role' to a categorical numerical value if necessary
challenger_df['role_encoded'] = challenger_df['role'].astype('category').cat.codes

# Use pairplot to uncover relationships (this is just an example, you can change columns)
sns.pairplot(challenger_df[['season', 'role_encoded']])
st.plt.show()


# Encode 'role' and 'lane' to numerical values
label_encoder = LabelEncoder()
challenger_df['role_encoded'] = label_encoder.fit_transform(challenger_df['role'])
challenger_df['lane_encoded'] = label_encoder.fit_transform(challenger_df['lane'])

# Extract important variables
features = challenger_df[['season', 'role_encoded', 'lane_encoded']]
target = challenger_df['gameId']  # Assuming 'gameId' is the target, can change based on your analysis

# Drop rows with missing values
features = features.dropna()
target = target.loc[features.index]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

# Train Decision Tree Regressor model
model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)

# Get feature importances
importance = model.feature_importances_
feature_importance = pd.Series(importance, index=features.columns)

# Display feature importance
st.write("Feature Importances:")
st.write(feature_importance.sort_values(ascending=False))

# Assuming 'challenger_df' is your DataFrame

# Select numeric columns
numeric_cols = challenger_df.select_dtypes(include=['float64', 'int64']).columns

# Calculate Z-scores for numeric columns
z_scores = challenger_df[numeric_cols].apply(zscore)

# Detect outliers (Z-score > 3 or < -3 indicates outlier)
outliers = (z_scores.abs() > 3).any(axis=1)

# Display rows with outliers
st.write("Outliers Detected (Rows):")
st.write(challenger_df[outliers])
data = challenger_df[['season', 'role_encoded']].dropna()

# Sort data by 'season' to ensure it's chronological
data = data.sort_values('season')

# Set 'season' as the index (time variable)
data.set_index('season', inplace=True)

# Check for data consistency (look at a summary of 'role_encoded')
print(data.describe())

# Fit ARIMA model for 'role_encoded'
arima_model = ARIMA(data['role_encoded'], order=(1, 1, 1))  # Adjust the order based on the data
arima_model_fit = arima_model.fit()

# Forecast for the next 5 seasons
forecast_seasons = 5
forecast_dates = pd.date_range(start=data.index[-1], periods=forecast_seasons+1, freq='A')[1:]  # 'A' for annual frequency
arima_forecast = arima_model_fit.forecast(steps=forecast_seasons)

# Plotting the forecast
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['role_encoded'], label='Historical Role Encoded Data', color='blue')
plt.plot(forecast_dates, arima_forecast, label='ARIMA Forecast', color='orange')
plt.title("Role Encoded Forecast by Season")
plt.xlabel("Season")
plt.ylabel("Role Encoded Value")
plt.legend()
st.plt.show()

# Visualize the distribution of 'role_encoded' across the different seasons
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='season', hue='role_encoded')
plt.title('Role Distribution Across Seasons')
plt.xlabel('Season')
plt.ylabel('Count of Roles')
st.plt.show()
# Residuals
residuals = arima_model_fit.resid

# Plot residuals to check for any patterns
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals of ARIMA Model')
plt.show()

# Check residuals summary
st.write(residuals.describe())

# Check if residuals are normally distributed
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
st.plt.show()


# Find the best ARIMA model automatically
auto_model = auto_arima(data['role_encoded'], seasonal=True, m=1, stepwise=True, trace=True)

# Fit the selected model
auto_model_fit = auto_model.fit(data['role_encoded'])

# Forecast with the new model
arima_forecast_auto = auto_model_fit.predict(n_periods=forecast_seasons)
st.write(arima_forecast_auto)

test_data = data.tail(5)

mae = mean_absolute_error(test_data['role_encoded'], arima_forecast[:len(test_data)]) # Ensure you are comparing the same number of data points.
st.write(f'Mean Absolute Error: {mae}')
# Get forecast with confidence intervals
forecast_values = arima_model_fit.get_forecast(steps=forecast_seasons)
forecast_mean = forecast_values.predicted_mean
forecast_ci = forecast_values.conf_int()

# Plot forecast with confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['role_encoded'], label='Historical Role Encoded Data', color='blue')
plt.plot(forecast_dates, forecast_mean, label='ARIMA Forecast', color='orange')
# Access confidence intervals using column names instead of slicing
plt.fill_between(forecast_dates, forecast_ci['lower role_encoded'], forecast_ci['upper role_encoded'], color='orange', alpha=0.3)
plt.title("Role Encoded Forecast by Season with Confidence Intervals")
plt.xlabel("Season")
plt.ylabel("Role Encoded Value")
plt.legend()
st.plt.show()
# Check if seasonality exists in the data
plt.figure(figsize=(10, 6))
sns.boxplot(x='season', y='role_encoded', data=data)
plt.title('Role Encoded Distribution Across Seasons')
plt.xlabel('Season')
plt.ylabel('Role Encoded')
plt.show()

# Check if seasonality exists in the data
plt.figure(figsize=(10, 6))
sns.boxplot(x='season', y='role_encoded', data=data)
plt.title('Role Encoded Distribution Across Seasons')
plt.xlabel('Season')
plt.ylabel('Role Encoded')
st.plt.show()

st.plt.tight_layout()
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit SARIMA model
sarima_model = SARIMAX(data['role_encoded'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))  # Example seasonal order
sarima_model_fit = sarima_model.fit()

# Forecast with SARIMA
sarima_forecast = sarima_model_fit.forecast(steps=forecast_seasons)
st.write(sarima_forecast)


data = challenger_df[['season', 'role_encoded']].dropna()

st.write(data.head())

# Applying K-Means Clustering with 3 clusters (you can change this value)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Check the data with assigned clusters
st.write(data.head())
# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(data['season'], data['role_encoded'], c=data['Cluster'], cmap='viridis', alpha=0.6)
plt.title('Clustering: Role Encoded vs Season')
plt.xlabel('Season')
plt.ylabel('Role Encoded Value')
plt.colorbar(label='Cluster')
st.plt.show()

# Elbow method to determine the best number of clusters
inertia = []
for k in range(1, 11):  # Test for 1 to 10 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
st.plt.show()

df = challenger_df.dropna()  # Remove rows with missing values

# Rename columns for easier reference (adjust to your dataset as needed)
df = df.rename(columns={'Unnamed: 0': 'Index', 'gameId': 'GameID', 'season': 'Season', 'role': 'Role', 'lane': 'Lane', 'accountId': 'AccountID'})

# Optional: Sorting by 'season' or any other relevant column (e.g., 'gameId' if it’s time-based)
df = df.sort_values('Season')

# Check the cleaned data
st.write(df.head())

# Example: Plotting the number of games by lane per season
lane_counts = df.groupby(['Season', 'Lane']).size().unstack()

# Plot the data
lane_counts.plot(kind='line', figsize=(12, 6), marker='o')
plt.title("Number of Games by Lane Over Time")
plt.xlabel("Season")
plt.ylabel("Number of Games")
plt.grid(True)
st.plt.show()

data_counts = df.groupby(['Season']).size()

# Compute a 5-season rolling average (adjust window as needed)
data_counts_rolling_avg = data_counts.rolling(window=5).mean()

# Plotting the number of games with a 5-season rolling average
plt.figure(figsize=(12, 6))
plt.plot(data_counts.index, data_counts, label="Number of Games", color='blue', marker='o', linestyle='-')
plt.plot(data_counts_rolling_avg.index, data_counts_rolling_avg, label="5-Season Rolling Average", color='orange', linestyle='--')
plt.title("Games Played Per Season with 5-Season Rolling Average")
plt.xlabel("Season")
plt.ylabel("Number of Games")
plt.legend()
plt.grid(True)
st.plt.show()

X = data[['season']]  # Assuming 'season' is the feature you're using
y = data['gameId']    # Replace 'gameId' with the target you're predicting (for example, earnings, etc.)

# Split the data into train and test sets (keeping time order intact)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # shuffle=False keeps time order

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(data['season'], data['gameId'], label='Actual Data', color='blue')  # Actual data
plt.plot(X_test, y_pred, label='Predicted Data (Linear Regression)', color='orange', linestyle='--')  # Predicted data
plt.title("Game Data: Actual vs. Predicted (Linear Regression)")
plt.xlabel("Season")
plt.ylabel("Game Data (or whatever metric you use)")
plt.legend()
plt.grid(True)
st.plt.show()

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
st.write(f'Mean Squared Error: {mse}')

X = data[['season']]  # Assuming 'season' is the feature you're using
y = data['gameId']    # Replace 'gameId' with the target you're predicting (for example, earnings)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # shuffle=False keeps time order

# Initialize and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Forecast for the next 5 years
future_years = pd.DataFrame({'season': np.arange(data['season'].max() + 1, data['season'].max() + 6)})

# Make predictions for the future years
future_predictions = model.predict(future_years)

# Plot the historical data and the future forecast
plt.figure(figsize=(10, 6))
plt.plot(data['season'], data['gameId'], label='Historical Data', color='blue')  # Historical data
plt.plot(future_years, future_predictions, label='Future Forecast', color='red', linestyle='--')  # Predicted future data
plt.title("Game Data Forecast with Linear Regression")
plt.xlabel("Season")
plt.ylabel("Game Data (or whatever target you're predicting)")
plt.legend()
plt.grid(True)
st.plt.show()

# You can print future predictions if needed:
st.write(f"Future Predictions for the next 5 years:\n{future_predictions}")
