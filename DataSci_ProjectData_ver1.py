

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

# Load Data
def load_data(filepath):
    data = pd.read_csv(filepath, encoding='ISO-8859-1')
    print("First 5 rows of the dataset:\n", data.head())
    print("\nDataset Info:\n", data.info())
    print("\nMissing values:\n", data.isnull().sum())
    return data

# Preprocess Data
def preprocess_data(data):
    data = data.dropna(subset=['role', 'lane'])
    preprocessor = ColumnTransformer(
        transformers=[
            ('role_lane', OneHotEncoder(), ['role', 'lane'])
        ],
        remainder='passthrough'
    )
    X = data[['role', 'lane']]
    y = data['season']
    return preprocessor, X, y

# Train the Model
def train_model(preprocessor, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Model Evaluation:")
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R²):", r2)
    return model, X_test, y_test, y_pred

# Plot Actual vs Predicted Values
def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    plt.xlabel("Actual Season")
    plt.ylabel("Predicted Season")
    plt.title("Actual vs Predicted Season")
    plt.grid(True)
    plt.show()

# Main Execution
data = load_data('/content/challenger_match.csv')
preprocessor, X, y = preprocess_data(data)
model, X_test, y_test, y_pred = train_model(preprocessor, X, y)
plot_results(y_test, y_pred)

# Count the frequency of each role
role_counts = data['role'].value_counts()
top_roles = role_counts.head(10)
print("Top 10 Roles by Frequency:\n", top_roles)

plt.figure(figsize=(10, 6))
sns.countplot(y='role', data=data, order=data['role'].value_counts().index)
plt.title('Top 10 Roles by Frequency in League of Legends (Challenger Dataset)')
plt.xlabel('Count')
plt.ylabel('Role')
plt.show()

# Encode 'role' and 'lane' to numerical values
label_encoder = LabelEncoder()
data['role_encoded'] = label_encoder.fit_transform(data['role'])
data['lane_encoded'] = label_encoder.fit_transform(data['lane'])

# Extract important variables
features = data[['season', 'role_encoded', 'lane_encoded']]
target = data['gameId']

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
print("Feature Importances:\n", feature_importance.sort_values(ascending=False))

# Detect outliers
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
z_scores = data[numeric_cols]. apply(zscore)
outliers = (z_scores.abs() > 3).any(axis=1)
print("Outliers Detected (Rows):")
display(data[outliers])

# Time Series Analysis
data_counts = data[['season', 'role_encoded']].dropna().sort_values('season')
data_counts.set_index('season', inplace=True)
print(data_counts.describe())

# Fit ARIMA model for 'role_encoded'
arima_model = ARIMA(data_counts['role_encoded'], order=(1, 1, 1))
arima_model_fit = arima_model.fit()

# Forecast for the next 5 seasons
forecast_seasons = 5
forecast_dates = pd.date_range(start=data_counts.index[-1], periods=forecast_seasons + 1, freq='A')[1:]
arima_forecast = arima_model_fit.forecast(steps=forecast_seasons)

# Plotting the forecast
plt.figure(figsize=(10, 6))
plt.plot(data_counts.index, data_counts['role_encoded'], label='Historical Role Encoded Data', color='blue')
plt.plot(forecast_dates, arima_forecast, label='ARIMA Forecast', color='orange')
plt.title("Role Encoded Forecast by Season")
plt.xlabel("Season")
plt.ylabel("Role Encoded Value")
plt.legend()
plt.show()

# Visualize the distribution of 'role_encoded' across the different seasons
plt.figure(figsize=(10, 6))
sns.countplot(data=data_counts.reset_index(), x='season', hue='role_encoded')
plt.title('Role Distribution Across Seasons')
plt.xlabel('Season')
plt.ylabel('Count of Roles')
plt.show()

# Residuals
residuals = arima_model_fit.resid
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals of ARIMA Model')
plt.show()

# Check residuals summary
print(residuals.describe())
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.show()

# Find the best ARIMA model automatically
auto_model = auto_arima(data_counts['role_encoded'], seasonal=True, m=1, stepwise=True, trace=True)
auto_model_fit = auto_model.fit(data_counts['role_encoded'])
arima_forecast_auto = auto_model_fit.predict(n_periods=forecast_seasons)

# Evaluate the model
test_data = data_counts.tail(5)
mae = mean_absolute_error(test_data['role_encoded'], arima_forecast[:len(test_data)])
print(f'Mean Absolute Error: {mae}')

# Get forecast with confidence intervals
forecast_values = arima_model_fit.get_forecast(steps=forecast_seasons)
forecast_mean = forecast_values.predicted_mean
forecast_ci = forecast_values.conf_int()

# Plot forecast with confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(data_counts.index, data_counts['role_encoded'], label='Historical Role Encoded Data', color='blue')
plt.plot(forecast_dates, forecast_mean, label='ARIMA Forecast', color='orange')
plt.fill_between(forecast_dates, forecast_ci['lower role_encoded'], forecast_ci['upper role_encoded'], color='orange', alpha=0.3)
plt.title("Role Encoded Forecast by Season with Confidence Intervals")
plt.xlabel("Season")
plt.ylabel("Role Encoded Value")
plt.legend()
plt.show()

# Check if seasonality exists in the data
plt.figure(figsize=(10, 6))
sns.boxplot(x='season', y='role_encoded', data=data_counts.reset_index())
plt.title('Role Encoded Distribution Across Seasons')
plt.xlabel('Season')
plt.ylabel('Role Encoded')
plt.show()

# Fit SARIMA model
from statsmodels.tsa.statespace.sarimax import SARIMAX
sarima_model = SARIMAX(data_counts['role_encoded'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
sarima_model_fit = sarima_model.fit()

# Forecast with SARIMA
sarima_forecast = sarima_model_fit.forecast(steps=forecast_seasons)

# Clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_counts[['season', 'role_encoded']])
kmeans = KMeans(n_clusters=3, random_state=42)
data_counts['Cluster'] = kmeans.fit_predict(scaled_data)

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(data_counts['season'], data_counts['role_encoded'], c=data_counts['Cluster'], cmap='viridis', alpha=0.6)
plt.title('Clustering: Role Encoded vs Season')
plt.xlabel('Season')
plt.ylabel('Role Encoded Value')
plt.colorbar(label='Cluster')
plt.show()

# Elbow method to determine the best number of clusters
inertia = []
for k in range (1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Example: Plotting the number of games by lane per season
lane_counts = data.groupby(['season', 'lane']).size().unstack()
lane_counts.plot(kind='line', figsize=(12, 6), marker='o')
plt.title("Number of Games by Lane Over Time")
plt.xlabel("Season")
plt.ylabel("Number of Games")
plt.grid(True)
plt.show()

# Compute a 5-season rolling average
data_counts = data.groupby(['season']).size()
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
plt.show()

# Prepare the data for linear regression
X = data[['season']]
y = data['gameId']  # Replace 'gameId' with the target you're predicting

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(data['season'], data['gameId'], label='Actual Data', color='blue')
plt.plot(X_test, y_pred, label='Predicted Data (Linear Regression)', color='orange', linestyle='--')
plt.title("Game Data: Actual vs. Predicted (Linear Regression)")
plt.xlabel("Season")
plt.ylabel("Game Data")
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Forecast for the next 5 years
future_years = pd.DataFrame({'season': np.arange(data['season'].max() + 1, data['season'].max() + 6)})
future_predictions = model.predict(future_years)

# Plot the historical data and the future forecast
plt.figure(figsize=(10, 6))
plt.plot(data['season'], data['gameId'], label='Historical Data', color='blue')
plt.plot(future_years, future_predictions, label='Future Forecast', color='red', linestyle='--')
plt.title("Game Data Forecast with Linear Regression")
plt.xlabel("Season")
plt.ylabel("Game Data")
plt.legend()
plt.grid(True)
plt.show()

# Print future predictions if needed
print(f"Future Predictions for the next 5 years:\n{future_predictions}")

!streamlit run DataSci_ProjectData_ver1.py & npx localtunnel --port 8501
