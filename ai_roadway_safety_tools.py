import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# Load the US Accidents dataset
us_accidents = pd.read_csv('US_Accidents_March23.csv')

# Data preprocessing
us_accidents = us_accidents[['Start_Lat', 'Start_Lng', 'Severity', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition']]
us_accidents = us_accidents.dropna()

# Encoding categorical variables
us_accidents = pd.get_dummies(us_accidents, columns=['Weather_Condition'])

# Splitting data into features and labels
X = us_accidents.drop(columns=['Severity'])
y = us_accidents['Severity']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predicting and evaluating
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accident Severity Prediction Accuracy: {accuracy * 100:.2f}%')

# Load the Traffic Prediction dataset
traffic_data = pd.read_csv('traffic.csv')

# Data preprocessing
traffic_data = traffic_data[['Junction', 'DateTime', 'Vehicles']]
traffic_data['DateTime'] = pd.to_datetime(traffic_data['DateTime'])
traffic_data['Hour'] = traffic_data['DateTime'].dt.hour
traffic_data = traffic_data.drop(columns=['DateTime'])

# Encoding categorical variables
traffic_data = pd.get_dummies(traffic_data, columns=['Junction'])

# Splitting data into features and labels
X_traffic = traffic_data.drop(columns=['Vehicles'])
y_traffic = traffic_data['Vehicles']

# Splitting into training and testing sets
X_train_traffic, X_test_traffic, y_train_traffic, y_test_traffic = train_test_split(X_traffic, y_traffic, test_size=0.3, random_state=42)

# Training the RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_traffic, y_train_traffic)

# Predicting and evaluating
y_pred_traffic = reg.predict(X_test_traffic)
mse = mean_squared_error(y_test_traffic, y_pred_traffic)
print(f'Traffic Volume Prediction Mean Squared Error: {mse:.2f}')
