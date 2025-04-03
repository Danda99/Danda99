import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt

def predict_high_risk_zones(traffic_data):
    traffic_data = pd.get_dummies(traffic_data, columns=['weather_condition', 'time_of_day'])
    features = traffic_data[['speed', 'traffic_density'] + [col for col in traffic_data.columns if col.startswith('weather_condition') or col.startswith('time_of_day')]]
    labels = traffic_data['accident_occurred']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, model.predict(features)

def obstacle_detection(vehicle_data):
    obstacles = ['car', 'pedestrian', 'traffic_light', 'animal', 'none']
    detected_obstacle = random.choice(obstacles)
    if detected_obstacle != 'none':
        print(f"Obstacle detected: {detected_obstacle}")
        return True
    return False

def accident_prediction(historical_data):
    historical_data = pd.get_dummies(historical_data, columns=['weather_conditions', 'road_type', 'time_of_day'])
    features = historical_data[['traffic_volume'] + [col for col in historical_data.columns if col.startswith('weather_conditions') or col.startswith('road_type') or col.startswith('time_of_day')]]
    labels = historical_data['accident_occurrence']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, model.predict(historical_data[['traffic_volume'] + [col for col in historical_data.columns if col.startswith('weather_conditions') or col.startswith('road_type') or col.startswith('time_of_day')]])

def collect_real_time_traffic_data():
    real_time_data = {
        'speed': random.randint(20, 80),
        'traffic_density': random.randint(1, 100),
        'weather_condition': random.choice(['clear', 'rainy', 'foggy', 'snowy']),
        'time_of_day': random.choice(['morning', 'afternoon', 'evening', 'night'])
    }
    return pd.DataFrame([real_time_data])

def ethical_decision_in_case_of_accident(vehicle_speed, pedestrian_distance):
    if vehicle_speed > 50 and pedestrian_distance < 10:
        decision = "Braking"
    else:
        decision = "Maintain Speed"
    print(f"Decision: {decision}")
    return decision

traffic_data = pd.DataFrame({
    'speed': np.random.randint(20, 100, 100),
    'traffic_density': np.random.randint(1, 100, 100),
    'weather_condition': np.random.choice(['clear', 'rainy', 'foggy', 'snowy'], 100),
    'time_of_day': np.random.choice(['morning', 'afternoon', 'evening', 'night'], 100),
    'accident_occurred': np.random.choice([0, 1], 100)
})

accuracy, predictions = predict_high_risk_zones(traffic_data)
print(f"Traffic Risk Zone Prediction Accuracy: {accuracy * 100:.2f}%\n")

obstacle_detected = obstacle_detection(None)
if obstacle_detected:
    print("Obstacle Avoidance System Activated\n")

historical_data = pd.DataFrame({
    'traffic_volume': np.random.randint(100, 1000, 100),
    'weather_conditions': np.random.choice(['clear', 'rainy', 'foggy', 'snowy'], 100),
    'road_type': np.random.choice(['highway', 'city', 'rural'], 100),
    'time_of_day': np.random.choice(['morning', 'afternoon', 'evening', 'night'], 100),
    'accident_occurrence': np.random.choice([0, 1], 100)
})

accident_accuracy, accident_predictions = accident_prediction(historical_data)
print(f"Accident Prediction Accuracy: {accident_accuracy * 100:.2f}%\n")

real_time_data = collect_real_time_traffic_data()
print("Real-Time Traffic Data:")
print(real_time_data.to_string(index=False))

vehicle_speed = 60
pedestrian_distance = 8
ethical_decision_in_case_of_accident(vehicle_speed, pedestrian_distance)

plt.figure(figsize=(10, 6))
plt.hist(traffic_data['traffic_density'], bins=15, alpha=0.7, color='purple', edgecolor='black')
plt.title("Traffic Density Distribution", fontsize=16)
plt.xlabel("Traffic Density", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid(True)
plt.xticks(np.arange(0, 101, step=10))
plt.show()
