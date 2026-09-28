import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, roc_auc_score,
    r2_score
)

RANDOM_STATE = 42
TEST_SIZE = 0.40

# =========================================================
# 1. WINE QUALITY - CLASIFICACION
# =========================================================
wine = pd.read_csv("winequality-white.csv", sep=";")

# Convertimos la calidad en una clasificacion binaria:
# 0 = calidad < 6, 1 = calidad >= 6.
wine["quality_class"] = (wine["quality"] >= 6).astype(int)

features_wine = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates",
    "alcohol"
]

X = wine[features_wine]
y = wine["quality_class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

wine_model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,
    random_state=RANDOM_STATE,
    early_stopping=True
)

wine_model.fit(X_train, y_train)
wine_pred = wine_model.predict(X_test)
wine_prob = wine_model.predict_proba(X_test)[:, 1]

print("\n===== WINE QUALITY =====")
print("Precision:", precision_score(y_test, wine_pred, zero_division=0))
print("Exactitud:", accuracy_score(y_test, wine_pred))
print("ROC-AUC:", roc_auc_score(y_test, wine_prob))


# =========================================================
# 2. ENERGY EFFICIENCY - REGRESION MULTISALIDA
# =========================================================
energy = pd.read_excel("ENB2012_data.xlsx")

energy.columns = [
    "relative_compactness", "surface_area", "wall_area",
    "roof_area", "overall_height", "orientation",
    "glazing_area", "glazing_area_distribution",
    "heating_load", "cooling_load"
]

features_energy = [
    "relative_compactness", "surface_area", "wall_area",
    "roof_area", "overall_height", "orientation",
    "glazing_area", "glazing_area_distribution"
]
targets_energy = ["heating_load", "cooling_load"]

X = energy[features_energy]
y = energy[targets_energy]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

energy_model = MLPRegressor(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=RANDOM_STATE,
    early_stopping=True
)

energy_model.fit(X_train, y_train)
energy_pred = energy_model.predict(X_test)

print("\n===== ENERGY EFFICIENCY =====")
for i, target in enumerate(targets_energy):
    r2 = r2_score(y_test.iloc[:, i], energy_pred[:, i])
    r = np.corrcoef(y_test.iloc[:, i], energy_pred[:, i])[0, 1]
    print(target, "- R2:", r2, "- R:", r)


# =========================================================
# 3. BIKE SHARING - REGRESION
# =========================================================
bike = pd.read_csv("hour.csv")
features_bike = [
    "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"
]

X = bike[features_bike]
y = bike["cnt"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

bike_model = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=RANDOM_STATE,
    early_stopping=True
)

bike_model.fit(X_train, y_train)
bike_pred = bike_model.predict(X_test)

print("\n===== BIKE SHARING =====")
print("R2:", r2_score(y_test, bike_pred))
print("R:", np.corrcoef(y_test, bike_pred)[0, 1])

print("\nProceso terminado.")
