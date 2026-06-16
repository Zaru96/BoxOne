# =====================================================
# TAHAP 1 — IMPORT LIBRARY
# =====================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =====================================================
# TAHAP 2 — LOAD DATASET
# =====================================================

races = pd.read_csv("data/races.csv")
results = pd.read_csv("data/results.csv")
drivers = pd.read_csv("data/drivers.csv")
constructors = pd.read_csv("data/constructors.csv")
circuits = pd.read_csv("data/circuits.csv")


# =====================================================
# TAHAP 3 — DATA MERGING
# =====================================================

df = results.merge(races, on="raceId", how="left")
df = df.merge(drivers, on="driverId", how="left")
df = df.merge(
    constructors,
    on="constructorId",
    how="left",
    suffixes=("", "_constructor")
)
df = df.merge(
    circuits,
    on="circuitId",
    how="left",
    suffixes=("", "_circuit")
)


# =====================================================
# TAHAP 4 — DATA CLEANING
# =====================================================

df["driver_name"] = df["forename"] + " " + df["surname"]

df = df.rename(columns={
    "name": "race_name",
    "name_constructor": "constructor_name",
    "name_circuit": "circuit_name"
})

f1 = df[[
    "year",
    "race_name",
    "circuit_name",
    "driver_name",
    "constructor_name",
    "grid",
    "positionOrder",
    "points",
    "laps",
    "statusId"
]].copy()


# =====================================================
# TAHAP 5 — FEATURE ENGINEERING
# =====================================================

f1["top_10"] = f1["positionOrder"].apply(lambda x: 1 if x <= 10 else 0)
f1["podium"] = f1["positionOrder"].apply(lambda x: 1 if x <= 3 else 0)
f1["win"] = f1["positionOrder"].apply(lambda x: 1 if x == 1 else 0)


# =====================================================
# TAHAP 6 — PERSIAPAN DATA MODELING
# =====================================================

model_data = f1[[
    "year",
    "grid",
    "constructor_name",
    "circuit_name",
    "top_10"
]].copy()

model_data = model_data.dropna()
model_data = model_data[model_data["grid"] > 0]

X = model_data[[
    "year",
    "grid",
    "constructor_name",
    "circuit_name"
]]

y = model_data["top_10"]


# =====================================================
# TAHAP 7 — PREPROCESSING DATA
# =====================================================

categorical_features = ["constructor_name", "circuit_name"]
numeric_features = ["year", "grid"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)


# =====================================================
# TAHAP 8 — TRAINING MODEL
# =====================================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)


# =====================================================
# TAHAP 9 — EVALUASI MODEL
# =====================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("====================================")
print("HASIL EVALUASI MODEL")
print("====================================")
print(f"Accuracy: {accuracy:.2f}")
print()
print(classification_report(y_test, y_pred))


# =====================================================
# TAHAP 10 — SIMPAN MODEL
# =====================================================

joblib.dump(model, "model.pkl", compress=3)

print("====================================")
print("Model berhasil disimpan sebagai model.pkl")
print("====================================")