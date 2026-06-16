# =====================================================
# TAHAP 1 — IMPORT LIBRARY
# =====================================================

from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib


# =====================================================
# TAHAP 2 — KONFIGURASI HALAMAN
# =====================================================

st.set_page_config(
    page_title="BoxONE",
    page_icon="🏎️",
    layout="wide"
)


# =====================================================
# TAHAP 3 — PATH FILE
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "model.pkl"

REQUIRED_FILES = {
    "races": DATA_DIR / "races.csv",
    "results": DATA_DIR / "results.csv",
    "drivers": DATA_DIR / "drivers.csv",
    "constructors": DATA_DIR / "constructors.csv",
    "circuits": DATA_DIR / "circuits.csv",
}


# =====================================================
# TAHAP 4 — LOAD DATASET
# =====================================================

@st.cache_data
def load_data():
    missing_files = []

    for name, path in REQUIRED_FILES.items():
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        return None, missing_files

    races = pd.read_csv(REQUIRED_FILES["races"])
    results = pd.read_csv(REQUIRED_FILES["results"])
    drivers = pd.read_csv(REQUIRED_FILES["drivers"])
    constructors = pd.read_csv(REQUIRED_FILES["constructors"])
    circuits = pd.read_csv(REQUIRED_FILES["circuits"])

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

    f1["top_10"] = f1["positionOrder"].apply(lambda x: 1 if x <= 10 else 0)
    f1["podium"] = f1["positionOrder"].apply(lambda x: 1 if x <= 3 else 0)
    f1["win"] = f1["positionOrder"].apply(lambda x: 1 if x == 1 else 0)

    return f1, []


# =====================================================
# TAHAP 5 — LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


# =====================================================
# TAHAP 6 — HEADER APP
# =====================================================

st.title("BoxOne")
st.write("F1 Data Analysis untuk analisis driver, constructor, dan prediksi finish Top 10.")

f1, missing_files = load_data()

if missing_files:
    st.error("Beberapa file dataset belum ditemukan.")
    st.write("File yang hilang:")
    for file in missing_files:
        st.code(file)
    st.stop()

if f1 is None or f1.empty:
    st.error("Dataset kosong atau gagal dimuat.")
    st.stop()


# =====================================================
# TAHAP 7 — SIDEBAR MENU
# =====================================================

st.sidebar.title("🏎️ Menu")

menu = st.sidebar.selectbox(
    "Pilih Halaman",
    [
        "Overview",
        "Driver Analysis",
        "Constructor Analysis",
        "Top 10 Predictor",
        "Data Preview"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Project: BoxOne")
st.sidebar.caption("Data: Formula 1 Historical Dataset")


# =====================================================
# TAHAP 8 — HALAMAN OVERVIEW
# =====================================================

if menu == "Overview":
    st.header("📊 Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Data Race", len(f1))
    col2.metric("Total Driver", f1["driver_name"].nunique())
    col3.metric("Total Constructor", f1["constructor_name"].nunique())
    col4.metric("Season", f'{int(f1["year"].min())} - {int(f1["year"].max())}')

    st.subheader("Jumlah Race per Tahun")

    races_per_year = (
        f1.groupby("year")["race_name"]
        .nunique()
        .reset_index()
        .rename(columns={"race_name": "total_race"})
    )

    fig = px.line(
        races_per_year,
        x="year",
        y="total_race",
        markers=True,
        title="Jumlah Race Formula 1 per Tahun"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Driver Berdasarkan Total Points")

    driver_points = (
        f1.groupby("driver_name")["points"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig2 = px.bar(
        driver_points,
        x="driver_name",
        y="points",
        title="Top 10 Driver Berdasarkan Total Points"
    )

    st.plotly_chart(fig2, use_container_width=True)


# =====================================================
# TAHAP 9 — HALAMAN DRIVER ANALYSIS
# =====================================================

elif menu == "Driver Analysis":
    st.header("👤 Driver Analysis")

    driver = st.selectbox(
        "Pilih Driver",
        sorted(f1["driver_name"].dropna().unique())
    )

    driver_data = f1[f1["driver_name"] == driver]

    total_race = len(driver_data)
    total_wins = int(driver_data["win"].sum())
    total_podiums = int(driver_data["podium"].sum())
    total_points = round(driver_data["points"].sum(), 2)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Race", total_race)
    col2.metric("Wins", total_wins)
    col3.metric("Podiums", total_podiums)
    col4.metric("Total Points", total_points)

    st.subheader(f"Points per Season - {driver}")

    points_per_year = (
        driver_data.groupby("year")["points"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        points_per_year,
        x="year",
        y="points",
        markers=True,
        title=f"Perolehan Points per Season - {driver}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Riwayat Race Driver")

    st.dataframe(
        driver_data.sort_values(["year", "race_name"], ascending=[False, True]),
        use_container_width=True
    )


# =====================================================
# TAHAP 10 — HALAMAN CONSTRUCTOR ANALYSIS
# =====================================================

elif menu == "Constructor Analysis":
    st.header("🏁 Constructor Analysis")

    constructor_points = (
        f1.groupby("constructor_name")["points"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    fig = px.bar(
        constructor_points,
        x="constructor_name",
        y="points",
        title="Top 15 Constructor Berdasarkan Total Points"
    )

    st.plotly_chart(fig, use_container_width=True)

    constructor = st.selectbox(
        "Pilih Constructor",
        sorted(f1["constructor_name"].dropna().unique())
    )

    constructor_data = f1[f1["constructor_name"] == constructor]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Race Data", len(constructor_data))
    col2.metric("Wins", int(constructor_data["win"].sum()))
    col3.metric("Podiums", int(constructor_data["podium"].sum()))
    col4.metric("Total Points", round(constructor_data["points"].sum(), 2))

    points_per_year = (
        constructor_data.groupby("year")["points"]
        .sum()
        .reset_index()
    )

    fig2 = px.line(
        points_per_year,
        x="year",
        y="points",
        markers=True,
        title=f"Performa {constructor} per Season"
    )

    st.plotly_chart(fig2, use_container_width=True)


# =====================================================
# TAHAP 11 — HALAMAN TOP 10 PREDICTOR
# =====================================================

elif menu == "Top 10 Predictor":
    st.header("🤖 Top 10 Predictor")

    st.write(
        "Prediksi sederhana apakah pembalap berpeluang finish Top 10 "
        "berdasarkan tahun, grid start, constructor, dan circuit."
    )

    model = load_model()

    if model is None:
        st.warning(
            "File model.pkl belum ditemukan di repository. "
            "Upload model.pkl atau jalankan train_model.py terlebih dahulu."
        )
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input(
            "Year",
            min_value=int(f1["year"].min()),
            max_value=int(f1["year"].max()),
            value=int(f1["year"].max())
        )

        grid = st.number_input(
            "Grid Start",
            min_value=1,
            max_value=30,
            value=5
        )

    with col2:
        constructor_name = st.selectbox(
            "Constructor",
            sorted(f1["constructor_name"].dropna().unique())
        )

        circuit_name = st.selectbox(
            "Circuit",
            sorted(f1["circuit_name"].dropna().unique())
        )

    if st.button("Prediksi"):
        sample = pd.DataFrame([{
            "year": year,
            "grid": grid,
            "constructor_name": constructor_name,
            "circuit_name": circuit_name
        }])

        try:
            prediction = model.predict(sample)[0]
            probability = model.predict_proba(sample)[0][1]

            if prediction == 1:
                st.success("Prediksi: Finish Top 10 ✅")
            else:
                st.error("Prediksi: Tidak Finish Top 10 ❌")

            st.metric("Probabilitas Top 10", f"{probability * 100:.2f}%")

        except Exception as e:
            st.error("Terjadi error saat prediksi.")
            st.code(str(e))




elif menu == "Data Preview":
    st.header("🧾 Data Preview")

    st.write("Preview dataset hasil gabungan.")

    st.dataframe(f1.head(100), use_container_width=True)

    st.subheader("Informasi Kolom")
    st.write(list(f1.columns))

    st.subheader("Cek Missing Value")
    missing = f1.isnull().sum().reset_index()
    missing.columns = ["column", "missing_value"]

    st.dataframe(missing, use_container_width=True)