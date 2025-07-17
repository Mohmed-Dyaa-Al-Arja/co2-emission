import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from catboost import CatBoostRegressor
import warnings
import plotly.express as px
warnings.filterwarnings("ignore")

MODEL_PATH = "final_cat_model.cbm"
HISTORY_CSV = "Agrofood_co2_emission.csv"

try:
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load the model : {e}")
    st.stop()

try:
    history = pd.read_csv(HISTORY_CSV)
except Exception as e:
    st.error(f" Could not load the data : {e}")
    st.stop()

history.columns = history.columns.str.replace(" ", "_")
MODEL_COLS = model.feature_names_

texts = {
    "ar": {
        "title": "توقع وتحليل درجات الحرارة المستقبلية",
        "method": "اختر طريقة الإدخال",
        "manual": "إدخال يدوي",
        "csv": "رفع ملف CSV",
        "area": "المنطقة",
        "year": "السنة المستهدفة",
        "predict": "تنبؤ",
        "upload_here": "ارفع ملف CSV (Area, Year, total_emission, Total_Population_-_Male, Total_Population_-_Female...)",
        "result": "النتيجة:",
        "completed": "تم! النتائج بالأسفل",
        "error": "خطأ: ",
        "temp_direction": "اتجاه درجة الحرارة:",
        "temp_difference": "فرق درجة الحرارة (°م):",
        "language_select": "اختر اللغة",
        "direction_up": "ارتفاع",
        "direction_down": "انخفاض",
        "direction_stable": "ثبات",
        "explore_charts": "استكشاف البيانات",
        "chart_type": "اختر نوع الرسم:",
        "x_column": "المحور X",
        "y_column": "المحور Y",
        "color_by": "التلوين حسب:",
        "heatmap": "خريطة حرارية (لو الإحداثيات متاحة)",
        "avg_change": "متوسط التغير في درجة الحرارة للدول (%)",
         "start_color": "لون البداية",
        "end_color": "لون النهاية",
    },
    "en": {
        "title": "Future Temperature Forecast & Analysis",
        "method": "Choose Input Method",
        "manual": "Manual Input",
        "csv": "Upload CSV File",
        "area": "Area",
        "year": "Target Year",
        "predict": "Predict",
        "upload_here": "Upload CSV (Area, Year, total_emission, Total_Population_-_Male, Total_Population_-_Female...)",
        "result": "Prediction:",
        "completed": "Done! Results below",
        "error": "Error: ",
        "temp_direction": "Temperature Direction:",
        "temp_difference": "Temperature Difference (°C):",
        "language_select": "Select Language",
        "direction_up": "Increase",
        "direction_down": "Decrease",
        "direction_stable": "Stable",
        "explore_charts": "Explore Data",
        "chart_type": "Chart Type:",
        "x_column": "X-axis",
        "y_column": "Y-axis",
        "color_by": "Color by:",
        "heatmap": "Heatmap (if coordinates available)",
        "avg_change": "Average Temperature Change by Country (%)",
        "start_color": "Start Color",
        "end_color": "End Color",
    },
}

lang_choice = st.sidebar.selectbox("🌐", ["العربية", "English"], index=0)
T = texts["ar"] if lang_choice == "العربية" else texts["en"]

st.set_page_config(page_title=T["title"], layout="wide")
st.title(T["title"])


def build_features(df_input: pd.DataFrame, hist_df: pd.DataFrame) -> pd.DataFrame:
    df_input = df_input.copy()
    df_input.columns = df_input.columns.str.replace(" ", "_")

    for col in MODEL_COLS:
        if col not in df_input.columns:
            df_input[col] = np.nan

    df_all = pd.concat([hist_df, df_input], ignore_index=True)
    df_all["Area_enc"] = df_all["Area"].astype("category").cat.codes
    df_all = df_all.sort_values(["Area_enc", "Year"]).reset_index(drop=True)

    for lag in [1, 2, 3, 5]:
        df_all[f"temp_lag{lag}"] = df_all.groupby("Area_enc")["Average_Temperature_°C"].shift(lag)
        df_all[f"emission_lag{lag}"] = df_all.groupby("Area_enc")["total_emission"].shift(lag)

    num_cols = [c for c in df_all.columns if df_all[c].dtype != "object" and c != "Area_enc"]
    df_all[num_cols] = df_all.groupby("Area")[num_cols].transform(lambda g: g.fillna(method="ffill").fillna(method="bfill"))
    df_all[num_cols] = df_all.groupby("Area")[num_cols].transform(lambda g: g.fillna(g.median()))

    for col in num_cols:
        if df_all[col].isna().any():
            df_all[f"{col}_missing"] = df_all[col].isna().astype(int)
            df_all[col] = df_all[col].fillna(df_all[col].median())

    df_all["temp_roll3"] = df_all.groupby("Area_enc")["Average_Temperature_°C"].rolling(3, min_periods=2).mean().reset_index(level=0, drop=True)
    df_all["Year_idx"] = df_all["Year"] - df_all["Year"].min()

    df_feat = df_all.loc[df_input.index].copy()
    for col in MODEL_COLS:
        if col not in df_feat.columns:
            df_feat[col] = hist_df[col].median() if col in hist_df.columns else 0

    return df_feat[MODEL_COLS]

def temperature_direction(pred_c: float, base_c: float) -> str:
    diff = pred_c - base_c
    if diff > 0:
        return T["direction_up"]
    if diff < 0:
        return T["direction_down"]
    return T["direction_stable"]

def c_to_f(c: float) -> float:
    return c * 9 / 5 + 32

mode = st.radio(T["method"], [T["manual"], T["csv"]])

MANUAL_COLS = [col for col in history.columns if col not in ["Area", "Year", "Average_Temperature_°C"]]
#yy
if mode == T["manual"]:
    st.subheader(T["manual"])
    area = st.selectbox(T["area"], sorted(history["Area"].dropna().unique()))
    year = st.number_input(T["year"], min_value=2025, max_value=2100, step=1, value=2030)

    input_data = {}
    cols_per_row = 3  

    for i in range(0, len(MANUAL_COLS), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(MANUAL_COLS[i:i+cols_per_row]):
            with cols[j]:
                input_data[col] = st.number_input(col.replace("_", " "), value=0.0)

    df_manual = pd.DataFrame([input_data])
    df_manual["Area"] = area
    df_manual["Year"] = year

    if st.button(T["predict"]):
        df_feat = build_features(df_manual, history)
        prediction = model.predict(df_feat)[0]
        latest_temp = history.loc[
            (history["Area"] == area) & (history["Year"] < year),
            "Average_Temperature_°C"
        ].iloc[-1] if not history.empty else np.nan
        diff = prediction - latest_temp
        direction = temperature_direction(prediction, latest_temp)

        st.success(T["completed"])
        st.metric(T["result"], f"{prediction:.2f}°C")
        st.metric(T["temp_difference"], f"{diff:.2f}°C")
        st.metric(T["temp_direction"], direction)

else:
    file = st.file_uploader(T["upload_here"], type="csv")
    if file:
        try:
            df_up = pd.read_csv(file)
            df_up.columns = df_up.columns.str.replace(" ", "_")

            basic_required = {
                "Area", "Year", "total_emission",
                "Total_Population_-_Male", "Total_Population_-_Female"
            }
            missing_req = basic_required - set(df_up.columns)
            if missing_req:
                st.error(f"{T['error']}{missing_req}")
                st.stop()
            for col in model.feature_names_:
                if col not in df_up.columns:
                    df_up[col] = np.nan

            X = build_features(df_up, history)
            df_up["Pred_Temp_°C"] = np.round(model.predict(X), 3)
            df_up["Pred_Temp_°F"] = np.round(c_to_f(df_up["Pred_Temp_°C"]), 2)

            def get_base(row):
                past = history[(history["Area"] == row["Area"]) & (history["Year"] < row["Year"])]
                return past["Average_Temperature_°C"].iloc[-1] if not past.empty else np.nan

            df_up["Base_Temp_°C"] = df_up.apply(get_base, axis=1)
            df_up["Temp_Diff_°C"] = df_up["Pred_Temp_°C"] - df_up["Base_Temp_°C"]
            df_up["Temp_Diff_%"] = (df_up["Temp_Diff_°C"] / df_up["Base_Temp_°C"]).replace([np.inf, -np.inf], np.nan) * 100
            df_up["Direction"] = df_up.apply(
                lambda r: temperature_direction(r["Pred_Temp_°C"], r["Base_Temp_°C"]), axis=1
            )

            st.success(f"{T['completed']} (Rows: {len(df_up)})")
            st.dataframe(df_up)

            with st.expander("📊 " + T["avg_change"]):
                avg_df = (
                    df_up.groupby("Area")["Temp_Diff_%"]
                    .mean()
                    .reset_index()
                    .sort_values("Temp_Diff_%", ascending=False)
                )

                col1, col2 = st.columns(2)
                start_color = col1.color_picker(T["start_color"], "#636EFA")
                end_color   = col2.color_picker(T["end_color"], "#EF553B")
                cscale = [start_color, end_color]

                fig_bar = px.bar(
                    avg_df, x="Area", y="Temp_Diff_%", text="Temp_Diff_%",
                    color="Temp_Diff_%", color_continuous_scale=cscale
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                st.dataframe(avg_df)

                fig_world = px.choropleth(
                    avg_df, locations="Area", locationmode="country names",
                    color="Temp_Diff_%", color_continuous_scale=cscale,
                    labels={"Temp_Diff_%": "% Δ Temp"}, projection="natural earth"
                )
                st.plotly_chart(fig_world, use_container_width=True)

            if {"Latitude", "Longitude"}.issubset(df_up.columns):
                with st.expander("🗺️ " + T["heatmap"]):
                    mdf = df_up.dropna(subset=["Latitude", "Longitude"])

                    hcol1, hcol2 = st.columns(2)
                    start_h = hcol1.color_picker(T["start_color"], "#636EFA", key="heat_start")
                    end_h   = hcol2.color_picker(T["end_color"], "#EF553B", key="heat_end")
                    cscale_h = [start_h, end_h]

                    fig_map = px.scatter_geo(
                        mdf, lat="Latitude", lon="Longitude",
                        color="Temp_Diff_%", hover_name="Area",
                        color_continuous_scale=cscale_h, size_max=18,
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                    st.map(mdf.rename(columns={"Latitude": "lat", "Longitude": "lon"}))

            with st.expander("📈 " + T["explore_charts"]):
                st.markdown("### أفكار لاختيار الألوان:")
                st.markdown("- لون واحد ثابت")
                st.markdown("- تدريج لونين")
                st.markdown("- لوحة فئات مميزة")

                chart_type = st.selectbox(
                    T["chart_type"],
                    ["bar", "line", "area", "scatter", "box", "violin", "strip", "hist", "density"]
                )
                x_col = st.selectbox(T["x_column"], ["Pesticides_Manufacturing_missing"] + list(df_up.columns))
                y_col = st.selectbox(T["y_column"], df_up.columns)
                color_mode = st.selectbox(T["color_by"], ["No Color", "Two‑Tone Gradient", "Categorical Palette"])

                if color_mode == "No Color":
                    color_kw = {}
                elif color_mode == "Two‑Tone Gradient":
                    grad_start = st.color_picker("Start", "#636EFA", key="grad_start")
                    grad_end   = st.color_picker("End",   "#EF553B", key="grad_end")
                    color_kw = {"color": y_col, "color_continuous_scale": [grad_start, grad_end]}
                else:
                    color_kw = {"color": "Area"}

                fig = None
                if chart_type == "bar":
                    fig = px.bar(df_up, x=x_col, y=y_col, **color_kw)
                elif chart_type == "line":
                    fig = px.line(df_up, x=x_col, y=y_col, **color_kw)
                elif chart_type == "area":
                    fig = px.area(df_up, x=x_col, y=y_col, **color_kw)
                elif chart_type == "scatter":
                    fig = px.scatter(df_up, x=x_col, y=y_col, **color_kw)
                elif chart_type == "box":
                    fig = px.box(df_up, x=x_col, y=y_col, **color_kw)
                elif chart_type == "violin":
                    fig = px.violin(df_up, x=x_col, y=y_col, **color_kw, box=True, points="all")
                elif chart_type == "strip":
                    fig = px.strip(df_up, x=x_col, y=y_col, **color_kw)
                elif chart_type == "hist":
                    fig = px.histogram(df_up, x=x_col, **color_kw)
                elif chart_type == "density":
                    fig = px.density_contour(df_up, x=x_col, y=y_col, **color_kw)

                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(T["error"] + str(e))

st.caption("© Mohamed Dyaa 2025")
