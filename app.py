##############################
# ADVANCED AIR QUALITY DASHBOARD — FINAL VERSION
# Zero Exclusion • Vulnerability Metrics • Clustering • Seasonality • Forecasting
#
# RUN:
#       streamlit run app.py
#
# Place the CSV files in the same folder:
#       stations.csv / Copy of stations.csv
#       city_day.csv
#       station_day.csv
#       city_hour.csv
#       station_hour.csv
##############################

import os
import math
import hashlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Silence the HDBSCAN syntax warning (library issue)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="hdbscan")

# Silence the Pandas GroupBy deprecation warning
warnings.filterwarnings("ignore", message=".*DataFrameGroupBy.apply operated on the grouping columns.*")

# Silence the UMAP n_jobs warning
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")

# Silence the Plotly deprecation warning (if you didn't fix it manually)
warnings.filterwarnings("ignore", message=".*scatter_mapbox is deprecated.*")

# Optional imports
try:
    import umap
    UMAP_AVAILABLE = True
except:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except:
    HDBSCAN_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False


#############################################
# UTILITIES
#############################################

def find_csv(name):
    candidates = [name, f"Copy of {name}"]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Missing required file: {name} or 'Copy of {name}'")


def numeric_columns(df, exclude=None):
    if exclude is None: exclude = []
    return [c for c in df.columns if (pd.api.types.is_numeric_dtype(df[c]) and c not in exclude)]


#############################################
# LOAD + CLEAN (ZERO EXCLUSION)
#############################################
@st.cache_data
def load_and_clean_all():
    url1 = "https://github.com/Sidb973/AQI_MONITORING_DASHBOARD/releases/download/filed/Copy.of.city_hour.csv"
    url2 = "https://github.com/Sidb973/AQI_MONITORING_DASHBOARD/releases/download/filed/Copy.of.station_hour.csv"
    stations = pd.read_csv(find_csv("stations.csv"))
    city_day = pd.read_csv(find_csv("city_day.csv"))
    station_day = pd.read_csv(find_csv("station_day.csv"))
    city_hour = pd.read_csv(url1)
    station_hour = pd.read_csv(url2 , low_memory=False)
    raw_city_day = city_day.copy()
    raw_station_day = station_day.copy()


    # Normalize columns
    def norm(df):
        df = df.copy()
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(".", "_").str.replace("-", "_")
        return df
    stations = norm(stations)
    city_day = norm(city_day)
    station_day = norm(station_day)
    city_hour = norm(city_hour)
    station_hour = norm(station_hour)

    

    # Dates
    for df, c in [(city_day, "Date"), (station_day, "Date")]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for df, c in [(city_hour, "Datetime"), (station_hour, "Datetime")]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Calendar fields
    def enrich(df, date_col):
        if date_col not in df.columns: return df
        df["Year"] = df[date_col].dt.year
        df["Month"] = df[date_col].dt.month
        df["DayOfYear"] = df[date_col].dt.dayofyear
        df["MonthName"] = df[date_col].dt.month_name()
        df["MonthYear"] = df[date_col].dt.to_period("M").astype(str)
        def season(m):
            if pd.isna(m): return np.nan
            m = int(m)
            if m in [12,1,2]: return "Winter"
            if m in [3,4,5]: return "Summer"
            if m in [6,7,8,9]: return "Monsoon"
            return "Post-Monsoon"
        df["Season"] = df["Month"].apply(season)
        return df

    city_day = enrich(city_day, "Date")
    station_day = enrich(station_day, "Date")
    city_hour = enrich(city_hour, "Datetime")
    station_hour = enrich(station_hour, "Datetime")

    # ---------------------------------------------------------   
    # Impute-numerics only by group (keeping rows)
    def impute(df, group_cols):
        df = df.copy()
        nums = numeric_columns(df)
        for col in nums:
            try:
                df[col] = df[col].fillna(df.groupby(group_cols)[col].transform("median"))
            except:
                df[col] = df[col].fillna(df[col].median())
        return df

    if "City" in city_day.columns:
        city_day = impute(city_day, ["City", "Year"])
    if "StationId" in station_day.columns:
        station_day = impute(station_day, ["StationId", "Year"])

    # Attach station metadata
    if "StationId" in station_day.columns and "StationId" in stations.columns:
        station_day = station_day.merge(stations, on="StationId", how="left")

    return dict(
        stations=stations,
        city_day=city_day,
        station_day=station_day,              # cleaned
        city_hour=city_hour,
        station_hour=station_hour,
        raw_city_day=raw_city_day,            # NEW
        raw_station_day=raw_station_day       # NEW
    )

#############################################
# CITY KPIs + VULNERABILITY METRIC (OPTION A)
#############################################
@st.cache_data
def compute_city_kpis(city_day):

    df = city_day.copy()

    if "AQI" not in df.columns:
        return pd.DataFrame()

    g = df.groupby("City").agg(
        Mean_AQI=("AQI","mean"),
        Median_AQI=("AQI","median"),
        Max_AQI=("AQI","max"),
        Min_AQI=("AQI","min"),
        Observed_Days=("Date","nunique")
    )

    # Risk days = AQI > 200
    risky = df.assign(Risky = df["AQI"]>200).groupby("City")["Risky"].sum()
    g["Risk_Days"] = risky
    g["Risk_Day_Fraction"] = g["Risk_Days"] / g["Observed_Days"]

    # Risk Score
    g["Risk_Score"] = g["Mean_AQI"] * g["Risk_Day_Fraction"]

    # Population Vulnerability Score (Option A)
    # Combination of:
    # - High Mean AQI
    # - Seasonal variance
    # - Diwali spikes
    # - Crop-burning spikes
    season_mean = df.groupby(["City","Season"])["AQI"].mean().unstack()
    season_range = season_mean.max(axis=1) - season_mean.min(axis=1)
    g["Seasonal_Spread"] = season_range

    # Vulnerability = weighted AQI + risk-day fraction + seasonal instability
    g["Vulnerability_Score"] = (
        0.5 * (g["Mean_AQI"] / g["Mean_AQI"].max()) +
        0.3 * g["Risk_Day_Fraction"] +
        0.2 * (g["Seasonal_Spread"] / g["Seasonal_Spread"].max())
    )

    # Tiers
    g = g.reset_index()
    g["Risk_Tier"] = pd.qcut(
        g["Vulnerability_Score"].rank(method="first"),
        4,
        labels=["Low","Moderate","High","Very High"]
    )

    return g


#############################################
# DEFENSIVE STATION HEALTH FUNCTION (FINAL VERSION)
#############################################
#############################################
# DEFENSIVE STATION HEALTH FUNCTION (with chemical-gap logic)
#############################################
@st.cache_data
def compute_station_health(station_day):
    """
    Computes station-level reliability metrics. A day is considered *present* (not a gap)
    if it has at most 3 missing pollutant measurements. If >3 pollutant values are missing
    on a given day for that station, the day is treated as a gap when computing observed days
    and longest downtime.
    """

    df = station_day.copy()

    if "StationId" not in df.columns:
        return pd.DataFrame()

    # Ensure basic metadata exists
    df["StationName"] = df.get("StationName", df["StationId"].astype(str))
    df["City"] = df.get("City", "Unknown")
    df["State"] = df.get("State", "Unknown")

    # parse dates if present
    has_date = "Date" in df.columns
    if has_date:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # --- identify pollutant columns (conservatively: non-metadata columns that contain numeric data) ---
    meta_cols = {
        "StationId", "StationName", "City", "State", "Date",
        "Year", "Month", "DayOfYear", "MonthName", "MonthYear",
        "Latitude", "Longitude"
    }

    pollutant_cols = []
    for c in df.columns:
        if c in meta_cols:
            continue
        # try a numeric conversion and accept column if any numeric value exists
        ser_num = pd.to_numeric(df[c], errors="coerce")
        if ser_num.notna().sum() > 0:
            # replace column in df with numeric-converted values for reliable missingness checks
            df[c] = ser_num
            pollutant_cols.append(c)

    # fallback: if none detected, treat AQI as only pollutant if it exists
    if not pollutant_cols:
        if "AQI" in df.columns:
            pollutant_cols = ["AQI"]
        else:
            pollutant_cols = []

    # GROUP-BY station base metadata
    base = df.groupby("StationId").agg(
        StationName=("StationName", "first"),
        City=("City", "first"),
        State=("State", "first")
    ).reset_index()

    # initialize stats with base
    stats = base.copy()

    if has_date and "Date" in df.columns:
        # first/last dates using all available rows (even gaps) so span is meaningful
        dt = df.groupby("StationId").agg(
            First_Date=("Date", "min"),
            Last_Date=("Date", "max")
        ).reset_index()

        # For gap logic we will compute, per station, the set of dates considered VALID (not a gap)
        gap_rows = []
        obs_days = []

        # iterate stations to compute: Observed_Days (days with <=3 missing pollutants),
        # Max_Gap_Days computed from valid dates sequence (gaps between valid days)
        for sid, sub in df.groupby("StationId"):
            # ensure Date present and sorted; dedupe dates
            dates = sub["Date"].dropna()
            if dates.empty:
                obs_days.append({"StationId": sid, "Observed_Days": 0})
                gap_rows.append({"StationId": sid, "Max_Gap_Days": 0})
                continue

            # Use daily-level rows — station_day should be one row per date; if multiple,
            # aggregate by date by taking first row (missingness counted per-date)
            daily = sub.set_index("Date").sort_index()

            # compute, for each row (which corresponds to a date), how many pollutant values are missing
            if pollutant_cols:
                missing_count = daily[pollutant_cols].isna().sum(axis=1)
            else:
                # no pollutant columns -> treat AQI-only or mark everything missing
                missing_count = pd.Series(1, index=daily.index)

            # valid day = missing_count <= 3 (i.e., at most 3 missing values)
            valid_mask = missing_count <= 3
            valid_dates = daily.index[valid_mask].unique().sort_values()

            # Observed days = number of valid unique dates
            observed = int(len(valid_dates))
            obs_days.append({"StationId": sid, "Observed_Days": observed})

            # compute max gap using the valid_dates array,
            # if there are <=1 valid day then Max_Gap_Days = 0
            if len(valid_dates) <= 1:
                max_gap = 0
            else:
                # convert to integer days since epoch for diff
                days = (pd.to_datetime(valid_dates).astype("int64") // 86400000000000).astype(int)
                max_gap = int(np.diff(days).max())

            gap_rows.append({"StationId": sid, "Max_Gap_Days": max_gap})

        # merge computed pieces
        dt = base.merge(dt, on="StationId", how="left")
        stats = dt.merge(pd.DataFrame(obs_days), on="StationId", how="left")
        stats = stats.merge(pd.DataFrame(gap_rows), on="StationId", how="left")

        # span and coverage
        stats["Span_Days"] = (stats["Last_Date"] - stats["First_Date"]).dt.days + 1
        stats["Span_Days"] = stats["Span_Days"].fillna(1).clip(lower=1)

        # Observed_Days maybe NaN for some -> fill 0
        stats["Observed_Days"] = stats["Observed_Days"].fillna(0).astype(int)

        stats["Coverage_Ratio"] = stats["Observed_Days"] / stats["Span_Days"]

        # AQI missing fraction: compute fraction of valid-days where AQI is missing (gives sense of AQI avail)
        # If AQI column missing entirely, mark as 1 (fully missing)
        aqi_missing = {}
        if "AQI" in df.columns and pollutant_cols:
            for sid, sub in df.groupby("StationId"):
                daily = sub.set_index("Date").sort_index()
                # compute missing_count again (same logic as above)
                missing_count = daily[pollutant_cols].isna().sum(axis=1) if pollutant_cols else pd.Series(1, index=daily.index)
                valid_mask = missing_count <= 3
                valid_daily = daily[valid_mask]
                if valid_daily.empty:
                    aqi_missing[sid] = 1.0
                else:
                    aqi_missing[sid] = float(valid_daily["AQI"].isna().mean())
        else:
            for sid in stats["StationId"].unique():
                aqi_missing[sid] = 1.0

        stats["AQI_Missing_Fraction"] = stats["StationId"].map(aqi_missing).fillna(1.0)

    else:
        # no date column present, fallback behavior
        stats = base.copy()
        stats["First_Date"] = pd.NaT
        stats["Last_Date"] = pd.NaT
        stats["Observed_Days"] = 0
        stats["Span_Days"] = 0
        stats["Coverage_Ratio"] = 0.0
        stats["AQI_Missing_Fraction"] = 1.0
        stats["Max_Gap_Days"] = 0

    # Reliability score: coverage * (1 - missing_fraction); clipped to [0,1]
    cov = stats["Coverage_Ratio"].fillna(0)
    miss = stats["AQI_Missing_Fraction"].fillna(1)
    stats["Reliability_Score"] = (cov * (1 - miss)).clip(0, 1)

    return stats


#############################################
# STATION MARKER CLASSIFICATION
#############################################
@st.cache_data
def classify_station_markers(station_day):
    df = station_day.copy()

    pollutant_cols = numeric_columns(df, exclude=["Year","Month","DayOfYear"])
    if not pollutant_cols: pollutant_cols=["AQI"]

    g = df.groupby("StationId")[pollutant_cols].mean().fillna(0).reset_index()

    # Ensure placeholders
    for col in ["SO2","NO2","CO","NOx"]:
        if col not in g.columns:
            g[col]=0

    g["industrial_score"] = g["SO2"] + g["NOx"]
    g["vehicular_score"] = g["NO2"] + g["CO"]

    # seasonal range
    try:
        s = df.groupby(["StationId","Season"])["AQI"].mean().unstack().fillna(0)
        g = g.merge(s.max(axis=1)-s.min(axis=1), left_on="StationId", right_index=True, how="left")
        g = g.rename(columns={0:"seasonal_range"})
    except:
        g["seasonal_range"]=0

    # thresholds
    g["Industrial_Flag"] = (g["industrial_score"] >= g["industrial_score"].quantile(0.75)).astype(int)
    g["Vehicular_Flag"] = (g["vehicular_score"] >= g["vehicular_score"].quantile(0.75)).astype(int)
    g["Seasonal_Flag"] = (g["seasonal_range"] >= g["seasonal_range"].quantile(0.75)).astype(int)

    return g




#############################################
# CLUSTER CITIES
#############################################
@st.cache_data
def cluster_cities(city_day):
    df = city_day.copy()

    pollutant_cols = numeric_columns(df, exclude=["Year","Month","DayOfYear","AQI"])
    if not pollutant_cols: pollutant_cols = ["AQI"]

    profile = df.groupby("City")[pollutant_cols].mean().fillna(0)
    X = profile.values

    if SKLEARN_AVAILABLE:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        pca = PCA(n_components=min(2, Xs.shape[1]))
        Xp = pca.fit_transform(Xs)
    else:
        Xp = X[:, :2]

    # clustering
    if HDBSCAN_AVAILABLE and len(profile)>=5:
        labels = hdbscan.HDBSCAN(min_cluster_size=max(3, len(profile)//10)).fit_predict(Xp)
    else:
        km = KMeans(n_clusters=min(4,max(2,len(profile)//5)), random_state=42)
        labels = km.fit_predict(Xp)

    out = profile.copy()
    out["Cluster"]=labels
    out = out.reset_index()
    return out, Xp, labels


#############################################
# SEASONALITY + FESTIVAL EFFECTS
#############################################
@st.cache_data
def seasonality_analysis(city_day):
    df = city_day.copy()

    monthly = df.groupby(["City","Year","Month","MonthName"])["AQI"].mean().reset_index()

    # Diwali periods
    diwali_dates = {
        2015:"2015-11-11",
        2016:"2016-10-30",
        2017:"2017-10-19",
        2018:"2018-11-07",
        2019:"2019-10-27",
        2020:"2020-11-14",
        2021:"2021-11-04",
        2022:"2022-10-24",
        2023:"2023-11-12",
        2024:"2024-11-01",
    }

    df["Is_Diwali"]=False
    for y,d in diwali_dates.items():
        d = pd.to_datetime(d)
        mask = (df["Date"]>=d-timedelta(days=7)) & (df["Date"]<=d+timedelta(days=7))
        df.loc[mask, "Is_Diwali"]=True

    diwali = df.groupby(["City","Is_Diwali"])["AQI"].mean().reset_index()

    # Crop-burning
    north = {"Punjab","Haryana","Delhi","UP","Uttar Pradesh","Bihar","Rajasthan"}
    df["Is_North"]=df["State"].isin(north) if "State" in df.columns else False
    df["Is_Crop"]=df["Month"].isin([10,11])
    crop = df.groupby(["City","Is_North","Is_Crop"])["AQI"].mean().reset_index()

    return dict(monthly=monthly, diwali=diwali, crop=crop)



#############################################
# ANOMALIES
#############################################
@st.cache_data
def detect_anomalies(df, pollutant="AQI"):
    df=df.copy().sort_values("Date")
    if pollutant not in df.columns:
        df["Anomaly"]=0
        return df

    if SKLEARN_AVAILABLE:
        X = df[[pollutant]].fillna(0)
        iso = IsolationForest(contamination=0.01, random_state=42)
        df["Anomaly"] = (iso.fit_predict(X)==-1).astype(int)
    else:
        z = (df[pollutant]-df[pollutant].mean())/(df[pollutant].std()+1e-9)
        df["Anomaly"] = (z.abs()>3).astype(int)
    return df


#############################################
# STATION EMBEDDING with fallback logic
#############################################
@st.cache_data
def station_embedding(station_day):

    df = station_day.copy()
    nums = numeric_columns(df, exclude=["Year","Month","DayOfYear"])
    if not nums: nums=["AQI"]

    profile = df.groupby("StationId")[nums].mean().fillna(0)

    if SKLEARN_AVAILABLE:
        scaler=StandardScaler()
        Xs=scaler.fit_transform(profile.values)
        try:
            if UMAP_AVAILABLE:
                emb = umap.UMAP(n_components=2, random_state=42).fit_transform(Xs)
            else:
                emb = PCA(n_components=2).fit_transform(Xs)
        except:
            emb = PCA(n_components=2).fit_transform(Xs)
    else:
        emb = profile.values[:, :2]

    profile["x"]=emb[:,0]
    profile["y"]=emb[:,1]

    # cluster
    if HDBSCAN_AVAILABLE and len(profile)>=5:
        try:
            labels = hdbscan.HDBSCAN(min_cluster_size=max(3,len(profile)//10)).fit_predict(emb)
        except:
            labels = np.zeros(len(profile))
    else:
        km = KMeans(n_clusters=min(4,max(2,len(profile)//5)), random_state=42)
        labels = km.fit_predict(emb)

    profile["cluster"]=labels
    return profile.reset_index()



#############################################
# STREAMLIT UI — START
#############################################
st.set_page_config(page_title="Air Quality Dashboard (Final)", layout="wide")
st.title("🌍 Advanced Air Quality Dashboard — Final Version (Zero-Exclusion)")
st.caption("All data is used. No city or station is dropped. Advanced clustering, vulnerability scoring, seasonality, event detection, anomaly detection, and policy suggestions included.")

data = load_and_clean_all()
stations = data["stations"]
city_day = data["city_day"]
station_day = data["station_day"]
city_hour = data["city_hour"]
station_hour = data["station_hour"]

#############################################
# SIDEBAR FILTERS
#############################################
st.sidebar.header("Filters")

cities = sorted(city_day["City"].dropna().unique())
selected_cities = st.sidebar.multiselect("Cities", options=cities, default=cities)

min_date = city_day["Date"].min()
max_date = city_day["Date"].max()
date_range = st.sidebar.date_input("Date Range", (min_date.date(), max_date.date()))

pollutants = numeric_columns(city_day)
default_poll = "PM2_5" if "PM2_5" in pollutants else ("AQI" if "AQI" in pollutants else pollutants[0])
pollutant = st.sidebar.selectbox("Pollutant", pollutants, index=pollutants.index(default_poll))

def filter_df(df):
    df=df.copy()
    if selected_cities:
        df=df[df["City"].isin(selected_cities)]
    if "Date" in df.columns:
        df=df[(df["Date"]>=pd.to_datetime(date_range[0])) & (df["Date"]<=pd.to_datetime(date_range[1]))]
    return df

city_day_view = filter_df(city_day)
station_day_view = filter_df(station_day)
city_hour_view = filter_df(city_hour)


#############################################
# EXECUTIVE SUMMARY
#############################################
st.header("Executive Summary")

city_kpis = compute_city_kpis(city_day)
kpis_view = city_kpis[city_kpis["City"].isin(selected_cities)]

col1,col2,col3,col4 = st.columns(4)
col1.metric("Avg AQI", f"{kpis_view['Mean_AQI'].mean():.1f}")
col2.metric("Avg Risk Score", f"{kpis_view['Risk_Score'].mean():.2f}")
col3.metric("Avg Vulnerability Score", f"{kpis_view['Vulnerability_Score'].mean():.2f}")
col4.metric("Cities Count", len(kpis_view))

st.markdown("### Vulnerability Ranking (Selected Cities)")
st.dataframe(kpis_view.sort_values("Vulnerability_Score", ascending=False)[["City","Mean_AQI","Risk_Days","Risk_Day_Fraction","Seasonal_Spread","Vulnerability_Score","Risk_Tier"]])


#############################################
# TABS
#############################################
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📈 Deep Dive",
    "🔗 Correlation & Clustering",
    "🏭 Station Health",
    "📅 Seasonality & Events",
    "⚠️ Forecast & Anomalies",
    "📘 Policy Recommendations"
])

#############################################
# TAB 1 — DEEP DIVE
#############################################
with tab1:
    st.subheader("Pollutant Trend Over Time")

    fig = go.Figure()
    for city in city_day_view["City"].unique():
        sub = city_day_view[city_day_view["City"]==city].sort_values("Date")
        fig.add_trace(go.Scatter(
            x=sub["Date"], y=sub[pollutant],
            mode='lines', name=city
        ))
    fig.update_layout(height=500, title=f"{pollutant} Trend")
    st.plotly_chart(fig, width="stretch")

    st.caption("Interpretation: Higher volatility or consistently elevated lines indicate systemic pollution issues. Sudden spikes often correspond to events or seasonal factors.")


    st.subheader("Hourly Heatmap (Daily Cycle Patterns)")
    if not city_hour_view.empty:
        city_hour_view["Hour"] = city_hour_view["Datetime"].dt.hour
        hourly = city_hour_view.groupby(["City","Hour"])[pollutant].mean().reset_index()
        heat = hourly.pivot(index="City", columns="Hour", values=pollutant)
        fig2 = px.imshow(heat, aspect="auto", color_continuous_scale="Reds")
        fig2.update_layout(height=700, title="Hourly Pollution Profile")
        st.plotly_chart(fig2, width="stretch")

        st.caption("Interpretation: Cities with heavy vehicular dominance show AM/PM peaks. Industrial cities show flatter patterns.")

#############################################
# TAB 2 — CORRELATION & CLUSTERING
#############################################
with tab2:
    st.subheader("Correlation Heatmap")

    corr = city_day_view[numeric_columns(city_day_view)].corr()
    figc = px.imshow(corr, text_auto=True, title="Pollutant Correlation Matrix", aspect="auto")
    st.plotly_chart(figc, width="stretch")

    st.caption("Interpretation: High correlation with AQI indicates pollution sources strongly influencing overall air quality. NO₂ ~ traffic, SO₂ ~ industry, PM2.5 ~ mixed sources.")

    st.subheader("City Clustering (Pollution Profile Similarity)")
    cluster_df, Xp, labels = cluster_cities(city_day)
    figcl = px.scatter(
        x=Xp[:,0], y=Xp[:,1],
        color=cluster_df["Cluster"].astype(str),
        hover_name=cluster_df["City"],
        title="City Clusters (UMAP/PCA)"
    )
    st.plotly_chart(figcl, width="stretch")
    st.caption("Cities in the same cluster share similar pollutant signatures — meaning similar sources and policy needs.")


#############################################
# TAB 3 — STATION HEALTH
#############################################
with tab3:
    
    st.subheader("Station Reliability Table")
    sh = compute_station_health(data["raw_station_day"])
    st.dataframe(sh.sort_values("Reliability_Score", ascending=False))

    st.subheader("Longest Downtime Stations")
    figgap = px.bar(
        sh.sort_values("Max_Gap_Days", ascending=False).head(40),
        x="Max_Gap_Days", y="StationName", orientation="h"
    )
    figgap.update_layout(height=700)
    st.plotly_chart(figgap, width="stretch")

    st.subheader("Station Marker Classification")
    sm = classify_station_markers(station_day)
    st.dataframe(sm[["StationId","industrial_score","vehicular_score","seasonal_range","Industrial_Flag","Vehicular_Flag","Seasonal_Flag"]].sort_values("industrial_score",ascending=False))

    st.caption("Interpretation: Industrial stations show elevated SO₂ / NOx. Vehicular stations show NO₂ / CO dominance. Seasonal stations exhibit strong seasonal fluctuations.")

# ---------------------------
# Enhanced Marker Visualizations
# (Station-level scatter, City bubble & Cluster summaries)
# ---------------------------

st.subheader("Visual Analysis — Industrial vs Vehicular vs Seasonal")

# station markers (ensure variable exists)
station_markers = classify_station_markers(station_day)  # columns: StationId, industrial_score, vehicular_score, seasonal_range, Industrial_Flag, Vehicular_Flag, Seasonal_Flag

# merge with station metadata for City/StationName if available
if "StationId" in stations.columns:
    sm_full = station_markers.merge(stations[["StationId","StationName","City","State"]], on="StationId", how="left")
else:
    sm_full = station_markers.copy()
    sm_full["StationName"] = sm_full["StationId"].astype(str)
    sm_full["City"] = sm_full["StationId"].astype(str)

# 1) Station-level scatter: industrial vs vehicular
fig_scatter = px.scatter(
    sm_full,
    x="industrial_score",
    y="vehicular_score",
    color=sm_full["Seasonal_Flag"].astype(str),
    size=(sm_full["seasonal_range"].fillna(0) + 1),   # +1 so we don't get size=0
    size_max=15,
    opacity=0.8,
    hover_name="StationName",
    hover_data=["StationId","City","industrial_score","vehicular_score","seasonal_range"],
    title="Station-level: Industrial score vs Vehicular score (size = seasonal range)"
)

fig_scatter.update_layout(legend_title="Seasonal_Flag (0/1)", height=600)
st.plotly_chart(fig_scatter, width="stretch")

st.markdown(
    "Interpretation: Stations high on the x-axis are industrial-characteristic (high SO₂/NOx). "
    "Stations high on the y-axis are vehicle-characteristic (high NO₂/CO). "
    "Large bubbles indicate strong seasonal swings (seasonal_flag likely 1)."
)

# 2) City-level aggregation for bubble chart + merge clusters
city_station_summary = (
    sm_full.groupby("City").agg(
        mean_industrial=("industrial_score","mean"),
        mean_vehicular=("vehicular_score","mean"),
        station_count=("StationId","nunique"),
        seasonal_count=("Seasonal_Flag","sum"),
        industrial_count=("Industrial_Flag","sum"),
        vehicular_count=("Vehicular_Flag","sum")
    ).reset_index()
)
city_station_summary["pct_seasonal"] = city_station_summary["seasonal_count"] / city_station_summary["station_count"]

# get city clusters from earlier clustering routine (cluster_df)
try:
    city_profiles, Xp, city_labels = cluster_cities(city_day)  # returns DataFrame with City + Cluster
except Exception:
    # fallback: mark cluster as unknown
    city_profiles = pd.DataFrame({"City": city_station_summary["City"].unique(), "Cluster": -1})

city_station_summary = city_station_summary.merge(city_profiles[["City","Cluster"]], on="City", how="left")

# Bubble chart: mean industrial vs mean vehicular per city
fig_city = px.scatter(
    city_station_summary,
    x="mean_industrial",
    y="mean_vehicular",
    size=(city_station_summary["pct_seasonal"].fillna(0) + 0.01),  # tiny epsilon so bubbles show
    size_max=20,
    color=city_station_summary["Cluster"].astype(str),
    opacity=0.8,
    hover_name="City",
    hover_data=["station_count","industrial_count","vehicular_count","seasonal_count","pct_seasonal"],
    title="City-level pollution signature (bubble size = % seasonal stations)"
)

fig_city.update_layout(height=650)
st.plotly_chart(fig_city, width="stretch")

st.markdown(
    "Interpretation: Cities in the upper-right have both industrial & vehicular signatures. "
    "Color groups show city clusters (similar pollution profiles). Large bubbles indicate cities where many stations show seasonal behaviour."
)

# 3) Cluster summary: stacked bar of flags per cluster
cluster_flag_summary = (
    sm_full.merge(city_profiles[["City","Cluster"]], on="City", how="left")
    .groupby(["Cluster"])
    .agg(
        stations_total=("StationId","nunique"),
        industrial_total=("Industrial_Flag","sum"),
        vehicular_total=("Vehicular_Flag","sum"),
        seasonal_total=("Seasonal_Flag","sum")
    ).reset_index()
)
# compute percentages
for col in ["industrial_total","vehicular_total","seasonal_total"]:
    cluster_flag_summary[col + "_pct"] = cluster_flag_summary[col] / cluster_flag_summary["stations_total"]

# stacked bar: absolute counts (interactive)
fig_stack = go.Figure()
fig_stack.add_trace(go.Bar(
    x=cluster_flag_summary["Cluster"].astype(str),
    y=cluster_flag_summary["industrial_total"],
    name="Industrial flagged (count)"
))
fig_stack.add_trace(go.Bar(
    x=cluster_flag_summary["Cluster"].astype(str),
    y=cluster_flag_summary["vehicular_total"],
    name="Vehicular flagged (count)"
))
fig_stack.add_trace(go.Bar(
    x=cluster_flag_summary["Cluster"].astype(str),
    y=cluster_flag_summary["seasonal_total"],
    name="Seasonal flagged (count)"
))
fig_stack.update_layout(barmode="stack", title="Cluster-wise counts of flagged stations", height=500)
st.plotly_chart(fig_stack, width="stretch")

st.markdown("Interpretation: This shows which clusters are dominated by industrial/vehicular/seasonal flagged stations. Use this to prioritize cluster-level interventions.")

# 4) (Optional) station reliability vs industrial/vehicular scatter (use `sh` computed above)
# correct - use the 'sh' computed above from raw_station_day
if "Reliability_Score" in sh.columns:
    merged = sm_full.merge(sh[["StationId","Reliability_Score"]], on="StationId", how="left")
    fig_rel = px.scatter(
        merged,
        x="industrial_score", y="vehicular_score",
        color="Reliability_Score",
        size=(merged["seasonal_range"].fillna(0) + 1),
        hover_name="StationName",
        title="Station markers colored by Reliability Score (size = seasonal_range)"
    )
    fig_rel.update_layout(height=600)
    st.plotly_chart(fig_rel, width="stretch")




# 5) Map overlay: stations colored by dominant marker
st.subheader("Map: Station markers (industrial / vehicular / seasonal)")

# ensure coords exist
if not {"Latitude","Longitude"}.issubset(stations.columns):
    # deterministic coords based on City if missing
    def det_coord(city):
        import hashlib
        base_lat, base_lon = 22.0, 78.0
        h = int(hashlib.sha256(str(city).encode()).hexdigest()[:8],16)
        lat = base_lat + ((h % 1000)/1000 - 0.5) * 10
        lon = base_lon + (((h//1000) %1000)/1000 - 0.5) * 15
        return lat, lon
    stations["Latitude"], stations["Longitude"] = zip(*stations["City"].astype(str).map(det_coord))

map_df = sm_full.merge(stations[["StationId","Latitude","Longitude"]], on="StationId", how="left")

# marker type column
def dominant_marker(row):
    # pick highest of industrial, vehicular, seasonal flags as label (ties possible)
    labels=[]
    if row.get("Industrial_Flag",0): labels.append("Industrial")
    if row.get("Vehicular_Flag",0): labels.append("Vehicular")
    if row.get("Seasonal_Flag",0): labels.append("Seasonal")
    return ",".join(labels) if labels else "None"

map_df["MarkerType"] = map_df.apply(dominant_marker, axis=1)

fig_map = px.scatter_map(  # <--- CHANGED NAME
    map_df,
    lat="Latitude", lon="Longitude",
    color="MarkerType",
    size="industrial_score",
    hover_name="StationName",
    hover_data=["StationId","City","industrial_score","vehicular_score","seasonal_range"],
    zoom=4,
    height=600
)
# Update style for the new map type
fig_map.update_layout(map_style="open-street-map") # <--- CHANGED PARAMETER NAME
st.plotly_chart(fig_map, width="stretch")

st.markdown("Map legend: Color by MarkerType (stations can have multiple flags). Bubble size = industrial score (you can inspect hover data for vehicular/seasonal).")


#############################################
# TAB 4 — SEASONALITY
#############################################
with tab4:
    st.subheader("Monthly AQI Heatmap")
    seas = seasonality_analysis(city_day)
    monthly = seas["monthly"]

    # --- Robust pivot for monthly heatmap (aggregate across years so each City x Month is unique)
# monthly: DataFrame with columns ['City','Year','Month','MonthName','AQI_Monthly_Mean'] or ['AQI']
# if monthly has Year, collapse across years to get mean monthly profile per city

# If monthly includes Year, aggregate across years first
    if {"City","Month","Year"}.issubset(monthly.columns):
        monthly_agg = monthly.groupby(["City","Month"]).agg(AQI_Monthly_Mean=("AQI", "mean")).reset_index()
    else:
    # If monthly is already aggregated per city-month, use as-is but ensure column name
        if "AQI" in monthly.columns and "AQI_Monthly_Mean" not in monthly.columns:
            monthly_agg = monthly.rename(columns={"AQI":"AQI_Monthly_Mean"}).copy()
        else:
            monthly_agg = monthly.copy()

# Create pivot_table (safe against duplicates; uses mean if duplicates persist)
    pivot = monthly_agg.pivot_table(index="City", columns="Month", values="AQI_Monthly_Mean", aggfunc="mean")

# Ensure months go 1..12 in order (if present)
    month_cols = [c for c in range(1,13) if c in pivot.columns]
    pivot = pivot.reindex(columns=month_cols)

# Replace any NA with 0 or keep NaN (your choice). We'll keep NaN but fill for visualization:
    pivot_filled = pivot.fillna(0)  # safe for plotting heatmap

    figm = px.imshow(
    pivot_filled,
    aspect="auto",
    color_continuous_scale="Reds",
    labels=dict(x="Month", y="City", color="Mean AQI"),
    title="Monthly mean AQI by City (aggregated across years)"
)

    figm.update_layout(height=800)
    st.plotly_chart(figm, width="stretch")

    st.subheader("Diwali Impact (±7 days)")
    di = seas["diwali"].copy()
# compute mean during window and outside for each city (safe aggregation)
    di_agg = di.groupby("City").apply(
        lambda g: pd.Series({
        "AQI_Diwali": g.loc[g["Is_Diwali"], "AQI"].mean() if g["Is_Diwali"].any() else np.nan,
        "AQI_Other" : g.loc[~g["Is_Diwali"], "AQI"].mean() if (~g["Is_Diwali"]).any() else np.nan
        })
    ).reset_index()
    di_agg["Delta"] = di_agg["AQI_Diwali"] - di_agg["AQI_Other"]
    st.dataframe(di_agg.sort_values("AQI_Diwali", ascending=False).head(50))


    st.subheader("Crop Burning Impact (North India, Oct–Nov)")
    cp = seas["crop"].copy()
    crop_agg = cp.groupby("City").apply(
        lambda g: pd.Series({
        "AQI_CropMonths": g.loc[g["Is_Crop"], "AQI"].mean() if g["Is_Crop"].any() else np.nan,
        "AQI_Other"     : g.loc[~g["Is_Crop"], "AQI"].mean() if (~g["Is_Crop"]).any() else np.nan
        })
    ).reset_index()
    crop_agg["Delta"] = crop_agg["AQI_CropMonths"] - crop_agg["AQI_Other"]
    st.dataframe(crop_agg.sort_values("AQI_CropMonths", ascending=False).head(50))


    st.caption("Interpretation: Cities showing strong Diwali or crop-burning spikes need targeted seasonal interventions.")


#############################################
# TAB 5 — FORECASTS & ANOMALIES
#############################################
with tab5:

    st.subheader("Anomaly Detection")
    anomalies = detect_anomalies(city_day_view, pollutant)
    figan = go.Figure()

    for city in anomalies["City"].unique():
        sub = anomalies[anomalies["City"]==city]
        figan.add_trace(go.Scatter(
            x=sub["Date"], y=sub[pollutant],
            mode="lines", name=city, opacity=0.4
        ))
        anom = sub[sub["Anomaly"]==1]
        figan.add_trace(go.Scatter(
            x=anom["Date"], y=anom[pollutant],
            mode="markers", marker=dict(color="red",size=6),
            name=f"{city} anomalies"
        ))

    figan.update_layout(height=600, title="Anomaly Markers (Red Points)")
    st.plotly_chart(figan, width="stretch")

    st.caption("Anomalies often correspond to industrial leaks, construction fires, festival bursts, or meteorological stagnation events.")


    st.subheader("Forecasting")
    fc_city = st.selectbox("City", cities)
    fc_poll = st.selectbox("Pollutant for Forecast", pollutants)

    horizon = st.slider("Forecast Horizon (days)", 7, 180, 30)

    if st.button("Run Forecast"):

        ts = city_day[city_day["City"]==fc_city][["Date",fc_poll]].dropna()
        ts = ts.rename(columns={"Date":"ds", fc_poll:"y"})
        ts = ts.set_index("ds").resample("D").mean().reset_index()

        if ts.empty:
            st.error("Not enough data for forecasting.")
        else:
            if PROPHET_AVAILABLE:
                m=Prophet(daily_seasonality=True, yearly_seasonality=True)
                m.fit(ts)
                future=m.make_future_dataframe(periods=horizon)
                forecast=m.predict(future)

                figf=go.Figure()
                figf.add_trace(go.Scatter(x=ts["ds"], y=ts["y"], name="History"))
                figf.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
                figf.update_layout(height=600)
                st.plotly_chart(figf, width="stretch")
            else:
                st.error("Prophet not installed. Install it to enable forecasting.")


#############################################
# TAB 6 — POLICY
#############################################
with tab6:
    st.subheader("High-Risk Cities (Top Priority)")
    st.dataframe(kpis_view.sort_values("Vulnerability_Score", ascending=False).head(15))

    st.subheader("Automated Policy Suggestions")
    st.markdown("""
### Key Recommendations:
1. **Strengthen monitoring** in cities with low reliability & high vulnerability.
2. **Target seasonal spikes** (Oct–Nov crop burning, Diwali period).
3. **Vehicular emission controls** where NO₂/CO patterns dominate.
4. **Industrial compliance checks** where SO₂ / NOx are high.
5. **Issue public advisories** during high-risk seasons and anomaly peaks.
6. **Expand sensor density** in Tier-1 & Tier-2 cities with high variability.
    """)



#############################################
# STATION CLUSTER EXPLORER (BOTTOM SECTION)
#############################################
st.header("Interactive Station Cluster Explorer")

embed = station_embedding(station_day)
embed = embed.merge(stations[["StationId","StationName","City"]], on="StationId", how="left")

fig_emb = px.scatter(
    embed, x="x", y="y",
    color=embed["cluster"].astype(str),
    hover_data=["StationId","StationName","City"],
    title="Station Embedding (UMAP/PCA)"
)
fig_emb.update_traces(marker=dict(size=8))
st.plotly_chart(fig_emb, width="stretch")

sel_station = st.selectbox("Select Station", options=embed["StationId"].astype(str))
prof = station_day[station_day["StationId"].astype(str)==sel_station]

if not prof.empty:
    numeric = numeric_columns(prof, exclude=["Year","Month","DayOfYear"])
    if not numeric: numeric=["AQI"]
    fp = prof[numeric].mean()
    fig_fp = px.bar(x=fp.index, y=fp.values, labels={"x":"Pollutant","y":"Mean Value"},
                    title=f"Pollutant Fingerprint of Station {sel_station}")
    st.plotly_chart(fig_fp, width="stretch")


st.caption("Dashboard enforces zero-exclusion. All rows retained. All charts update dynamically.")
