"""
app.py - Flask API for the Rossmann daily store-sales model.

Endpoints
  GET  /health   liveness + artifact status + model metrics
  GET  /schema   what /predict expects
  POST /predict  predict daily sales for one or many (Store, Date) rows

Artifacts (created by export_artifacts.py) live in ./artifacts/
"""
import json
import os

import holidays
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
app = Flask(__name__)

# ---------------- load artifacts once ----------------
try:
    model = joblib.load(os.path.join(ART, "model.joblib"))
    scaler = joblib.load(os.path.join(ART, "scaler.joblib"))
    encoder = joblib.load(os.path.join(ART, "encoder.joblib"))
    with open(os.path.join(ART, "feature_config.json")) as f:
        CFG = json.load(f)
    STORES = pd.read_csv(os.path.join(ART, "store.csv")).set_index("Store")
    STATS = pd.read_csv(os.path.join(ART, "store_statistics.csv")).set_index("Store")
    WEATHER = pd.read_csv(os.path.join(ART, "weather.csv"), parse_dates=["Date"])
    LOADED, LOAD_ERROR = True, None
except Exception as e:  # keep app alive so /health can explain what is wrong
    model = scaler = encoder = CFG = STORES = STATS = WEATHER = None
    LOADED, LOAD_ERROR = False, f"{type(e).__name__}: {e}"

NUMERIC_COLS = CFG["numeric_cols"] if CFG else []
CATEGORICAL_COLS = CFG["categorical_cols"] if CFG else []
ENCODED_COLS = CFG["encoded_cols"] if CFG else []
COMP_MEDIAN = CFG["competition_distance_median"] if CFG else None

if LOADED:
    _w = WEATHER.copy()
    _w["Month"] = _w["Date"].dt.month
    MONTHLY_WEATHER = _w.groupby("Month")[["Max_Temp", "Rain_Sum", "Snow_Sum"]].mean()

STATE_HOLIDAY_MAP = {"0": "None", "a": "Public", "b": "Easter", "c": "Christmas"}
STORE_TYPE_MAP = {"a": "Typical Store", "b": "Flagship Dept Store", "c": "Smaller Urban Store", "d": "Extended Store"}
ASSORTMENT_MAP = {"a": "Basic", "b": "Extra", "c": "Extended"}
MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"}
REQUIRED = ["Store", "Date", "Promo", "StateHoliday", "SchoolHoliday"]
OPTIONAL = ["Open", "Max_Temp", "Rain_Sum", "Snow_Sum"]


def build_raw_features(records):
    """Mirror the notebook's cleaning + feature engineering (before encoding/scaling)."""
    df = pd.DataFrame.from_records(records)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    df["Store"] = df["Store"].astype(int)
    unknown = sorted(set(df["Store"]) - set(STORES.index))
    if unknown:
        raise ValueError(f"Unknown Store id(s): {unknown}. Valid ids are 1-1115.")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    if "Open" not in df.columns:
        df["Open"] = 1
    df["Open"] = df["Open"].fillna(1).astype(int)
    for c in ("Promo", "SchoolHoliday"):
        df[c] = df[c].astype(int)

    sh = df["StateHoliday"].astype(str).str.strip().str.lower()
    bad = sorted(set(sh) - set(STATE_HOLIDAY_MAP))
    if bad:
        raise ValueError(f"StateHoliday must be one of 0, a, b, c (got {bad}).")
    df["StateHoliday"] = sh.map(STATE_HOLIDAY_MAP)

    # store attributes + the same NaN handling as the notebook
    df = df.merge(STORES.reset_index(), how="left", on="Store")
    df["StoreType"] = df["StoreType"].map(STORE_TYPE_MAP)
    df["Assortment"] = df["Assortment"].map(ASSORTMENT_MAP)
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(COMP_MEDIAN)
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(1)
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(1990)
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)
    df["PromoInterval"] = df["PromoInterval"].fillna("Unknown")

    # calendar features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.dayofweek + 1  # Rossmann: 1=Mon ... 7=Sun

    de = holidays.Germany(years=sorted(df["Year"].unique().tolist()))
    df["Public_Holiday"] = df["Date"].apply(lambda d: int(d.date() in de))

    # weather: use supplied values, else the cached Frankfurt data, else monthly average
    df = df.merge(WEATHER, how="left", on="Date", suffixes=("", "_hist"))
    for c in ("Max_Temp", "Rain_Sum", "Snow_Sum"):
        if c not in df.columns:
            df[c] = np.nan
        if c + "_hist" in df.columns:
            df[c] = df[c].fillna(df[c + "_hist"])
            df = df.drop(columns=[c + "_hist"])
        df[c] = df[c].fillna(df["Month"].map(MONTHLY_WEATHER[c]))

    # competition features
    opened = pd.to_datetime(
        df["CompetitionOpenSinceYear"].astype(int).astype(str) + "-"
        + df["CompetitionOpenSinceMonth"].astype(int).astype(str) + "-01", errors="coerce")
    df["CompetitionActive"] = (df["Date"] >= opened).astype(int)
    df["CompetitionMonthsOpen"] = ((df["Year"] - opened.dt.year) * 12 + (df["Month"] - opened.dt.month)).clip(lower=0)
    df["Log_CompetitionDistance"] = np.log1p(df["CompetitionDistance"])

    # Promo2 active this month?
    month_str = df["Month"].map(MONTH_ABBR)
    started = (df["Year"] > df["Promo2SinceYear"]) | (
        (df["Year"] == df["Promo2SinceYear"]) & (df["WeekOfYear"] >= df["Promo2SinceWeek"]))
    in_interval = [(m in iv) if (p == 1 and iv != "Unknown") else False
                   for m, iv, p in zip(month_str, df["PromoInterval"], df["Promo2"])]
    df["IsPromo2MonthActive"] = ((df["Promo2"] == 1) & started & pd.Series(in_interval, index=df.index)).astype(int)

    # store history features (computed on the training split in the notebook)
    df = df.merge(STATS.reset_index(), how="left", on="Store")
    return df


def predict_records(records):
    df = build_raw_features(records)
    X_num = pd.DataFrame(scaler.transform(df[NUMERIC_COLS]), columns=NUMERIC_COLS, index=df.index)
    X_cat = pd.DataFrame(encoder.transform(df[CATEGORICAL_COLS]), columns=ENCODED_COLS, index=df.index)
    X = pd.concat([X_num, X_cat], axis=1)[NUMERIC_COLS + ENCODED_COLS]
    sales = np.expm1(model.predict(X))          # model predicts log1p(Sales)
    sales = np.clip(sales, 0, None)
    sales[df["Open"].values == 0] = 0.0         # closed store => 0 sales (as in the notebook)
    return [round(float(s), 2) for s in sales]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if LOADED else "artifacts_missing",
        "artifacts_loaded": LOADED,
        "error": LOAD_ERROR,
        "metrics": CFG.get("metrics") if CFG else None,
        "trained_on": CFG.get("train_date_range") if CFG else None,
    }), (200 if LOADED else 503)


@app.route("/schema", methods=["GET"])
def schema():
    return jsonify({
        "required_fields": REQUIRED,
        "optional_fields": OPTIONAL,
        "notes": {
            "Store": "integer 1-1115",
            "Date": "YYYY-MM-DD",
            "StateHoliday": "0 (none), a (public), b (Easter), c (Christmas)",
            "Open": "defaults to 1; if 0 the prediction is 0",
            "weather": "Max_Temp / Rain_Sum / Snow_Sum are optional; if omitted, cached "
                       "Frankfurt data is used for 2013-01-01..2015-09-17, otherwise monthly averages",
        },
        "example": {"Store": 1, "Date": "2015-09-17", "Promo": 1, "StateHoliday": "0", "SchoolHoliday": 0},
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not LOADED:
        return jsonify({"error": "Artifacts not loaded.", "detail": LOAD_ERROR}), 503
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON."}), 400
    is_batch = isinstance(payload, list)
    records = payload if is_batch else [payload]
    if not records:
        return jsonify({"error": "Empty request."}), 400
    try:
        preds = predict_records(records)
    except (ValueError, KeyError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {e}"}), 500
    return jsonify({"predictions": preds} if is_batch else {"predicted_sales": preds[0]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)