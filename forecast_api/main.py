# ─────────────────────────────────────────────────────────────
# main.py  –  Beverage Sales Forecasting API
# ─────────────────────────────────────────────────────────────

# ── Section 1: Imports ───────────────────────────────────────
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
MODELS_DIR       = BASE_DIR / "models"
REGISTRY_PATH    = BASE_DIR / "model_registry.json"
FORECASTS_PATH   = BASE_DIR / "forecasts.json"

FORECAST_HORIZON = 8
LAG_WEEKS        = [1, 7, 30]
ROLL_WINDOWS     = [4, 8]

# ── App initialisation ───────────────────────────────────────
app = FastAPI(
    title="Beverage Sales Forecasting API",
    description=(
        "Weekly beverage sales forecasts for 43 US states. "
        "Powered by per-state best models: XGBoost, ARIMA, Prophet, or LSTM."
    ),
    version="1.0.0",
)

# ── CORS ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ────────────────────────────────────────────
FRONTEND_DIR = BASE_DIR.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
FEATURE_COLS = [
    "week_of_year", "month", "quarter", "year",
    "lag_1", "lag_7", "lag_30",
    "rolling_mean_4", "rolling_mean_8", "rolling_std_4",
]


# ── File-name helper ─────────────────────────────────────────
def state_prefix(state: str) -> str:
    """
    Convert a state name to the underscore-separated Title_Case prefix
    used in model filenames, e.g. "New Mexico" → "New_Mexico".
    """
    return "_".join(word.capitalize() for word in state.split())


# ─────────────────────────────────────────────────────────────
# Section 2: Startup – load registry, forecasts & models
# ─────────────────────────────────────────────────────────────

# In-memory stores
registry:  Dict = {}
forecasts: Dict = {}
models:    Dict = {}   # { state: { "model": <obj>, "scaler": <obj>|None, "type": str } }


def load_xgboost(state: str):
    import xgboost as xgb
    prefix      = state_prefix(state)
    model_path  = MODELS_DIR / f"{prefix}_XGBoost.json"
    scaler_path = MODELS_DIR / f"{prefix}_XGBoost_scaler.pkl"
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return {"model": model, "scaler": scaler, "type": "xgboost"}


def load_arima(state: str):
    prefix     = state_prefix(state)
    model_path = MODELS_DIR / f"{prefix}_ARIMA.pkl"
    model = joblib.load(model_path)
    return {"model": model, "scaler": None, "type": "arima"}


def load_prophet(state: str):
    prefix     = state_prefix(state)
    model_path = MODELS_DIR / f"{prefix}_Prophet.pkl"
    model = joblib.load(model_path)
    return {"model": model, "scaler": None, "type": "prophet"}


def load_lstm(state: str):
    import tensorflow as tf
    prefix      = state_prefix(state)
    model_path  = MODELS_DIR / f"{prefix}_LSTM.keras"
    scaler_path = MODELS_DIR / f"{prefix}_LSTM_scaler.pkl"
    model  = tf.keras.models.load_model(str(model_path))
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return {"model": model, "scaler": scaler, "type": "lstm"}


LOADERS = {
    "xgboost": load_xgboost,
    "arima":   load_arima,
    "prophet": load_prophet,
    "lstm":    load_lstm,
}


@app.on_event("startup")
async def startup_event():
    global registry, forecasts, models

    # Load registry
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    # Load pre-computed forecasts
    with open(FORECASTS_PATH) as f:
        forecasts = json.load(f)

    # Load each state's best model
    for state, info in registry.items():
        best = info["best_model"].lower()
        try:
            loader        = LOADERS[best]
            models[state] = loader(state)
        except Exception as e:
            print(f"[WARN] Could not load model for {state}: {e}")

    print(f"[INFO] Loaded {len(models)}/{len(registry)} models successfully.")


# ─────────────────────────────────────────────────────────────
# Section 3: Pydantic schemas
# ─────────────────────────────────────────────────────────────

class ForecastPoint(BaseModel):
    date:              str
    forecasted_sales:  float


class ForecastResponse(BaseModel):
    state:       str
    model_used:  str
    mape:        float
    forecast:    List[ForecastPoint]


class LiveForecastRequest(BaseModel):
    state:            str
    historical_sales: List[float]   # at least 30 weekly values, most-recent last


class ModelInfo(BaseModel):
    state:       str
    best_model:  str
    mape:        float
    all_metrics: Dict[str, float]


# ─────────────────────────────────────────────────────────────
# Section 4: Feature engineering helper (XGBoost / LSTM)
# ─────────────────────────────────────────────────────────────

def build_features(series: List[float], target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Given a history of weekly sales (most-recent last) and the date we are
    predicting, construct the same feature set used during training (quickhyre.py Cell 11).
    Column names and order must match FEATURE_COLS exactly.
    """
    s = np.array(series, dtype=float)

    row = {
        "week_of_year":   int(target_date.isocalendar()[1]),
        "month":          target_date.month,
        "quarter":        target_date.quarter,
        "year":           target_date.year,
        "lag_1":          s[-1],
        "lag_7":          s[-7]  if len(s) >= 7  else s[0],
        "lag_30":         s[-30] if len(s) >= 30 else s[0],
        "rolling_mean_4": np.mean(s[-4:]) if len(s) >= 4  else np.mean(s),
        "rolling_mean_8": np.mean(s[-8:]) if len(s) >= 8  else np.mean(s),
        "rolling_std_4":  np.std(s[-4:])  if len(s) >= 4  else np.std(s),
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)


# ─────────────────────────────────────────────────────────────
# Section 5: Per-model inference helpers
# ─────────────────────────────────────────────────────────────

def predict_xgboost(state: str, history: List[float], dates: List[pd.Timestamp]) -> List[float]:
    obj    = models[state]
    model  = obj["model"]
    scaler = obj["scaler"]
    preds  = []
    series = list(history)

    for dt in dates:
        X = build_features(series, dt)
        if scaler:
            X = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLS)
        pred = float(model.predict(X)[0])
        preds.append(pred)
        series.append(pred)          # roll forward

    return preds


def predict_arima(state: str, horizon: int) -> List[float]:
    model = models[state]["model"]
    fc    = model.predict(n_periods=horizon)
    return [float(v) for v in fc]


def predict_prophet(state: str, dates: List[pd.Timestamp]) -> List[float]:
    model  = models[state]["model"]
    future = pd.DataFrame({"ds": dates})
    fc     = model.predict(future)
    return [float(v) for v in fc["yhat"].values]


def predict_lstm(state: str, history: List[float], dates: List[pd.Timestamp]) -> List[float]:
    obj    = models[state]
    model  = obj["model"]
    scaler = obj["scaler"]
    preds  = []
    series = list(history)

    # infer sequence length from model input shape
    seq_len = model.input_shape[1]

    for dt in dates:
        window = np.array(series[-seq_len:], dtype=float).reshape(1, seq_len, 1)
        if scaler:
            flat   = window.reshape(-1, 1)
            flat   = scaler.transform(flat)
            window = flat.reshape(1, seq_len, 1)
        raw = model.predict(window, verbose=0)[0][0]
        if scaler:
            raw = scaler.inverse_transform([[raw]])[0][0]
        pred = float(raw)
        preds.append(pred)
        series.append(pred)

    return preds


# ─────────────────────────────────────────────────────────────
# Section 6: Routes
# ─────────────────────────────────────────────────────────────

# ── GET /states ──────────────────────────────────────────────
@app.get("/states", response_model=List[str], summary="List all available states")
def list_states():
    """Return the list of all 43 states covered by the API."""
    return sorted(registry.keys())


# ── GET /model-info/{state} ───────────────────────────────────
@app.get("/model-info/{state}", response_model=ModelInfo, summary="Model details for a state")
def model_info(state: str):
    """Return the best model type, its MAPE, and all four model MAPEs for a state."""
    state = state.title()
    if state not in registry:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")

    info = registry[state]
    return ModelInfo(
        state       = state,
        best_model  = info["best_model"],
        mape        = info["metrics"][info["best_model"]]["MAPE"],
        all_metrics = {k: v["MAPE"] for k, v in info["metrics"].items()},
    )


# ── GET /forecast/{state} ─────────────────────────────────────
@app.get("/forecast/{state}", response_model=ForecastResponse, summary="Pre-computed 8-week forecast")
def get_forecast(state: str):
    """
    Return the pre-computed 8-week forecast for a state (starting 2023-05-14).
    Fastest endpoint — no inference at request time.
    """
    state = state.title()
    if state not in forecasts:
        raise HTTPException(status_code=404, detail=f"No forecast found for '{state}'.")

    info      = registry[state]
    best      = info["best_model"]
    mape      = info["metrics"][best]["MAPE"]
    fc_points = [
        ForecastPoint(date=row["date"], forecasted_sales=row["forecast"])
        for row in forecasts[state]
    ]
    return ForecastResponse(state=state, model_used=best, mape=mape, forecast=fc_points)


# ── POST /forecast/live ───────────────────────────────────────
@app.post("/forecast/live", response_model=ForecastResponse, summary="Live 8-week forecast from custom history")
def live_forecast(req: LiveForecastRequest):
    """
    Accept at least 30 weeks of historical sales (most-recent last) for a state
    and run the state's best model to produce a fresh 8-week forecast.
    """
    state = req.state.title()

    if state not in models:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found or model failed to load.")
    if len(req.historical_sales) < 30:
        raise HTTPException(status_code=422, detail="At least 30 weeks of historical sales are required.")

    info     = registry[state]
    best     = info["best_model"].lower()
    mape     = info["metrics"][info["best_model"]]["MAPE"]
    history  = req.historical_sales

    # Generate future dates (weekly, Monday-anchored)
    last_date    = pd.Timestamp("2023-05-07")   # last known date in dataset
    future_dates = [last_date + pd.Timedelta(weeks=i + 1) for i in range(FORECAST_HORIZON)]

    try:
        if best == "xgboost":
            preds = predict_xgboost(state, history, future_dates)
        elif best == "arima":
            preds = predict_arima(state, FORECAST_HORIZON)
        elif best == "prophet":
            preds = predict_prophet(state, future_dates)
        elif best == "lstm":
            preds = predict_lstm(state, history, future_dates)
        else:
            raise HTTPException(status_code=500, detail=f"Unknown model type '{best}'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    fc_points = [
        ForecastPoint(date=str(dt.date()), forecasted_sales=round(pred, 2))
        for dt, pred in zip(future_dates, preds)
    ]
    return ForecastResponse(state=state, model_used=info["best_model"], mape=mape, forecast=fc_points)


# ── GET /health ───────────────────────────────────────────────
@app.get("/health", summary="Health check")
def health():
    return {
        "status":        "ok",
        "states_loaded": len(models),
        "registry_size": len(registry),
    }


# ─────────────────────────────────────────────────────────────
# Section 7: Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)