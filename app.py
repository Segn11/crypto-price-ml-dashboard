from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "model/crypto_kfold_pipeline.pkl"
EPSILON = 1e-9


@st.cache_resource
def load_model_pipeline(path: str = MODEL_PATH) -> Tuple[List, List[str]]:
    pipeline = joblib.load(path)
    models = pipeline.get("models")
    feature_order = pipeline.get("features")
    if not models or not feature_order:
        raise RuntimeError("Pipeline file must expose both models and feature names.")
    return models, list(feature_order)


def sanitize_number(value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return float(result) if np.isfinite(result) else 0.0


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    denominator = sanitize_number(denominator)
    if abs(denominator) < EPSILON:
        return default
    return float(sanitize_number(numerator)) / denominator


def build_feature_dataframe(raw_inputs: Dict[str, float], feature_order: List[str]) -> pd.DataFrame:
    open_price = sanitize_number(raw_inputs.get("open", 0.0))
    high_price = sanitize_number(raw_inputs.get("high", 0.0))
    low_price = sanitize_number(raw_inputs.get("low", 0.0))
    volume = sanitize_number(raw_inputs.get("volume", 0.0))
    market_cap = sanitize_number(raw_inputs.get("market_cap", 0.0))
    market_cap_global = sanitize_number(raw_inputs.get("market_cap_global", 0.0))

    close_proxy = (open_price + high_price + low_price) / 3.0
    spread = high_price - low_price
    volatility_like = abs(spread)
    body = abs(close_proxy - open_price)
    body_ratio = safe_div(body, max(abs(open_price), EPSILON))
    body_sign = float(np.sign(close_proxy - open_price))

    log_volume = float(np.log1p(max(volume, 0.0)))
    market_cap_log = float(np.log1p(max(market_cap, 0.0)))
    market_cap_global_log = float(np.log1p(max(market_cap_global, 0.0)))

    sqrt_volume = float(np.sqrt(max(volume, 0.0)))
    sqrt_market_cap = float(np.sqrt(max(market_cap, 0.0)))
    sqrt_market_cap_global = float(np.sqrt(max(market_cap_global, 0.0)))

    cbrt_volume = float(np.cbrt(volume))
    cbrt_market_cap = float(np.cbrt(market_cap))
    cbrt_market_cap_global = float(np.cbrt(market_cap_global))

    inv_volume = safe_div(1.0, volume)
    inv_market_cap = safe_div(1.0, market_cap)
    inv_market_cap_global = safe_div(1.0, market_cap_global)

    volume_relative = safe_div(volume, market_cap_global + volume + 1.0)
    volume_per_marketcap = safe_div(volume, market_cap)
    price_relative_global = safe_div(open_price, market_cap_global)
    marketcap_per_global = safe_div(market_cap, market_cap_global)
    volatility_per_marketcap = safe_div(volatility_like, market_cap)

    num_array = np.array([open_price, high_price, low_price, volume, market_cap, market_cap_global], dtype=float)
    num_mean = float(np.mean(num_array))
    num_std = float(np.std(num_array))
    num_min = float(np.min(num_array))
    num_max = float(np.max(num_array))
    num_range = num_max - num_min

    feature_data = {
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "volume": volume,
        "market_cap": market_cap,
        "market_cap_global": market_cap_global,
        "high_low_ratio": safe_div(high_price, low_price),
        "close_open_ratio": safe_div(close_proxy, open_price),
        "high_open_ratio": safe_div(high_price, open_price),
        "low_open_ratio": safe_div(low_price, open_price),
        "wick_upper": max(0.0, high_price - max(open_price, close_proxy)),
        "wick_lower": max(0.0, min(open_price, close_proxy) - low_price),
        "body": body,
        "body_ratio": body_ratio,
        "volume_log": log_volume,
        "market_cap_log": market_cap_log,
        "market_cap_global_log": market_cap_global_log,
        "open_x_volume": open_price * volume,
        "high_x_volume": high_price * volume,
        "marketcap_x_volume": market_cap * volume,
        "volume_relative": volume_relative,
        "price_relative_global": price_relative_global,
        "volatility_like": volatility_like,
        "spread": spread,
        "volume_x_spread": volume * spread,
        "volume_x_volatility": volume * volatility_like,
        "marketcap_x_spread": market_cap * spread,
        "marketcap_x_volatility": market_cap * volatility_like,
        "log_volume": log_volume,
        "sqrt_volume": sqrt_volume,
        "cbrt_volume": cbrt_volume,
        "inv_volume": inv_volume,
        "log_market_cap": market_cap_log,
        "sqrt_market_cap": sqrt_market_cap,
        "cbrt_market_cap": cbrt_market_cap,
        "inv_market_cap": inv_market_cap,
        "log_market_cap_global": market_cap_global_log,
        "sqrt_market_cap_global": sqrt_market_cap_global,
        "cbrt_market_cap_global": cbrt_market_cap_global,
        "inv_market_cap_global": inv_market_cap_global,
        "num_mean": num_mean,
        "num_std": num_std,
        "num_min": num_min,
        "num_max": num_max,
        "num_range": num_range,
        "volume_per_marketcap": volume_per_marketcap,
        "marketcap_per_global": marketcap_per_global,
        "high_minus_open": high_price - open_price,
        "low_minus_open": low_price - open_price,
        "body_sign": body_sign,
        "volatility_per_marketcap": volatility_per_marketcap,
    }

    feature_df = pd.DataFrame([feature_data])
    feature_df = feature_df.reindex(columns=feature_order)
    feature_df = feature_df.fillna(0.0)
    return feature_df


def predict_close(models: List, feature_df: pd.DataFrame) -> Tuple[float, List[float]]:
    predictions = []
    for model in models:
        prediction = model.predict(feature_df)[0]
        predictions.append(float(prediction))
    return float(np.mean(predictions)), predictions


def main() -> None:
    st.set_page_config(
        page_title="Crypto Closing Price Predictor",
        page_icon="💹",
        layout="wide",
    )

    st.title("    Cryptocurrency Closing Price Predictor")
    st.markdown(
        "Enter the current market snapshot for a cryptocurrency to predict its closing price. "
    )

    models, feature_order = load_model_pipeline()

    with st.form("input_form"):
        st.header("Market snapshot")
        price_column, market_column = st.columns(2)
        with price_column:
            open_price = st.number_input("Open price", min_value=0.0, value=30000.0, step=1.0, format="%.2f")
            high_price = st.number_input("High price", min_value=0.0, value=30500.0, step=1.0, format="%.2f")
            low_price = st.number_input("Low price", min_value=0.0, value=29500.0, step=1.0, format="%.2f")
        with market_column:
            volume = st.number_input("Volume", min_value=0.0, value=2000000000.0, step=1000000.0, format="%.2f")
            market_cap = st.number_input("Market Cap", min_value=0.0, value=600000000000.0, step=1000000.0, format="%.2f")
            market_cap_global = st.number_input(
                "Global Market Cap", min_value=0.0, value=1500000000000.0, step=1000000.0, format="%.2f"
            )
        predict_button = st.form_submit_button("Predict")

    if predict_button:
        if low_price > high_price:
            st.error("Low price cannot be higher than High price. Adjust the inputs and try again.")
            return

        raw_inputs = {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "volume": volume,
            "market_cap": market_cap,
            "market_cap_global": market_cap_global,
        }

        try:
            feature_df = build_feature_dataframe(raw_inputs, feature_order)
            prediction, _ = predict_close(models, feature_df)
        except Exception as exc:
            st.error(f"Could not compute prediction: {exc}")
            return

        trend = "UP" if prediction > open_price else "DOWN"

        result_column, chart_column = st.columns((2, 1))
        with result_column:
            st.metric("Predicted Closing Price", f"${prediction:,.2f}")
            st.markdown(f"**Market Trend:** {trend}")

        with chart_column:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(["Open", "Predicted Close"], [open_price, prediction], marker="o", color="#FF6600")
            ax.set_ylabel("Price")
            ax.set_title("Open vs Predicted Close")
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)
            plt.close(fig)

        st.caption("Predictions are averaged across the stored fold models to smooth the estimate.")


if __name__ == "__main__":
    main()
