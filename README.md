# Cryptocurrency Closing Price Predictor

A production-grade Streamlit experience that lets analysts and traders predict a valid closing price for a cryptocurrency session using a pre-trained Random Forest KFold ensemble. Every run mirrors the original training pipeline so the user can trust feature parity, while the UI highlights trend direction, charts, and confidence through averaged fold predictions.

![](photo/Screenshot%202026-02-11%20142804.png)

## Why this matters
- **Feature-accurate ensemble**: Loads the serialized `crypto_kfold_pipeline.pkl`, reconstructs engineered signals from raw open/high/low/volume/market cap data, and feeds them to every fold model for averaging.
- **Responsive, professional UI**: Streamlit columns, form validation, and a dynamic matplotlib chart keep the experience polished on both desktop and Streamlit Cloud.
- **Actionable trend insight**: Along with the predicted closing price, users get an immediate “Market Trend” direction and can visually compare the forecast against the open price.

## Quickstart
1. Clone this repository and activate the virtual environment (`.venv` is already configured for Python 3.13.2).
2. Install runtime dependencies:

```bash
C:/Users/USER/Desktop/crypto_ml/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
C:/Users/USER/Desktop/crypto_ml/.venv/Scripts/streamlit run app.py
```

4. Enter today’s market snapshot (open, high, low, volume, market cap, global market cap) and hit **Predict** to see the ensemble output plus the price comparison chart.

## Data & Model
- The training and test datasets powering this project were sourced from **Zindi Africa**, giving the application access to real-world market snapshots aligned with the original competition data.
- The `model` folder contains `crypto_kfold_pipeline.pkl`, which wraps the trained Random Forest fold list and the exact feature order used during training.

## Architecture at a glance
1. **Input sanitization**: Converts every field to finite floats, guards against division-by-zero, and fills missing engineered features with zero when necessary.
2. **Feature engineering**: Computes logs, ratios, spreads, statistics, and volatility interactions so the DataFrame exactly matches the serialized pipeline’s feature list.
3. **Model inference**: Each fold in the ensemble predicts, and the output is the mean value. UI shows a trend direction and renders a plot.

This consistent flow keeps the Streamlit deployment in lockstep with the model that a data science team already validated.

## Screenshots
![Input form and results](photo/Screenshot%202026-02-11%20143146.png)

## Maintenance notes
- If you retrain the ensemble, regenerate `crypto_kfold_pipeline.pkl` with the same `models` + `features` structure so the app can still load it.
- Deploy to Streamlit Cloud by pushing this repository along with the `model` assets; the app expects the `model/crypto_kfold_pipeline.pkl` file at runtime.
- Use the provided `.venv` path or update the commands with your own Python interpreter.

## Credits & contact
- **Data**: Zindi Africa (train + test datasets)
- **Modeling**: Random Forest KFold ensemble
- **App**: Streamlit with matplotlib for visualization

For questions or collaboration ideas, feel free to reach out via the repository’s issue tracker or preferred contact channel.
