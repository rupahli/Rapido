# Rapido Mobility Insights Dashboard

A separate Streamlit dashboard for the four project use cases described in the Rapido brief:

- Ride outcome prediction
- Customer cancellation risk
- Fare forecasting
- Driver delay and incomplete-ride risk

The dashboard reads the existing CSV files from `src/DataCSVFiles` and does not modify the existing notebooks.

## Run

From the Rapido project root:

```powershell
.\.rapido\Scripts\python.exe -m pip install -r src\streamlit_dashboard\requirements.txt
.\.rapido\Scripts\python.exe -m streamlit run src\streamlit_dashboard\app.py
```

The models are trained and cached when the dashboard starts. The customer-risk model uses historical customer fields as inputs and the current booking cancellation status as its training target.

## Standalone model files

Each model has a simple training script and saves a pickle plus a JSON metrics file in `model_artifacts/`:

- `train_cancellation_model.py`
- `train_outcome_model.py`
- `train_fare_model.py`
- `train_driver_delay_model.py`

Run an individual model from the dashboard directory, for example:

```powershell
cd src\streamlit_dashboard
..\..\.rapido\Scripts\python.exe train_cancellation_model.py
```

The dashboard's **Model performance** tab reads the saved metrics and displays classification accuracy, macro precision, macro recall, macro F1, or regression RMSE, MAE, and R2.
