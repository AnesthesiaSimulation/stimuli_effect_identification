import pandas as pd
import numpy as np

def compute_kpis_with_std(
    df_meas,
    df_sim,
    case_col="caseid",
    time_col="Time",
    value_col="value"
):
    # merge signals
    df = (
        df_meas[[case_col, time_col, value_col]]
        .merge(
            df_sim[[case_col, time_col, value_col]],
            on=[case_col, time_col],
            suffixes=("_meas", "_sim")
        )
    )

    def per_patient(group):
        y_true = group[f"{value_col}_meas"].values
        y_pred = group[f"{value_col}_sim"].values
        err = y_pred - y_true

        rmse = np.sqrt(np.mean(err**2))

        if np.any(y_true != 0):
            mdape = np.median(np.abs(err[y_true != 0] / y_true[y_true != 0])) * 100
            mdpe = np.median(err[y_true != 0] / y_true[y_true != 0]) * 100
        else:
            mape = np.nan

        return pd.Series({
            "MDPE": mdpe,
            "RMSE": rmse,
            "MDAPE": mdape
        })

    # per-patient KPIs
    patient_kpis = df.groupby(case_col).apply(per_patient)

    # aggregate across patients
    out = pd.DataFrame({
        "MDPE_mean": [patient_kpis["MDPE"].mean()],
        "MDPE_std":  [patient_kpis["MDPE"].std()],
        "RMSE_mean": [patient_kpis["RMSE"].mean()],
        "RMSE_std":  [patient_kpis["RMSE"].std()],
        "MDAPE_mean": [patient_kpis["MDAPE"].mean()],
        "MDAPE_std":  [patient_kpis["MDAPE"].std()],
        "N_patients": [len(patient_kpis)]
    })

    return out
