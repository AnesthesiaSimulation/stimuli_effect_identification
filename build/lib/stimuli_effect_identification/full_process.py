import pandas as pd

from .identification import identify_intubation, identify_surgery
from .simulation import dataset_simulation


def full_identification_process(
        df_measure: pd.DataFrame,
        param_dist: dict = None,
        model_propo: str = 'Eleveld',
        model_remi: str = 'Eleveld',
        model_bis: str = 'Bouillon',
):
    sampling_time = df_measure.Time.iloc[1] - df_measure.Time.iloc[0]
    if param_dist is not None:
        param_dist_intub = {
            key: param_dist[key] for key in param_dist.keys() if 'intub' in key
        }
        param_dist_surg = {
            key: param_dist[key] for key in param_dist.keys() if 'surg' in key
        }
    else:
        param_dist_intub = None
        param_dist_surg = None
    intub_param = identify_intubation(
        df_measure,
        param_dist_intub,
        sampling_time,
        model_propo,
        model_remi,
        model_bis
    )
    surg_param = identify_surgery(
        df_measure,
        param_dist_surg,
        sampling_time,
        model_propo,
        model_remi,
        model_bis,
    )

    param_dist_out = {key: intub_param[key] for key in intub_param.keys() if 'intub' in key}
    param_dist_out.update({key: surg_param[key] for key in surg_param.keys() if 'surg' in key})
    param_hemo_out = {}
    for key in intub_param.keys():
        if key not in param_dist_out.keys():
            param_hemo_out[key] = intub_param[key]

    df_simulated = dataset_simulation(
        df_measure,
        sampling_time,
        model_propo,
        model_remi,
        model_bis,
        param_dist_out,
        param_hemo_out,
        bar=True,
    )

    return param_dist_out, param_hemo_out, df_simulated
