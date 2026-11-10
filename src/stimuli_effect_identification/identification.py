import numpy as np
import pandas as pd
import python_anesthesia_simulator as pas
import optuna
import multiprocessing as mp
from pathlib import Path

from .simulation import dataset_simulation


def loss_function(df_simu, df_measure):

    loss = 0

    for caseid, df_simu_case in df_simu.groupby('caseid'):
        df_measure_case = df_measure.query('(caseid==@caseid) & (Time <= @df_simu_case.Time.iloc[-1])')
        for signal in ['HR', 'SAP', 'DAP']:
            abs_error = np.abs((df_simu_case[signal] - df_measure_case[signal]))

            if signal != 'TPR':
                abs_error = abs_error[~(df_measure_case[signal] < 10)]
                signal_values = df_measure_case[signal][~(df_measure_case[signal] < 10)]
            else:
                abs_error = abs_error[~(df_measure_case[signal] < 0.0075)]
                signal_values = df_measure_case[signal][~(df_measure_case[signal] < 0.0075)]

            med_abs_err = np.median(abs_error)

            # median absolute deviation of the signal itself
            med_signal = np.median(signal_values)
            med_dev_signal = np.median(np.abs(signal_values - med_signal))

            # metric: median absolute error divided by median deviation of the signal
            loss += med_abs_err / med_dev_signal
    loss = loss / 3 / df_simu.caseid.nunique()
    return loss


def optuna_obj_recalibration(
        trial,
        df_measure,
        sampling_time,
        model_propo,
        model_remi,
        model_bis,
):
    int_hr = trial.suggest_float("int_hr", -0.119*5, -0.119/22, log=False)
    emax_propo_tpr = trial.suggest_float("emax_propo_tpr", -0.778*5, -1e-3, log=False)
    fb = trial.suggest_float("fb", -0.661*5, -0.661/5, log=False)

    param_hemo = {
        'int_hr': int_hr,
        'fb': fb,
        'emax_propo_tpr': emax_propo_tpr,
    }

    result_df = dataset_simulation(
        df=df_measure,
        sampling_time=sampling_time,
        model_propo=model_propo,
        model_remi=model_remi,
        model_bis=model_bis,
        param_hemo=param_hemo,
        dist_profil=None,
    )
    loss = loss_function(result_df, df_measure)
    return loss

def optuna_obj_intubation(
        trial,
        df_measure,
        param_dist,
        sampling_time,
        model_propo,
        model_remi,
        model_bis,
):
    int_hr = trial.suggest_float("int_hr", -0.119*2, -0.119/22, log=False)
    emax_propo_tpr = trial.suggest_float("emax_propo_tpr", -0.778*2, -1e-3, log=False)
    fb = trial.suggest_float("fb", -0.661*2, -0.661/2, log=False)
    dist_intub_tpr_k = trial.suggest_float("dist_intub_tpr_k", 1e-6, 1e-4, log=False)
    dist_intub_sv_k = trial.suggest_float("dist_intub_sv_k", 0.05, 1, log=False)
    dist_intub_hr_k = trial.suggest_float("dist_intub_hr_k", 1, 50, log=False)

    param_hemo = {
        'int_hr': int_hr,
        'fb': fb,
        'emax_propo_tpr': emax_propo_tpr,
    }
    if param_dist is None:
        param_dist = {}
    param_dist['dist_intub_tpr_k'] = dist_intub_tpr_k
    param_dist['dist_intub_sv_k'] = dist_intub_sv_k
    param_dist['dist_intub_hr_k'] = dist_intub_hr_k

    result_df = dataset_simulation(
        df=df_measure,
        sampling_time=sampling_time,
        model_propo=model_propo,
        model_remi=model_remi,
        model_bis=model_bis,
        param_dist=param_dist,
        param_hemo=param_hemo,
        time_max_strat='before_surgery',
    )
    loss = loss_function(result_df, df_measure)
    return loss


def optuna_obj_surgery(
        trial,
        df_measure,
        param_dist,
        param_hemo,
        sampling_time,
        model_propo,
        model_remi,
        model_bis,
):
    dist_surg_tpr_k = trial.suggest_float("dist_surg_tpr_k", 1e-6, 1e-4, log=False)
    dist_surg_sv_k = trial.suggest_float("dist_surg_sv_k", 0.05, 1, log=False)
    dist_surg_hr_k = trial.suggest_float("dist_surg_hr_k", 1, 50, log=False)

    if param_dist is None:
        param_dist = {}
    param_dist['dist_surg_tpr_k'] = dist_surg_tpr_k
    param_dist['dist_surg_sv_k'] = dist_surg_sv_k,
    param_dist['dist_surg_hr_k'] = dist_surg_hr_k,

    result_df = dataset_simulation(
        df=df_measure,
        sampling_time=sampling_time,
        model_propo=model_propo,
        model_remi=model_remi,
        model_bis=model_bis,
        param_dist=param_dist,
        param_hemo=param_hemo,
    )
    loss = loss_function(result_df, df_measure)
    return loss


def identify_intubation(
        df_measure,
        param_dist: dict = None,
        sampling_time: float = 1,
        model_propo: str = 'Eleveld',
        model_remi: str = 'Eleveld',
        model_bis: str = 'Bouillon',
        study_name: str = None,
        first_guess: dict = None,
        n_trials: int = 200,
) -> dict:
    data_path = Path('./data/')
    if not data_path.exists():
        data_path.mkdir()

    study = optuna.create_study(direction='minimize', study_name=study_name,
                                storage='sqlite:///./data/optimization.db', load_if_exists=True)
    if first_guess is not None:
        study.enqueue_trial(first_guess)
    args = (
        df_measure,
        param_dist,
        sampling_time,
        model_propo,
        model_remi,
        model_bis,
    )
    study.optimize(lambda trial: optuna_obj_intubation(trial, *args), n_trials=n_trials, show_progress_bar=True, )
    print(study.best_params)

    best_params = pd.DataFrame(study.best_params, index=[0])
    best_params.to_csv('data/best_params_step_1.csv', index=False)
    return study.best_params

def identify_hemo_only(
        df_measure,
        sampling_time: float = 1,
        model_propo: str = 'Eleveld',
        model_remi: str = 'Eleveld',
        model_bis: str = 'Bouillon',
        study_name: str = None,
        first_guess: dict = None,
        n_trials: int = 200,
) -> dict:
    data_path = Path('./data/')
    if not data_path.exists():
        data_path.mkdir()

    study = optuna.create_study(direction='minimize', study_name=study_name,
                                storage='sqlite:///./data/optimization.db', load_if_exists=True)
    if first_guess is not None:
        study.enqueue_trial(first_guess)
    args = (
        df_measure,
        sampling_time,
        model_propo,
        model_remi,
        model_bis,
    )
    study.optimize(lambda trial: optuna_obj_recalibration(trial, *args), n_trials=n_trials, show_progress_bar=True, )
    print(study.best_params)

    best_params = pd.DataFrame(study.best_params, index=[0])
    best_params.to_csv('data/best_params_recab_only.csv', index=False)
    return study.best_params



def identify_surgery(
        df_measure,
        param_dist: dict = None,
        sampling_time: float = 1,
        model_propo: str = 'Eleveld',
        model_remi: str = 'Eleveld',
        model_bis: str = 'Bouillon',
        study_name: str = None,
        first_guess: dict = None,
        n_trials: int = 200,
        result_induction_path: str = 'data/best_params_step_1.csv',
) -> dict:
    """ Identify the disturbance parameters related to surgery using Optuna optimization.

    Parameters
    ----------
    df_measure : _type_
        DataFrame including the measured data for all patients.
    param_dist : dict, optional
        guess of the time constant parameters, by default None
    sampling_time : float, optional
        Sampling time in s, by default 1
    model_propo : str, optional
        Model used for propofol PK simulation, by default 'Eleveld'
    model_remi : str, optional
        Model used for remifentanil PK simulation, by default 'Eleveld'
    model_bis : str, optional
        Model used for BIS PD simulation, by default 'Bouillon'
    study_name : str, optional
        Name of the optuna study to save it, by default None
    first_guess : dict, optional
        First guess of gain parameters, by default None
    n_trials : int, optional
        Number of step in the optimization process, by default 200
    result_induction_path : str, optional
        Path of the intubation identification results, by default 'data/best_params_step_1.csv'

    Returns
    -------
    dict
        _description_
    """

    param_intub = pd.read_csv(result_induction_path)

    if param_dist is None:
        param_dist = {}
    param_dist['dist_intub_tpr_k'] = param_intub.dist_intub_tpr_k.iloc[0]
    param_dist['dist_intub_sv_k'] = param_intub.dist_intub_sv_k.iloc[0]
    param_dist['dist_intub_hr_k'] = param_intub.dist_intub_hr_k.iloc[0]

    param_hemo = {
        'int_hr': param_intub.int_hr.iloc[0],
        'fb': param_intub.fb.iloc[0],
        'emax_propo_tpr': param_intub.emax_propo_tpr.iloc[0],
    }

    study = optuna.create_study(direction='minimize', study_name=study_name,
                                storage='sqlite:///./data/optimization.db', load_if_exists=True)
    if first_guess is not None:
        study.enqueue_trial(first_guess)
    args = (
        df_measure,
        param_dist,
        param_hemo,
        sampling_time,
        model_propo,
        model_remi,
        model_bis,
    )
    study.optimize(lambda trial: optuna_obj_surgery(trial, *args), n_trials=n_trials, show_progress_bar=True, )
    print(study.best_params)

    best_params = pd.DataFrame(study.best_params, index=[0])
    best_params.to_csv('data/best_params_step_2.csv', index=False)

    return study.best_params
