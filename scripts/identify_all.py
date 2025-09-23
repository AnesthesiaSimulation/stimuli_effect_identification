import numpy as np
import pandas as pd
import python_anesthesia_simulator as pas
from new_hemo_class import NewHemoModel
import optuna
import multiprocessing as mp
from scipy.signal import TransferFunction, lfilter

smooth_stimuli = TransferFunction([1], [50, 1])


def one_simu(args):
    data_case, param_vec, sampling_time, model = args
    data_case = data_case.iloc[::sampling_time]
    time_max = int(data_case.opstart.iloc[0] - 5*60/sampling_time)
    data_case = data_case.query('Time <= @time_max')
    age = data_case.age.iloc[0]
    height = data_case.height.iloc[0]
    weight = data_case.weight.iloc[0]
    sex = data_case.sex.iloc[0]
    bis_base = data_case.bis_base.iloc[0]
    hr_base = data_case.hr_base.iloc[0]
    map_base = data_case.map_base.iloc[0]
    sv_base = data_case.sv_base.iloc[0]

    intubation_time = data_case.intubation_time.iloc[0]
    dist_input = np.zeros(time_max)
    dist_input[np.max([0, intubation_time - int(2*60/sampling_time)])
                      :intubation_time] = 1  # step disturbance after 5 seconds
    discret = smooth_stimuli.to_discrete(sampling_time, method='bilinear')

    dist_input = lfilter(discret.num, discret.den, dist_input)

    Patient_simu_dist = pas.Patient(
        [age, height, weight, sex],
        model_propo=model,  # \"Schnider\",
        model_remi=model,  # \"Minto\",
        model_bis='Fuentes',
        ts=sampling_time,
    )
    Patient_simu_dist.bis_pd.E0 = bis_base

    Patient_simu_dist.hemo_pd = NewHemoModel(
        age=Patient_simu_dist.age,
        ts=Patient_simu_dist.ts,
        random=False,
        hr_base=hr_base,
        sv_base=sv_base,
        map_base=map_base,
    )
    Patient_simu_dist.hemo_pd.int_hr = param_vec[0]
    Patient_simu_dist.hemo_pd.emax_propo_tpr = param_vec[1]
    Patient_simu_dist.hemo_pd.fb = param_vec[2]
    Patient_simu_dist.hemo_pd.dist_tpr_tau = 3
    Patient_simu_dist.hemo_pd.dist_tpr_k = param_vec[3]
    Patient_simu_dist.hemo_pd.dist_sv_tau = 2.7
    Patient_simu_dist.hemo_pd.dist_sv_k = param_vec[4]
    Patient_simu_dist.hemo_pd.dist_hr_tau = 25
    Patient_simu_dist.hemo_pd.dist_hr_k = param_vec[5]
    Patient_simu_dist.hemo_pd.dist_input = dist_input

    df_simu_dist = Patient_simu_dist.full_sim(
        data_case['Propofol']*20/3600,
        data_case['Remifentanil']*20/3600
    )
    loss = 0
    # for signal in ['HR', 'TPR', 'SV', 'MAP']:
    #     error = (df_simu_dist[signal] - data_case[signal]) / df_simu_dist[signal]
    #     if signal != 'TPR':
    #         error = error[~(data_case[signal] < 10)]
    #     else:
    #         error = error[~(data_case[signal] < 0.0075)]
    #     loss += np.linalg.norm(error, 2)
    for signal in ['HR', 'SAP', 'DAP']:
        abs_error = np.abs((df_simu_dist[signal] - data_case[signal]))

        if signal != 'TPR':
            abs_error = abs_error[~(data_case[signal] < 10)]
            signal_values = data_case[signal][~(data_case[signal] < 10)]
        else:
            abs_error = abs_error[~(data_case[signal] < 0.0075)]
            signal_values = data_case[signal][~(data_case[signal] < 0.0075)]

        med_abs_err = np.median(abs_error)

        # median absolute deviation of the signal itself
        med_signal = np.median(signal_values)
        med_dev_signal = np.median(np.abs(signal_values - med_signal))

        # metric: median absolute error divided by median deviation of the signal
        loss += med_abs_err / med_dev_signal

    return loss


def run_with_timeout(func, args, timeout):
    def worker(q, func, args):
        try:
            q.put(func(*args))
        except Exception as e:
            q.put(e)

    q = mp.Queue()
    p = mp.Process(target=worker, args=(q, func, args))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        raise optuna.TrialPruned()  # prune if it took too long

    result = q.get()
    if isinstance(result, Exception):
        raise result
    return result


data = pd.read_csv('./data/database.csv')
data['TPR'] = (data['MAP'] / (data['HR'] * data['SV'])).clip(0, 0.04)
caseid = data.caseid.unique()[:16]
data_cases = data  # .query('caseid in @caseid')


model = "Eleveld"
sampling_time = 1


def optuna_obj(trial):
    sl_remi_hr = -trial.suggest_float("int_hr", 0.119/2, 0.119*2, log=True)
    k_in_sv = -trial.suggest_float("emax_propo_tpr", 1e-3, 0.778*2, log=True)
    fb = trial.suggest_float("fb", -0.661*2, -0.661/2, log=False)
    dist_tpr_k = trial.suggest_float("tpr_k", 1e-6, 1e-4, log=False)
    dist_sv_k = trial.suggest_float("sv_k", 0.05, 1, log=False)
    dist_hr_k = trial.suggest_float("hr_k", 1, 50, log=False)
    param_vec = [sl_remi_hr,
                 k_in_sv,
                 fb,
                 dist_tpr_k,
                 dist_sv_k,
                 dist_hr_k
                 ]

    loss = 0  # np.sum([np.linalg.norm(vec, 1) for vec in [hr_vec, sv_vec, tpr_vec]]) * 100
    args_list = [
        (data_case, param_vec, sampling_time, model)
        for caseid, data_case in data_cases.groupby('caseid')
    ]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(one_simu, args_list)

    loss = sum(results)/3/data_cases['caseid'].nunique()
    return loss


study = optuna.create_study(direction='minimize', study_name='identif_all_v6',
                            storage='sqlite:///./data/tuning.db', load_if_exists=True)
study.enqueue_trial({
    "int_hr": 0.1,
    "emax_propo_tpr": 6e-3,
    "fb": -0.9,
    "tpr_k": 02e-06,
    "sv_k": 0.16,
    "hr_k": 9.5,
})
study.optimize(optuna_obj, n_trials=200, show_progress_bar=True, )
print(study.best_params)

best_params = pd.DataFrame(study.best_params, index=[0])
best_params.to_csv('data/best_params_hr.csv', index=False)
