import numpy as np
import pandas as pd
import python_anesthesia_simulator as pas
from new_hemo_class import NewHemoModel
import optuna
import multiprocessing as mp


def one_simu(args):
    data_case, dist_vec, sampling_time, model = args
    data_case = data_case.iloc[::sampling_time]
    time_max = 30*60//sampling_time
    data_case = data_case.query('Time <= @time_max')
    age = data_case.age.iloc[0]
    height = data_case.height.iloc[0]
    weight = data_case.weight.iloc[0]
    sex = data_case.sex.iloc[0]
    bis_base = data_case.bis_base.iloc[0]
    hr_base = data_case.hr_base.iloc[0]
    map_base = data_case.map_base.iloc[0]
    sv_base = data_case.sv_base.iloc[0]

    dist_input = np.zeros(time_max)
    intubation_time = data_case.intubation_time.iloc[0]
    dist_input[np.max([0, intubation_time - int(2*60/sampling_time)])
                      :intubation_time] = 1  # step disturbance after 5 seconds

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
    Patient_simu_dist.hemo_pd.dist_tpr_tau = dist_vec[0]
    Patient_simu_dist.hemo_pd.dist_tpr_k = dist_vec[1]
    Patient_simu_dist.hemo_pd.dist_sv_tau = dist_vec[2]
    Patient_simu_dist.hemo_pd.dist_sv_k = dist_vec[3]
    Patient_simu_dist.hemo_pd.dist_hr_tau = dist_vec[4]
    Patient_simu_dist.hemo_pd.dist_hr_k = dist_vec[5]
    Patient_simu_dist.hemo_pd.dist_input = dist_input

    df_simu_dist = Patient_simu_dist.full_sim(
        data_case['Propofol']*20/3600,
        data_case['Remifentanil']*20/3600
    )
    loss = 0
    for signal in ['HR', 'MAP', 'SV', 'TPR']:
        error = (df_simu_dist[signal] - data_case[signal]) / df_simu_dist[signal]
        if signal != 'TPR':
            error = error[~(data_case[signal] < 10)]
        else:
            error = error[~(data_case[signal] < 0.0075)]
        loss += np.linalg.norm(error, 2)
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
# caseid = data.caseid.unique()[0]
data_cases = data  # .query('caseid == @caseid')


model = "Eleveld"
sampling_time = 1


def optuna_obj(trial):
    dist_tpr_tau = trial.suggest_float("tpr_tau", 0.1, 10, log=True)
    dist_tpr_k = trial.suggest_float("tpr_k", 1e-5, 1e-4, log=False)
    dist_sv_tau = trial.suggest_float("sv_tau", 0.1, 10, log=True)
    dist_sv_k = trial.suggest_float("sv_k", 0.05, 1, log=False)
    dist_hr_tau = trial.suggest_float("hr_tau", 1, 60, log=True)
    dist_hr_k = trial.suggest_float("hr_k", 1, 50, log=False)
    dist_vec = [dist_tpr_tau,
                dist_tpr_k,
                dist_sv_tau,
                dist_sv_k,
                dist_hr_tau,
                dist_hr_k]

    loss = 0  # np.sum([np.linalg.norm(vec, 1) for vec in [hr_vec, sv_vec, tpr_vec]]) * 100
    args_list = [
        (data_case, dist_vec, sampling_time, model)
        for caseid, data_case in data_cases.groupby('caseid')
    ]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(one_simu, args_list)

    loss = sum(results)
    return loss


study = optuna.create_study(direction='minimize', study_name='identif_v9',
                            storage='sqlite:///./data/tuning.db', load_if_exists=True)
study.enqueue_trial({
    "tpr_tau": np.float64(0.7834387152523279),
    "tpr_k": np.float64(3.756912845347241e-05),
    "sv_tau": np.float64(2.7699152215508143),
    "sv_k": np.float64(0.16028047674063062),
    "hr_tau": np.float64(24.568943848896044),
    "hr_k": np.float64(11.111759255124891)})
study.optimize(optuna_obj, n_trials=100, show_progress_bar=True, )
print(study.best_params)

best_params = pd.DataFrame(study.best_params, index=[0])
best_params.to_csv('data/best_params_dist_v2.csv', index=False)
