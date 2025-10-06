import numpy as np
import pandas as pd
import python_anesthesia_simulator as pas
from new_hemo_class import NewHemoModel
import optuna
import multiprocessing as mp


def one_simu(args):
    data_case, param_vec, sampling_time, model = args
    data_case = data_case.iloc[::sampling_time]
    time_max = int(data_case.opstart.iloc[0] - 5*60/sampling_time)
    # data_case = data_case.query('Time <= @time_max')
    age = data_case.age.iloc[0]
    height = data_case.height.iloc[0]
    weight = data_case.weight.iloc[0]
    sex = data_case.sex.iloc[0]
    bis_base = data_case.bis_base.iloc[0]
    hr_base = data_case.hr_base.iloc[0]
    map_base = data_case.map_base.iloc[0]
    sv_base = data_case.sv_base.iloc[0]

    intubation_time = data_case.intubation_time.iloc[0]
    dist_input = np.zeros(10000)
    dist_input[np.max([0, intubation_time - int(2*60/sampling_time)])
                      :intubation_time] = 1  # step disturbance after 5 seconds
    surg_input = np.zeros(10000)
    surg_input[int(data_case.opstart.iloc[0]/sampling_time):] = 1

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
    param = {'int_hr': 0.09690889513781517, 'emax_propo_tpr': 0.030045146063674148, 'fb': -0.5009912989433524,
             'tpr_k': 8.854740714663977e-06, 'sv_k': 0.17603064373166186, 'hr_k': 11.936291973917072}

    Patient_simu_dist.hemo_pd.int_hr = -param['int_hr']
    Patient_simu_dist.hemo_pd.emax_propo_tpr = -param['emax_propo_tpr']
    Patient_simu_dist.hemo_pd.fb = param['fb']
    Patient_simu_dist.hemo_pd.dist_intub_tpr_tau = 3
    Patient_simu_dist.hemo_pd.dist_intub_tpr_k = param['tpr_k']
    Patient_simu_dist.hemo_pd.dist_intub_sv_tau = 2.7
    Patient_simu_dist.hemo_pd.dist_intub_sv_k = param['sv_k']
    Patient_simu_dist.hemo_pd.dist_intub_hr_tau = 25
    Patient_simu_dist.hemo_pd.dist_intub_hr_k = param['hr_k']
    Patient_simu_dist.hemo_pd.intub_input = dist_input

    Patient_simu_dist.hemo_pd.dist_surg_tpr_tau = 5*60
    Patient_simu_dist.hemo_pd.dist_surg_tpr_k = param_vec[0]
    Patient_simu_dist.hemo_pd.dist_surg_sv_tau = 3*60
    Patient_simu_dist.hemo_pd.dist_surg_sv_k = param_vec[1]
    Patient_simu_dist.hemo_pd.dist_surg_hr_tau = 8*60
    Patient_simu_dist.hemo_pd.dist_surg_hr_k = param_vec[2]
    Patient_simu_dist.hemo_pd.surg_input = surg_input

    Patient_simu_dist.hemo_pd.disturbance_dynamic()

    df_simu_dist = Patient_simu_dist.full_sim(
        data_case['Propofol']*20/3600,
        data_case['Remifentanil']*20/3600
    )
    loss = 0
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
    dist_tpr_k = trial.suggest_float("tpr_k", 1e-6, 1e-3, log=True)
    dist_sv_k = trial.suggest_float("sv_k", 1, 50, log=False)
    dist_hr_k = trial.suggest_float("hr_k", 1, 50, log=False)
    param_vec = [
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


study = optuna.create_study(direction='minimize', study_name='identif_surg_v3',
                            storage='sqlite:///./data/tuning.db', load_if_exists=True)
study.enqueue_trial({
    "tpr_k": 1e-3,
    "sv_k": 17,
    "hr_k": 12,
})
study.optimize(optuna_obj, n_trials=200, show_progress_bar=True, )

best_params = pd.DataFrame(study.best_params, index=[0])
best_params.to_csv('data/best_params_surg.csv', index=False)
