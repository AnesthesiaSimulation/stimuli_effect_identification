import pandas as pd
import numpy as np
import python_anesthesia_simulator as pas
from new_hemo_class import NewHemoModel
import multiprocessing as mp
import tqdm


def one_simu(args):
    data_case, dist_vec, sampling_time, model, caseid = args
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
    dist_input[np.max([0, intubation_time - int(2*60/sampling_time)]):intubation_time] = 1  # step disturbance after 5 seconds

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
    for signal in ['HR', 'MAP', 'SV', 'TPR']:  # , 'TPR'
        error = (df_simu_dist[signal] - data_case[signal]) / df_simu_dist[signal]
        if signal != 'TPR':
            error = error[~(data_case[signal] < 10)]
        else:
            error = error[~(data_case[signal] < 0.0075)]
        loss += np.linalg.norm(error, 2)
    df_simu_dist['caseid'] = caseid
    return [loss, df_simu_dist]


data = pd.read_csv('./data/database.csv')
data['TPR'] = (data['MAP'] / (data['HR'] * data['SV'])).clip(0, 0.04)
# caseid = data.caseid.unique()[0]
data_cases = data  # .query('caseid == @caseid')


model = "Eleveld"
sampling_time = 1

best_param = pd.read_csv('data/best_params_dist_v2.csv')
dist_tpr_tau = best_param["tpr_tau"].iloc[0]
dist_tpr_k = best_param["tpr_k"].iloc[0]
dist_sv_tau = best_param["sv_tau"].iloc[0]
dist_sv_k = best_param["sv_k"].iloc[0]
dist_hr_tau = best_param["hr_tau"].iloc[0]
dist_hr_k = best_param["hr_k"].iloc[0]
param = {
    "tpr_tau": np.float64(0.7834387152523279),
    "tpr_k": np.float64(3.756912845347241e-05),
    "sv_tau": np.float64(2.7699152215508143),
    "sv_k": np.float64(0.16028047674063062),
    "hr_tau": np.float64(24.568943848896044),
    "hr_k": np.float64(11.111759255124891)}
# dist_tpr_tau = param["tpr_tau"]
# dist_tpr_k = param["tpr_k"]
# dist_sv_tau = param["sv_tau"]
# dist_sv_k = param["sv_k"]
# dist_hr_tau = param["hr_tau"]
# dist_hr_k = param["hr_k"]

dist_vec = [dist_tpr_tau,
            dist_tpr_k,
            dist_sv_tau,
            dist_sv_k,
            dist_hr_tau,
            dist_hr_k]

print(dist_vec)

loss = 0  # np.sum([np.linalg.norm(vec, 1) for vec in [hr_vec, sv_vec, tpr_vec]]) * 100
args_list = [
    (data_case, dist_vec, sampling_time, model, caseid)
    for caseid, data_case in data_cases.groupby('caseid')
]

with mp.Pool(processes=mp.cpu_count()) as pool:
    results = list(tqdm.tqdm(pool.imap(one_simu, args_list), total=len(args_list)))

loss_list = [result[0] for result in results]
data_out = [result[1] for result in results]
loss = sum(loss_list)
print(f"Loss: {loss}")

data_out = pd.concat(data_out)
data_out.to_csv('./data/simulation_disturbance_hand.csv', index=False)
