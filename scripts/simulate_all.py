import pandas as pd
import numpy as np
import python_anesthesia_simulator as pas
from new_hemo_class import NewHemoModel
import multiprocessing as mp
import tqdm


def one_simu(args):
    data_case, param_vec, sampling_time, model, caseid = args
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
    dist_input[np.max([0, intubation_time - int(2*60/sampling_time)]):intubation_time] = 1  # step disturbance after 5 seconds
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
    # Patient_simu_dist.hemo_pd.sl_remi_hr = param_vec[0]
    Patient_simu_dist.hemo_pd.int_hr = -param_vec[0]
    Patient_simu_dist.hemo_pd.emax_propo_tpr = -param_vec[1]
    Patient_simu_dist.hemo_pd.fb = param_vec[2]
    Patient_simu_dist.hemo_pd.dist_intub_tpr_tau = 3
    Patient_simu_dist.hemo_pd.dist_intub_tpr_k = param_vec[3]
    Patient_simu_dist.hemo_pd.dist_intub_sv_tau = 2.7
    Patient_simu_dist.hemo_pd.dist_intub_sv_k = param_vec[4]
    Patient_simu_dist.hemo_pd.dist_intub_hr_tau = 40
    Patient_simu_dist.hemo_pd.dist_intub_hr_k = param_vec[5]
    Patient_simu_dist.hemo_pd.intub_input = dist_input

    Patient_simu_dist.hemo_pd.dist_surg_tpr_tau = 5*60
    Patient_simu_dist.hemo_pd.dist_surg_tpr_k = 0.001
    Patient_simu_dist.hemo_pd.dist_surg_sv_tau = 3*60
    Patient_simu_dist.hemo_pd.dist_surg_sv_k = 17
    Patient_simu_dist.hemo_pd.dist_surg_hr_tau = 8*60
    Patient_simu_dist.hemo_pd.dist_surg_hr_k = 12
    Patient_simu_dist.hemo_pd.surg_input = surg_input

    Patient_simu_dist.hemo_pd.disturbance_dynamic()

    df_simu_dist = Patient_simu_dist.full_sim(
        data_case['Propofol']*20/3600,
        data_case['Remifentanil']*20/3600
    )
    loss = 0
    # for signal in ['HR', 'SAP', 'DAP']:
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
    df_simu_dist['caseid'] = caseid
    return [loss, df_simu_dist]


data = pd.read_csv('./data/database.csv')
data['TPR'] = (data['MAP'] / (data['HR'] * data['SV'])).clip(0, 0.04)
# caseid = data.caseid.unique()[0]
data_cases = data  # .query('caseid == @caseid')


model = "Eleveld"
sampling_time = 1

# best_param = pd.read_csv('data/best_params_hr.csv')

# dist_vec = best_param[['int_hr', 'emax_propo_tpr', 'fb', 'tpr_k', 'sv_k', 'hr_k']].values

param = {'int_hr': 0.0849954176401289, 'emax_propo_tpr': 0.023733132184657085, 'fb': -0.3672899549950461,
         'tpr_k': 1.1256297341273485e-05, 'sv_k': 0.18035095427137257, 'hr_k': 8.677453095961003}
# 9.133701583688122
# {'int_hr': 0.08306804315460889, 'emax_propo_tpr': 0.002790666280043327, 'fb': -0.9435659394707905,
#          'tpr_k': 2.0765872824840383e-05, 'sv_k': 0.2186964343369317, 'hr_k': 8.081070313428453}
#     "int_hr": 0.1,
#     "emax_propo_tpr": 6e-3,
#     "fb": -0.9,
#     "tpr_k": 02e-06,
#     "sv_k": 0.16,
#     "hr_k": 9.5,
# }
# {'int_hr': 0.08932675186010991, 'emax_propo_tpr': 0.046080277096856054, 'fb': -1.1572051153728211,
#  'tpr_k': 1.0373731555017309e-05, 'sv_k': 0.14800087578490084, 'hr_k': 9.423398268190823}

dist_vec = [param[key] for key in param.keys()]

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
    loss = sum(loss_list)/3/data_cases['caseid'].nunique()
    print(f"Loss: {loss}")

    data_out = pd.concat(data_out)
    data_out.to_csv('./data/simulation_new_param.csv', index=False)
