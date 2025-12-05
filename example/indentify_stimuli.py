from stimuli_effect_identification import full_identification_process
import pandas as pd

data_measure = pd.read_csv('data/database.csv')


data_measure.rename(columns={'opstart': 'surgery_time'}, inplace=True)
data_measure['Propofol'] = data_measure['Propofol']*20/3600  # convert in mg/s
data_measure['Remifentanil'] = data_measure['Remifentanil']*20/3600  # convert in µg/s
data_measure.intubation_time = (data_measure.intubation_time - 2*60).clip(0, 600)  # from end to start of intubation


param_dist = {  # guess of the time constant by hand
    'dist_intub_tpr_tau': 3,
    'dist_intub_sv_tau': 2.7,
    'dist_intub_hr_tau': 40,
    'smooth_factor': 50,
    'dist_surg_tpr_tau': 5 * 60,
    'dist_surg_sv_tau': 3 * 60,
    'dist_surg_hr_tau': 8 * 60,
}

first_guess = {  # to ensure reproductibility of the results
    'int_hr': -0.097,
    'fb': -0.5,
    'emax_propo_tpr': - 0.03,
    'dist_intub_tpr_k': 8.8e-6,
    'dist_intub_sv_k': 0.176,
    'dist_intub_hr_k': 11.94,
    'dist_surg_tpr_k': 8.8e-4,
    'dist_surg_sv_k': 13.65,
    'dist_surg_hr_k': 9.92,
}


param_dist_out, param_hemo_out, df_simulated = full_identification_process(
    data_measure,
    nb_steps=200,
    first_guess=first_guess,
    param_dist=param_dist,
)

print("Disturbance Parameters Identified:")
print(param_dist_out)
print("\nHemodynamic Parameters Identified:")
print(param_hemo_out)

df_simulated.to_csv('data/simulated_vitaldb.csv', index=False)
