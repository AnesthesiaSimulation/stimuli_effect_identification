from stimuli_effect_identification import dataset_simulation, identify_hemo_only
import pandas as pd

data_measure = pd.read_csv('data/database.csv')

data_measure.rename(columns={'opstart': 'surgery_time'}, inplace=True)
data_measure['Propofol'] = data_measure['Propofol']*20/3600  # convert in mg/s
data_measure['Remifentanil'] = data_measure['Remifentanil']*20/3600  # convert in µg/s
data_measure.intubation_time = (data_measure.intubation_time - 2*60).clip(0, 600)  # from end to start of intubation


data_measure_train = data_measure.query('split == "train"').reset_index()
data_measure_test = data_measure.query('split == "test"').reset_index()




first_guess = {  # to ensure reproductibility of the results
    'int_hr': -0.097,
    'fb': -0.5,
    'emax_propo_tpr': - 0.03,
}


param_hemo_out = identify_hemo_only(
    data_measure_train,
    first_guess = first_guess,
 )

df_simulated_train = dataset_simulation(
    data_measure_train,
    sampling_time = 1,
    model_propo = 'Eleveld' ,
    model_remi = 'Eleveld',
    model_bis = 'Bouillon',
    param_hemo = param_hemo_out,
    dist_profil=None,
    bar=True,
)

print("\nHemodynamic Parameters Identified:")
print(param_hemo_out)

df_simulated_train.to_csv('data/simulated_vitaldb_train_recalibration.csv', index=False)



df_simulated_test = dataset_simulation(
    data_measure_test,
    sampling_time = 1,
    model_propo = 'Eleveld' ,
    model_remi = 'Eleveld',
    model_bis = 'Bouillon',
    param_hemo = param_hemo_out,
    dist_profil=None,
    bar=True,
)

df_simulated_test.to_csv('data/simulated_vitaldb_test_recalibration.csv', index=False)