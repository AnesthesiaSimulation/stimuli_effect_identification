"""Create a simple dataset from vitalDB."""

# %% Import
import vitaldb as vdb
import sys
import warnings
import pandas as pd
import numpy as np
import python_anesthesia_simulator as pas
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
# local
try:
    sys.path.append('../utilities')
    from vitaldb_local import load_cases

except ImportError:
    print('Could not import vitaldb_local, import online version')

    def load_cases(track_names: list, caseids: list):
        """Import a list of cases from vitaldb in a dataframe format."""
        dataframe_final = pd.DataFrame()
        for caseid in caseids:
            cases = vdb.VitalFile(caseid, track_names)
            dataframe_temp = cases.to_pandas(track_names, 1)
            dataframe_temp.insert(0, 'caseid', caseid)
            dataframe_final = pd.concat([dataframe_final, dataframe_temp], ignore_index=True)
        return dataframe_final


warnings.filterwarnings("ignore", category=FutureWarning)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# %% Load data
perso_data = pd.read_csv("https://api.vitaldb.net/cases", decimal='.')  # personal data


# id_list


with open('./data/caseid_list.txt', 'r') as f:
    caselist = f.read().splitlines()

caselist = [int(i) for i in caselist]
# import the cases
cases = load_cases(['BIS/BIS', 'BIS/SQI', 'Orchestra/PPF20_RATE', 'Orchestra/RFTN20_RATE',
                    'Solar8000/ART_MBP', 'Solar8000/ART_SBP', 'Solar8000/ART_DBP',
                    'Solar8000/NIBP_MBP', 'Solar8000/NIBP_SBP', 'Solar8000/NIBP_DBP',
                    'Solar8000/PLETH_HR', 'Orchestra/RFTN20_VOL', 'Orchestra/PPF20_VOL',
                    'Solar8000/RR_CO2',
                    ], caseids=caselist)  # load the case from vitalDB

cases.rename(columns={'BIS/BIS': 'BIS',
                      'Orchestra/PPF20_RATE': 'Propofol',
                      'Orchestra/RFTN20_RATE': "Remifentanil",
                      'Solar8000/ART_MBP': "MAP",
                      'Solar8000/ART_SBP': "SAP",
                      'Solar8000/ART_DBP': "DAP",
                      'BIS/SQI': "SQI",
                      'Solar8000/PLETH_HR': "HR",
                      'Orchestra/RFTN20_VOL': 'Vol_Rem',
                      'Orchestra/PPF20_VOL': 'Vol_Prop',
                      'Solar8000/NIBP_MBP': 'NI_MAP',
                      'Solar8000/NIBP_SBP': 'NI_SAP',
                      'Solar8000/NIBP_DBP': 'NI_DAP',
                      'Solar8000/ETCO2': 'ETCO2',
                      'Solar8000/RR_CO2': 'RR',
                      'Solar8000/PLETH_SPO2': 'SPO2'
                      }, inplace=True)
# define bound for the values
cols = ['BIS', 'MAP', 'SAP', 'DAP', 'HR', 'Propofol',
        'Remifentanil', "SQI", 'age', 'gender', 'height', 'weight', 'bmi']

min_val = {'BIS': 10,
           'MAP': 40,
           'SAP': 50,
           'DAP': 30,
           'HR': 30,
           'ETCO2': 5,
           'SPO2': 10,
           'RR': 5,
           'Propofol': -1,
           'Remifentanil': -1,
           "SQI": 0
           }
max_val = {'BIS': 100,
           'MAP': 160,
           'SAP': 200,
           'DAP': 150,
           'HR': 200,
           'ETCO2': 100,
           'SPO2': 105,
           'RR': 60,
           'Propofol': 1e3,
           'Remifentanil': 1e3,
           "SQI": 100}

# %%
Full_data = pd.DataFrame()

bar = tqdm(cases['caseid'].unique())

for caseid, Patient_df in cases.groupby('caseid'):
    bar.update()
    # find MAP baseline
    Patient_df = Patient_df.copy()
    Patient_df.reset_index(inplace=True)
    Patient_df.drop(columns=['index'], inplace=True)
    # Add time column

    # replace nan by previous value in drug rates
    Patient_df['Propofol'] = Patient_df['Propofol'].bfill()
    Patient_df['Remifentanil'] = Patient_df['Remifentanil'].bfill()
    # replace nan by previous measurement for non-invasive BP measure

    # find first drug injection
    istart = 0
    for i in range(len(Patient_df)):
        if Patient_df.loc[i, 'Propofol'] != 0 or Patient_df.loc[i, 'Remifentanil'] != 0:
            istart = max(0, i-60)  # start 1 min before
            break

    map_base = Patient_df['NI_MAP'].iloc[:istart+10].median()
    sv_base = (Patient_df['NI_SAP'].iloc[:istart+10] - Patient_df['NI_DAP'].iloc[:istart+10]).median() * 1.5
    hr_base = Patient_df['HR'].iloc[:istart+10].median()

    if np.isnan(map_base) or np.isnan(sv_base) or np.isnan(hr_base):
        print(f"No value before induction for case {caseid}")
        print(f"Map_base id nan: {np.isnan(map_base)}")
        continue

    # removed before starting of anesthesia
    Patient_df = Patient_df[istart:]
    Patient_df.reset_index(inplace=True)
    Patient_df.insert(1, "Time", np.arange(0, len(Patient_df['BIS'])))

    # find intubation time
    constant_values = Patient_df['RR'].ffill().rolling(window=3*60).apply(
        lambda x: np.unique(x).size == 1 and x[0] > 5, raw=True).shift(-3*60)
    intubation_time = constant_values.idxmax()

    # smooth BIS and MAP, SAP, DAP

    for signal in ['BIS', 'MAP', 'SAP', 'DAP']:
        Patient_df[signal].mask(Patient_df[signal] < min_val[signal], np.nan, inplace=True)
        Patient_df[signal].mask(Patient_df[signal] > max_val[signal], np.nan, inplace=True)

        if signal != 'BIS':
            # fill Nan using NI_BP
            Patient_df[signal] = Patient_df[signal].combine_first(Patient_df['NI_'+signal])

            # remove peaks on the mean arterial pressure
            rolling_mean = Patient_df[signal].rolling(
                window=500, center=True, min_periods=10
            ).mean()
            dict_peaks = {'MAP': 30, 'SAP': 45, 'DAP': 30}
            # Identify peaks based on the difference from the rolling mean
            Patient_df[signal].mask(
                (Patient_df[signal] - rolling_mean).abs() > dict_peaks[signal],
                inplace=True,
            )
        elif signal == 'BIS':
            Patient_df[signal].mask(Patient_df['SQI'] < 50, np.nan, inplace=True)

    Patient_df['BIS'] = Patient_df['BIS'].rolling(window=10, min_periods=1).mean()
    Patient_df['MAP'] = Patient_df['MAP'].rolling(window=40, min_periods=1).mean()
    Patient_df['SAP'] = Patient_df['SAP'].rolling(window=40, min_periods=1).mean()
    Patient_df['DAP'] = Patient_df['DAP'].rolling(window=40, min_periods=1).mean()
    Patient_df.insert(len(Patient_df.columns), 'SV', (Patient_df['SAP'] - Patient_df['DAP'])*1.5)

    Patient_df = Patient_df.ffill()
    Patient_df = Patient_df.bfill()

    for col in cols:
        if col in min_val.keys():
            Patient_df.loc[(Patient_df[col] <= min_val[col]) | (Patient_df[col] >= max_val[col]), col] = np.nan

    # count the proportion of missing values
    missing_values = Patient_df.isna().sum()/len(Patient_df)
    # print the percent of missing values for each columns
    if missing_values.mean() > 0.2:
        print(f"Case {caseid} has {missing_values.mean():2%} of missing values")
        continue

    Patient_df = Patient_df.ffill()
    Patient_df = Patient_df.bfill()

    # Add personnal information to the dataframe
    case_data = perso_data.query('caseid==@caseid')
    age = case_data['age'].iloc[0]
    Patient_df.insert(len(Patient_df.columns), "age", age)
    sex = case_data['sex'].iloc[0] == 'M'
    Patient_df.insert(len(Patient_df.columns), "sex", sex)
    weight = case_data['weight'].iloc[0]
    Patient_df.insert(len(Patient_df.columns), "weight", weight)
    height = case_data['height'].iloc[0]
    Patient_df.insert(len(Patient_df.columns), "height", height)
    bmi = case_data['bmi'].iloc[0]
    Patient_df.insert(len(Patient_df.columns), "bmi", bmi)
    opstart = case_data['opstart'].iloc[0]
    Patient_df.insert(len(Patient_df.columns), "opstart", opstart - istart)
    Patient_df.insert(len(Patient_df.columns), "intubation_time", intubation_time)
    approach = case_data['approach'].iloc[0]
    Patient_df.insert(len(Patient_df.columns), "approach", approach)

    # Use the Python Anesthesia Simulator (PAS) to simulate the pharmacokinetics of propofol and remifentanil
    model = "Eleveld"  # Eleveld model
    Patient_simu = pas.Patient(
        [age, height, weight, sex],
        model_propo=model,
        model_remi=model,
        model_bis='Fuentes',
        map_base=map_base,
        hr_base=hr_base,
        co_base=hr_base*sv_base / 1000,
    )
    # Patient_simu.hemo_pd.ltde_sv = 0
    # Patient_simu.hemo_pd.sv_base = sv_base

    # Patient_simu.hemo_pd.tpr_base = map_base / (hr_base * sv_base)
    # Patient_simu.hemo_pd.k_in_tpr = Patient_simu.hemo_pd.k_out * Patient_simu.hemo_pd.tpr_base
    # Patient_simu.hemo_pd.k_in_sv = Patient_simu.hemo_pd.k_out * sv_base
    # Patient_simu.hemo_pd.x = np.array([
    #     Patient_simu.hemo_pd.tpr_base,
    #     Patient_simu.hemo_pd.sv_base,
    #     Patient_simu.hemo_pd.hr_base,
    #     Patient_simu.hemo_pd.sv_base * Patient_simu.hemo_pd.ltde_sv,
    #     Patient_simu.hemo_pd.hr_base * Patient_simu.hemo_pd.ltde_hr,
    # ])

    Patient_df.insert(len(Patient_df.columns), "lbm", Patient_simu.lbm)
    df_simu = Patient_simu.full_sim(u_propo=Patient_df['Propofol']*20/3600,
                                    u_remi=Patient_df['Remifentanil']*20/3600)

    Patient_df["HR_sim"] = df_simu['HR']
    Patient_df["MAP_sim"] = df_simu['MAP']
    Patient_df["BIS_Eleveld"] = df_simu['BIS']
    Patient_df["SV_sim"] = df_simu['SV']
    Patient_df["Ce_prop"] = df_simu['x_propo_4']
    Patient_df["Ce_remi"] = df_simu['x_remi_4']
    Patient_df["TOL_sim"] = df_simu['TOL']

    if False:
        plt.plot(Patient_df['Time'], Patient_df['RR'].ffill(), label='BIS')
        plt.vlines(intubation_time, 5, 50, 'g')
        plt.show()

        plt.plot(Patient_df['Time'], Patient_df['BIS'], label='BIS')
        plt.plot(Patient_df['Time'], Patient_df['MAP'], label='MAP')
        plt.plot(Patient_df['Time'], df_simu['BIS'], label='BIS sim')
        plt.plot(Patient_df['Time'], df_simu['MAP'], label='MAP sim')
        plt.vlines(opstart, 50, 150, 'r')
        plt.vlines(intubation_time, 50, 150, 'g')

        plt.legend()
        plt.grid()
        plt.title(f"caseid: {caseid}")
        plt.show()

    Full_data = pd.concat([Full_data, Patient_df], ignore_index=True)


# final number of caseid
print(f"Number of caseid: {Full_data['caseid'].nunique()}")
print(f" data columns: {Full_data.columns}")
# Save Patients DataFrame
Full_data.drop(columns=['index', 'NI_MAP', 'NI_DAP', 'NI_SAP'], inplace=True)
Full_data.to_csv("./data/database.csv")
