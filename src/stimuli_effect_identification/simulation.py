import pandas as pd
import python_anesthesia_simulator as pas
from tqdm.contrib.concurrent import thread_map
import multiprocessing as mp


def simulation(
        data_case: pd.DataFrame,
        caseid: int,
        sampling_time: float = 1,
        model_propo: str = 'Eleveld',
        model_remi: str = 'Eleveld',
        model_bis: str = 'Bouillon',
        param_dist: dict = None,
        param_hemo: dict = None,
        dist_profil: str = 'VitalDB',
        time_max: int = None,
) -> pd.DataFrame:
    """Perform simulation with given model and sampling time for a single patient.
    Parameters
    ----------
    data_case : pd.DataFrame
        Dataframe including patient demographic characteristics and inputs drug rates for a single patient. Especially the dataframe should have the column: age (in yr), height (in cm), weight(in kg), sex(0=female, 1=male ), hr_base (beat/min), sv_base (ml), map_base (mmHg), intubation_time (in s), surgery_time (in s), propofol (in mg/s), remifentanil (in µg/s)
    caseid : int
        Unique identifier of the patient.
    sampling_time : float
    model_propo: str
        Name of the PK model for propofol used in the simulation. The default is Eleveld.
    model_remi: str
        Name of the PK model for remifentanil used in the simulation. The default is Eleveld.
    model_bis: str
        Name of the PD model for BIS used in the simulation. The default is Bouillon.
    param_dist: dict
        Dictionary including the disturbance parameters to set in the simulation. The default is None.
    param_hemo: dict
        Dictionary including the hemodynamic parameters to set in the simulation. The default is None.
    time_max: int
        Maximum simulation time for the patient. If None, the maximum time is the maximum time available in the dataframe for the patient. The default is None.
    Returns
    -------
    pd.DataFrame
        Output dataframe including all the simulation results for the patient.
    """

    data_case = data_case.iloc[::sampling_time]
    if time_max is None:
        time_max = data_case.Time.iloc[-1]
    data_case = data_case.query('Time <= @time_max')
    age = data_case.age.iloc[0]
    height = data_case.height.iloc[0]
    weight = data_case.weight.iloc[0]
    sex = data_case.sex.iloc[0]
    hr_base = data_case.hr_base.iloc[0]
    map_base = data_case.map_base.iloc[0]
    sv_base = data_case.sv_base.iloc[0]

    intubation_time = data_case.intubation_time.iloc[0]
    surgery_time = data_case.surgery_time.iloc[0]

    patient = pas.Patient(
        [age, height, weight, sex],
        model_propo=model_propo,
        model_remi=model_remi,
        model_bis=model_bis,
        ts=sampling_time,
        hr_base=hr_base,
        map_base=map_base,
        co_base=hr_base*sv_base/1000,
    )
    e0_init = patient.bis_pd.E0
    patient.bis_pd.E0 = data_case['BIS'].iloc[0]
    patient.bis_pd.Emax *= patient.bis_pd.E0 / e0_init

    simu = pas.Simulator(
        patient,
        disturbance_profil=dist_profil,
        arg_disturbance={
            'start_intub_time': intubation_time,
            'start_surgery_time': surgery_time,
        },
    )
    if param_hemo is not None:
        for key, value in param_hemo.items():
            if hasattr(simu.patient.hemo_pd, key):
                setattr(simu.patient.hemo_pd, key, value)

    if param_dist is not None:
        for key, value in param_dist.items():
            if hasattr(simu.disturbances, key):
                setattr(simu.disturbances, key, value)

    if dist_profil == 'VitalDB':
        simu.disturbances.disturbance_dynamic()

    df_simu_dist = simu.full_sim(
        data_case['Propofol'],
        data_case['Remifentanil']
    )
    df_simu_dist['caseid'] = caseid

    return df_simu_dist


def dataset_simulation(
        df: pd.DataFrame,
        sampling_time: float = 1,
        model_propo: str = 'Eleveld',
        model_remi: str = 'Eleveld',
        model_bis: str = 'Bouillon',
        param_dist: dict = None,
        param_hemo: dict = None,
        dist_profil: str = 'VitalDB',
        time_max_strat: str = None,
        bar: bool = False
) -> pd.DataFrame:
    """Perform simulation with given model and sampling time for each patient of the database.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe including patient demographic characteristics and inputs drug rates. Especially the dataframe should have the column: age (in yr), height (in cm), weight(in kg), sex(0=female, 1=male), hr_base (beat/min), sv_base (ml), map_base (mmHg), intubation_time (in s), surgery_time (in s), propofol (in mg/s), remifentanil (in µg/s)
    sampling_time : float
    model_propo: str
        Name of the PK model for propofol used in the simulation. The default is Eleveld.
    model_remi: str
        Name of the PK model for remifentanil used in the simulation. The default is Eleveld.
    model_bis: str
        Name of the PD model for BIS used in the simulation. The default is Bouillon.
    param_dist: dict
        Dictionary including the disturbance parameters to set in the simulation. The default is None.
    param_hemo: dict
        Dictionary including the hemodynamic parameters to set in the simulation. The default is None.
    time_max_strat: str
        Strategy to set the maximum simulation time for each patient. If None, the maximum time is the maximum time available in the dataframe for each patient. If 'before_surgery', the maximum time is set to 5 minutes before surgery start time for each patient. The default is None.
    bar: bool
        If True, display a progress bar during the simulation. The default is False.

    Returns
    -------
    pd.DataFrame
        Output dataframe including all the simulation results for each patients.
    """
    print(dist_profil)
    if time_max_strat is None:
        args_list = [
            (
                data_case,
                caseid,
                sampling_time,
                model_propo,
                model_remi,
                model_bis,
                param_dist,
                param_hemo,
                dist_profil,
                None
            )
            for caseid, data_case in df.groupby('caseid')
        ]
    elif time_max_strat == 'before_surgery':
        args_list = [
            (
                data_case,
                caseid,
                sampling_time,
                model_propo,
                model_remi,
                model_bis,
                param_dist,
                param_hemo,
                dist_profil,
                data_case.surgery_time.iloc[0] - 5*60
            )
            for caseid, data_case in df.groupby('caseid')
        ]
    nb_cpu = mp.cpu_count() - 2
    if bar:
        results = thread_map(lambda x: simulation(*x), args_list, max_workers=nb_cpu)
    else:
        with mp.Pool(processes=nb_cpu) as pool:
            results = pool.starmap(simulation, args_list)
    return pd.concat(results)
