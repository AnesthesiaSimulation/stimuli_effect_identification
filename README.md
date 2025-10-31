# PAS_vs_vitalDB

Compare output of the simulator against [vitalDB](https://vitaldb.net/dataset/) data.

## Usage

### Install Environement

with uv:

```bash
uv sync
```

With pip:

```bash
python -m venv your_environment
source your_environment/bin/activate
pip install .
```

### dataset creation

- ```scripts/select_cases.py``` Select the caseid to download
- ```scripts/create_database.py``` Download the data, perform the simulations with raw model and save the data

### Identification of the stimuli and noise

- ```identify_bis_dist.py``` Compute the median error over the intubation period and approximate the results with a polinomial function of time.
- ```identify_intub.py``` Optimize main hemodynamic parameters + intubation stimuli over the intubation period.
- ```identify_surg.py``` Optimize surgery stimuli to fit surgery data.
- ```identify_noise.ipynb``` Identify noise for MAP, HR, and BIS over a steday state period of clinical data.

### Compare clinical data and model output

- ```bis_model_comparison.ipynb``` Compare all bis model with clinical data.
- ```raw_model_vs_data.ipynb``` Compare raw model with clinical data at intubation and surgery time.
- ```simulate_all.py``` perform simulation withstimuli model.
- ```compare_stimuli.ipynb``` Compare Stimuli model versus Non-stimuli model versus Clinical data.


