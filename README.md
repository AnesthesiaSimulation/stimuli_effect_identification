# Stimuli Effect Identification

Package to identify stimuli model from clinical data. It is intended to be use as a plugin for the [Python Anesthesia Simulator](https://github.com/AnesthesiaSimulation/Python_Anesthesia_Simulator) to calibrate the model on a given dataset. In this repository the identification procedure is applied to as subset of the [VitalDB dataset](https://vitaldb.net/dataset/).

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

## Exemple files

### Create database from VitalDB

- ```example/select_cases.py``` Select the caseid to download
- ```example/create_database.py``` Download the data, perform the simulations with raw model and save the data

### Identification of the stimuli and noise
- ```example/raw_model_vs_data.ipynb``` Make simulation without stimuli input and compare raw model with clinical data at intubation and surgery time.
- ```example/identify_bis_dist.py``` Compute the median error over the intubation period and approximate the results with a polinomial function of time.
- ```example/identify_stimuli.py``` Run the identification prdecure as describe in the paper.
- ```example/identify_noise.ipynb``` Identify noise for MAP, HR, and BIS over a steday state period of clinical data.

### Compare clinical data and model output

- ```example/bis_model_comparison.ipynb``` Compare all bis model with clinical data.
- ```example/compare_stimuli.ipynb``` Compare Stimuli model versus Non-stimuli model versus Clinical data.
