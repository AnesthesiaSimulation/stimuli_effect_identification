# PAS_vs_vitalDB

Compare output of the simulator against [vitalDB](https://vitaldb.net/dataset/) data.

## Usage

To run the analysis, premilinary scripts must be launch in the following order:

- ```scripts/select_cases.py``` Select the caseid to download
- ```scripts/create_database.py``` Download the data, perform the simulations and save the data

Then the notebooks ```look_disturb``` and ```bis_model_comparison``` can be launched to make the analysis.


