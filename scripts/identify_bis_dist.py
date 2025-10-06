import pandas as pd
import numpy as np
import python_anesthesia_simulator as pas
from matplotlib import pyplot as plt

data = pd.read_csv('data/database.csv')
data['Time_aligned'] = data['Time'] - data['intubation_time']
bis = []
for caseid, data_case in data.groupby('caseid'):
    bis_model = pas.BIS_model(hill_model='Fuentes', age=data_case['age'].iloc[0])
    # e0_init = bis_model.E0
    # bis_model.E0 = data_case['BIS'].iloc[0]
    # bis_model.Emax *= bis_model.E0 / e0_init
    bis_case = bis_model.compute_bis(data_case['Ce_prop'].values,
                                     data_case['Ce_remi'].values)
    bis_case = np.clip(bis_case, 0, 100)

    bis += list(bis_case)
data = data.drop(columns=['BIS_sim'])
data.insert(0, "BIS_sim", bis)


Time = [i*60 for i in range(-7, 30)]
bis_delta = []
for t in Time:
    median_diff = data.query('Time_aligned >= @t and Time_aligned < @t + 60')['BIS'].median() - \
        data.query('Time_aligned >= @t and Time_aligned < @t + 60')['BIS_sim'].median()
    bis_delta.append(median_diff)

data.groupby('Time_aligned')['BIS'].median().plot(label='BIS measured', color='C0')
data.groupby('Time_aligned')['BIS_sim'].median().plot(label='BIS simulated', color='C1')
plt.xlim(-5*60, 30*60)
plt.show()

tnew = np.linspace(-6, 30, 100)*60

plt.plot(Time, bis_delta, label='Median(BIS - BIS_sim)')
for deg in [4]:
    p = np.polyfit(Time, bis_delta, deg)
    plt.plot(tnew, np.polyval(p, tnew), '-.', label='polinomial approximation')
    print(p)
plt.xlabel('Time after intubation (min)')
plt.ylabel('Median(BIS - BIS_sim)')
# plt.title('BIS difference between measured and simulated')

plt.legend()
plt.savefig('output/bis_diff.png', dpi=300)
plt.show()
