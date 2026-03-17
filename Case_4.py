import matplotlib.pyplot as plt

from read_telemetry_ecg import read_ecg_mat

ecg, fs, t = read_ecg_mat("../Data_E2/005_Pimpel.mat", plotresult=True)

fig = plt.figure(figsize=(9, 12))
plt.plot(t, ecg)
plt.xlabel("Time (s)")
plt.ylabel("ECG (mV)")
plt.title("Raw ECG signal")
plt.xlim(0, t[-1])


%%% Deel 2 - Ventriculaire activiteit %%%

