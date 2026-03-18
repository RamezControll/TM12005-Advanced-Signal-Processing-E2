import matplotlib.pyplot as plt

from read_telemetry_ecg import read_ecg_mat
from scipy import signal
import numpy as np

ecg, fs, t = read_ecg_mat("../Data_E2/005_Pimpel.mat", plotresult=True)

fig = plt.figure(figsize=(9, 12))
plt.plot(t, ecg)
plt.xlabel("Time (s)")
plt.ylabel("ECG (mV)")
plt.title("Raw ECG signal")
plt.xlim(0, t[-1])


% Deel 2 - Ventriculaire activiteit
def bandpass_ecg(x, fs, low=5.0, high=15.0, order=3):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low/nyq, high/nyq], btype="bandpass")
    return signal.filtfilt(b, a, x)

def derivative_filter(x, fs):
    h = np.array([1, 2, 0, -2, -1]) * (1.0/8.0) * fs
    return signal.lfilter(h, 1, x)

def moving_window_integration(x, fs, win_ms=150):
    win = int(win_ms * fs / 1000)
    win = max(win, 1)
    return signal.lfilter(np.ones(win)/win, 1, x)

def pan_tompkins_preprocess(ecg, fs):
    y_bp = bandpass_ecg(ecg, fs, low=5, high=15, order=3)
    y_der = derivative_filter(y_bp, fs)
    y_sq = y_der ** 2
    y_mwi = moving_window_integration(y_sq, fs, win_ms=150)
    return y_bp, y_der, y_sq, y_mwi

def detect_peaks_on_mwi(y_mwi, fs, thr_factor=0.5, refractory_ms=250):
    refractory = int(refractory_ms * fs / 1000)
    thr = thr_factor * np.max(y_mwi)
    peaks, _ = signal.find_peaks(y_mwi, height=thr, distance=refractory)
    return peaks, thr

y_bp, y_der, y_sq, y_mwi = pan_tompkins_preprocess(ecg, fs)
peaks, thr = detect_peaks_on_mwi(y_mwi, fs, thr_factor=0.5)

plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(t, ecg)
plt.title("Raw ECG")
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(t, y_bp)
plt.title("Bandpass Filtered (5-15 Hz)")
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(t, y_sq)
plt.title("Squared Signal")
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(t, y_mwi, label="MWI")
plt.axhline(thr, linestyle="--", color="r", label="Threshold")
plt.plot(t[peaks], y_mwi[peaks], "x", label="Detected Peaks")
plt.title("Moving Window Integration + Peak Detection")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
