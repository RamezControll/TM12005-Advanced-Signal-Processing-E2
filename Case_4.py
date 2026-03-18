import matplotlib.pyplot as plt
from read_telemetry_ecg import read_ecg_mat
from scipy import signal
import numpy as np

# -----------------------------
# Load ECG from .mat file
# -----------------------------
ecg, fs, t = read_ecg_mat("../Data_E2/005_Pimpel.mat", plotresult=True)

# Make sure t is in seconds as float
t = np.arange(len(ecg)) / fs

# Plot raw ECG
plt.figure(figsize=(9, 4))
plt.plot(t, ecg)
plt.xlabel("Time (s)")
plt.ylabel("ECG (mV)")
plt.title("Raw ECG signal")
plt.xlim(0, t[-1])
plt.grid(True)
plt.tight_layout()
plt.show()

# Deel 2 - Ventriculaire activiteit %%%

def bandpass_ecg(x, fs, low=5.0, high=15.0, order=3):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="bandpass")
    return signal.filtfilt(b, a, x)  # zero-phase filtering

def derivative_filter(x, fs):
    # 5-point derivative filter
    h = np.array([1, 2, 0, -2, -1]) * (1.0 / 8.0) * fs
    return signal.lfilter(h, 1, x)

def moving_window_integration(x, fs, win_ms=150):
    win = int(win_ms * fs / 1000)
    win = max(win, 1)
    return signal.lfilter(np.ones(win) / win, 1, x)

def pan_tompkins_preprocess(ecg, fs, bp_low=5, bp_high=15, mwi_ms=150):
    y_bp = bandpass_ecg(ecg, fs, low=bp_low, high=bp_high, order=3)
    y_der = derivative_filter(y_bp, fs)
    y_sq = y_der ** 2
    y_mwi = moving_window_integration(y_sq, fs, win_ms=mwi_ms)
    return y_bp, y_der, y_sq, y_mwi

def detect_peaks_on_mwi(y_mwi, fs, thr_factor=0.5, refractory_ms=250):
    refractory = int(refractory_ms * fs / 1000)
    thr = thr_factor * np.max(y_mwi)
    peaks, props = signal.find_peaks(y_mwi, height=thr, distance=refractory)
    return peaks, thr, props

def rr_hr_from_peaks(t, peaks):
    if len(peaks) < 2:
        return np.nan, np.nan, np.array([])
    rr = np.diff(t[peaks])      # seconds
    mean_rr = np.mean(rr)
    mean_hr = 60.0 / mean_rr
    return mean_rr, mean_hr, rr

# -----------------------------
# Run Pan-Tompkins pipeline
# -----------------------------
y_bp, y_der, y_sq, y_mwi = pan_tompkins_preprocess(ecg, fs)

# Peak detection
peaks, thr, _ = detect_peaks_on_mwi(y_mwi, fs, thr_factor=0.04)

# RR and HR
mean_rr, mean_hr, rr = rr_hr_from_peaks(t, peaks)

# -----------------------------
# Plot results
# -----------------------------
fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True)

ax[0].plot(t, ecg)
ax[0].set_title("Raw ECG")
ax[0].set_ylabel("ECG (mV)")
ax[0].grid(True)

ax[1].plot(t, y_bp)
ax[1].set_title("Bandpass (5-15 Hz)")
ax[1].grid(True)

ax[2].plot(t, y_der)
ax[2].set_title("Derivative")
ax[2].grid(True)

ax[3].plot(t, y_sq)
ax[3].set_title("Squared")
ax[3].grid(True)

ax[4].plot(t, y_mwi, label="MWI")
ax[4].axhline(thr, linestyle="--", color="r", label="Threshold")
ax[4].plot(t[peaks], y_mwi[peaks], "x", label="Detected peaks")
ax[4].set_title("MWI + peak detection")
ax[4].set_xlabel("Time (s)")
ax[4].set_ylabel("MWI")
ax[4].grid(True)
ax[4].legend()

# Show first 10 seconds
ax[0].set_xlim(0, 100)

plt.tight_layout()
plt.show()

# -----------------------------
# Print results
# -----------------------------
print(f"Detected peaks: {len(peaks)}")
print(f"Mean RR: {mean_rr:.3f} s")
print(f"Mean HR: {mean_hr:.2f} bpm")