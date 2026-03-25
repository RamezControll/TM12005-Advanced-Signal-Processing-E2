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


## Deel 2 - Ventriculaire activiteit %%%

def bandpass_ecg(x, fs, low=5.0, high=15.0, order=3):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="bandpass")
    return signal.filtfilt(b, a, x)  # zero-phase filtering

def derivative_filter(x):
    # Derivative-based filter coefficients
    b = np.array([1.0, -1.0])
    a = np.array([1.0, -0.999])
    return signal.lfilter(b, a, x)

def moving_window_integration(x, fs, win_ms=150):
    win = int(win_ms * fs / 1000)
    win = max(win, 1)
    return signal.lfilter(np.ones(win) / win, 1, x)

def pan_tompkins_preprocess(ecg, fs, bp_low=5, bp_high=15, mwi_ms=150):
    y_bp = bandpass_ecg(ecg, fs, low=bp_low, high=bp_high, order=3)
    y_der = derivative_filter(y_bp)
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
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].plot(t, ecg)
ax[0].set_title("Raw ECG")
ax[0].set_ylabel("ECG (mV)")
ax[0].grid(True)

ax[1].plot(t, y_mwi, label="MWI")
ax[1].axhline(thr, linestyle="--", color="r", label="Threshold")
ax[1].plot(t[peaks], y_mwi[peaks], "x", label="Detected peaks")
ax[1].set_title("MWI + peak detection")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("MWI")
ax[1].set_ylim(-1e6, 5e6)
ax[1].grid(True)
ax[1].legend()

# Show first 10 seconds
ax[0].set_xlim(0, 120)

plt.tight_layout()
plt.show()

# # -----------------------------
# # Print metricresults for the plotted range
# # -----------------------------
# plot_start, plot_end = 0, 120  # seconds
# plot_mask = (t >= plot_start) & (t <= plot_end)
# plot_peaks = peaks[(t[peaks] >= plot_start) & (t[peaks] <= plot_end)]

# if len(plot_peaks) > 1:
#     plot_mean_rr, plot_mean_hr, _ = rr_hr_from_peaks(t[plot_mask], plot_peaks)
#     print(f"Measured length of the plotted range: {plot_end - plot_start} seconds")
#     print(f"Detected peaks in plot range: {len(plot_peaks)}")
#     print(f"Mean RR in plot range: {plot_mean_rr:.3f} s")
#     print(f"Mean HR in plot range: {plot_mean_hr:.2f} bpm")
# else:
#     print("Not enough peaks detected in the plot range to calculate RR and HR.")
