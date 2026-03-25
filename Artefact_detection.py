import matplotlib.pyplot as plt
from read_telemetry_ecg import read_ecg_mat
from scipy import signal
import numpy as np

def remove_pacemaker_artifacts(ecg, threshold=-1000, max_extension=6):
    # Every sample below the threshold belongs to an artifact candidate.
    below_threshold = np.flatnonzero(ecg < threshold)
    if below_threshold.size == 0:
        return ecg.copy(), np.array([], dtype=int), np.ones(len(ecg), dtype=bool)

    # Split separate artifacts first.
    split_points = np.where(np.diff(below_threshold) > 1)[0] + 1
    groups = np.split(below_threshold, split_points)

    trough_indices = []
    mask = np.ones(len(ecg), dtype=bool)

    for group in groups:
        trough_idx = group[np.argmin(ecg[group])]
        trough_indices.append(trough_idx)

        # Grow the whole trough around the minimum using the local slope.
        left = trough_idx
        right = trough_idx

        while left > 0 and (trough_idx - left) < max_extension:
            if ecg[left] <= ecg[left - 1]:
                left -= 1
            else:
                break

        while right < len(ecg) - 1 and (right - trough_idx) < max_extension:
            if ecg[right] <= ecg[right + 1]:
                right += 1
            else:
                break

        mask[left:right + 1] = False

    clean_ecg = ecg.copy()
    valid_idx = np.flatnonzero(mask)
    missing_idx = np.flatnonzero(~mask)

    if valid_idx.size >= 2:
        clean_ecg[missing_idx] = np.interp(missing_idx, valid_idx, ecg[valid_idx])
    elif valid_idx.size == 1:
        clean_ecg[missing_idx] = ecg[valid_idx[0]]

    return clean_ecg, np.array(trough_indices, dtype=int), mask

def load_ecg_seconds(
    path="../Data_E2/005_Pimpel_3.mat",
    plotresult=True,
    plot_start=0,
    plot_end=120,
):
    ecg, fs, _ = read_ecg_mat(path, plotresult=plotresult)
    t = np.arange(len(ecg)) / fs
    plot_mask = (t >= plot_start) & (t <= plot_end)
    return ecg, fs, t, plot_mask


def bandpass_ecg(x, fs, low=5.0, high=15.0, order=3):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="bandpass")
    return signal.filtfilt(b, a, x)


def derivative_filter(x):
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


def detect_peaks_on_mwi(y_mwi, fs, prominence_factor=0.05, refractory_ms=350):
    refractory = int(refractory_ms * fs / 1000)
    robust_span = np.percentile(y_mwi, 99) - np.percentile(y_mwi, 50)
    prominence = prominence_factor * robust_span
    peaks, props = signal.find_peaks(y_mwi, prominence=prominence, distance=refractory)
    return peaks, prominence, props


def rr_hr_from_peaks(t, peaks):
    if len(peaks) < 2:
        return np.nan, np.nan, np.array([])
    rr = np.diff(t[peaks])
    mean_rr = np.mean(rr)
    mean_hr = 60.0 / mean_rr
    return mean_rr, mean_hr, rr


def main():
    plot_start, plot_end = 0, 120
    ecg, fs, t, plot_mask = load_ecg_seconds(plot_start=plot_start, plot_end=plot_end)

    artifactfree_ecg, pacemaker_indices, artifact_mask = remove_pacemaker_artifacts(
        ecg,
        threshold=-1000,
        max_extension=6,
    )

    plt.figure(figsize=(10, 4))
    plt.plot(t[plot_mask], ecg[plot_mask], label="Raw ECG", alpha=0.35)
    plt.plot(t[plot_mask], artifactfree_ecg[plot_mask], label="Artifact-Free ECG", alpha=0.9)
    plt.plot(
        t[pacemaker_indices[(t[pacemaker_indices] >= plot_start) & (t[pacemaker_indices] <= plot_end)]],
        ecg[pacemaker_indices[(t[pacemaker_indices] >= plot_start) & (t[pacemaker_indices] <= plot_end)]],
        "rx",
        label="Removed pacemaker spikes",
    )
    plt.plot(
        t[plot_mask][~artifact_mask[plot_mask]],
        artifactfree_ecg[plot_mask][~artifact_mask[plot_mask]],
        ".",
        markersize=3,
        label="Interpolated samples",
    )
    plt.title("ECG with Pacemaker Artifact Removal (First 120 seconds)")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG (mV)")
    plt.legend()
    plt.grid(True)
    plt.show()

    y_bp, y_der, y_sq, y_mwi = pan_tompkins_preprocess(artifactfree_ecg, fs)
    plot_t = t[plot_mask]
    plot_ecg = artifactfree_ecg[plot_mask]
    plot_mwi = y_mwi[plot_mask]
    prominence_factor = 0.05
    refractory_ms = 350
    peaks, prominence, _ = detect_peaks_on_mwi(
        plot_mwi,
        fs,
        prominence_factor=prominence_factor,
        refractory_ms=refractory_ms,
    )
    mean_rr, mean_hr, rr = rr_hr_from_peaks(plot_t, peaks)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(plot_t, plot_ecg, alpha=0.85, label="Artifact-Free ECG")
    ax[0].set_title("Artifact-Free ECG used for MWI analysis")
    ax[0].set_ylabel("ECG (mV)")
    ax[0].grid(True)
    ax[0].legend()
    ecg_margin = 0.1 * np.ptp(plot_ecg) if np.ptp(plot_ecg) > 0 else 1.0
    ax[0].set_ylim(plot_ecg.min() - ecg_margin, plot_ecg.max() + ecg_margin)

    ax[1].plot(plot_t, plot_mwi, label="MWI")
    ax[1].plot(plot_t[peaks], plot_mwi[peaks], "x", label="Detected peaks")
    ax[1].set_title("MWI + prominence-based peak detection")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("MWI")
    ax[1].grid(True)
    ax[1].legend()
    mwi_margin = 0.1 * np.ptp(plot_mwi) if np.ptp(plot_mwi) > 0 else 1.0
    ax[1].set_ylim(
        max(0.0, plot_mwi.min() - mwi_margin),
        plot_mwi.max() + mwi_margin,
    )

    ax[0].set_xlim(plot_start, plot_end)
    plt.tight_layout()
    plt.show()

    if len(rr) > 0:
        print(f"Prominence factor: {prominence_factor:.3f}")
        print(f"Absolute prominence: {prominence:.1f}")
        print(f"Refractory period: {refractory_ms} ms")
        print(f"Mean RR interval: {mean_rr:.3f} s")
        print(f"Mean heart rate: {mean_hr:.2f} bpm")
    else:
        print("Not enough peaks detected to calculate RR and HR.")


if __name__ == "__main__":
    main()
