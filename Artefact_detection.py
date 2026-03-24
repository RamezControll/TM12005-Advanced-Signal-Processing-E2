import matplotlib.pyplot as plt
from read_telemetry_ecg import read_ecg_mat
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
    path="../Data_E2/005_Pimpel.mat",
    plotresult=True,
    plot_start=0,
    plot_end=120,
):
    ecg, fs, _ = read_ecg_mat(path, plotresult=plotresult)
    t = np.arange(len(ecg)) / fs
    plot_mask = (t >= plot_start) & (t <= plot_end)
    return ecg, fs, t, plot_mask


def main():
    ecg, fs, t, plot_mask = load_ecg_seconds()

    artifactfree_ecg, pacemaker_indices, artifact_mask = remove_pacemaker_artifacts(
        ecg[plot_mask],
        threshold=-1000,
        max_extension=6,
    )

    plt.figure(figsize=(10, 4))
    plt.plot(t[plot_mask], ecg[plot_mask], label="Raw ECG", alpha=0.35)
    plt.plot(t[plot_mask], artifactfree_ecg, label="Artifact-Free ECG", alpha=0.9)
    plt.plot(
        t[plot_mask][pacemaker_indices],
        ecg[plot_mask][pacemaker_indices],
        "rx",
        label="Removed pacemaker spikes",
    )
    plt.plot(
        t[plot_mask][~artifact_mask],
        artifactfree_ecg[~artifact_mask],
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


if __name__ == "__main__":
    main()
