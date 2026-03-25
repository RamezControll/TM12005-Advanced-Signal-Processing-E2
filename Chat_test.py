import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from read_telemetry_ecg import read_ecg_mat


# ============================================================
# 1. Inladen van ECG
# ============================================================

def load_ecg_seconds(
    path="../Data_E2/005_Pimpel.mat",
    plotresult=True,
    plot_start=0,
    plot_end=120,
):
    """
    Laadt ECG-signaal uit .mat-bestand.

    Returns
    -------
    ecg : np.ndarray
        Ruwe ECG-signaalwaarden.
    fs : float
        Samplefrequentie in Hz.
    t : np.ndarray
        Tijdas in seconden.
    plot_mask : np.ndarray of bool
        Masker voor het gewenste plot-interval.
    """
    ecg, fs, _ = read_ecg_mat(path, plotresult=plotresult)
    t = np.arange(len(ecg)) / fs
    plot_mask = (t >= plot_start) & (t <= plot_end)
    return ecg, fs, t, plot_mask


# ============================================================
# 2. Verwijderen van pacemaker-artefacten
# ============================================================

def remove_pacemaker_artifacts(ecg, threshold=-1000, max_extension=6):
    """
    Detecteert pacing-spikes als diepe negatieve artefacten onder een drempel
    en vervangt deze lokaal via lineaire interpolatie.

    Parameters
    ----------
    ecg : np.ndarray
        Ruwe ECG.
    threshold : float
        Alle samples onder deze waarde worden als mogelijke pacing-artefacten gezien.
    max_extension : int
        Maximaal aantal samples waarmee links/rechts wordt uitgebreid rond het minimum.

    Returns
    -------
    clean_ecg : np.ndarray
        ECG waarin pacing-spikes zijn weggefilterd.
    trough_indices : np.ndarray
        Index van het minimum per gedetecteerd artefact.
    mask : np.ndarray of bool
        True voor behouden samples, False voor weggefilterde samples.
    """
    below_threshold = np.flatnonzero(ecg < threshold)

    if below_threshold.size == 0:
        return ecg.copy(), np.array([], dtype=int), np.ones(len(ecg), dtype=bool)

    # Split in afzonderlijke groepen als artefacten niet aaneengesloten zijn
    split_points = np.where(np.diff(below_threshold) > 1)[0] + 1
    groups = np.split(below_threshold, split_points)

    trough_indices = []
    mask = np.ones(len(ecg), dtype=bool)

    for group in groups:
        trough_idx = group[np.argmin(ecg[group])]
        trough_indices.append(trough_idx)

        left = trough_idx
        right = trough_idx

        # Uitbreiden naar links
        while left > 0 and (trough_idx - left) < max_extension:
            if ecg[left] <= ecg[left - 1]:
                left -= 1
            else:
                break

        # Uitbreiden naar rechts
        while right < len(ecg) - 1 and (right - trough_idx) < max_extension:
            if ecg[right] <= ecg[right + 1]:
                right += 1
            else:
                break

        mask[left:right + 1] = False

    clean_ecg = ecg.copy()
    valid_idx = np.flatnonzero(mask)
    removed_idx = np.flatnonzero(~mask)

    # Interpoleer de weggehaalde punten
    if valid_idx.size >= 2:
        clean_ecg[removed_idx] = np.interp(removed_idx, valid_idx, ecg[valid_idx])
    elif valid_idx.size == 1:
        clean_ecg[removed_idx] = ecg[valid_idx[0]]

    return clean_ecg, np.array(trough_indices, dtype=int), mask


# ============================================================
# 3. Voorbewerking voor QRS-detectie
# ============================================================

def bandpass_ecg(x, fs, low=5.0, high=15.0, order=3):
    """
    Bandpassfilter voor QRS-versterking.
    """
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="bandpass")
    return signal.filtfilt(b, a, x)


def derivative_filter(x):
    """
    Simpele differentiatiefilter om steile overgangen te benadrukken.
    """
    b = np.array([1.0, -1.0])
    a = np.array([1.0, -0.999])
    return signal.lfilter(b, a, x)


def moving_window_integration(x, fs, win_ms=150):
    """
    Moving window integration zoals in Pan-Tompkins.
    """
    win = int(win_ms * fs / 1000)
    win = max(win, 1)
    return signal.lfilter(np.ones(win) / win, 1, x)


def pan_tompkins_preprocess(ecg, fs, bp_low=5, bp_high=15, mwi_ms=150):
    """
    Volledige preprocessing-keten:
    1. bandpass
    2. differentiatie
    3. kwadrateren
    4. moving window integration
    """
    y_bp = bandpass_ecg(ecg, fs, low=bp_low, high=bp_high, order=3)
    y_der = derivative_filter(y_bp)
    y_sq = y_der ** 2
    y_mwi = moving_window_integration(y_sq, fs, win_ms=mwi_ms)
    return y_bp, y_der, y_sq, y_mwi


# ============================================================
# 4. QRS-detectie
# ============================================================

def detect_qrs_peaks_on_mwi(y_mwi, fs, prominence_factor=0.05, refractory_ms=350):
    """
    Detecteert pieken op de MWI-output.

    Parameters
    ----------
    y_mwi : np.ndarray
        Output van moving window integration.
    fs : float
        Samplefrequentie.
    prominence_factor : float
        Factor voor prominencedrempel.
    refractory_ms : float
        Minimale afstand tussen twee QRS-detecties.

    Returns
    -------
    peaks : np.ndarray
        Gedetecteerde piekindices.
    prominence : float
        Gebruikte absolute prominence.
    properties : dict
        Output van scipy.signal.find_peaks.
    """
    refractory = int(refractory_ms * fs / 1000)
    robust_span = np.percentile(y_mwi, 99) - np.percentile(y_mwi, 50)
    prominence = prominence_factor * robust_span
    peaks, properties = signal.find_peaks(
        y_mwi,
        prominence=prominence,
        distance=refractory
    )
    return peaks, prominence, properties


def rr_hr_from_peaks(t, peaks):
    """
    Berekent RR-intervallen en gemiddelde hartfrequentie.
    """
    if len(peaks) < 2:
        return np.nan, np.nan, np.array([])
    rr = np.diff(t[peaks])
    mean_rr = np.mean(rr)
    mean_hr = 60.0 / mean_rr
    return mean_rr, mean_hr, rr


# ============================================================
# 5. QRS/P schatting en classificatie van pacing-momenten
# ============================================================

def refine_qrs_peaks_to_ecg(clean_ecg, qrs_indices, fs, search_back_ms=180, search_forward_ms=60):
    """
    Verplaatst MWI-detecties naar de dominante QRS-excursie in de schone ECG.
    """
    qrs_indices = np.asarray(qrs_indices, dtype=int)
    search_back_samples = int(search_back_ms * fs / 1000)
    search_forward_samples = int(search_forward_ms * fs / 1000)

    refined_indices = []

    for qrs_idx in qrs_indices:
        start = max(0, qrs_idx - search_back_samples)
        end = min(len(clean_ecg), qrs_idx + search_forward_samples + 1)

        segment = clean_ecg[start:end]
        if segment.size == 0:
            continue

        baseline = np.median(segment[:max(3, segment.size // 4)])
        local_idx = int(np.argmax(np.abs(segment - baseline)))
        refined_indices.append(start + local_idx)

    return np.asarray(refined_indices, dtype=int)


def estimate_qrs_onsets(clean_ecg, qrs_peak_indices, fs, search_back_ms=180, threshold_ratio=0.12):
    """
    Schat het begin van elk QRS-complex op basis van de QRS-envelope.
    """
    qrs_peak_indices = np.asarray(qrs_peak_indices, dtype=int)
    search_back_samples = int(search_back_ms * fs / 1000)

    qrs_band = bandpass_ecg(clean_ecg, fs, low=5.0, high=25.0, order=2)
    qrs_envelope = np.abs(qrs_band)
    win = max(int(20 * fs / 1000), 1)
    qrs_envelope = np.convolve(qrs_envelope, np.ones(win) / win, mode="same")

    qrs_onsets = []

    for peak_idx in qrs_peak_indices:
        start = max(0, peak_idx - search_back_samples)
        segment = qrs_envelope[start:peak_idx + 1]

        if segment.size == 0:
            qrs_onsets.append(peak_idx)
            continue

        peak_val = np.max(segment)
        threshold = threshold_ratio * peak_val
        above_threshold = np.flatnonzero(segment >= threshold)

        if above_threshold.size == 0:
            qrs_onsets.append(peak_idx)
        else:
            qrs_onsets.append(start + int(above_threshold[0]))

    return np.asarray(qrs_onsets, dtype=int)


def estimate_p_peaks(clean_ecg, qrs_indices, fs, p_search_start_ms=250, p_search_end_ms=60):
    """
    Schat per QRS een mogelijke P-top in het venster ervoor.

    De P-top wordt benaderd als het punt met de grootste absolute amplitude
    in een mild gefilterd signaal tussen 250 en 60 ms voor het QRS-complex.
    """
    qrs_indices = np.asarray(qrs_indices, dtype=int)

    p_search_start_samples = int(p_search_start_ms * fs / 1000)
    p_search_end_samples = int(p_search_end_ms * fs / 1000)

    p_band = bandpass_ecg(clean_ecg, fs, low=0.5, high=12.0, order=2)
    p_peaks = []

    for qrs_idx in qrs_indices:
        start = max(0, qrs_idx - p_search_start_samples)
        end = max(start + 1, qrs_idx - p_search_end_samples)

        if end <= start:
            continue

        segment = p_band[start:end]
        if segment.size == 0:
            continue

        local_idx = int(np.argmax(np.abs(segment)))
        p_peaks.append(start + local_idx)

    return np.asarray(p_peaks, dtype=int)


def classify_pacing_events(
    spike_indices,
    qrs_peak_indices,
    qrs_onset_indices,
    p_peak_indices,
    fs,
    ventricular_max_ms=100,
    atrial_peak_min_ms=15,
    atrial_peak_max_ms=120,
    qrs_error_post_ms=80,
):
    """
    Classificeert alleen gedetecteerde pacing-spikes in drie categorieen:
    - atriaal: spike kort voor een geschatte P-top
    - ventriculair: spike vlak voor het begin van een QRS
    - pacingfout: spike tijdens of direct rond een QRS-complex
    """
    spike_indices = np.asarray(spike_indices, dtype=int)
    qrs_peak_indices = np.asarray(qrs_peak_indices, dtype=int)
    qrs_onset_indices = np.asarray(qrs_onset_indices, dtype=int)
    p_peak_indices = np.asarray(p_peak_indices, dtype=int)

    ventricular_max_samples = int(ventricular_max_ms * fs / 1000)
    atrial_peak_min_samples = int(atrial_peak_min_ms * fs / 1000)
    atrial_peak_max_samples = int(atrial_peak_max_ms * fs / 1000)
    qrs_error_post_samples = int(qrs_error_post_ms * fs / 1000)

    pacing_events = []

    for spike_idx in spike_indices:
        prev_pos = np.searchsorted(qrs_onset_indices, spike_idx, side="right") - 1
        next_pos = np.searchsorted(qrs_onset_indices, spike_idx, side="left")

        prev_qrs_onset = qrs_onset_indices[prev_pos] if prev_pos >= 0 else None
        next_qrs_onset = qrs_onset_indices[next_pos] if next_pos < len(qrs_onset_indices) else None
        next_qrs_peak = qrs_peak_indices[next_pos] if next_pos < len(qrs_peak_indices) else None

        label = None
        related_qrs_idx = None
        related_p_idx = None

        if prev_qrs_onset is not None:
            dt_after_prev_qrs = spike_idx - prev_qrs_onset
            if 0 <= dt_after_prev_qrs <= qrs_error_post_samples:
                label = "pacingfout"
                related_qrs_idx = prev_qrs_onset

        if label is None and next_qrs_onset is not None:
            dt_before_next_qrs = next_qrs_onset - spike_idx

            if 0 <= dt_before_next_qrs <= ventricular_max_samples:
                label = "ventriculair"
                related_qrs_idx = next_qrs_peak
            else:
                p_candidates = p_peak_indices[
                    (p_peak_indices > spike_idx) &
                    (p_peak_indices < next_qrs_onset)
                ]

                if p_candidates.size > 0:
                    p_idx = p_candidates[0]
                    dt_to_p = p_idx - spike_idx

                    if atrial_peak_min_samples <= dt_to_p <= atrial_peak_max_samples:
                        label = "atriaal"
                        related_qrs_idx = next_qrs_peak
                        related_p_idx = p_idx

        if label is not None:
            pacing_events.append({
                "spike_index": spike_idx,
                "time_s": spike_idx / fs,
                "label": label,
                "related_qrs_index": related_qrs_idx,
                "related_p_index": related_p_idx,
            })

    return pacing_events


def summarize_pacing_events(pacing_events):
    """
    Telt hoe vaak elk pacing-label voorkomt.
    """
    summary = {
        "atriaal": 0,
        "ventriculair": 0,
        "pacingfout": 0,
    }

    for event in pacing_events:
        if event["label"] in summary:
            summary[event["label"]] += 1

    return summary


# ============================================================
# 6. Plotfuncties
# ============================================================

def plot_artifact_removal(t, ecg, clean_ecg, spike_indices, artifact_mask, plot_mask):
    """
    Laat zien:
    - ruwe ECG
    - artefactvrije ECG
    - gedetecteerde pacing-spikes
    - geïnterpoleerde stukken
    """
    plt.figure(figsize=(12, 4))
    plt.plot(t[plot_mask], ecg[plot_mask], label="Raw ECG", alpha=0.35)
    plt.plot(t[plot_mask], clean_ecg[plot_mask], label="Artifact-free ECG", alpha=0.9)

    visible_spikes = spike_indices[(t[spike_indices] >= t[plot_mask][0]) & (t[spike_indices] <= t[plot_mask][-1])]
    plt.plot(
        t[visible_spikes],
        ecg[visible_spikes],
        "rx",
        label="Detected pacing spikes"
    )

    plt.plot(
        t[plot_mask][~artifact_mask[plot_mask]],
        clean_ecg[plot_mask][~artifact_mask[plot_mask]],
        ".",
        markersize=3,
        label="Interpolated samples"
    )

    plt.title("ECG with removed pacemaker artifacts")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_qrs_detection(t, clean_ecg, y_mwi, qrs_indices, plot_mask):
    """
    Laat artifactvrije ECG en MWI zien met gedetecteerde QRS-complexen.
    """
    plot_t = t[plot_mask]
    plot_ecg = clean_ecg[plot_mask]
    plot_mwi = y_mwi[plot_mask]

    full_indices = np.flatnonzero(plot_mask)

    # Zet globale qrs_indices om naar lokale indices in plotsegment
    qrs_in_window = qrs_indices[(qrs_indices >= full_indices[0]) & (qrs_indices <= full_indices[-1])]
    qrs_local = qrs_in_window - full_indices[0]

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax[0].plot(plot_t, plot_ecg, label="Artifact-free ECG", alpha=0.9)
    ax[0].plot(plot_t[qrs_local], plot_ecg[qrs_local], "rx", label="Detected QRS")
    ax[0].set_title("Artifact-free ECG with QRS detections")
    ax[0].set_ylabel("ECG amplitude")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(plot_t, plot_mwi, label="MWI")
    ax[1].plot(plot_t[qrs_local], plot_mwi[qrs_local], "rx", label="Detected peaks")
    ax[1].set_title("MWI with detected QRS-related peaks")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("MWI")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def plot_pacing_labels(t, ecg, clean_ecg, pacing_events, plot_mask):
    """
    Visualiseert per pacing-spike welk label is toegekend op de schone ECG.
    """
    plt.figure(figsize=(14, 4))
    plt.plot(t[plot_mask], ecg[plot_mask], label="Raw ECG", alpha=0.25)
    plt.plot(t[plot_mask], clean_ecg[plot_mask], label="Artifact-free ECG", alpha=0.85)

    label_styles = {
        "atriaal": "bo",
        "ventriculair": "ro",
        "pacingfout": "mo",
    }

    plotted_labels = set()

    for event in pacing_events:
        spike_idx = event["spike_index"]
        if not plot_mask[spike_idx]:
            continue

        label = event["label"]
        style = label_styles[label]

        if label not in plotted_labels:
            plt.plot(t[spike_idx], clean_ecg[spike_idx], style, label=label)
            plotted_labels.add(label)
        else:
            plt.plot(t[spike_idx], clean_ecg[spike_idx], style)

    plt.title("Classificatie van pacing-momenten")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. Main
# ============================================================

def main():
    # -----------------------------
    # Instellingen
    # -----------------------------
    ecg_path = "../Data_E2/005_Pimpel_3.mat"
    plot_start = 0
    plot_end = 12000

    # Artefactdetectie
    artifact_threshold = -1000
    artifact_max_extension = 6

    # QRS-detectie
    bp_low = 5
    bp_high = 15
    mwi_ms = 150
    prominence_factor = 0.05
    refractory_ms = 350

    # Classificatie van pacing-momenten
    ventricular_max_ms = 100
    atrial_peak_min_ms = 15
    atrial_peak_max_ms = 120
    p_search_start_ms = 250
    p_search_end_ms = 60
    qrs_error_post_ms = 80

    # -----------------------------
    # 1. ECG laden
    # -----------------------------
    ecg, fs, t, plot_mask = load_ecg_seconds(
        path=ecg_path,
        plotresult=True,
        plot_start=plot_start,
        plot_end=plot_end,
    )

    # -----------------------------
    # 2. Pacing artefacten verwijderen
    # -----------------------------
    clean_ecg, pacemaker_indices, artifact_mask = remove_pacemaker_artifacts(
        ecg,
        threshold=artifact_threshold,
        max_extension=artifact_max_extension,
    )

    plot_artifact_removal(
        t=t,
        ecg=ecg,
        clean_ecg=clean_ecg,
        spike_indices=pacemaker_indices,
        artifact_mask=artifact_mask,
        plot_mask=plot_mask,
    )

    # -----------------------------
    # 3. Preprocessing voor QRS
    # -----------------------------
    y_bp, y_der, y_sq, y_mwi = pan_tompkins_preprocess(
        clean_ecg,
        fs,
        bp_low=bp_low,
        bp_high=bp_high,
        mwi_ms=mwi_ms,
    )

    # -----------------------------
    # 4. QRS-detectie over volledig signaal
    # -----------------------------
    qrs_indices, prominence, _ = detect_qrs_peaks_on_mwi(
        y_mwi,
        fs,
        prominence_factor=prominence_factor,
        refractory_ms=refractory_ms,
    )

    qrs_peak_indices = refine_qrs_peaks_to_ecg(
        clean_ecg=clean_ecg,
        qrs_indices=qrs_indices,
        fs=fs,
    )
    qrs_onset_indices = estimate_qrs_onsets(
        clean_ecg=clean_ecg,
        qrs_peak_indices=qrs_peak_indices,
        fs=fs,
    )

    mean_rr, mean_hr, rr = rr_hr_from_peaks(t, qrs_peak_indices)

    plot_qrs_detection(
        t=t,
        clean_ecg=clean_ecg,
        y_mwi=y_mwi,
        qrs_indices=qrs_indices,
        plot_mask=plot_mask,
    )

    # -----------------------------
    # 5. Classificatie van pacing-momenten
    # -----------------------------
    p_peak_indices = estimate_p_peaks(
        clean_ecg=clean_ecg,
        qrs_indices=qrs_onset_indices,
        fs=fs,
        p_search_start_ms=p_search_start_ms,
        p_search_end_ms=p_search_end_ms,
    )

    pacing_events = classify_pacing_events(
        spike_indices=pacemaker_indices,
        qrs_peak_indices=qrs_peak_indices,
        qrs_onset_indices=qrs_onset_indices,
        p_peak_indices=p_peak_indices,
        fs=fs,
        ventricular_max_ms=ventricular_max_ms,
        atrial_peak_min_ms=atrial_peak_min_ms,
        atrial_peak_max_ms=atrial_peak_max_ms,
        qrs_error_post_ms=qrs_error_post_ms,
    )

    summary = summarize_pacing_events(pacing_events)

    plot_pacing_labels(
        t=t,
        ecg=ecg,
        clean_ecg=clean_ecg,
        pacing_events=pacing_events,
        plot_mask=plot_mask,
    )

    # -----------------------------
    # 6. Resultaten printen
    # -----------------------------
    print("\n================ ECG ANALYSIS SUMMARY ================\n")
    print(f"Sampling frequency: {fs:.2f} Hz")
    print(f"Number of detected pacing spikes: {len(pacemaker_indices)}")
    print(f"Number of detected QRS complexes: {len(qrs_indices)}")
    print(f"Prominence factor: {prominence_factor:.3f}")
    print(f"Absolute prominence used: {prominence:.3f}")
    print(f"Refractory period: {refractory_ms} ms")

    if len(rr) > 0:
        print(f"Mean RR interval: {mean_rr:.3f} s")
        print(f"Mean heart rate: {mean_hr:.2f} bpm")
    else:
        print("Not enough QRS detections to compute RR/HR.")

    print("\nPacing event summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nFirst 20 classified pacing events:")
    for event in pacing_events[:20]:
        print(
            f"time={event['time_s']:.3f} s | "
            f"label={event['label']}"
        )


if __name__ == "__main__":
    main()
