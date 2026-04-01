import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from read_telemetry_ecg import read_ecg_mat


# ============================================================
# 1. Inladen van ECG
# ============================================================

def load_ecg_seconds(
    path="../Data_E2/005_Pimpel.mat",
    plotresult=False,
    plot_start=0,
    plot_end=120,
):
    """
    Laadt ECG-signaal uit een .mat-bestand en bouwt een tijdas in seconden.

    Gebruik `plot_end=None` om het volledige resterende signaal te plotten.
    """
    ecg, fs, _ = read_ecg_mat(path, plotresult=plotresult)
    t = np.arange(len(ecg)) / fs

    if plot_start is None:
        plot_start = 0

    if plot_end is None:
        plot_mask = t >= plot_start
    else:
        plot_mask = (t >= plot_start) & (t <= plot_end)

    return ecg, fs, t, plot_mask


# ============================================================
# 2. Detectie en verwijdering van pacemaker-artefacten
# ============================================================

def detect_pacemaker_artifacts(ecg, threshold=-1000, max_extension=6):
    """
    Detecteert smalle pacemaker-spikes als diepe negatieve uitschieters.

    Per artefact wordt het minimum, het begin/einde van het te verwijderen
    segment en het midden van de interpolatie opgeslagen.
    """
    below_threshold = np.flatnonzero(ecg < threshold)

    if below_threshold.size == 0:
        return []

    split_points = np.where(np.diff(below_threshold) > 1)[0] + 1
    groups = np.split(below_threshold, split_points)

    artifact_events = []

    for group in groups:
        trough_idx = int(group[np.argmin(ecg[group])])

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

        center_idx = int(round(0.5 * (left + right)))

        artifact_events.append({
            "trough_index": trough_idx,
            "start_index": left,
            "end_index": right,
            "center_index": center_idx,
            "width_samples": right - left + 1,
        })

    return artifact_events


def remove_pacemaker_artifacts(ecg, threshold=-1000, max_extension=6):
    """
    Vervangt gedetecteerde pacemaker-spikes lokaal via lineaire interpolatie.
    """
    artifact_events = detect_pacemaker_artifacts(
        ecg,
        threshold=threshold,
        max_extension=max_extension,
    )

    if not artifact_events:
        return ecg.copy(), [], np.ones(len(ecg), dtype=bool)

    mask = np.ones(len(ecg), dtype=bool)

    for event in artifact_events:
        start = event["start_index"]
        end = event["end_index"]
        mask[start:end + 1] = False

    clean_ecg = ecg.copy()
    valid_idx = np.flatnonzero(mask)
    removed_idx = np.flatnonzero(~mask)

    if valid_idx.size >= 2:
        clean_ecg[removed_idx] = np.interp(removed_idx, valid_idx, ecg[valid_idx])
    elif valid_idx.size == 1:
        clean_ecg[removed_idx] = ecg[valid_idx[0]]

    return clean_ecg, artifact_events, mask


def event_indices(events, key):
    """
    Haalt een numpy-array met indices uit een lijst event-dictionaries.
    """
    if not events:
        return np.array([], dtype=int)
    return np.asarray([event[key] for event in events], dtype=int)


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
    Volledige preprocessing-keten voor QRS-detectie.
    """
    y_bp = bandpass_ecg(ecg, fs, low=bp_low, high=bp_high, order=3)
    y_der = derivative_filter(y_bp)
    y_sq = y_der ** 2
    y_mwi = moving_window_integration(y_sq, fs, win_ms=mwi_ms)
    return y_bp, y_der, y_sq, y_mwi


def detect_qrs_peaks_on_mwi(y_mwi, fs, prominence_factor=0.05, refractory_ms=350):
    """
    Detecteert pieken op de MWI-output.
    """
    refractory = int(refractory_ms * fs / 1000)
    robust_span = np.percentile(y_mwi, 99) - np.percentile(y_mwi, 50)
    prominence = prominence_factor * robust_span
    peaks, properties = signal.find_peaks(
        y_mwi,
        prominence=prominence,
        distance=refractory,
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
# 4. Scherpere QRS/P schatting
# ============================================================

def refine_qrs_peaks_to_ecg(
    clean_ecg,
    qrs_indices,
    fs,
    search_back_ms=180,
    search_forward_ms=60,
):
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


def build_qrs_envelope(clean_ecg, fs, low=5.0, high=25.0, order=2, smooth_ms=20):
    """
    Bouwt een gladde QRS-envelope voor onset/offset-schatting.
    """
    qrs_band = bandpass_ecg(clean_ecg, fs, low=low, high=high, order=order)
    qrs_envelope = np.abs(qrs_band)
    win = max(int(smooth_ms * fs / 1000), 1)
    qrs_envelope = np.convolve(qrs_envelope, np.ones(win) / win, mode="same")
    return qrs_envelope


def estimate_qrs_windows(
    qrs_envelope,
    qrs_peak_indices,
    fs,
    search_back_ms=180,
    search_forward_ms=220,
    threshold_ratio=0.12,
):
    """
    Schat per QRS een onset en offset met een lokale drempel.

    Voor de onset wordt de laatste kruising voor de piek genomen, zodat de
    QRS-start scherper ligt dan bij de eerdere eerste-kruising-benadering.
    """
    qrs_peak_indices = np.asarray(qrs_peak_indices, dtype=int)
    search_back_samples = int(search_back_ms * fs / 1000)
    search_forward_samples = int(search_forward_ms * fs / 1000)

    qrs_onsets = []
    qrs_offsets = []

    for peak_idx in qrs_peak_indices:
        start = max(0, peak_idx - search_back_samples)
        pre_segment = qrs_envelope[start:peak_idx + 1]

        if pre_segment.size == 0:
            qrs_onsets.append(peak_idx)
        else:
            baseline_pre = np.percentile(pre_segment, 10)
            peak_pre = np.max(pre_segment)
            threshold_pre = baseline_pre + threshold_ratio * (peak_pre - baseline_pre)
            below_pre = np.flatnonzero(pre_segment < threshold_pre)
            onset = start if below_pre.size == 0 else start + int(below_pre[-1]) + 1
            qrs_onsets.append(min(onset, peak_idx))

        end = min(len(qrs_envelope), peak_idx + search_forward_samples + 1)
        post_segment = qrs_envelope[peak_idx:end]

        if post_segment.size == 0:
            qrs_offsets.append(peak_idx)
        else:
            baseline_post = np.percentile(post_segment, 10)
            peak_post = np.max(post_segment)
            threshold_post = baseline_post + threshold_ratio * (peak_post - baseline_post)
            below_post = np.flatnonzero(post_segment < threshold_post)
            offset = end - 1 if below_post.size == 0 else peak_idx + int(below_post[0])
            qrs_offsets.append(max(offset, peak_idx))

    return np.asarray(qrs_onsets, dtype=int), np.asarray(qrs_offsets, dtype=int)


def estimate_p_peaks(
    clean_ecg,
    qrs_onset_indices,
    fs,
    p_search_start_ms=260,
    p_search_end_ms=60,
    smooth_ms=30,
    prominence_ratio=0.25,
):
    """
    Schat per QRS een mogelijke P-top in het venster ervoor.

    In afleiding II zijn P-toppen meestal positieve deflecties. Daarom zoeken
    we hier alleen naar positieve toppen in een mild gefilterde P-band, zodat
    grote negatieve ventriculaire uitslagen niet per ongeluk als P-top worden
    gekozen.
    """
    qrs_onset_indices = np.asarray(qrs_onset_indices, dtype=int)

    p_search_start_samples = int(p_search_start_ms * fs / 1000)
    p_search_end_samples = int(p_search_end_ms * fs / 1000)
    smooth_samples = max(int(smooth_ms * fs / 1000), 1)
    peak_distance = max(int(40 * fs / 1000), 1)

    p_band = bandpass_ecg(clean_ecg, fs, low=0.5, high=12.0, order=2)
    p_band = np.convolve(p_band, np.ones(smooth_samples) / smooth_samples, mode="same")

    p_peaks = []

    for qrs_onset_idx in qrs_onset_indices:
        start = max(0, qrs_onset_idx - p_search_start_samples)
        end = max(start + 1, qrs_onset_idx - p_search_end_samples)

        if end <= start:
            continue

        segment = p_band[start:end]
        if segment.size == 0:
            continue

        centered = segment - np.median(segment)
        positive_centered = centered.copy()
        positive_centered[positive_centered < 0] = 0.0

        if np.allclose(positive_centered, 0.0):
            continue

        prominence = prominence_ratio * np.max(positive_centered)
        peaks, properties = signal.find_peaks(
            positive_centered,
            prominence=prominence,
            distance=peak_distance,
        )

        if peaks.size == 0:
            continue
        else:
            prominences = properties["prominences"]
            strong_mask = prominences >= 0.5 * np.max(prominences)
            candidate_peaks = peaks[strong_mask]
            local_idx = int(candidate_peaks[-1])

        p_peaks.append(start + local_idx)

    return np.asarray(p_peaks, dtype=int)


# ============================================================
# 5. Classificatie van pacing-momenten
# ============================================================

def classify_pacing_events(
    artifact_events,
    qrs_peak_indices,
    qrs_onset_indices,
    qrs_offset_indices,
    p_peak_indices,
    fs,
    ventricular_min_ms=5,
    ventricular_max_ms=130,
    ventricular_peak_max_ms=300,
    atrial_peak_min_ms=15,
    atrial_peak_max_ms=120,
    atrial_qrs_guard_ms=70,
    pacing_error_post_peak_ms=5,
):
    """
    Classificeert elke gedetecteerde stimulatie als:
    - atriaal: spike kort voor een P-top
    - ventriculair: spike vlak voor een QRS-onset
    - pacingfout: spike duidelijk in een al lopend QRS-complex
    - onbekend: geen duidelijke match
    """
    qrs_peak_indices = np.asarray(qrs_peak_indices, dtype=int)
    qrs_onset_indices = np.asarray(qrs_onset_indices, dtype=int)
    qrs_offset_indices = np.asarray(qrs_offset_indices, dtype=int)
    p_peak_indices = np.asarray(p_peak_indices, dtype=int)

    ventricular_min_samples = int(ventricular_min_ms * fs / 1000)
    ventricular_max_samples = int(ventricular_max_ms * fs / 1000)
    ventricular_peak_max_samples = int(ventricular_peak_max_ms * fs / 1000)
    atrial_peak_min_samples = int(atrial_peak_min_ms * fs / 1000)
    atrial_peak_max_samples = int(atrial_peak_max_ms * fs / 1000)
    atrial_qrs_guard_samples = int(atrial_qrs_guard_ms * fs / 1000)
    pacing_error_post_peak_samples = int(pacing_error_post_peak_ms * fs / 1000)

    pacing_events = []

    for artifact_event in artifact_events:
        spike_idx = artifact_event["center_index"]
        prev_pos = np.searchsorted(qrs_onset_indices, spike_idx, side="right") - 1
        next_pos = np.searchsorted(qrs_onset_indices, spike_idx, side="left")

        label = "onbekend"
        related_qrs_idx = None
        related_p_idx = None

        if prev_pos >= 0:
            prev_qrs_onset = qrs_onset_indices[prev_pos]
            prev_qrs_peak = qrs_peak_indices[prev_pos]
            prev_qrs_offset = qrs_offset_indices[prev_pos]

            if prev_qrs_onset <= spike_idx <= prev_qrs_offset:
                if spike_idx <= prev_qrs_peak + pacing_error_post_peak_samples:
                    label = "ventriculair"
                    related_qrs_idx = prev_qrs_peak
                else:
                    label = "pacingfout"
                    related_qrs_idx = prev_qrs_peak

        if label == "onbekend" and next_pos < len(qrs_onset_indices):
            next_qrs_onset = qrs_onset_indices[next_pos]
            next_qrs_peak = qrs_peak_indices[next_pos]
            dt_to_qrs = next_qrs_onset - spike_idx
            dt_to_qrs_peak = next_qrs_peak - spike_idx

            p_candidates = p_peak_indices[
                (p_peak_indices > spike_idx) &
                (p_peak_indices < next_qrs_onset)
            ]

            if p_candidates.size > 0:
                p_idx = int(p_candidates[0])
                dt_to_p = p_idx - spike_idx
                p_to_qrs = next_qrs_onset - p_idx

                if (
                    atrial_peak_min_samples <= dt_to_p <= atrial_peak_max_samples
                    and p_to_qrs >= atrial_qrs_guard_samples
                ):
                    label = "atriaal"
                    related_qrs_idx = next_qrs_peak
                    related_p_idx = p_idx

            if (
                label == "onbekend"
                and (
                    ventricular_min_samples <= dt_to_qrs <= ventricular_max_samples
                    or 0 <= dt_to_qrs_peak <= ventricular_peak_max_samples
                )
            ):
                label = "ventriculair"
                related_qrs_idx = next_qrs_peak

        pacing_events.append({
            "spike_index": artifact_event["center_index"],
            "trough_index": artifact_event["trough_index"],
            "start_index": artifact_event["start_index"],
            "end_index": artifact_event["end_index"],
            "time_s": artifact_event["center_index"] / fs,
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
        "onbekend": 0,
    }

    for event in pacing_events:
        if event["label"] in summary:
            summary[event["label"]] += 1

    return summary


# ============================================================
# 6. Plotfuncties
# ============================================================

def plot_raw_ecg(t, ecg, plot_mask):
    """
    Eerste plot: alleen het rauwe ECG-signaal.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(t[plot_mask], ecg[plot_mask], color="steelblue", linewidth=0.9)
    plt.title("Rauw ECG-signaal")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_artifact_removal(t, ecg, clean_ecg, artifact_events, artifact_mask, plot_mask):
    """
    Tweede plot: interpolatie na verwijdering van pacemaker-spikes.
    """
    trough_indices = event_indices(artifact_events, "trough_index")

    plt.figure(figsize=(12, 4))
    plt.plot(t[plot_mask], ecg[plot_mask], label="Raw ECG", alpha=0.3)
    plt.plot(t[plot_mask], clean_ecg[plot_mask], label="Interpolated ECG", alpha=0.9)

    visible_troughs = trough_indices[plot_mask[trough_indices]] if trough_indices.size > 0 else trough_indices
    if visible_troughs.size > 0:
        plt.plot(
            t[visible_troughs],
            ecg[visible_troughs],
            "rx",
            label="Detected pacing spikes",
        )

    plt.plot(
        t[plot_mask][~artifact_mask[plot_mask]],
        clean_ecg[plot_mask][~artifact_mask[plot_mask]],
        ".",
        markersize=2.5,
        label="Interpolated samples",
    )

    plt.title("ECG na piekverwijdering en interpolatie")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pacing_labels(
    t,
    clean_ecg,
    pacing_events,
    qrs_peak_indices,
    p_peak_indices,
    plot_mask,
):
    """
    Derde plot: pacing-labels op de geinterpoleerde ECG.
    """
    plt.figure(figsize=(14, 5))
    plt.plot(t[plot_mask], clean_ecg[plot_mask], label="Interpolated ECG", alpha=0.9)

    visible_qrs = qrs_peak_indices[plot_mask[qrs_peak_indices]] if len(qrs_peak_indices) > 0 else qrs_peak_indices
    visible_p = p_peak_indices[plot_mask[p_peak_indices]] if len(p_peak_indices) > 0 else p_peak_indices

    if visible_qrs.size > 0:
        plt.plot(
            t[visible_qrs],
            clean_ecg[visible_qrs],
            "k.",
            markersize=3,
            alpha=0.45,
            label="QRS peak",
        )

    if visible_p.size > 0:
        plt.plot(
            t[visible_p],
            clean_ecg[visible_p],
            ".",
            color="forestgreen",
            markersize=3,
            alpha=0.45,
            label="P peak",
        )

    label_styles = {
        "atriaal": {"fmt": "o", "color": "royalblue"},
        "ventriculair": {"fmt": "^", "color": "crimson"},
        "pacingfout": {"fmt": "x", "color": "magenta"},
        "onbekend": {"fmt": "s", "color": "gray"},
    }

    plotted_labels = set()

    for event in pacing_events:
        spike_idx = event["spike_index"]
        if not plot_mask[spike_idx]:
            continue

        style = label_styles[event["label"]]
        label = event["label"] if event["label"] not in plotted_labels else None

        plt.plot(
            t[spike_idx],
            clean_ecg[spike_idx],
            marker=style["fmt"],
            color=style["color"],
            linestyle="None",
            markersize=6,
            label=label,
        )

        if label is not None:
            plotted_labels.add(event["label"])

    plt.title("Classificatie van pacing op de geinterpoleerde ECG")
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
    ecg_path = "../Data_E2/005_Pimpel.mat"
    plot_start = 0
    plot_end = None # None betekent dat we tot het einde van het signaal plotten, anders gewoon aantal secondes

    # Artefactdetectie
    artifact_threshold = -1000
    artifact_max_extension = 6

    # QRS-detectie
    bp_low = 5
    bp_high = 15
    mwi_ms = 150
    prominence_factor = 0.05
    refractory_ms = 350
    qrs_threshold_ratio = 0.12

    # P-top schatting
    p_search_start_ms = 260
    p_search_end_ms = 60
    p_prominence_ratio = 0.25

    # Classificatie van pacing-momenten
    ventricular_min_ms = 5
    ventricular_max_ms = 130
    ventricular_peak_max_ms = 300
    atrial_peak_min_ms = 15
    atrial_peak_max_ms = 120
    atrial_qrs_guard_ms = 70
    pacing_error_post_peak_ms = 5

    # -----------------------------
    # 1. ECG laden
    # -----------------------------
    ecg, fs, t, plot_mask = load_ecg_seconds(
        path=ecg_path,
        plotresult=False,
        plot_start=plot_start,
        plot_end=plot_end,
    )

    # -----------------------------
    # 2. Rauw signaal plotten
    # -----------------------------
    plot_raw_ecg(t, ecg, plot_mask)

    # -----------------------------
    # 3. Pacing artefacten verwijderen
    # -----------------------------
    clean_ecg, artifact_events, artifact_mask = remove_pacemaker_artifacts(
        ecg,
        threshold=artifact_threshold,
        max_extension=artifact_max_extension,
    )

    plot_artifact_removal(
        t=t,
        ecg=ecg,
        clean_ecg=clean_ecg,
        artifact_events=artifact_events,
        artifact_mask=artifact_mask,
        plot_mask=plot_mask,
    )

    # -----------------------------
    # 4. Preprocessing voor QRS
    # -----------------------------
    y_bp, y_der, y_sq, y_mwi = pan_tompkins_preprocess(
        clean_ecg,
        fs,
        bp_low=bp_low,
        bp_high=bp_high,
        mwi_ms=mwi_ms,
    )

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

    qrs_envelope = build_qrs_envelope(clean_ecg, fs)
    qrs_onset_indices, qrs_offset_indices = estimate_qrs_windows(
        qrs_envelope=qrs_envelope,
        qrs_peak_indices=qrs_peak_indices,
        fs=fs,
        threshold_ratio=qrs_threshold_ratio,
    )

    p_peak_indices = estimate_p_peaks(
        clean_ecg=clean_ecg,
        qrs_onset_indices=qrs_onset_indices,
        fs=fs,
        p_search_start_ms=p_search_start_ms,
        p_search_end_ms=p_search_end_ms,
        prominence_ratio=p_prominence_ratio,
    )

    pacing_events = classify_pacing_events(
        artifact_events=artifact_events,
        qrs_peak_indices=qrs_peak_indices,
        qrs_onset_indices=qrs_onset_indices,
        qrs_offset_indices=qrs_offset_indices,
        p_peak_indices=p_peak_indices,
        fs=fs,
        ventricular_min_ms=ventricular_min_ms,
        ventricular_max_ms=ventricular_max_ms,
        ventricular_peak_max_ms=ventricular_peak_max_ms,
        atrial_peak_min_ms=atrial_peak_min_ms,
        atrial_peak_max_ms=atrial_peak_max_ms,
        atrial_qrs_guard_ms=atrial_qrs_guard_ms,
        pacing_error_post_peak_ms=pacing_error_post_peak_ms,
    )

    plot_pacing_labels(
        t=t,
        clean_ecg=clean_ecg,
        pacing_events=pacing_events,
        qrs_peak_indices=qrs_peak_indices,
        p_peak_indices=p_peak_indices,
        plot_mask=plot_mask,
    )

    # -----------------------------
    # 5. Resultaten printen
    # -----------------------------
    mean_rr, mean_hr, rr = rr_hr_from_peaks(t, qrs_peak_indices)
    summary = summarize_pacing_events(pacing_events)

    print("\n================ ECG ANALYSIS SUMMARY ================\n")
    print(f"Sampling frequency: {fs:.2f} Hz")
    print(f"Number of detected pacemaker spikes: {len(artifact_events)}")
    print(f"Number of detected QRS complexes: {len(qrs_peak_indices)}")
    print(f"Prominence factor: {prominence_factor:.3f}")
    print(f"Absolute prominence used: {prominence:.3f}")
    print(f"QRS threshold ratio: {qrs_threshold_ratio:.3f}")

    if len(rr) > 0:
        print(f"Mean RR interval: {mean_rr:.3f} s")
        print(f"Mean heart rate: {mean_hr:.2f} bpm")
    else:
        print("Not enough QRS detections to compute RR/HR.")

    print("\nPacing event summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nFirst 20 pacing events:")
    for event in pacing_events[:20]:
        print(
            f"time={event['time_s']:.3f} s | "
            f"label={event['label']}"
        )


if __name__ == "__main__":
    main()
