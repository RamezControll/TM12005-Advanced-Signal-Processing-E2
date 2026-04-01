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


def remove_baseline_wander(ecg, fs, cutoff_hz=0.5, order=2):
    """
    Verwijdert baseline-wander en DC-offset met een nul-fase high-pass filter.

    De gecorrigeerde ECG wordt gebruikt voor verdere detectie van P-toppen en
    QRS-complexen, zodat langzame drift minder snel tot foutieve labels leidt.
    """
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")
    corrected_ecg = signal.sosfiltfilt(sos, ecg)
    corrected_ecg = corrected_ecg - np.median(corrected_ecg)
    estimated_baseline = ecg - corrected_ecg
    return corrected_ecg, estimated_baseline


def event_indices(events, key):
    """
    Haalt een numpy-array met indices uit een lijst event-dictionaries.
    """
    if not events:
        return np.array([], dtype=int)
    return np.asarray([event[key] for event in events], dtype=int)


# ============================================================
# 3. Detectie van QRS-complexen en P-toppen
# ============================================================

def bandpass_ecg(x, fs, low=5.0, high=15.0, order=3):
    """
    Bandpassfilter op de ECG.
    """
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="bandpass")
    return signal.filtfilt(b, a, x)


def smooth_signal(x, fs, win_ms=30):
    """
    Gladde moving-average zonder faseverschuiving.
    """
    win = max(int(win_ms * fs / 1000), 1)
    return np.convolve(x, np.ones(win) / win, mode="same")


def robust_mad(x):
    """
    Robuuste maat voor spreiding.
    """
    x = np.asarray(x, dtype=float)
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return median, mad + np.finfo(float).eps


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


def build_qrs_detection_signal(clean_ecg, fs, low=8.0, high=30.0, order=3, smooth_ms=30):
    """
    Bouwt een QRS-activiteitssignaal dat brede paced QRS-complexen benadrukt.
    """
    qrs_band = bandpass_ecg(clean_ecg, fs, low=low, high=high, order=order)
    qrs_gradient = np.gradient(qrs_band)
    qrs_energy = qrs_band ** 2 + 0.5 * qrs_gradient ** 2
    qrs_energy = smooth_signal(qrs_energy, fs, win_ms=smooth_ms)
    return qrs_band, qrs_energy


def merge_nearby_peaks(candidate_indices, clean_ecg, min_distance_samples):
    """
    Voorkomt dubbele QRS-detecties binnen hetzelfde brede complex.
    """
    if len(candidate_indices) == 0:
        return np.array([], dtype=int)

    merged = [int(candidate_indices[0])]

    for candidate_idx in candidate_indices[1:]:
        candidate_idx = int(candidate_idx)
        if candidate_idx - merged[-1] > min_distance_samples:
            merged.append(candidate_idx)
            continue

        if abs(clean_ecg[candidate_idx]) > abs(clean_ecg[merged[-1]]):
            merged[-1] = candidate_idx

    return np.asarray(merged, dtype=int)


# ============================================================
# 4. Nieuwe QRS/P schatting
# ============================================================

def detect_qrs_complexes(
    clean_ecg,
    fs,
    qrs_low=8.0,
    qrs_high=30.0,
    smooth_ms=30,
    threshold_factor=3.0,
    refractory_ms=220,
    refine_search_ms=120,
    merge_ms=120,
    onset_search_back_ms=180,
    offset_search_forward_ms=220,
    local_threshold_ratio=0.15,
):
    """
    Detecteert QRS-complexen opnieuw op basis van lokale energie-activiteit.

    Deze detector is expliciet afgestemd op paced ECG: eerst brede QRS-energie,
    daarna een verfijning naar de dominante deflectie in het echte signaal.
    """
    qrs_band, qrs_energy = build_qrs_detection_signal(
        clean_ecg,
        fs,
        low=qrs_low,
        high=qrs_high,
        smooth_ms=smooth_ms,
    )
    energy_baseline, energy_mad = robust_mad(qrs_energy)
    energy_threshold = energy_baseline + threshold_factor * energy_mad
    refractory_samples = max(int(refractory_ms * fs / 1000), 1)
    refine_samples = max(int(refine_search_ms * fs / 1000), 1)
    merge_samples = max(int(merge_ms * fs / 1000), 1)
    onset_search_back_samples = max(int(onset_search_back_ms * fs / 1000), 1)
    offset_search_forward_samples = max(int(offset_search_forward_ms * fs / 1000), 1)
    prominence_floor = max(1.5 * energy_mad, 0.1 * (np.percentile(qrs_energy, 99) - energy_baseline))

    energy_peaks, _ = signal.find_peaks(
        qrs_energy,
        height=energy_threshold,
        prominence=prominence_floor,
        distance=refractory_samples,
    )

    refined_peak_indices = []

    for energy_peak_idx in energy_peaks:
        start = max(0, int(energy_peak_idx) - refine_samples)
        end = min(len(clean_ecg), int(energy_peak_idx) + refine_samples + 1)
        segment = clean_ecg[start:end]

        if segment.size == 0:
            continue

        local_baseline = np.median(segment)
        local_idx = int(np.argmax(np.abs(segment - local_baseline)))
        refined_peak_indices.append(start + local_idx)

    qrs_peak_indices = merge_nearby_peaks(sorted(refined_peak_indices), clean_ecg, merge_samples)

    if qrs_peak_indices.size == 0:
        empty_int = np.array([], dtype=int)
        empty_float = np.array([], dtype=float)
        return empty_int, empty_int, empty_int, empty_float, empty_float, qrs_energy, energy_threshold

    qrs_onsets = []
    qrs_offsets = []
    qrs_widths_ms = []
    qrs_excursions = []

    for peak_idx in qrs_peak_indices:
        start = max(0, int(peak_idx) - onset_search_back_samples)
        end = min(len(qrs_energy), int(peak_idx) + offset_search_forward_samples + 1)

        pre_segment = qrs_energy[start:int(peak_idx) + 1]
        post_segment = qrs_energy[int(peak_idx):end]

        local_baseline = np.percentile(qrs_energy[start:end], 20)
        local_peak = np.max(qrs_energy[max(0, int(peak_idx) - 3):min(len(qrs_energy), int(peak_idx) + 4)])
        local_threshold = max(
            0.65 * energy_threshold,
            local_baseline + local_threshold_ratio * (local_peak - local_baseline),
        )

        below_pre = np.flatnonzero(pre_segment < local_threshold)
        onset_idx = start if below_pre.size == 0 else start + int(below_pre[-1]) + 1
        onset_idx = min(onset_idx, int(peak_idx))

        below_post = np.flatnonzero(post_segment < local_threshold)
        offset_idx = end - 1 if below_post.size == 0 else int(peak_idx) + int(below_post[0])
        offset_idx = max(offset_idx, int(peak_idx))

        qrs_onsets.append(onset_idx)
        qrs_offsets.append(offset_idx)

        qrs_segment = clean_ecg[onset_idx:offset_idx + 1]
        qrs_widths_ms.append((offset_idx - onset_idx) * 1000.0 / fs)
        qrs_excursions.append(np.ptp(qrs_segment) if qrs_segment.size > 0 else 0.0)

    return (
        qrs_peak_indices.astype(int),
        np.asarray(qrs_onsets, dtype=int),
        np.asarray(qrs_offsets, dtype=int),
        np.asarray(qrs_widths_ms, dtype=float),
        np.asarray(qrs_excursions, dtype=float),
        qrs_energy,
        energy_threshold,
    )


def build_p_wave_signal(clean_ecg, fs, low=0.5, high=10.0, order=2, smooth_ms=35):
    """
    Bouwt een glad P-golf-signaal waarin kleine positieve P-toppen beter
    zichtbaar zijn dan in het ruwe, driftgevoelige ECG.
    """
    p_band = bandpass_ecg(clean_ecg, fs, low=low, high=high, order=order)
    return smooth_signal(p_band, fs, win_ms=smooth_ms)


def estimate_qr_landmarks(clean_ecg, qrs_onset_indices, qrs_offset_indices):
    """
    Bepaalt per QRS-complex een positieve R-landmark en het begin van de
    eerste negatieve Q-deflectie.

    In deze paced morfologie ligt de diepste negatieve trough vaak te laat.
    Daarom gebruiken we als Q-landmark het eerste duidelijke begin van de
    neergaande negatieve deflectie na de R-piek.
    """
    qrs_onset_indices = np.asarray(qrs_onset_indices, dtype=int)
    qrs_offset_indices = np.asarray(qrs_offset_indices, dtype=int)

    q_landmark_indices = []
    r_peak_indices = []

    for onset_idx, offset_idx in zip(qrs_onset_indices, qrs_offset_indices):
        segment = clean_ecg[int(onset_idx):int(offset_idx) + 1]
        if segment.size == 0:
            q_landmark_indices.append(int(onset_idx))
            r_peak_indices.append(int(onset_idx))
            continue

        local_r_idx = int(np.argmax(segment))
        r_peak_indices.append(int(onset_idx) + local_r_idx)

        q_landmark_local_idx = local_r_idx
        qrs_excursion = float(np.ptp(segment))
        sharp_drop_threshold = -max(0.12 * qrs_excursion, 8.0)

        if local_r_idx < segment.size - 1:
            post_r_segment = segment[local_r_idx:]
            post_r_diff = np.diff(post_r_segment)
            sharp_drop_indices = np.flatnonzero(post_r_diff <= sharp_drop_threshold)

            if sharp_drop_indices.size > 0:
                q_landmark_local_idx = local_r_idx + int(sharp_drop_indices[0]) + 1
            else:
                negative_candidates = np.flatnonzero(post_r_segment[1:] < 0)
                if negative_candidates.size > 0:
                    q_landmark_local_idx = local_r_idx + int(negative_candidates[0]) + 1

        q_landmark_indices.append(int(onset_idx) + q_landmark_local_idx)

    return (
        np.asarray(q_landmark_indices, dtype=int),
        np.asarray(r_peak_indices, dtype=int),
    )


def measure_local_positive_height(reference_signal, peak_idx, fs, flank_ms=80):
    """
    Schat hoe hoog een top lokaal boven zijn omgeving uitkomt.
    """
    flank_samples = max(int(flank_ms * fs / 1000), 1)
    start = max(0, int(peak_idx) - flank_samples)
    end = min(len(reference_signal), int(peak_idx) + flank_samples + 1)
    baseline = np.median(reference_signal[start:end])
    return float(reference_signal[int(peak_idx)] - baseline)


def find_positive_peak_candidate(
    filtered_signal,
    reference_signal,
    start_idx,
    end_idx,
    fs,
    min_prominence_abs=0.35,
    min_height_abs=0.2,
    peak_distance_ms=35,
    min_width_ms=18,
    max_width_ms=140,
    prefer="last",
):
    """
    Zoekt een positieve P-kandidaat in een afgebakend tijdvenster.
    """
    start_idx = max(int(start_idx), 0)
    end_idx = min(int(end_idx), len(filtered_signal))

    if end_idx <= start_idx:
        return None

    segment = filtered_signal[start_idx:end_idx]
    if segment.size == 0:
        return None

    centered = segment - np.median(segment)
    if np.max(centered) <= min_height_abs:
        return None

    peak_distance = max(int(peak_distance_ms * fs / 1000), 1)
    min_width = max(int(min_width_ms * fs / 1000), 1)
    max_width = max(int(max_width_ms * fs / 1000), min_width + 1)

    peaks, properties = signal.find_peaks(
        centered,
        height=min_height_abs,
        prominence=min_prominence_abs,
        distance=peak_distance,
        width=(min_width, max_width),
    )

    if peaks.size == 0:
        return None

    peak_heights = properties["peak_heights"]
    prominences = properties["prominences"]
    raw_heights = np.array([
        measure_local_positive_height(reference_signal, start_idx + local_idx, fs)
        for local_idx in peaks
    ])

    valid_mask = raw_heights > 0
    if not np.any(valid_mask):
        return None

    candidate_indices = np.flatnonzero(valid_mask)

    if prefer == "last":
        strong_mask = (
            (prominences >= 0.6 * np.max(prominences))
            | (raw_heights >= 0.6 * np.max(raw_heights))
        )
        candidate_indices = np.flatnonzero(valid_mask & strong_mask)
        if candidate_indices.size == 0:
            candidate_indices = np.flatnonzero(valid_mask)
        chosen_pos = int(candidate_indices[-1])
    else:
        scores = prominences + 0.35 * peak_heights + 0.25 * raw_heights
        scores[~valid_mask] = -np.inf
        chosen_pos = int(np.argmax(scores))

    best_local_idx = int(peaks[chosen_pos])

    return {
        "index": start_idx + best_local_idx,
        "height": float(peak_heights[chosen_pos]),
        "prominence": float(prominences[chosen_pos]),
        "raw_height": float(raw_heights[chosen_pos]),
        "width_samples": float(properties["widths"][chosen_pos]),
    }


def estimate_p_peaks(
    clean_ecg,
    p_wave_signal,
    qrs_onset_indices,
    qrs_offset_indices,
    qrs_excursions,
    fs,
    p_search_back_ms=320,
    p_qrs_guard_ms=60,
    previous_qrs_guard_ms=45,
    min_prominence_abs=0.35,
    min_height_abs=0.2,
    raw_height_min=0.35,
    raw_height_ratio=0.004,
):
    """
    Schat per beat een P-top in het pre-QRS venster.

    Deze detector zoekt alleen in het fysiologisch logische gebied voor de
    volgende QRS en valideert de kandidaat ook op ruwe lokale hoogte.
    """
    qrs_onset_indices = np.asarray(qrs_onset_indices, dtype=int)
    qrs_offset_indices = np.asarray(qrs_offset_indices, dtype=int)
    qrs_excursions = np.asarray(qrs_excursions, dtype=float)
    p_search_back_samples = max(int(p_search_back_ms * fs / 1000), 1)
    p_qrs_guard_samples = max(int(p_qrs_guard_ms * fs / 1000), 1)
    previous_qrs_guard_samples = max(int(previous_qrs_guard_ms * fs / 1000), 1)

    p_peaks = []

    for qrs_pos, qrs_onset_idx in enumerate(qrs_onset_indices):
        start = max(0, int(qrs_onset_idx) - p_search_back_samples)
        if qrs_pos > 0:
            start = max(start, int(qrs_offset_indices[qrs_pos - 1]) + previous_qrs_guard_samples)

        end = max(start + 1, int(qrs_onset_idx) - p_qrs_guard_samples)
        if end <= start:
            continue

        candidate = find_positive_peak_candidate(
            p_wave_signal,
            clean_ecg,
            start_idx=start,
            end_idx=end,
            fs=fs,
            min_prominence_abs=min_prominence_abs,
            min_height_abs=min_height_abs,
            prefer="last",
        )

        if candidate is None:
            continue

        min_raw_height = max(raw_height_min, raw_height_ratio * qrs_excursions[qrs_pos])
        if candidate["raw_height"] < min_raw_height:
            continue

        p_peaks.append(candidate["index"])

    return np.unique(np.asarray(p_peaks, dtype=int))


# ============================================================
# 5. Classificatie van pacing-momenten
# ============================================================

def classify_pacing_events(
    artifact_events,
    clean_ecg,
    p_wave_signal,
    qrs_peak_indices,
    q_landmark_indices,
    r_peak_indices,
    qrs_onset_indices,
    qrs_offset_indices,
    qrs_widths_ms,
    qrs_excursions,
    p_peak_indices,
    fs,
    max_cycle_pre_ms=420,
    qrs_error_pre_ms=35,
    qrs_error_post_ms=40,
    q_landmark_tolerance_ms=0,
    q_error_narrow_qrs_ms=80,
    q_error_narrow_min_qrs_amp=170,
    q_error_low_amp_qrs_ms=110,
    q_error_low_amp_max_qrs_amp=120,
    q_error_low_amp_delay_ms=15,
    atrial_min_delay_ms=15,
    atrial_max_delay_ms=140,
    atrial_same_peak_max_ms=20,
    p_qrs_guard_ms=60,
    p_min_height_abs=0.2,
    p_min_prominence_abs=0.35,
    raw_p_height_min=0.35,
    raw_p_height_ratio=0.004,
    inter_spike_guard_ms=20,
    ventricular_min_delay_ms=5,
    ventricular_max_delay_ms=180,
    ventricular_peak_max_ms=280,
    ventricular_wide_qrs_ms=115,
    ventricular_large_qrs_amp=170,
    fallback_ventricular_max_delay_ms=190,
    fallback_ventricular_min_qrs_width_ms=60,
    fallback_ventricular_min_qrs_amp=120,
):
    """
    Classificeert spikes opnieuw rond bekende stimulatiemomenten.

    Regels:
    - `pacingfout`: spike valt na het begin van de Q-deflectie binnen een QRS-complex
    - `atriaal`: na de spike volgt lokaal een P-top voor de volgende QRS
    - `ventriculair`: spike ligt na de P-top en voor de volgende brede/grote QRS
    - `onbekend`: geen betrouwbare match
    """
    qrs_peak_indices = np.asarray(qrs_peak_indices, dtype=int)
    q_landmark_indices = np.asarray(q_landmark_indices, dtype=int)
    r_peak_indices = np.asarray(r_peak_indices, dtype=int)
    qrs_onset_indices = np.asarray(qrs_onset_indices, dtype=int)
    qrs_offset_indices = np.asarray(qrs_offset_indices, dtype=int)
    qrs_widths_ms = np.asarray(qrs_widths_ms, dtype=float)
    qrs_excursions = np.asarray(qrs_excursions, dtype=float)
    p_peak_indices = np.asarray(p_peak_indices, dtype=int)
    artifact_center_indices = event_indices(artifact_events, "center_index")

    max_cycle_pre_samples = max(int(max_cycle_pre_ms * fs / 1000), 1)
    qrs_error_pre_samples = max(int(qrs_error_pre_ms * fs / 1000), 1)
    qrs_error_post_samples = max(int(qrs_error_post_ms * fs / 1000), 1)
    q_landmark_tolerance_samples = max(int(q_landmark_tolerance_ms * fs / 1000), 0)
    q_error_low_amp_delay_samples = max(int(q_error_low_amp_delay_ms * fs / 1000), 0)
    atrial_min_delay_samples = max(int(atrial_min_delay_ms * fs / 1000), 1)
    atrial_max_delay_samples = max(int(atrial_max_delay_ms * fs / 1000), 1)
    atrial_same_peak_max_samples = max(int(atrial_same_peak_max_ms * fs / 1000), 0)
    p_qrs_guard_samples = max(int(p_qrs_guard_ms * fs / 1000), 1)
    inter_spike_guard_samples = max(int(inter_spike_guard_ms * fs / 1000), 1)
    ventricular_min_samples = max(int(ventricular_min_delay_ms * fs / 1000), 1)
    ventricular_max_samples = max(int(ventricular_max_delay_ms * fs / 1000), 1)
    ventricular_peak_max_samples = int(ventricular_peak_max_ms * fs / 1000)
    fallback_ventricular_max_delay_samples = max(int(fallback_ventricular_max_delay_ms * fs / 1000), 1)

    pacing_events = []

    for event_pos, artifact_event in enumerate(artifact_events):
        spike_idx = artifact_event["center_index"]
        prev_pos = np.searchsorted(qrs_onset_indices, spike_idx, side="right") - 1
        next_pos = np.searchsorted(qrs_onset_indices, spike_idx, side="left")
        next_spike_idx = None

        if event_pos + 1 < len(artifact_center_indices):
            next_spike_idx = int(artifact_center_indices[event_pos + 1])

        label = "onbekend"
        related_qrs_idx = None
        related_p_idx = None

        if prev_pos >= 0 and spike_idx <= qrs_offset_indices[prev_pos] + qrs_error_post_samples:
            if spike_idx >= qrs_onset_indices[prev_pos] - qrs_error_pre_samples:
                related_qrs_idx = qrs_peak_indices[prev_pos]
                q_landmark_idx = int(q_landmark_indices[prev_pos])
                prev_qrs_width_ms = float(qrs_widths_ms[prev_pos])
                prev_qrs_excursion = float(qrs_excursions[prev_pos])
                dt_from_q_landmark = spike_idx - q_landmark_idx

                if spike_idx <= q_landmark_idx + q_landmark_tolerance_samples:
                    label = "ventriculair"
                elif (
                    (
                        prev_qrs_width_ms <= q_error_narrow_qrs_ms
                        and prev_qrs_excursion >= q_error_narrow_min_qrs_amp
                    )
                    or (
                        prev_qrs_width_ms <= q_error_low_amp_qrs_ms
                        and prev_qrs_excursion < q_error_low_amp_max_qrs_amp
                        and dt_from_q_landmark >= q_error_low_amp_delay_samples
                    )
                ):
                    label = "pacingfout"
                else:
                    label = "ventriculair"

        if label == "onbekend" and next_pos < len(qrs_onset_indices):
            next_qrs_onset = int(qrs_onset_indices[next_pos])
            next_qrs_peak = int(qrs_peak_indices[next_pos])
            next_qrs_width_ms = float(qrs_widths_ms[next_pos])
            next_qrs_excursion = float(qrs_excursions[next_pos])
            dt_to_qrs = next_qrs_onset - spike_idx
            dt_to_qrs_peak = next_qrs_peak - spike_idx
            ventricular_qrs = (
                next_qrs_width_ms >= ventricular_wide_qrs_ms
                or next_qrs_excursion >= ventricular_large_qrs_amp
            )

            cycle_start = max(0, next_qrs_onset - max_cycle_pre_samples)
            if next_pos > 0:
                cycle_start = max(
                    cycle_start,
                    int(qrs_offset_indices[next_pos - 1]) + inter_spike_guard_samples,
                )

            cycle_p_candidates = p_peak_indices[
                (p_peak_indices >= cycle_start) &
                (p_peak_indices < next_qrs_onset - p_qrs_guard_samples)
            ]
            cycle_p_idx = int(cycle_p_candidates[-1]) if cycle_p_candidates.size > 0 else None

            local_p_end = min(
                spike_idx + atrial_max_delay_samples,
                next_qrs_onset - p_qrs_guard_samples,
            )
            if next_spike_idx is not None and next_spike_idx < next_qrs_onset:
                local_p_end = min(local_p_end, next_spike_idx - inter_spike_guard_samples)

            local_p_candidate = find_positive_peak_candidate(
                p_wave_signal,
                clean_ecg,
                start_idx=spike_idx + atrial_min_delay_samples,
                end_idx=local_p_end,
                fs=fs,
                min_prominence_abs=p_min_prominence_abs,
                min_height_abs=p_min_height_abs,
                prefer="strongest",
            )

            if local_p_candidate is not None:
                min_raw_height = max(raw_p_height_min, raw_p_height_ratio * next_qrs_excursion)
                if local_p_candidate["raw_height"] < min_raw_height:
                    local_p_candidate = None

            if (
                local_p_candidate is not None
                and next_qrs_onset - int(local_p_candidate["index"]) >= p_qrs_guard_samples
            ):
                label = "atriaal"
                related_qrs_idx = next_qrs_peak
                related_p_idx = int(local_p_candidate["index"])

            elif cycle_p_idx is not None:
                dt_to_cycle_p = cycle_p_idx - spike_idx
                cycle_p_before_next_spike = (
                    next_spike_idx is None
                    or next_spike_idx >= next_qrs_onset
                    or cycle_p_idx <= next_spike_idx - inter_spike_guard_samples
                )

                if (
                    cycle_p_before_next_spike
                    and atrial_min_delay_samples <= dt_to_cycle_p <= atrial_max_delay_samples
                ):
                    label = "atriaal"
                    related_qrs_idx = next_qrs_peak
                    related_p_idx = cycle_p_idx
                elif (
                    cycle_p_before_next_spike
                    and 0 <= dt_to_cycle_p <= atrial_same_peak_max_samples
                    and next_qrs_onset - cycle_p_idx >= p_qrs_guard_samples
                ):
                    label = "atriaal"
                    related_qrs_idx = next_qrs_peak
                    related_p_idx = cycle_p_idx

            if label == "onbekend":
                if (
                    cycle_p_idx is not None
                    and cycle_p_idx < spike_idx
                    and ventricular_min_samples <= dt_to_qrs <= ventricular_max_samples
                ):
                    label = "ventriculair"
                    related_qrs_idx = next_qrs_peak
                    related_p_idx = cycle_p_idx
                elif (
                    ventricular_min_samples <= dt_to_qrs <= ventricular_max_samples
                    and ventricular_qrs
                ):
                    label = "ventriculair"
                    related_qrs_idx = next_qrs_peak
                elif (
                    0 <= dt_to_qrs_peak <= ventricular_peak_max_samples
                    and ventricular_qrs
                ):
                    label = "ventriculair"
                    related_qrs_idx = next_qrs_peak
                elif (
                    dt_to_qrs <= fallback_ventricular_max_delay_samples
                    and (
                        next_qrs_width_ms >= fallback_ventricular_min_qrs_width_ms
                        or next_qrs_excursion >= fallback_ventricular_min_qrs_amp
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


def plot_artifact_removal(
    t,
    ecg,
    clean_ecg,
    corrected_ecg,
    artifact_events,
    artifact_mask,
    plot_mask,
):
    """
    Tweede plot: interpolatie en baseline-correctie na verwijdering van
    pacemaker-spikes.
    """
    trough_indices = event_indices(artifact_events, "trough_index")

    plt.figure(figsize=(12, 4))
    plt.plot(t[plot_mask], ecg[plot_mask], label="Raw ECG", alpha=0.3)
    plt.plot(t[plot_mask], clean_ecg[plot_mask], label="Interpolated ECG", alpha=0.35)
    plt.plot(
        t[plot_mask],
        corrected_ecg[plot_mask],
        label="Interpolated + baseline-corrected ECG",
        alpha=0.9,
    )

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
        corrected_ecg[plot_mask][~artifact_mask[plot_mask]],
        ".",
        markersize=2.5,
        label="Interpolated samples",
    )

    plt.title("ECG na piekverwijdering, interpolatie en baseline-correctie")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pacing_labels(
    t,
    analysis_ecg,
    pacing_events,
    qrs_peak_indices,
    p_peak_indices,
    plot_mask,
):
    """
    Derde plot: pacing-labels op de geinterpoleerde ECG.
    """
    plt.figure(figsize=(14, 5))
    plt.plot(
        t[plot_mask],
        analysis_ecg[plot_mask],
        label="Interpolated + baseline-corrected ECG",
        alpha=0.9,
    )

    visible_qrs = qrs_peak_indices[plot_mask[qrs_peak_indices]] if len(qrs_peak_indices) > 0 else qrs_peak_indices
    visible_p = p_peak_indices[plot_mask[p_peak_indices]] if len(p_peak_indices) > 0 else p_peak_indices

    if visible_qrs.size > 0:
        plt.plot(
            t[visible_qrs],
            analysis_ecg[visible_qrs],
            "k.",
            markersize=3,
            alpha=0.45,
            label="QRS peak",
        )

    if visible_p.size > 0:
        plt.plot(
            t[visible_p],
            analysis_ecg[visible_p],
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
            analysis_ecg[spike_idx],
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
    qrs_low = 8
    qrs_high = 30
    qrs_smooth_ms = 30
    qrs_threshold_factor = 3.0
    qrs_refractory_ms = 220
    qrs_refine_search_ms = 120
    qrs_merge_ms = 120
    qrs_local_threshold_ratio = 0.15
    baseline_cutoff_hz = 0.5

    # P-top schatting
    p_search_back_ms = 320
    p_qrs_guard_ms = 60
    p_previous_qrs_guard_ms = 45
    p_min_height_abs = 0.2
    p_min_prominence_abs = 0.35
    raw_p_height_min = 0.35
    raw_p_height_ratio = 0.004
    inter_spike_guard_ms = 20

    # Classificatie van pacing-momenten
    max_cycle_pre_ms = 420
    qrs_error_pre_ms = 35
    qrs_error_post_ms = 40
    q_landmark_tolerance_ms = 0
    q_error_narrow_qrs_ms = 80
    q_error_narrow_min_qrs_amp = 170
    q_error_low_amp_qrs_ms = 110
    q_error_low_amp_max_qrs_amp = 120
    q_error_low_amp_delay_ms = 15
    atrial_min_delay_ms = 15
    atrial_max_delay_ms = 140
    atrial_same_peak_max_ms = 20
    ventricular_min_delay_ms = 5
    ventricular_max_delay_ms = 180
    ventricular_peak_max_ms = 280
    ventricular_wide_qrs_ms = 115
    ventricular_large_qrs_amp = 170
    fallback_ventricular_max_delay_ms = 190
    fallback_ventricular_min_qrs_width_ms = 60
    fallback_ventricular_min_qrs_amp = 120

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

    analysis_ecg, estimated_baseline = remove_baseline_wander(
        clean_ecg,
        fs,
        cutoff_hz=baseline_cutoff_hz,
    )

    plot_artifact_removal(
        t=t,
        ecg=ecg,
        clean_ecg=clean_ecg,
        corrected_ecg=analysis_ecg,
        artifact_events=artifact_events,
        artifact_mask=artifact_mask,
        plot_mask=plot_mask,
    )

    # -----------------------------
    # 4. Nieuwe detectie van QRS en P-toppen
    # -----------------------------
    (
        qrs_peak_indices,
        qrs_onset_indices,
        qrs_offset_indices,
        qrs_widths_ms,
        qrs_excursions,
        qrs_energy,
        qrs_energy_threshold,
    ) = detect_qrs_complexes(
        analysis_ecg,
        fs,
        qrs_low=qrs_low,
        qrs_high=qrs_high,
        smooth_ms=qrs_smooth_ms,
        threshold_factor=qrs_threshold_factor,
        refractory_ms=qrs_refractory_ms,
        refine_search_ms=qrs_refine_search_ms,
        merge_ms=qrs_merge_ms,
        local_threshold_ratio=qrs_local_threshold_ratio,
    )

    p_wave_signal = build_p_wave_signal(analysis_ecg, fs)
    p_peak_indices = estimate_p_peaks(
        clean_ecg=analysis_ecg,
        p_wave_signal=p_wave_signal,
        qrs_onset_indices=qrs_onset_indices,
        qrs_offset_indices=qrs_offset_indices,
        qrs_excursions=qrs_excursions,
        fs=fs,
        p_search_back_ms=p_search_back_ms,
        p_qrs_guard_ms=p_qrs_guard_ms,
        previous_qrs_guard_ms=p_previous_qrs_guard_ms,
        min_prominence_abs=p_min_prominence_abs,
        min_height_abs=p_min_height_abs,
        raw_height_min=raw_p_height_min,
        raw_height_ratio=raw_p_height_ratio,
    )

    q_landmark_indices, r_peak_indices = estimate_qr_landmarks(
        analysis_ecg,
        qrs_onset_indices,
        qrs_offset_indices,
    )

    pacing_events = classify_pacing_events(
        artifact_events=artifact_events,
        clean_ecg=analysis_ecg,
        p_wave_signal=p_wave_signal,
        qrs_peak_indices=qrs_peak_indices,
        q_landmark_indices=q_landmark_indices,
        r_peak_indices=r_peak_indices,
        qrs_onset_indices=qrs_onset_indices,
        qrs_offset_indices=qrs_offset_indices,
        qrs_widths_ms=qrs_widths_ms,
        qrs_excursions=qrs_excursions,
        p_peak_indices=p_peak_indices,
        fs=fs,
        max_cycle_pre_ms=max_cycle_pre_ms,
        qrs_error_pre_ms=qrs_error_pre_ms,
        qrs_error_post_ms=qrs_error_post_ms,
        q_landmark_tolerance_ms=q_landmark_tolerance_ms,
        q_error_narrow_qrs_ms=q_error_narrow_qrs_ms,
        q_error_narrow_min_qrs_amp=q_error_narrow_min_qrs_amp,
        q_error_low_amp_qrs_ms=q_error_low_amp_qrs_ms,
        q_error_low_amp_max_qrs_amp=q_error_low_amp_max_qrs_amp,
        q_error_low_amp_delay_ms=q_error_low_amp_delay_ms,
        atrial_min_delay_ms=atrial_min_delay_ms,
        atrial_max_delay_ms=atrial_max_delay_ms,
        atrial_same_peak_max_ms=atrial_same_peak_max_ms,
        p_qrs_guard_ms=p_qrs_guard_ms,
        p_min_height_abs=p_min_height_abs,
        p_min_prominence_abs=p_min_prominence_abs,
        raw_p_height_min=raw_p_height_min,
        raw_p_height_ratio=raw_p_height_ratio,
        inter_spike_guard_ms=inter_spike_guard_ms,
        ventricular_min_delay_ms=ventricular_min_delay_ms,
        ventricular_max_delay_ms=ventricular_max_delay_ms,
        ventricular_peak_max_ms=ventricular_peak_max_ms,
        ventricular_wide_qrs_ms=ventricular_wide_qrs_ms,
        ventricular_large_qrs_amp=ventricular_large_qrs_amp,
        fallback_ventricular_max_delay_ms=fallback_ventricular_max_delay_ms,
        fallback_ventricular_min_qrs_width_ms=fallback_ventricular_min_qrs_width_ms,
        fallback_ventricular_min_qrs_amp=fallback_ventricular_min_qrs_amp,
    )

    plot_pacing_labels(
        t=t,
        analysis_ecg=analysis_ecg,
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
    print(f"QRS energy threshold: {qrs_energy_threshold:.3f}")
    print(f"Baseline high-pass cutoff: {baseline_cutoff_hz:.2f} Hz")

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
