import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from collections import Counter

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
    ecg, fs, _ = read_ecg_mat(path, plotresult=plotresult)
    t = np.arange(len(ecg)) / fs
    plot_mask = (t >= plot_start) & (t <= plot_end)
    return ecg, fs, t, plot_mask


# ============================================================
# 2. Verwijderen van pacemaker-artefacten
# ============================================================

def remove_pacemaker_artifacts(ecg, threshold=-1000, max_extension=6):
    below_threshold = np.flatnonzero(ecg < threshold)

    if below_threshold.size == 0:
        return ecg.copy(), np.array([], dtype=int), np.ones(len(ecg), dtype=bool)

    split_points = np.where(np.diff(below_threshold) > 1)[0] + 1
    groups = np.split(below_threshold, split_points)

    trough_indices = []
    mask = np.ones(len(ecg), dtype=bool)

    for group in groups:
        trough_idx = group[np.argmin(ecg[group])]
        trough_indices.append(trough_idx)

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
    removed_idx = np.flatnonzero(~mask)

    if valid_idx.size >= 2:
        clean_ecg[removed_idx] = np.interp(removed_idx, valid_idx, ecg[valid_idx])
    elif valid_idx.size == 1:
        clean_ecg[removed_idx] = ecg[valid_idx[0]]

    return clean_ecg, np.array(trough_indices, dtype=int), mask


# ============================================================
# 3. Voorbewerking voor QRS-detectie
# ============================================================

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
    y_bp  = bandpass_ecg(ecg, fs, low=bp_low, high=bp_high, order=3)
    y_der = derivative_filter(y_bp)
    y_sq  = y_der ** 2
    y_mwi = moving_window_integration(y_sq, fs, win_ms=mwi_ms)
    return y_bp, y_der, y_sq, y_mwi


# ============================================================
# 4. QRS-detectie
# ============================================================

def detect_qrs_peaks_on_mwi(y_mwi, fs, prominence_factor=0.05, refractory_ms=350):
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
    if len(peaks) < 2:
        return np.nan, np.nan, np.array([])
    rr = np.diff(t[peaks])
    return np.mean(rr), 60.0 / np.mean(rr), rr


# ============================================================
# 5. QRS/P schatting en classificatie van pacing-momenten
# ============================================================

def refine_qrs_peaks_to_ecg(clean_ecg, qrs_indices, fs,
                              search_back_ms=180, search_forward_ms=60):
    qrs_indices = np.asarray(qrs_indices, dtype=int)
    sb = int(search_back_ms    * fs / 1000)
    sf = int(search_forward_ms * fs / 1000)
    refined = []
    for idx in qrs_indices:
        start = max(0, idx - sb)
        end   = min(len(clean_ecg), idx + sf + 1)
        seg   = clean_ecg[start:end]
        if seg.size == 0:
            continue
        baseline  = np.median(seg[:max(3, seg.size // 4)])
        local_idx = int(np.argmax(np.abs(seg - baseline)))
        refined.append(start + local_idx)
    return np.asarray(refined, dtype=int)


def estimate_qrs_onsets(clean_ecg, qrs_peak_indices, fs,
                         search_back_ms=180, threshold_ratio=0.12):
    qrs_peak_indices = np.asarray(qrs_peak_indices, dtype=int)
    sb = int(search_back_ms * fs / 1000)

    qrs_band     = bandpass_ecg(clean_ecg, fs, low=5.0, high=25.0, order=2)
    qrs_envelope = np.abs(qrs_band)
    win = max(int(20 * fs / 1000), 1)
    qrs_envelope = np.convolve(qrs_envelope, np.ones(win) / win, mode="same")

    onsets = []
    for peak_idx in qrs_peak_indices:
        start   = max(0, peak_idx - sb)
        segment = qrs_envelope[start:peak_idx + 1]
        if segment.size == 0:
            onsets.append(peak_idx)
            continue
        thresh = threshold_ratio * np.max(segment)
        above  = np.flatnonzero(segment >= thresh)
        onsets.append(peak_idx if above.size == 0 else start + int(above[0]))
    return np.asarray(onsets, dtype=int)


def estimate_p_peaks(clean_ecg, qrs_indices, fs,
                      p_search_start_ms=250, p_search_end_ms=60):
    qrs_indices = np.asarray(qrs_indices, dtype=int)
    ss = int(p_search_start_ms * fs / 1000)
    se = int(p_search_end_ms   * fs / 1000)
    p_band  = bandpass_ecg(clean_ecg, fs, low=0.5, high=12.0, order=2)
    p_peaks = []
    for qrs_idx in qrs_indices:
        start = max(0, qrs_idx - ss)
        end   = max(start + 1, qrs_idx - se)
        if end <= start:
            continue
        seg = p_band[start:end]
        if seg.size == 0:
            continue
        p_peaks.append(start + int(np.argmax(np.abs(seg))))
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
    spike_indices     = np.asarray(spike_indices,     dtype=int)
    qrs_peak_indices  = np.asarray(qrs_peak_indices,  dtype=int)
    qrs_onset_indices = np.asarray(qrs_onset_indices, dtype=int)
    p_peak_indices    = np.asarray(p_peak_indices,    dtype=int)

    vm  = int(ventricular_max_ms  * fs / 1000)
    apm = int(atrial_peak_min_ms  * fs / 1000)
    apx = int(atrial_peak_max_ms  * fs / 1000)
    qep = int(qrs_error_post_ms   * fs / 1000)

    events = []

    for spike_idx in spike_indices:
        prev_pos = np.searchsorted(qrs_onset_indices, spike_idx, side="right") - 1
        next_pos = np.searchsorted(qrs_onset_indices, spike_idx, side="left")

        prev_qrs_onset = qrs_onset_indices[prev_pos] if prev_pos >= 0                    else None
        next_qrs_onset = qrs_onset_indices[next_pos] if next_pos < len(qrs_onset_indices) else None
        next_qrs_peak  = qrs_peak_indices[next_pos]  if next_pos < len(qrs_peak_indices)  else None

        label         = None
        related_qrs   = None
        related_p     = None

        # Spike valt vlak ná een QRS-onset → pacingfout (sensing failure)
        if prev_qrs_onset is not None:
            if 0 <= spike_idx - prev_qrs_onset <= qep:
                label       = "pacingfout"
                related_qrs = prev_qrs_onset

        # Spike vlak vóór QRS-onset → ventriculaire pacing
        if label is None and next_qrs_onset is not None:
            dt_before = next_qrs_onset - spike_idx
            if 0 <= dt_before <= vm:
                label       = "ventriculair"
                related_qrs = next_qrs_peak
            else:
                # Zoek P-piek tussen spike en QRS-onset → atriale pacing
                candidates = p_peak_indices[
                    (p_peak_indices > spike_idx) &
                    (p_peak_indices < next_qrs_onset)
                ]
                if candidates.size > 0:
                    p_idx    = candidates[0]
                    dt_to_p  = p_idx - spike_idx
                    if apm <= dt_to_p <= apx:
                        label       = "atriaal"
                        related_qrs = next_qrs_peak
                        related_p   = p_idx

        # Geen QRS volgend → niet gevolgd (geen capture)
        if label is None:
            label = "geen_capture"

        events.append({
            "spike_index":     spike_idx,
            "time_s":          spike_idx / fs,
            "label":           label,
            "related_qrs_index": related_qrs,
            "related_p_index": related_p,
        })

    return events


def summarize_pacing_events(pacing_events):
    counts = Counter(e["label"] for e in pacing_events)
    return dict(counts)


# ============================================================
# 6. Vraag 3 – Pacemaker intervalanalyse (instellingen bepalen)
# ============================================================

def analyseer_pacemaker_intervallen(pacemaker_indices, t, fs):
    """
    Vraag 3: Bepaal de pacemaker-instelling(en) op basis van spike-intervallen.

    Strategie:
    - Bereken alle tijdsverschillen tussen opeenvolgende spikes.
    - Korte intervallen (~100–300 ms) → AV-delay (duaal kamer pacing).
    - Lange intervallen → basisfrequentie van de pacemaker.
    - Variabiliteit in lange intervallen → sensing aanwezig (rate-responsief
      of inhibitie door eigen slagen).
    """
    if len(pacemaker_indices) < 2:
        print("Te weinig spikes voor intervalanalyse.")
        return {}

    spike_t        = t[pacemaker_indices]
    intervallen_ms = np.diff(spike_t) * 1000   # ms

    # ── Splits in korte (AV-delay regio) en lange (basisritme) intervallen ──
    korte = intervallen_ms[intervallen_ms <  350]
    lange = intervallen_ms[intervallen_ms >= 350]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].hist(intervallen_ms, bins=300, color="steelblue", edgecolor="none")
    axes[0].set_xlabel("Interval (ms)")
    axes[0].set_ylabel("Aantal")
    axes[0].set_title("Alle spike-intervallen")
    axes[0].grid(True, alpha=0.3)

    if len(korte) > 0:
        axes[1].hist(korte, bins=60, color="coral", edgecolor="none")
        axes[1].set_xlabel("Interval (ms)")
        axes[1].set_title(f"Korte intervallen <350 ms  (n={len(korte)})\n→ mogelijke AV-delay")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Geen intervallen\n< 350 ms",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Korte intervallen <350 ms")

    if len(lange) > 0:
        axes[2].hist(lange, bins=100, color="mediumseagreen", edgecolor="none")
        axes[2].set_xlabel("Interval (ms)")
        axes[2].set_title(f"Lange intervallen ≥350 ms  (n={len(lange)})\n→ basisfrequentie regio")
        axes[2].grid(True, alpha=0.3)

    plt.suptitle("Vraag 3 – Pacemaker spike-intervallen", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

    # ── Statistieken ────────────────────────────────────────────────────────
    resultaten = {}

    print("\n======== Vraag 3 – Pacemaker intervalanalyse ========")

    if len(lange) > 0:
        basis_ms = np.median(lange)
        basis_bpm = 60_000 / basis_ms
        cv_basis = np.std(lange) / np.mean(lange)   # variatiecoëfficiënt
        resultaten["basis_interval_ms"] = basis_ms
        resultaten["basis_bpm"]         = basis_bpm
        resultaten["cv_basisinterval"]  = cv_basis
        print(f"  Basisinterval (mediaan lange intervallen): {basis_ms:.1f} ms  →  {basis_bpm:.1f} bpm")
        print(f"  Variatiecoëff. basisinterval:              {cv_basis:.3f}  "
              f"({'sensing aanwezig' if cv_basis > 0.05 else 'vaste rate / geen sensing'})")

    if len(korte) > 0:
        av_ms = np.median(korte)
        resultaten["av_delay_ms"] = av_ms
        print(f"  AV-delay (mediaan korte intervallen):      {av_ms:.1f} ms")
        print(f"  → Duaal kamer pacing waarschijnlijk (A→V paar)")
    else:
        print("  Geen korte intervallen → enkamer of alleen ventriculaire pacing")

    return resultaten


# ============================================================
# 7. Vraag 6 – Percentages gepacede slagen
# ============================================================

def bereken_pacing_percentages(pacing_events, qrs_peak_indices):
    """
    Vraag 6: Bereken het percentage atriale en ventriculaire slagen
    dat door de pacemaker is veroorzaakt, en de capture rate.

    Definities:
    - % atriale pacing   = A-spikes / totaal aantal QRS-complexen × 100
      (elke atriale spike levert uiteindelijk een QRS)
    - % ventriculaire pacing = V-spikes / totaal aantal QRS-complexen × 100
    - % capture          = (A + V spikes) / totaal spikes × 100
    """
    labels   = [e["label"] for e in pacing_events]
    n_A      = labels.count("atriaal")
    n_V      = labels.count("ventriculair")
    n_fout   = labels.count("pacingfout")
    n_geen   = labels.count("geen_capture")
    n_spikes = len(pacing_events)
    n_qrs    = len(qrs_peak_indices)

    pct_A_van_qrs  = 100 * n_A / n_qrs    if n_qrs    > 0 else float("nan")
    pct_V_van_qrs  = 100 * n_V / n_qrs    if n_qrs    > 0 else float("nan")
    pct_capture    = 100 * (n_A + n_V) / n_spikes if n_spikes > 0 else float("nan")

    print("\n======== Vraag 6 – Pacing percentages ========")
    print(f"  Totaal QRS-complexen:              {n_qrs:6d}")
    print(f"  Totaal pacemaker-spikes:           {n_spikes:6d}")
    print(f"    waarvan atriaal (A):             {n_A:6d}  →  {pct_A_van_qrs:5.1f}% van QRS")
    print(f"    waarvan ventriculair (V):        {n_V:6d}  →  {pct_V_van_qrs:5.1f}% van QRS")
    print(f"    waarvan pacingfout:              {n_fout:6d}")
    print(f"    waarvan geen capture:            {n_geen:6d}")
    print(f"  Capture rate (A+V / totaal spikes):{pct_capture:5.1f}%")

    return {
        "n_QRS":        n_qrs,
        "n_spikes":     n_spikes,
        "n_A":          n_A,
        "n_V":          n_V,
        "n_pacingfout": n_fout,
        "n_geen_cap":   n_geen,
        "pct_A_van_QRS":   pct_A_van_qrs,
        "pct_V_van_QRS":   pct_V_van_qrs,
        "pct_capture":  pct_capture,
    }


# ============================================================
# 8. Vraag 7 – Pacemaker modus detecteren
# ============================================================

def detecteer_pacemaker_modus(interval_resultaten, pct_resultaten):
    """
    Vraag 7: Bepaal automatisch de meest waarschijnlijke pacemaker-modus
    (NASPE/BPEG-code) op basis van de telemetriedata.

    NASPE/BPEG positie:
        I   – kamer die gestimuleerd wordt  (O/A/V/D)
        II  – kamer die gevoeld wordt       (O/A/V/D)
        III – respons op sensing            (O/I/T/D)

    Gebruikte signalen:
    - n_A, n_V, n_geen_cap  → welke kamers worden gestimuleerd?
    - cv_basisinterval      → is er sensing (hoge cv → inhibitie aanwezig)?
    - av_delay_ms           → is er een AV-delay (duaal kamer)?
    - pct_capture           → worden spikes gevolgd door complexen?
    """
    n_A    = pct_resultaten.get("n_A", 0)
    n_V    = pct_resultaten.get("n_V", 0)
    n_geen = pct_resultaten.get("n_geen_cap", 0)
    n_tot  = pct_resultaten.get("n_spikes", 1)

    pct_A = 100 * n_A / n_tot if n_tot > 0 else 0
    pct_V = 100 * n_V / n_tot if n_tot > 0 else 0

    cv_basis = interval_resultaten.get("cv_basisinterval", 0)
    av_delay = interval_resultaten.get("av_delay_ms",      None)
    basis_bpm = interval_resultaten.get("basis_bpm",       None)

    sensing_aanwezig = cv_basis > 0.05
    duaal_kamer      = (av_delay is not None) and (pct_A > 5) and (pct_V > 5)

    mogelijke_modi = []
    onderbouwing   = []

    # ── Beslisboom ──────────────────────────────────────────────────────────
    if duaal_kamer:
        onderbouwing.append(f"  ✓ Zowel A ({pct_A:.0f}%) als V ({pct_V:.0f}%) spikes → duaal kamer pacing")
        onderbouwing.append(f"  ✓ AV-delay ~{av_delay:.0f} ms gedetecteerd")

        if sensing_aanwezig:
            mogelijke_modi.append("DDD")
            onderbouwing.append(f"  ✓ Variabele basisintervallen (CV={cv_basis:.3f}) → sensing aanwezig → DDD")
            onderbouwing.append("     Positie I=D (stimuleert A+V), II=D (voelt A+V), III=D (inhibitie + trigger)")
        else:
            mogelijke_modi.extend(["DOO", "DDD"])
            onderbouwing.append(f"  ~ Vaste basisintervallen (CV={cv_basis:.3f}) → mogelijk geen sensing → DOO")
            onderbouwing.append("     Maar DDD met volledig gepace ritme ziet er identiek uit → beide mogelijk")

    elif pct_V > 80:
        onderbouwing.append(f"  ✓ Voornamelijk ventriculaire spikes ({pct_V:.0f}%)")

        if sensing_aanwezig:
            mogelijke_modi.extend(["VVI", "VDD"])
            onderbouwing.append(f"  ✓ Variabele intervallen (CV={cv_basis:.3f}) → sensing aanwezig")
            onderbouwing.append("     VVI: voelt V, inhibitie bij eigen slag")
            onderbouwing.append("     VDD: voelt A+V, V getriggerd door eigen P-golf (geen A-spikes zichtbaar)")
        else:
            mogelijke_modi.append("VOO")
            onderbouwing.append(f"  ~ Vaste intervallen (CV={cv_basis:.3f}) → vaste rate, geen sensing → VOO")

    elif pct_A > 80:
        onderbouwing.append(f"  ✓ Voornamelijk atriale spikes ({pct_A:.0f}%)")

        if sensing_aanwezig:
            mogelijke_modi.append("AAI")
            onderbouwing.append(f"  ✓ Variabele intervallen → sensing → AAI")
        else:
            mogelijke_modi.extend(["AOO", "AAI"])
            onderbouwing.append("  ~ Vaste intervallen → AOO of AAI met volledig gepace ritme")

    else:
        mogelijke_modi.append("onbepaald")
        onderbouwing.append("  ⚠  Gemengd patroon zonder duidelijke meerderheid → handmatige inspectie nodig")

    # Niet gevolgd door complex → capture failure of sensing failure
    if n_geen > 0.05 * n_tot:
        onderbouwing.append(f"\n  ⚠  {n_geen} spikes zonder volgend complex ({100*n_geen/n_tot:.1f}%)")
        onderbouwing.append("     → mogelijke capture failure of output te laag")

    # ── Output ──────────────────────────────────────────────────────────────
    print("\n======== Vraag 7 – Pacemaker modus detectie ========")
    for regel in onderbouwing:
        print(regel)

    print(f"\n  ► Meest waarschijnlijke modus:  {' / '.join(mogelijke_modi)}")
    if basis_bpm is not None:
        print(f"  ► Basisfrequentie:              {basis_bpm:.0f} bpm")
    if av_delay is not None:
        print(f"  ► AV-delay:                     {av_delay:.0f} ms")

    print("\n  Niet onderscheidbaar op basis van telemetrie alleen:")
    if "DDD" in mogelijke_modi and "DOO" in mogelijke_modi:
        print("    - DDD vs DOO: als eigen ritme hoger is dan pacemaker-rate en")
        print("      toch altijd gepaced → DDD (eigen activiteit niet gevoeld); anders DOO.")
    if "VVI" in mogelijke_modi and "VDD" in mogelijke_modi:
        print("    - VVI vs VDD: VDD heeft duaal sensing maar geen A-pacing;")
        print("      bij ontbreken van A-spikes en aanwezigheid van eigen P-golven → VDD.")

    return mogelijke_modi


# ============================================================
# 9. Plotfuncties (ongewijzigd + uitgebreid)
# ============================================================

def plot_artifact_removal(t, ecg, clean_ecg, spike_indices, artifact_mask, plot_mask):
    plt.figure(figsize=(12, 4))
    plt.plot(t[plot_mask], ecg[plot_mask],       label="Raw ECG",           alpha=0.35)
    plt.plot(t[plot_mask], clean_ecg[plot_mask], label="Artifact-free ECG", alpha=0.90)

    vis = spike_indices[(t[spike_indices] >= t[plot_mask][0]) &
                        (t[spike_indices] <= t[plot_mask][-1])]
    plt.plot(t[vis], ecg[vis], "rx", label="Detected pacing spikes")
    plt.plot(
        t[plot_mask][~artifact_mask[plot_mask]],
        clean_ecg[plot_mask][~artifact_mask[plot_mask]],
        ".", markersize=3, label="Interpolated samples",
    )
    plt.title("ECG with removed pacemaker artifacts")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_qrs_detection(t, clean_ecg, y_mwi, qrs_indices, plot_mask):
    plot_t   = t[plot_mask]
    plot_ecg = clean_ecg[plot_mask]
    plot_mwi = y_mwi[plot_mask]
    full_idx = np.flatnonzero(plot_mask)

    qrs_in  = qrs_indices[(qrs_indices >= full_idx[0]) & (qrs_indices <= full_idx[-1])]
    qrs_loc = qrs_in - full_idx[0]

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(plot_t, plot_ecg,         label="Artifact-free ECG", alpha=0.9)
    ax[0].plot(plot_t[qrs_loc], plot_ecg[qrs_loc], "rx", label="Detected QRS")
    ax[0].set_title("Artifact-free ECG with QRS detections")
    ax[0].set_ylabel("ECG amplitude")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(plot_t, plot_mwi,         label="MWI")
    ax[1].plot(plot_t[qrs_loc], plot_mwi[qrs_loc], "rx", label="Detected peaks")
    ax[1].set_title("MWI with detected QRS-related peaks")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("MWI")
    ax[1].grid(True)
    ax[1].legend()
    plt.tight_layout()
    plt.show()


def plot_pacing_labels(t, ecg, clean_ecg, pacing_events, plot_mask):
    plt.figure(figsize=(14, 4))
    plt.plot(t[plot_mask], ecg[plot_mask],       label="Raw ECG",           alpha=0.25)
    plt.plot(t[plot_mask], clean_ecg[plot_mask], label="Artifact-free ECG", alpha=0.85)

    stijlen   = {"atriaal": "bo", "ventriculair": "ro",
                 "pacingfout": "mo", "geen_capture": "kx"}
    geplot    = set()

    for event in pacing_events:
        idx   = event["spike_index"]
        if not plot_mask[idx]:
            continue
        label = event["label"]
        stijl = stijlen.get(label, "ko")
        lbl   = label if label not in geplot else None
        plt.plot(t[idx], clean_ecg[idx], stijl, label=lbl)
        geplot.add(label)

    plt.title("Classificatie van pacing-momenten")
    plt.xlabel("Time (s)")
    plt.ylabel("ECG amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_spike_timeline(t, pacemaker_indices, labels_per_spike, plot_mask):
    """
    Vraag 6 – Tijdlijn van A en V spikes over het hele signaal,
    zodat het pacing-patroon in een oogopslag zichtbaar is.
    """
    spike_t = t[pacemaker_indices]
    kleur_map = {
        "atriaal":      "blue",
        "ventriculair": "red",
        "pacingfout":   "magenta",
        "geen_capture": "black",
    }

    fig, ax = plt.subplots(figsize=(18, 2))
    for label, kleur in kleur_map.items():
        sel = np.array([i for i, l in enumerate(labels_per_spike) if l == label])
        if len(sel) > 0:
            ax.vlines(spike_t[sel], 0, 1, colors=kleur,
                      linewidth=0.6, alpha=0.7, label=label)

    ax.set_yticks([])
    ax.set_xlabel("Tijd (s)")
    ax.set_title("Tijdlijn pacemaker-spikes (A=blauw, V=rood, fout=magenta, geen capture=zwart)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 10. Main
# ============================================================

def main():
    # ── Instellingen ────────────────────────────────────────
    ecg_path   = "../Data_E2/005_Pimpel_3.mat"
    plot_start = 0
    plot_end   = 12000

    artifact_threshold    = -1000
    artifact_max_ext      = 6

    bp_low            = 5
    bp_high           = 15
    mwi_ms            = 150
    prominence_factor = 0.05
    refractory_ms     = 350

    ventricular_max_ms  = 100
    atrial_peak_min_ms  = 15
    atrial_peak_max_ms  = 120
    p_search_start_ms   = 250
    p_search_end_ms     = 60
    qrs_error_post_ms   = 80

    # ── 1. ECG laden ────────────────────────────────────────
    ecg, fs, t, plot_mask = load_ecg_seconds(
        path=ecg_path, plotresult=True,
        plot_start=plot_start, plot_end=plot_end,
    )

    # ── 2. Artefacten verwijderen ────────────────────────────
    clean_ecg, pacemaker_indices, artifact_mask = remove_pacemaker_artifacts(
        ecg, threshold=artifact_threshold, max_extension=artifact_max_ext,
    )
    plot_artifact_removal(t, ecg, clean_ecg, pacemaker_indices, artifact_mask, plot_mask)

    # ── 3. Preprocessing voor QRS ───────────────────────────
    _, _, _, y_mwi = pan_tompkins_preprocess(clean_ecg, fs, bp_low, bp_high, mwi_ms)

    # ── 4. QRS-detectie ─────────────────────────────────────
    qrs_indices, prominence, _ = detect_qrs_peaks_on_mwi(
        y_mwi, fs, prominence_factor=prominence_factor, refractory_ms=refractory_ms,
    )
    qrs_peak_indices  = refine_qrs_peaks_to_ecg(clean_ecg, qrs_indices, fs)
    qrs_onset_indices = estimate_qrs_onsets(clean_ecg, qrs_peak_indices, fs)

    mean_rr, mean_hr, rr = rr_hr_from_peaks(t, qrs_peak_indices)
    plot_qrs_detection(t, clean_ecg, y_mwi, qrs_indices, plot_mask)

    # ── 5. Classificatie pacing-momenten ────────────────────
    p_peak_indices = estimate_p_peaks(
        clean_ecg, qrs_onset_indices, fs, p_search_start_ms, p_search_end_ms,
    )
    pacing_events = classify_pacing_events(
        pacemaker_indices, qrs_peak_indices, qrs_onset_indices, p_peak_indices, fs,
        ventricular_max_ms, atrial_peak_min_ms, atrial_peak_max_ms, qrs_error_post_ms,
    )
    summary = summarize_pacing_events(pacing_events)

    plot_pacing_labels(t, ecg, clean_ecg, pacing_events, plot_mask)

    labels_per_spike = [e["label"] for e in pacing_events]
    plot_spike_timeline(t, pacemaker_indices, labels_per_spike, plot_mask)

    # ── 6. Vraag 3 – Intervalanalyse ────────────────────────
    interval_resultaten = analyseer_pacemaker_intervallen(pacemaker_indices, t, fs)

    # ── 7. Vraag 6 – Pacing percentages ─────────────────────
    pct_resultaten = bereken_pacing_percentages(pacing_events, qrs_peak_indices)

    # ── 8. Vraag 7 – Modus detectie ─────────────────────────
    modus = detecteer_pacemaker_modus(interval_resultaten, pct_resultaten)

    # ── 9. Samenvatting ─────────────────────────────────────
    print("\n================ ECG ANALYSIS SUMMARY ================")
    print(f"  Sample freq.:          {fs:.1f} Hz")
    print(f"  Pacemaker spikes:      {len(pacemaker_indices)}")
    print(f"  QRS-complexen:         {len(qrs_indices)}")
    if len(rr) > 0:
        print(f"  Gemiddeld RR:          {mean_rr*1000:.1f} ms")
        print(f"  Gemiddelde hartfreq.:  {mean_hr:.1f} bpm")
    print("\n  Pacing events:")
    for k, v in summary.items():
        print(f"    {k:<20}: {v}")
    print(f"\n  Waarschijnlijke modus: {' / '.join(modus)}")


if __name__ == "__main__":
    main()