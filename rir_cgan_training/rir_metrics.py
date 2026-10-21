import numpy as np


def schroeder_t60(rir, sample_rate, decay_db=20):
    """Estimate T60 by T20 extrapolation; returns nan when invalid."""
    x = np.asarray(rir, dtype=np.float64).reshape(-1)
    peak = int(np.argmax(np.abs(x)))
    x = x[peak:]
    if len(x) < 32 or np.max(np.abs(x)) == 0:
        return float("nan")
    energy = np.cumsum((x * x)[::-1])[::-1]
    db = 10.0 * np.log10(energy / (energy[0] + 1e-12) + 1e-12)
    idx = np.where((db <= 0.0) & (db >= -float(decay_db)))[0]
    if len(idx) < 8:
        return float("nan")
    slope, _ = np.polyfit(idx / sample_rate, db[idx], 1)
    if slope >= 0:
        return float("nan")
    return float(-60.0 / slope)


def accept_rir(rir, target_t60, sample_rate, tolerance_sec=0.05,
               min_rms=1e-6, max_peak=1.2):
    x = np.asarray(rir).reshape(-1)
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    measured = schroeder_t60(x, sample_rate)
    ok = (np.isfinite(measured) and abs(measured - target_t60) <= tolerance_sec
          and rms >= min_rms and peak <= max_peak)
    return {"accepted": bool(ok), "measured_t60": measured,
            "t60_error": abs(measured - target_t60) if np.isfinite(measured) else float("nan"),
            "peak": peak, "rms": rms}

