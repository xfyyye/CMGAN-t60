"""Create the fixed AIR three-room case study without running any model.

This is a separate experiment entry point. It never changes the original audio,
training code, checkpoints, paper, or existing experiment output directories.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import fftconvolve

if __package__ in (None, ""):
    from common import (
        REPO_ROOT,
        config_sha256,
        data_config_sha256,
        data_root,
        git_revision,
        load_config,
        raw_path,
        reserve_output,
        sha256_file,
        validate_manifest,
        write_json,
    )
else:
    from .common import (
        REPO_ROOT,
        config_sha256,
        data_config_sha256,
        data_root,
        git_revision,
        load_config,
        raw_path,
        reserve_output,
        sha256_file,
        validate_manifest,
        write_json,
    )


LABEL_CHANNEL_POLICY = "unknown_file_level_historical"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_id(value: Any, field: str) -> str:
    text = str(value)
    _require(
        re.fullmatch(r"[A-Za-z0-9_-]+", text) is not None,
        f"Invalid {field}: {text!r}",
    )
    return text


def _read_audio(
    path: Path,
    sample_rate: int,
    channels: int,
    frames: int | None = None,
) -> np.ndarray:
    _require(path.is_file(), f"Audio source does not exist: {path}")
    audio, sr = sf.read(str(path), dtype="float64", always_2d=True)
    _require(sr == sample_rate, f"Unexpected sample rate {sr}: {path}")
    _require(audio.shape[1] == channels, f"Unexpected channel count: {path}")
    _require(audio.shape[0] > 0, f"Empty audio: {path}")
    if frames is not None:
        _require(audio.shape[0] == frames, f"Unexpected sample count: {path}")
    _require(bool(np.isfinite(audio).all()), f"Non-finite audio: {path}")
    return audio


def _power(audio: np.ndarray, min_power: float, description: str) -> float:
    power = float(np.mean(np.square(audio)))
    _require(
        math.isfinite(power) and power > min_power,
        f"Invalid or insufficient power ({power}): {description}",
    )
    return power


def _source_record(cfg: dict[str, Any], relative: str) -> dict[str, str]:
    path = raw_path(cfg, relative)
    _require(path.is_file(), f"Source file does not exist: {path}")
    return {
        "relative_path": relative,
        "absolute_path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def _check_label_record(
    cfg: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    path = raw_path(cfg, case["label_csv"])
    target_line = int(case["label_csv_line"])
    found: dict[str, str] | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if reader.line_num == target_line:
                found = row
                break
    _require(found is not None, f"Missing label CSV line {target_line}: {path}")
    assert found is not None
    _require(
        found.get("original_filename") == case["original_filename"],
        f"Original RIR filename does not match label CSV line: {case['id']}",
    )
    _require(
        found.get("new_filename") == Path(case["rir_path"]).name,
        f"Renamed RIR filename does not match label CSV line: {case['id']}",
    )
    _require(
        found.get("success", "").strip().lower() == "true",
        f"Label CSV row is not marked successful: {case['id']}",
    )
    csv_t60 = float(found["t60_value"])
    _require(
        math.isfinite(csv_t60)
        and math.isclose(csv_t60, float(case["t60_s"]), rel_tol=0.0, abs_tol=1e-9),
        f"T60 label does not match label CSV line: {case['id']}",
    )
    return {
        "source": _source_record(cfg, case["label_csv"]),
        "line": target_line,
        "record": found,
    }


def _load_speeches(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    settings = cfg["synthesis"]
    directory = raw_path(cfg, settings["speech_directory"])
    _require(directory.is_dir(), f"Speech directory does not exist: {directory}")
    paths = sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    _require(
        len(paths) == int(settings["speech_count"]),
        f"Expected {settings['speech_count']} speech files, found {len(paths)}",
    )
    result: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        audio = _read_audio(
            path,
            int(settings["sample_rate_hz"]),
            channels=1,
            frames=int(settings["speech_input_samples"]),
        )[:, 0]
        _power(audio, float(settings["min_power"]), str(path))
        relative = (
            Path(settings["speech_directory"]) / path.relative_to(directory)
        ).as_posix()
        result.append(
            {
                "id": f"speech{index:06d}",
                "audio": audio,
                "source": _source_record(cfg, relative),
                "original_filename": path.name,
            }
        )
    return result


def _load_noises(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    settings = cfg["synthesis"]
    output_samples = int(settings["output_samples"])
    result: list[dict[str, Any]] = []
    for item in settings["noise_files"]:
        noise_id = _safe_id(item["id"], "noise id")
        path = raw_path(cfg, item["path"])
        audio = _read_audio(path, int(settings["sample_rate_hz"]), channels=1)[:, 0]
        _require(len(audio) >= output_samples, f"Noise is too short: {path}")
        result.append(
            {
                "id": noise_id,
                "audio": audio,
                "source": _source_record(cfg, item["path"]),
                "original_filename": path.name,
            }
        )
    return result


def _noise_templates(
    cfg: dict[str, Any],
    speeches: list[dict[str, Any]],
    noises: list[dict[str, Any]],
    rng: np.random.Generator,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Draw once in sorted-speech x configured-noise order, before case loops."""
    settings = cfg["synthesis"]
    count = int(settings["output_samples"])
    templates: dict[tuple[str, str], dict[str, Any]] = {}
    for speech in speeches:
        for noise in noises:
            # The final valid offset must be reachable; the upper bound is exclusive.
            offset = int(rng.integers(0, len(noise["audio"]) - count + 1))
            segment = noise["audio"][offset : offset + count]
            power = _power(segment, float(settings["min_power"]), f"{speech['id']} / {noise['id']}")
            templates[(speech["id"], noise["id"])] = {
                "audio": segment,
                "offset_samples": offset,
                "power": power,
            }
    _require(
        len(templates) == len(speeches) * len(noises),
        "Noise template identifiers are not unique",
    )
    return templates


def _write_and_verify_triplet(
    directory: Path,
    sample_id: str,
    reverberant: np.ndarray,
    noise_segment: np.ndarray,
    noise_gain: float,
    snr_db: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    scaled_noise = noise_segment * noise_gain
    noisy = reverberant + scaled_noise
    peak_before = float(
        max(np.max(np.abs(reverberant)), np.max(np.abs(scaled_noise)), np.max(np.abs(noisy)))
    )
    _require(math.isfinite(peak_before) and peak_before > 0.0, f"Invalid peak: {sample_id}")
    peak_gain = min(1.0, float(settings["peak_limit"]) / peak_before)
    _require(math.isfinite(peak_gain) and peak_gain > 0.0, f"Invalid peak gain: {sample_id}")
    arrays = {
        "noisy": noisy * peak_gain,
        "clean": reverberant * peak_gain,
        "noise": scaled_noise * peak_gain,
    }
    filenames = {
        "noisy": f"{sample_id}.wav",
        "clean": f"{sample_id}_denoised.wav",
        "noise": f"{sample_id}_noise.wav",
    }
    directory.mkdir(parents=True, exist_ok=False)
    sample_rate = int(settings["sample_rate_hz"])
    count = int(settings["output_samples"])
    reloaded: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for role, audio in arrays.items():
        _require(bool(np.isfinite(audio).all()), f"Non-finite {role}: {sample_id}")
        path = directory / filenames[role]
        sf.write(str(path), audio.astype(np.float32), sample_rate, format="WAV", subtype="FLOAT")
        decoded = _read_audio(path, sample_rate, channels=1, frames=count)[:, 0]
        _require(sf.info(str(path)).subtype == "FLOAT", f"Unexpected WAV subtype: {path}")
        _require(
            float(np.max(np.abs(decoded))) <= float(settings["peak_limit"]) + 1e-6,
            f"Peak limit exceeded after reading WAV: {path}",
        )
        reloaded[role] = decoded
        hashes[role] = sha256_file(path)
    clean_power = _power(reloaded["clean"], 0.0, f"Stored clean: {sample_id}")
    noise_power = _power(reloaded["noise"], 0.0, f"Stored noise: {sample_id}")
    actual_snr = 10.0 * math.log10(clean_power / noise_power)
    snr_error = abs(actual_snr - snr_db)
    additive_error = float(
        np.max(np.abs(reloaded["noisy"] - reloaded["clean"] - reloaded["noise"]))
    )
    _require(
        snr_error <= float(settings["snr_tolerance_db"]),
        f"Stored SNR error {snr_error:.8g} dB: {sample_id}",
    )
    _require(
        additive_error <= float(settings["additivity_tolerance"]),
        f"Stored additivity error {additive_error:.8g}: {sample_id}",
    )
    return {
        "filenames": filenames,
        "hashes": hashes,
        "peak_before_common_gain": peak_before,
        "common_peak_gain": peak_gain,
        "actual_snr_db": actual_snr,
        "snr_error_db": snr_error,
        "additivity_max_abs_error": additive_error,
        "stored_clean_power": clean_power,
        "stored_noise_power": noise_power,
    }


def synthesize(cfg: dict[str, Any]) -> dict[str, Any]:
    settings = cfg["synthesis"]
    _require(int(settings["channel_index"]) == 0, "This experiment requires RIR channel 0")
    _require(settings["wav_subtype"] == "FLOAT", "This experiment requires FLOAT WAV")
    output = data_root(cfg)
    reserve_output(output)
    write_json(output / "config.snapshot.json", cfg)
    started = datetime.now(timezone.utc).isoformat()

    speeches = _load_speeches(cfg)
    noises = _load_noises(cfg)
    rng = np.random.default_rng(int(settings["seed"]))
    templates = _noise_templates(cfg, speeches, noises, rng)
    cases: list[dict[str, Any]] = []
    source_records: dict[str, dict[str, str]] = {}
    for source in [item["source"] for item in speeches + noises]:
        source_records[source["relative_path"]] = source
    for case_cfg in cfg["cases"]:
        case = dict(case_cfg)
        case["id"] = _safe_id(case["id"], "case id")
        rir_source = _source_record(cfg, case["rir_path"])
        rir_audio = _read_audio(
            raw_path(cfg, case["rir_path"]),
            int(settings["sample_rate_hz"]),
            channels=int(case["expected_channels"]),
        )
        # Keep channel 0 exactly as stored: no rescaling, alignment, or RIR crop.
        case["rir_audio"] = rir_audio[:, int(settings["channel_index"])]
        _power(case["rir_audio"], float(settings["min_power"]), case["rir_path"])
        case["rir_source"] = rir_source
        case["label"] = _check_label_record(cfg, case)
        cases.append(case)
        source_records[rir_source["relative_path"]] = rir_source
        label_source = case["label"]["source"]
        source_records[label_source["relative_path"]] = label_source

    start = int(settings["crop_start_sample"])
    count = int(settings["output_samples"])
    end = start + count
    rows: list[dict[str, Any]] = []
    for case in cases:
        for speech in speeches:
            # Perform the full linear convolution before taking the fixed crop.
            full_reverberant = fftconvolve(speech["audio"], case["rir_audio"], mode="full")
            _require(len(full_reverberant) >= end, f"Convolution is too short: {speech['id']}")
            _require(bool(np.isfinite(full_reverberant).all()), f"Invalid convolution: {speech['id']}")
            reverberant = full_reverberant[start:end]
            signal_power = _power(
                reverberant, float(settings["min_power"]), f"{case['id']} / {speech['id']}",
            )
            for noise in noises:
                template = templates[(speech["id"], noise["id"])]
                for configured_snr in settings["snr_db"]:
                    snr_db = float(configured_snr)
                    noise_gain = math.sqrt(signal_power / template["power"]) * 10.0 ** (-snr_db / 20.0)
                    _require(math.isfinite(noise_gain) and noise_gain > 0.0, "Invalid noise gain")
                    sample_id = (
                        f"{case['id']}_{speech['id']}_{noise['id']}_reverb_"
                        f"{float(case['t60_s']):.3f}_{snr_db:g}dB"
                    )
                    relative_directory = Path("audio") / case["id"] / sample_id
                    verified = _write_and_verify_triplet(
                        output / relative_directory,
                        sample_id,
                        reverberant,
                        template["audio"],
                        noise_gain,
                        snr_db,
                        settings,
                    )
                    row = {
                        "sample_id": sample_id,
                        "case_id": case["id"],
                        "room_name": case["room_name"],
                        "speech_id": speech["id"],
                        "noise_id": noise["id"],
                        "snr_db": snr_db,
                        "t60_s": float(case["t60_s"]),
                        "channel_index": int(settings["channel_index"]),
                        "label_channel_policy": LABEL_CHANNEL_POLICY,
                        "sample_rate_hz": int(settings["sample_rate_hz"]),
                        "crop_start_sample": start,
                        "output_samples": count,
                        "speech_input_samples": int(settings["speech_input_samples"]),
                        "speech_relative_path": speech["source"]["relative_path"],
                        "speech_source_path": speech["source"]["absolute_path"],
                        "speech_original_filename": speech["original_filename"],
                        "speech_sha256": speech["source"]["sha256"],
                        "noise_source_relative_path": noise["source"]["relative_path"],
                        "noise_source_path": noise["source"]["absolute_path"],
                        "noise_original_filename": noise["original_filename"],
                        "noise_source_sha256": noise["source"]["sha256"],
                        "noise_offset_samples": template["offset_samples"],
                        "noise_template_power": template["power"],
                        "rir_source_relative_path": case["rir_source"]["relative_path"],
                        "rir_source_path": case["rir_source"]["absolute_path"],
                        "rir_sha256": case["rir_source"]["sha256"],
                        "rir_original_filename": case["original_filename"],
                        "rir_channels": int(case["expected_channels"]),
                        "rir_samples": int(len(case["rir_audio"])),
                        "label_source_relative_path": case["label"]["source"]["relative_path"],
                        "label_source_path": case["label"]["source"]["absolute_path"],
                        "label_source_sha256": case["label"]["source"]["sha256"],
                        "label_source_line": case["label"]["line"],
                        "label_source_record": case["label"]["record"],
                        "reverberant_crop_power": signal_power,
                        "noise_snr_gain": noise_gain,
                        "common_peak_gain": verified["common_peak_gain"],
                        "peak_before_common_gain": verified["peak_before_common_gain"],
                        "actual_snr_db": verified["actual_snr_db"],
                        "snr_error_db": verified["snr_error_db"],
                        "additivity_max_abs_error": verified["additivity_max_abs_error"],
                        "stored_clean_power": verified["stored_clean_power"],
                        "stored_noise_power": verified["stored_noise_power"],
                    }
                    for role in ("noisy", "clean", "noise"):
                        row[f"{role}_relative_path"] = (
                            relative_directory / verified["filenames"][role]
                        ).as_posix()
                        row[f"{role}_sha256"] = verified["hashes"][role]
                    rows.append(row)
        print(f"Completed audio for {case['id']}: {len(rows)} cumulative samples", flush=True)

    _require(len(rows) == int(settings["expected_samples"]), "Unexpected final sample count")
    validate_manifest(cfg, rows)
    # Refuse completion if an input changed after its provenance was captured.
    for source in source_records.values():
        _require(
            sha256_file(Path(source["absolute_path"])) == source["sha256"],
            f"Source changed during synthesis: {source['absolute_path']}",
        )
    manifest = output / "manifest.jsonl"
    with manifest.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")

    scripts = [Path(__file__).resolve(), REPO_ROOT / "case_study" / "common.py"]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": cfg["experiment_id"],
        "sample_count": len(rows),
        "expected_samples": int(settings["expected_samples"]),
        "case_count": len(cases),
        "speech_count": len(speeches),
        "noise_count": len(noises),
        "noise_template_count": len(templates),
        "snr_db": [float(value) for value in settings["snr_db"]],
        "sample_rate_hz": int(settings["sample_rate_hz"]),
        "channel_index": int(settings["channel_index"]),
        "label_channel_policy": LABEL_CHANNEL_POLICY,
        "seed": int(settings["seed"]),
        "software_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "soundfile": sf.__version__,
            "libsndfile": sf.__libsndfile_version__,
        },
        "noise_template_rng": {
            "api": "numpy.random.default_rng",
            "generator_type": type(rng).__name__,
            "bit_generator_type": type(rng.bit_generator).__name__,
            "seed": int(settings["seed"]),
        },
        "source_hashes_reverified": True,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256(cfg),
        "data_config_sha256": data_config_sha256(cfg),
        "manifest_sha256": sha256_file(manifest),
        "git_revision": git_revision(),
        "script_sha256": {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in scripts},
        "source_sha256": {key: value["sha256"] for key, value in sorted(source_records.items())},
        "source_records": [value for _, value in sorted(source_records.items())],
        "max_snr_error_db": max(row["snr_error_db"] for row in rows),
        "max_additivity_error": max(row["additivity_max_abs_error"] for row in rows),
        "noise_template_offsets": [
            {
                "speech_id": speech_id,
                "noise_id": noise_id,
                "offset_samples": template["offset_samples"],
                "power": template["power"],
            }
            for (speech_id, noise_id), template in templates.items()
        ],
    }
    # The completion marker is deliberately the last file written.
    write_json(output / "synthesis_summary.json", summary)
    print(f"Synthesis complete: {len(rows)} samples in {output}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "case_study" / "config.json",
        help="Validated case-study JSON configuration",
    )
    args = parser.parse_args()
    synthesize(load_config(args.config))


if __name__ == "__main__":
    main()
