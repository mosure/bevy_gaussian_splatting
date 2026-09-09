#!/usr/bin/env python3
"""Pair attested GPU PNG captures at identical logical poses.

Metrics describe linearized, premultiplied RGB reconstructed from RGBA8 sRGB
output. They are teacher-relative and include output quantization. This is not
photo-ground-truth quality, floating-point framebuffer PSNR, or a release gate.
Dependencies: numpy and Pillow. No renderer or GPU is started by this command.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from check_lod_capture import CaptureError, read_captures, require, verify_images


def blur_valid(image):
    """Separable 11-tap Gaussian, sigma 1.5; omit the five-pixel border."""
    x = np.arange(-5, 6, dtype=np.float64)
    weights = np.exp(-(x * x) / (2 * 1.5**2))
    weights /= weights.sum()
    height, width = image.shape[:2]
    require(height >= 11 and width >= 11, "SSIM requires at least 11x11 pixels")
    horizontal = sum(weight * image[:, i : width - 10 + i] for i, weight in enumerate(weights))
    return sum(weight * horizontal[i : height - 10 + i] for i, weight in enumerate(weights))


def ssim_map(reference, candidate):
    # Population covariance, fixed unit-range constants; average RGB channels.
    rx, cx = blur_valid(reference), blur_valid(candidate)
    rv = np.maximum(0, blur_valid(reference * reference) - rx * rx)
    cv = np.maximum(0, blur_valid(candidate * candidate) - cx * cx)
    cov = blur_valid(reference * candidate) - rx * cx
    return (((2 * rx * cx + 0.01**2) * (2 * cov + 0.03**2)) /
            ((rx * rx + cx * cx + 0.01**2) * (rv + cv + 0.03**2))).mean(axis=2)


def rgba(path):
    with Image.open(path) as image:
        require(image.mode == "RGBA", f"expected RGBA PNG: {path}")
        pixels = np.asarray(image, dtype=np.float64) / 255
    srgb = pixels[..., :3]
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return linear, pixels[..., 3]


def metrics(reference_rgb, candidate_rgb, reference_alpha, candidate_alpha):
    require(reference_rgb.shape == candidate_rgb.shape, "image shape mismatch")
    foreground = (reference_alpha > 0.02) | (candidate_alpha > 0.02)
    require(bool(foreground.any()), "empty foreground: do not qualify a blank image")
    squared = ((reference_rgb - candidate_rgb) ** 2).mean(axis=2)
    similarity = ssim_map(reference_rgb, candidate_rgb)
    interior_foreground = foreground[5:-5, 5:-5]
    require(bool(interior_foreground.any()), "empty SSIM foreground")
    full_mse, foreground_mse = float(squared.mean()), float(squared[foreground].mean())

    def psnr(mse):
        # JSON does not admit infinity. A null PSNR with zero MSE means exact.
        return -10 * math.log10(mse) if mse > 0 else None

    union = np.count_nonzero(foreground)
    intersection = np.count_nonzero((reference_alpha > 0.02) & (candidate_alpha > 0.02))
    return {
        "full_mse": full_mse, "full_psnr_db": psnr(full_mse),
        "foreground_mse": foreground_mse, "foreground_psnr_db": psnr(foreground_mse),
        "full_ssim": float(similarity.mean()),
        "foreground_ssim": float(similarity[interior_foreground].mean()),
        "foreground_pixels": int(union), "foreground_fraction": float(foreground.mean()),
        "alpha_mae": float(np.abs(reference_alpha - candidate_alpha).mean()),
        "alpha_max_error": float(np.abs(reference_alpha - candidate_alpha).max()),
        "silhouette_iou": float(intersection / union),
    }


def require_renderer_traversal_mapping(evidence, prefix):
    """Match observed renderer inputs to the separately numbered producer output."""
    mapping = evidence.get("renderer_traversal_mapping")
    require(isinstance(mapping, dict) and mapping.get("schema_version") == 1 and
            mapping.get("association") == "current_render_encoded_inputs",
            "missing explicit renderer/traversal input mapping")
    renderer_submission = evidence.get(f"{prefix}_submission")
    traversal_submission = evidence.get("traversal_submission")
    require(type(renderer_submission) is int and renderer_submission > 0 and
            type(traversal_submission) is int and traversal_submission > 0 and
            mapping.get("renderer_submission") == renderer_submission,
            "renderer submission mapping mismatch")
    inputs, output = mapping.get("renderer_inputs"), mapping.get("traversal_output")
    require(isinstance(inputs, list) and len(inputs) == 1 and
            isinstance(inputs[0], dict) and isinstance(output, dict),
            "capture requires exactly one mapped renderer hierarchy input")
    require(output.get("submission") == traversal_submission,
            "traversal submission mapping mismatch")
    require(inputs[0].get("submission") == traversal_submission,
            "renderer input traversal submission mismatch")
    for field in ("cloud", "source_asset", "residency_generation", "allocation_generation"):
        value = evidence.get(field)
        valid = (isinstance(value, str) and bool(value)) if field in ("cloud", "source_asset") \
            else type(value) is int and value >= 0
        require(valid and output.get(field) == value, f"traversal output {field} mismatch")
        require(inputs[0].get(field) == value, f"renderer input {field} mismatch")


def require_attested_image(record, evidence):
    """Validate the receipt belonging to this pipeline and exact generation."""
    pipeline = record["counts"].get("pipeline", "hierarchy")
    require(record["counts"]["source"] == "gpu_readback", "image lacks GPU counts")
    require(record["stamp"]["frame"] == evidence["frame"], "receipt frame mismatch")
    for field in ("run_id", "view_id"):
        value = evidence.get(field, evidence.get("stamp", {}).get(field))
        if value is not None:
            require(record["stamp"].get(field) == value, f"receipt {field} mismatch")
    if pipeline in ("hierarchy_ordered", "hierarchy_point"):
        prefix = "ordered" if pipeline == "hierarchy_ordered" else "point"
        receipt = "ordered_draw_receipt" if prefix == "ordered" else "point_image_receipt"
        require(evidence.get("pipeline") == pipeline, "receipt pipeline mismatch")
        require(evidence.get(receipt) is True and
                evidence.get(f"{prefix}_image_attested") is True and
                evidence.get("counts_valid") is True, "image lacks renderer attestation")
        require_renderer_traversal_mapping(evidence, prefix)
        require(record["stamp"]["generation"] == evidence.get("residency_generation"),
                "image and residency generation mismatch")
    else:
        require(evidence.get("draw_command_attested") is True, "draw lacks command attestation")


def index_capture(path):
    records = read_captures(path)
    verify_images(records, path)
    evidence_path = path.parent / "submission_evidence.jsonl"
    evidence, receipt_ids, frame_contexts = {}, set(), {}
    for record in records:
        stamp = record["stamp"]
        frame_contexts.setdefault(stamp["frame"], set()).add((stamp["run_id"], stamp["view_id"]))
    with evidence_path.open() as file:
        for line in file:
            item = json.loads(line)
            frame = item["frame"]
            context = tuple(item.get(field, item.get("stamp", {}).get(field))
                            for field in ("run_id", "view_id"))
            require((frame, context) not in receipt_ids, "duplicate submission evidence identity")
            receipt_ids.add((frame, context))
            evidence.setdefault(frame, []).append((context, item))
    indexed = {}
    for record in records:
        frame = record["stamp"]["frame"]
        stamp = record["stamp"]
        context = (stamp["run_id"], stamp["view_id"])
        matches = []
        for supplied, item in evidence.get(frame, []):
            possible = {value for value in frame_contexts[frame]
                        if all(given is None or given == actual for given, actual in zip(supplied, value))}
            if context in possible:
                require(len(possible) == 1, "ambiguous submission receipt: run/view identity required")
                matches.append(item)
        require(len(matches) <= 1, "multiple submission receipts match the image")
        item = matches[0] if matches else None
        if not item or not record.get("image") or record["counts"].get("drawn") is None:
            continue
        require_attested_image(record, item)
        key = (record["scenario"], item["path_frame"], stamp["view_id"])
        # Startup can repeat a pose while shaders compile. Keep its last
        # attested observation; actual frame identity stays in the result.
        previous = indexed.get(key)
        require(previous is None or previous["stamp"]["run_id"] == stamp["run_id"],
                "ambiguous logical sample across runs; compare one run at a time")
        indexed[key] = record
    require(bool(indexed), "no attested image captures")
    return indexed


def compare(reference_path, candidate_path, require_complete_pairs=False):
    reference, candidate = index_capture(reference_path), index_capture(candidate_path)
    reference_views = {key[2] for key in reference}
    candidate_views = {key[2] for key in candidate}
    # Separate single-view runs can allocate different entity IDs. Multiview
    # captures instead pair by their actual stable view IDs, never last-writer.
    single_view = len(reference_views) == len(candidate_views) == 1
    if single_view:
        reference = {(key[0], key[1], None): value for key, value in reference.items()}
        candidate = {(key[0], key[1], None): value for key, value in candidate.items()}
    if require_complete_pairs:
        require(reference.keys() == candidate.keys(), "incomplete logical sample pairs")
    pairs = []
    for key in sorted(reference.keys() & candidate.keys()):
        r, c = reference[key], candidate[key]
        require(r["camera"] == c["camera"], f"camera mismatch at {key}")
        require(r["identity"]["source_sha256"] == c["identity"]["source_sha256"], "source mismatch")
        require(r["identity"]["camera_path_sha256"] == c["identity"]["camera_path_sha256"], "camera path mismatch")
        rr, ra = rgba(reference_path.parent / r["image"]["path"])
        cr, ca = rgba(candidate_path.parent / c["image"]["path"])
        drawn = c["counts"]["drawn"]
        pairs.append({
            "scenario": key[0], "path_frame": key[1],
            "reference_view_id": r["stamp"]["view_id"], "candidate_view_id": c["stamp"]["view_id"],
            "reference_stamp": r["stamp"], "candidate_stamp": c["stamp"],
            "reference_image": r["image"], "candidate_image": c["image"],
            "reference_drawn": r["counts"]["drawn"], "candidate_drawn": drawn,
            "draw_reduction": r["counts"]["drawn"] / drawn if drawn else None,
            "metrics": metrics(rr, cr, ra, ca),
            "reference_timings": r.get("timings"), "candidate_timings": c.get("timings"),
        })
    require(bool(pairs), "no matching logical camera samples")
    return {
        "schema_version": 1, "release_qualified": False,
        "metric_domain": "RGBA8_sRGB_to_linear_premultiplied_RGB;teacher_relative;SSIM_11x11_sigma1.5_population_valid",
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_capture_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "candidate_capture_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "view_pairing": "one_view_per_run" if single_view else "matching_view_id",
        "complete_pairs_required": require_complete_pairs,
        "complete_pairs": reference.keys() == candidate.keys(),
        "paired_samples": len(pairs), "unpaired_reference_samples": len(reference) - len(pairs),
        "unpaired_candidate_samples": len(candidate) - len(pairs), "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-complete-pairs", action="store_true",
                        help="fail if either run has unpaired attested scenario/path-frame/view samples")
    args = parser.parse_args()
    try:
        report = compare(args.reference, args.candidate, args.require_complete_pairs)
        with args.output.open("x") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
    except (CaptureError, ValueError, OSError) as error:
        parser.exit(1, f"compare_lod_captures: {error}\n")
    print(f"paired {report['paired_samples']} samples; wrote {args.output}")


if __name__ == "__main__":
    main()
