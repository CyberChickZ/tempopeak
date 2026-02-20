#!/usr/bin/env python3
"""Build a stage-1 raw dataset from video sources using SAM3 text prompts.

This utility is designed for a human-in-the-loop dataset pipeline:
1. Run SAM3 text prompting + propagation on video sources.
2. Export masks/boxes/scores to a machine-readable raw dataset.
3. Let annotators refine the generated draft labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from sam3.model_builder import build_sam3_video_predictor


@dataclass
class PromptSpec:
    text: str
    frame_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess videos with SAM3 text prompts to produce a stage-1 raw "
            "dataset for human refinement."
        )
    )
    parser.add_argument(
        "--source",
        nargs="+",
        required=True,
        help="Video source paths. Each item can be an MP4 file or an image-folder video.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Text prompt for SAM3. Repeat this argument to add multiple prompts.",
    )
    parser.add_argument(
        "--prompt-frame-index",
        type=int,
        action="append",
        default=None,
        help=(
            "Frame index for prompt injection. "
            "Provide once to share across all prompts, or once per prompt. "
            "Default: 0."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for exported stage-1 raw dataset artifacts.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sam3_stage1_raw",
        help="Name recorded in the generated dataset manifest.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.0,
        help="Threshold applied when exporting binary mask PNGs.",
    )
    parser.add_argument(
        "--skip-mask-export",
        action="store_true",
        help="If set, do not export mask PNG files (metadata only).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining sources if one source fails.",
    )
    return parser.parse_args()


def validate_sources(source_paths: Sequence[str]) -> List[Path]:
    sources: List[Path] = []
    for raw in source_paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Source path does not exist: {path}")
        if not (path.is_file() or path.is_dir()):
            raise ValueError(f"Source must be a file or directory: {path}")
        sources.append(path)
    return sources


def build_prompt_specs(prompts: Sequence[str], frame_indices: Optional[Sequence[int]]) -> List[PromptSpec]:
    if not prompts:
        raise ValueError("At least one --prompt is required.")

    if not frame_indices:
        return [PromptSpec(text=p, frame_index=0) for p in prompts]

    if len(frame_indices) == 1:
        idx = frame_indices[0]
        return [PromptSpec(text=p, frame_index=idx) for p in prompts]

    if len(frame_indices) != len(prompts):
        raise ValueError(
            "--prompt-frame-index must be provided once, or once per --prompt. "
            f"Got {len(frame_indices)} frame indices for {len(prompts)} prompts."
        )

    return [PromptSpec(text=p, frame_index=fi) for p, fi in zip(prompts, frame_indices)]


def make_sample_id(source_path: Path) -> str:
    stem = source_path.stem if source_path.is_file() else source_path.name
    safe_stem = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem).strip("_")
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:8]
    return f"{safe_stem}_{digest}" if safe_stem else digest


def to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return np.array([])
        if torch is not None and isinstance(value[0], torch.Tensor):
            return np.stack([v.detach().cpu().numpy() for v in value], axis=0)
        return np.asarray(value)
    return np.asarray(value)


def normalize_masks(raw_masks: Any) -> Optional[np.ndarray]:
    masks = to_numpy(raw_masks)
    if masks is None:
        return None
    if masks.size == 0:
        return masks.reshape(0, 0, 0)

    if masks.ndim == 2:
        return masks[None, ...]
    if masks.ndim == 3:
        return masks
    if masks.ndim == 4:
        if masks.shape[1] == 1:
            return masks[:, 0, ...]
        if masks.shape[0] == 1:
            return masks[0, ...]
    raise ValueError(f"Unsupported mask shape: {masks.shape}")


def normalize_boxes(raw_boxes: Any) -> Optional[np.ndarray]:
    boxes = to_numpy(raw_boxes)
    if boxes is None:
        return None
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    if boxes.ndim == 1:
        if boxes.shape[0] != 4:
            raise ValueError(f"Expected box with 4 elements, got shape: {boxes.shape}")
        return boxes[None, ...]
    return boxes


def normalize_scores(raw_scores: Any) -> Optional[np.ndarray]:
    scores = to_numpy(raw_scores)
    if scores is None:
        return None
    if scores.size == 0:
        return scores.reshape(0)
    return scores.reshape(-1)


def normalize_output_entries(outputs: Any) -> List[Dict[str, Any]]:
    if outputs is None:
        return []

    if isinstance(outputs, list):
        return [entry for entry in outputs if isinstance(entry, dict)]

    if isinstance(outputs, dict):
        if "predictions" in outputs and isinstance(outputs["predictions"], list):
            return [entry for entry in outputs["predictions"] if isinstance(entry, dict)]

        if "frame_outputs" in outputs and isinstance(outputs["frame_outputs"], list):
            return [entry for entry in outputs["frame_outputs"] if isinstance(entry, dict)]

        if "masks" in outputs or "mask" in outputs:
            return [outputs]

        frame_entries: List[Dict[str, Any]] = []
        for key, value in outputs.items():
            if isinstance(value, dict) and str(key).isdigit():
                entry = dict(value)
                entry.setdefault("frame_index", int(key))
                frame_entries.append(entry)
        if frame_entries:
            return frame_entries

    return []


def export_mask(mask: np.ndarray, out_path: Path, threshold: float) -> None:
    binary = (mask > threshold).astype(np.uint8) * 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary).save(out_path)


def parse_response_instances(
    response_outputs: Any,
    prompt_spec: PromptSpec,
    prompt_index: int,
    sample_dir: Path,
    threshold: float,
    skip_mask_export: bool,
) -> Tuple[Dict[int, List[Dict[str, Any]]], int]:
    frame_to_instances: Dict[int, List[Dict[str, Any]]] = {}
    exported_mask_count = 0

    for frame_seq_idx, entry in enumerate(normalize_output_entries(response_outputs)):
        frame_index = int(entry.get("frame_index", frame_seq_idx))

        raw_masks = entry.get("masks", entry.get("mask"))
        raw_boxes = entry.get("boxes")
        raw_scores = entry.get("scores")

        masks = normalize_masks(raw_masks) if raw_masks is not None else None
        boxes = normalize_boxes(raw_boxes)
        scores = normalize_scores(raw_scores)

        counts = [
            masks.shape[0] if masks is not None and masks.ndim >= 3 else 0,
            boxes.shape[0] if boxes is not None and boxes.ndim >= 2 else 0,
            scores.shape[0] if scores is not None else 0,
        ]
        num_instances = max(counts)
        if num_instances == 0:
            continue

        instances = frame_to_instances.setdefault(frame_index, [])

        for inst_idx in range(num_instances):
            score = float(scores[inst_idx]) if scores is not None and inst_idx < scores.shape[0] else None

            box: Optional[List[float]] = None
            if boxes is not None and inst_idx < boxes.shape[0]:
                box_arr = boxes[inst_idx].reshape(-1)
                box = [float(x) for x in box_arr.tolist()]

            mask_rel_path: Optional[str] = None
            if masks is not None and inst_idx < masks.shape[0] and not skip_mask_export:
                mask_path = (
                    sample_dir
                    / "masks"
                    / f"frame_{frame_index:06d}"
                    / f"prompt_{prompt_index:02d}_instance_{inst_idx:04d}.png"
                )
                export_mask(masks[inst_idx], mask_path, threshold=threshold)
                exported_mask_count += 1
                mask_rel_path = str(mask_path.relative_to(sample_dir.parent.parent))

            instances.append(
                {
                    "instance_index": inst_idx,
                    "prompt_index": prompt_index,
                    "prompt_text": prompt_spec.text,
                    "prompt_frame_index": prompt_spec.frame_index,
                    "score": score,
                    "box_xyxy": box,
                    "mask_path": mask_rel_path,
                }
            )

    return frame_to_instances, exported_mask_count


def process_source(
    predictor: Any,
    source_path: Path,
    prompt_specs: Sequence[PromptSpec],
    output_root: Path,
    mask_threshold: float,
    skip_mask_export: bool,
) -> Dict[str, Any]:
    start_response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": str(source_path),
        }
    )
    session_id = start_response.get("session_id")
    if session_id is None:
        raise RuntimeError(f"SAM3 did not return a session_id for source: {source_path}")

    sample_id = make_sample_id(source_path)
    sample_dir = output_root / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    merged_frames: Dict[int, List[Dict[str, Any]]] = {}
    total_exported_masks = 0

    for prompt_index, prompt_spec in enumerate(prompt_specs):
        response = predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": prompt_spec.frame_index,
                "text": prompt_spec.text,
            }
        )

        outputs = response.get("outputs")
        frame_map, exported_count = parse_response_instances(
            response_outputs=outputs,
            prompt_spec=prompt_spec,
            prompt_index=prompt_index,
            sample_dir=sample_dir,
            threshold=mask_threshold,
            skip_mask_export=skip_mask_export,
        )
        total_exported_masks += exported_count

        for frame_index, items in frame_map.items():
            merged_frames.setdefault(frame_index, []).extend(items)

    frame_entries = [
        {
            "frame_index": frame_index,
            "instances": instances,
        }
        for frame_index, instances in sorted(merged_frames.items(), key=lambda kv: kv[0])
    ]

    sample_metadata: Dict[str, Any] = {
        "sample_id": sample_id,
        "resource_path": str(source_path),
        "session_id": session_id,
        "prompts": [
            {
                "prompt_index": idx,
                "text": p.text,
                "frame_index": p.frame_index,
            }
            for idx, p in enumerate(prompt_specs)
        ],
        "frames": frame_entries,
        "num_frames_with_instances": len(frame_entries),
        "num_instances": sum(len(frame["instances"]) for frame in frame_entries),
        "num_exported_masks": total_exported_masks,
    }

    metadata_path = sample_dir / "metadata.json"
    metadata_path.write_text(json.dumps(sample_metadata, indent=2), encoding="utf-8")

    return {
        "sample_id": sample_id,
        "resource_path": str(source_path),
        "metadata_path": str(metadata_path.relative_to(output_root)),
        "num_frames_with_instances": sample_metadata["num_frames_with_instances"],
        "num_instances": sample_metadata["num_instances"],
        "num_exported_masks": sample_metadata["num_exported_masks"],
    }


def main() -> None:
    args = parse_args()

    sources = validate_sources(args.source)
    prompt_specs = build_prompt_specs(args.prompt, args.prompt_frame_index)

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    predictor = build_sam3_video_predictor()

    processed_samples: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for source in sources:
        try:
            sample_summary = process_source(
                predictor=predictor,
                source_path=source,
                prompt_specs=prompt_specs,
                output_root=output_root,
                mask_threshold=args.mask_threshold,
                skip_mask_export=args.skip_mask_export,
            )
            processed_samples.append(sample_summary)
            print(
                "[OK]",
                source,
                f"frames={sample_summary['num_frames_with_instances']}",
                f"instances={sample_summary['num_instances']}",
            )
        except Exception as exc:  # pragma: no cover
            msg = str(exc)
            errors.append({"resource_path": str(source), "error": msg})
            print(f"[ERROR] {source}: {msg}")
            if not args.continue_on_error:
                raise

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_stage": "stage1_raw_auto_generated",
        "generator": "sam3_video_raw_dataset_preprocessor",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompts": [
            {"text": p.text, "frame_index": p.frame_index}
            for p in prompt_specs
        ],
        "mask_threshold": args.mask_threshold,
        "skip_mask_export": args.skip_mask_export,
        "num_sources_requested": len(sources),
        "num_sources_processed": len(processed_samples),
        "num_errors": len(errors),
        "samples": processed_samples,
        "errors": errors,
    }

    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nManifest written to: {manifest_path}")
    print(f"Processed sources: {len(processed_samples)}/{len(sources)}")


if __name__ == "__main__":
    main()
