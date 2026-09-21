from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _normalize_caption(text: str, capitalize: bool) -> str:
    text = str(text).strip()
    return text.capitalize() if capitalize else text


def resolve_image_paths(images_dir: Path, filenames: list[str]) -> list[Path]:
    images_dir = Path(images_dir).resolve()
    if not images_dir.is_dir():
        raise NotADirectoryError(images_dir)

    all_files = [p for p in images_dir.rglob("*") if p.is_file()]
    basename_map: dict[str, list[Path]] = {}
    for path in all_files:
        basename_map.setdefault(path.name, []).append(path)

    resolved, missing, ambiguous = [], [], []
    for filename in filenames:
        direct = images_dir / filename
        if direct.is_file():
            resolved.append(direct.resolve())
            continue
        matches = basename_map.get(Path(filename).name, [])
        if len(matches) == 1:
            resolved.append(matches[0].resolve())
        elif len(matches) == 0:
            missing.append(filename)
        else:
            ambiguous.append((filename, [str(x) for x in matches[:5]]))

    if missing or ambiguous:
        raise FileNotFoundError(
            f"Image resolution failed: missing={len(missing)}, ambiguous={len(ambiguous)}; "
            f"missing_preview={missing[:10]!r}; ambiguous_preview={ambiguous[:3]!r}"
        )
    return resolved


def read_dataset(
    dataset_root: Path,
    dataset_json: Path,
    images_dir: Path,
    *,
    split: str = "test",
    capitalize_text: bool = False,
    expected_captions_per_image: int | None = 5,
) -> tuple[dict[str, Any], list[int], list[list[int]]]:
    dataset_root = Path(dataset_root).resolve()
    dataset_json = Path(dataset_json).resolve()
    images_dir = Path(images_dir).resolve()

    payload = json.loads(dataset_json.read_text(encoding="utf-8"))
    if "images" not in payload:
        raise KeyError("dataset JSON does not contain an 'images' field")

    images: list[dict[str, Any]] = []
    captions: list[dict[str, Any]] = []
    caption_to_image: list[int] = []
    positives: list[list[int]] = []

    for source_index, record in enumerate(payload["images"]):
        if record.get("split") != split:
            continue
        filename = record.get("filename")
        if not filename:
            raise ValueError(f"Record {source_index} has no filename")
        raw = [
            _normalize_caption(s.get("raw", ""), capitalize_text)
            for s in record.get("sentences", [])
        ]
        raw = [x for x in raw if x]
        if not raw:
            raise ValueError(f"No valid captions for {filename}")
        if expected_captions_per_image is not None and len(raw) != expected_captions_per_image:
            raise ValueError(
                f"Expected {expected_captions_per_image} captions for {filename}, got {len(raw)}"
            )

        image_index = len(images)
        group: list[int] = []
        images.append({
            "image_index": image_index,
            "source_index": source_index,
            "image_id": record.get("imgid", record.get("image_id", image_index)),
            "filename": str(filename),
            "caption_count": len(raw),
        })
        for caption in raw:
            caption_index = len(captions)
            captions.append({
                "caption_index": caption_index,
                "image_index": image_index,
                "caption": caption,
            })
            caption_to_image.append(image_index)
            group.append(caption_index)
        positives.append(group)

    if not images:
        raise RuntimeError(f"No images found for split={split!r}")

    paths = resolve_image_paths(images_dir, [x["filename"] for x in images])
    for rec, path in zip(images, paths):
        try:
            rel = path.relative_to(dataset_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Resolved image is outside dataset root and is not portable: {path}"
            ) from exc
        rec["resolved_path"] = rel.as_posix()

    metadata = {
        "schema": "ra-scg.query-metadata.v1",
        "split": split,
        "capitalize_text": bool(capitalize_text),
        "images": images,
        "captions": captions,
    }
    return metadata, caption_to_image, positives


def validate_metadata(metadata: dict[str, Any]) -> tuple[list[int], list[list[int]]]:
    images = metadata.get("images", [])
    captions = metadata.get("captions", [])
    if [int(x["image_index"]) for x in images] != list(range(len(images))):
        raise RuntimeError("image indices are not contiguous")
    if [int(x["caption_index"]) for x in captions] != list(range(len(captions))):
        raise RuntimeError("caption indices are not contiguous")

    caption_to_image = [int(x["image_index"]) for x in captions]
    positives = [[] for _ in images]
    for ci, ii in enumerate(caption_to_image):
        if not (0 <= ii < len(images)):
            raise RuntimeError(f"caption {ci} has invalid image index {ii}")
        positives[ii].append(ci)
    if any(len(g) != 5 for g in positives):
        bad = [(i, len(g)) for i, g in enumerate(positives) if len(g) != 5][:10]
        raise RuntimeError(f"expected five captions per image; bad groups={bad}")
    return caption_to_image, positives


def load_metadata(path: Path) -> tuple[dict[str, Any], list[int], list[list[int]]]:
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    cti, positives = validate_metadata(meta)
    return meta, cti, positives
