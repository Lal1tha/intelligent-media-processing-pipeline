"""Run bounded, stage-by-stage plate OCR diagnostics for a supplied image.

Usage inside the API container:
    python scripts/debug_plate_ocr.py /data/uploads/2026/08/example.jpg --output /tmp/plate-debug
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from app.analyzers import ocr
from app.analyzers.image_checks import plate_candidates_from_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/tmp/plate-debug"))
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Cannot decode image: {args.image}")
    args.output.mkdir(parents=True, exist_ok=True)
    ocr._configure_tesseract()
    regions = ocr.detect_plate_regions(image)
    print(f"candidate_regions={len(regions)} image_shape={image.shape}")
    annotated = image.copy()
    for index, region in enumerate(regions):
        x, y, width, height = region["bbox"]
        crop = region["crop"]
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 0, 255), 2)
        cv2.putText(annotated, str(index), (x, max(16, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        print(f"region={index} bbox={region['bbox']} score={region['score']} crop_shape={crop.shape} signals={region.get('signals', {})}")
        cv2.imwrite(str(args.output / f"region_{index}.png"), crop)
        for variant_index, variant in enumerate(ocr._plate_variants(crop)):
            cv2.imwrite(str(args.output / f"region_{index}_variant_{variant_index}.png"), variant)
            for config in ocr.PLATE_CONFIGS:
                text, confidence = ocr._ocr(variant, f"{config} -c tessedit_char_whitelist={ocr.PLATE_WHITELIST}")
                candidates = plate_candidates_from_text(text)
                print(f"region={index} variant={variant_index} config={config} confidence={confidence} text={text!r} candidates={candidates}")
    cv2.imwrite(str(args.output / "annotated_candidates.png"), annotated)
    print(f"wrote_debug_artifacts={args.output}")


if __name__ == "__main__":
    main()
