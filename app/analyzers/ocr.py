"""Diagnostic OCR pipeline for Indian vehicle-registration detection."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pytesseract

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None

from app.analyzers.common import result
from app.analyzers.image_checks import (
    normalize_ocr,
    plate_candidates_from_text,
)
from app.core.config import settings


# ---------------------------------------------------------------------------
# TESSERACT CONFIGURATION
# ---------------------------------------------------------------------------

PLATE_CONFIGS = (
    "--psm 6",
    "--psm 7",
    "--psm 8",
    "--psm 11",
    "--psm 13",
)

PLATE_WHITELIST = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

# Maximum number of plate regions considered.
MAX_PLATE_REGIONS = 10

# To avoid excessive OCR processing, only the strongest regions
# are sent through the full OCR pipeline.
OCR_REGIONS_TO_PROCESS = 5

DEBUG_DIR = Path(
    "/data/uploads/plate_debug"
)


# ---------------------------------------------------------------------------
# RAPIDOCR
# ---------------------------------------------------------------------------

_rapidocr_engine = None


def _get_rapidocr():
    """
    Create the RapidOCR engine lazily and reuse it.

    Loading the OCR model for every image would be expensive.
    """
    global _rapidocr_engine

    if RapidOCR is None:
        return None

    if _rapidocr_engine is None:
        _rapidocr_engine = RapidOCR()

    return _rapidocr_engine


def _rapidocr(
    image: np.ndarray,
) -> tuple[str, float | None]:
    """
    Run RapidOCR on an image.

    Returns:
        (recognized_text, normalized_confidence)
    """

    engine = _get_rapidocr()

    if engine is None:
        return "", None

    if image is None or image.size == 0:
        return "", None

    try:
        result_data, _ = engine(image)

        if not result_data:
            return "", None

        texts: list[str] = []
        confidences: list[float] = []

        for item in result_data:

            if len(item) < 3:
                continue

            text = str(item[1]).strip()

            try:
                confidence = float(item[2])
            except (
                ValueError,
                TypeError,
            ):
                confidence = 0.0

            if text:
                texts.append(text)

            if confidence > 0:
                confidences.append(
                    confidence
                )

        text = " ".join(texts)

        average_confidence = (
            round(
                sum(confidences)
                / len(confidences),
                3,
            )
            if confidences
            else None
        )

        return (
            text,
            average_confidence,
        )

    except Exception as exc:
        print(
            f"[RapidOCR ERROR] {exc}"
        )

        return "", None


# ---------------------------------------------------------------------------
# TESSERACT
# ---------------------------------------------------------------------------

def _configure_tesseract() -> None:
    """Configure the Tesseract executable when explicitly provided."""

    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = (
            settings.tesseract_cmd
        )


def _ocr(
    image: np.ndarray,
    config: str,
) -> tuple[str, float | None]:
    """Run Tesseract OCR and return text plus normalized confidence."""

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config=config,
    )

    words = [
        word.strip()
        for word in data["text"]
        if word.strip()
    ]

    confidences: list[float] = []

    for value in data["conf"]:
        try:
            confidence = float(value)
        except (
            ValueError,
            TypeError,
        ):
            continue

        if confidence >= 0:
            confidences.append(
                confidence
            )

    text = " ".join(words)

    average_confidence = (
        round(
            sum(confidences)
            / len(confidences)
            / 100,
            3,
        )
        if confidences
        else None
    )

    return (
        text,
        average_confidence,
    )


# ---------------------------------------------------------------------------
# FULL IMAGE OCR
# ---------------------------------------------------------------------------

def extract_text(
    image: np.ndarray,
) -> dict:
    """Full-image OCR fallback."""

    _configure_tesseract()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    enlarged = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    prepared = cv2.adaptiveThreshold(
        enlarged,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )

    # ---------------------------------------------------------------
    # Try RapidOCR first
    # ---------------------------------------------------------------

    try:
        rapid_text, rapid_confidence = (
            _rapidocr(enlarged)
        )

        if rapid_text:

            return result(
                "pass",
                "Full-image RapidOCR text extracted.",
                raw_text=rapid_text,
                normalized_text=normalize_ocr(
                    rapid_text
                ),
                meaningful_text=True,
                confidence=rapid_confidence,
                engine="rapidocr",
            )

    except Exception as exc:

        print(
            f"[RapidOCR FULL IMAGE ERROR] "
            f"{exc}"
        )

    # ---------------------------------------------------------------
    # Tesseract fallback
    # ---------------------------------------------------------------

    try:

        text, confidence = _ocr(
            prepared,
            "--psm 6",
        )

        return result(
            "pass" if text else "warning",
            (
                "Full-image OCR text extracted."
                if text
                else (
                    "No meaningful full-image "
                    "OCR text detected."
                )
            ),
            raw_text=text,
            normalized_text=normalize_ocr(
                text
            ),
            meaningful_text=bool(text),
            confidence=confidence,
            engine="tesseract",
        )

    except (
        pytesseract.TesseractNotFoundError,
        pytesseract.TesseractError,
        RuntimeError,
    ) as exc:

        return result(
            "unavailable",
            (
                "OCR could not run; "
                "Tesseract may be unavailable."
            ),
            raw_text="",
            normalized_text="",
            meaningful_text=False,
            error=str(exc)[:180],
        )


# ---------------------------------------------------------------------------
# IOU
# ---------------------------------------------------------------------------

def _iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    """Calculate intersection-over-union for two bounding boxes."""

    ax, ay, aw, ah = first
    bx, by, bw, bh = second

    left = max(
        ax,
        bx,
    )

    top = max(
        ay,
        by,
    )

    right = min(
        ax + aw,
        bx + bw,
    )

    bottom = min(
        ay + ah,
        by + bh,
    )

    intersection = (
        max(
            0,
            right - left,
        )
        * max(
            0,
            bottom - top,
        )
    )

    union = (
        aw * ah
        + bw * bh
        - intersection
    )

    return (
        intersection / union
        if union
        else 0.0
    )


# ---------------------------------------------------------------------------
# YELLOW PLATE DETECTION
# ---------------------------------------------------------------------------

def _yellow_dark_text_mask(
    image: np.ndarray,
    gray: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect yellow plate areas and dark text appearing over yellow.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    hsv_yellow = cv2.inRange(
        hsv,
        np.array(
            [15, 45, 55]
        ),
        np.array(
            [45, 255, 255]
        ),
    )

    lab_yellow = cv2.inRange(
        lab[:, :, 2],
        145,
        255,
    )

    yellow = cv2.bitwise_and(
        hsv_yellow,
        lab_yellow,
    )

    if gray is None:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    dark = cv2.threshold(
        gray,
        115,
        255,
        cv2.THRESH_BINARY_INV,
    )[1]

    yellow_neighbourhood = cv2.dilate(
        yellow,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (9, 7),
        ),
    )

    dark_on_yellow = cv2.bitwise_and(
        dark,
        yellow_neighbourhood,
    )

    dark_on_yellow = cv2.morphologyEx(
        dark_on_yellow,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (13, 5),
        ),
    )

    return (
        yellow,
        dark_on_yellow,
    )


# ---------------------------------------------------------------------------
# PLATE REGION DETECTION
# ---------------------------------------------------------------------------

def detect_plate_regions(
    image: np.ndarray,
    limit: int = MAX_PLATE_REGIONS,
) -> list[dict]:
    """Find likely vehicle-registration plate regions."""

    height, width = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8),
    ).apply(gray)

    filtered = cv2.bilateralFilter(
        clahe,
        7,
        55,
        55,
    )

    canny = cv2.Canny(
        filtered,
        55,
        170,
    )

    gradient = cv2.morphologyEx(
        filtered,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3),
        ),
    )

    gradient = cv2.threshold(
        gradient,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )[1]

    adaptive = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        8,
    )

    yellow, dark_on_yellow = (
        _yellow_dark_text_mask(
            image,
            filtered,
        )
    )

    text_edges = cv2.bitwise_or(
        canny,
        cv2.bitwise_or(
            gradient,
            adaptive,
        ),
    )

    horizontal = cv2.morphologyEx(
        text_edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (35, 7),
        ),
    )

    square = cv2.morphologyEx(
        text_edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (9, 9),
        ),
    )

    yellow_text = cv2.morphologyEx(
        dark_on_yellow,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (35, 9),
        ),
    )

    masks = (
        horizontal,
        square,
        yellow_text,
    )

    sobel_x = cv2.convertScaleAbs(
        cv2.Sobel(
            gray,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )
    )

    min_width = max(
        28,
        width // 45,
    )

    min_height = max(
        10,
        height // 120,
    )

    candidates: list[dict] = []

    for mask_index, mask in enumerate(
        masks
    ):

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:

            x, y, box_width, box_height = (
                cv2.boundingRect(contour)
            )

            area = (
                box_width
                * box_height
            )

            ratio = (
                box_width
                / max(
                    box_height,
                    1,
                )
            )

            if (
                box_width < min_width
                or box_height < min_height
                or not 1.1 <= ratio <= 12.0
            ):
                continue

            if (
                area
                > height * width * 0.22
                or box_height
                > height * 0.38
            ):
                continue

            rectangularity = (
                cv2.contourArea(contour)
                / max(
                    area,
                    1,
                )
            )

            crop_gray = gray[
                y : y + box_height,
                x : x + box_width,
            ]

            vertical_edges = float(
                (
                    sobel_x[
                        y : y + box_height,
                        x : x + box_width,
                    ]
                    > 40
                ).mean()
            )

            text_density = float(
                (
                    text_edges[
                        y : y + box_height,
                        x : x + box_width,
                    ]
                    > 0
                ).mean()
            )

            yellow_support = float(
                (
                    yellow[
                        y : y + box_height,
                        x : x + box_width,
                    ]
                    > 0
                ).mean()
            )

            dark_on_yellow_support = float(
                (
                    dark_on_yellow[
                        y : y + box_height,
                        x : x + box_width,
                    ]
                    > 0
                ).mean()
            )

            contrast = (
                float(
                    crop_gray.std()
                )
                / 128.0
            )

            ratio_score = max(
                0.0,
                1
                - abs(
                    ratio - 3.5
                )
                / 5.0,
            )

            score = (
                0.27 * ratio_score
                + 0.15
                * min(
                    1.0,
                    rectangularity,
                )
                + 0.18
                * min(
                    1.0,
                    vertical_edges * 4,
                )
                + 0.15
                * min(
                    1.0,
                    text_density * 4,
                )
                + 0.07
                * min(
                    1.0,
                    contrast,
                )
                + 0.08
                * min(
                    1.0,
                    yellow_support * 3,
                )
                + 0.10
                * min(
                    1.0,
                    dark_on_yellow_support
                    * 6,
                )
            )

            padding_x = max(
                12,
                box_width // 4,
            )

            padding_y = max(
                6,
                box_height // 2,
            )

            left = max(
                0,
                x - padding_x,
            )

            top = max(
                0,
                y - padding_y,
            )

            right = min(
                width,
                x + box_width + padding_x,
            )

            bottom = min(
                height,
                y + box_height + padding_y,
            )

            candidates.append(
                {
                    "crop": image[
                        top:bottom,
                        left:right,
                    ],
                    "bbox": [
                        left,
                        top,
                        right - left,
                        bottom - top,
                    ],
                    "score": round(
                        score,
                        3,
                    ),
                    "mask": mask_index,
                    "signals": {
                        "rectangularity": round(
                            rectangularity,
                            3,
                        ),
                        "vertical_edge_density": round(
                            vertical_edges,
                            3,
                        ),
                        "text_edge_density": round(
                            text_density,
                            3,
                        ),
                        "yellow_support": round(
                            yellow_support,
                            3,
                        ),
                        "dark_on_yellow_support": round(
                            dark_on_yellow_support,
                            3,
                        ),
                        "contrast": round(
                            contrast,
                            3,
                        ),
                    },
                }
            )

    candidates.sort(
        key=lambda candidate: candidate[
            "score"
        ],
        reverse=True,
    )

    selected: list[dict] = []

    for candidate in candidates:

        if any(
            _iou(
                tuple(
                    candidate["bbox"]
                ),
                tuple(
                    other["bbox"]
                ),
            )
            > 0.55
            for other in selected
        ):
            continue

        selected.append(
            candidate
        )

        if len(selected) >= limit:
            break

    return selected


# ---------------------------------------------------------------------------
# PADDING
# ---------------------------------------------------------------------------

def _add_padding(
    crop: np.ndarray,
    percent: float,
) -> np.ndarray:
    """Add replicated border around a crop."""

    if (
        crop is None
        or crop.size == 0
    ):
        return crop

    h, w = crop.shape[:2]

    px = max(
        1,
        int(w * percent),
    )

    py = max(
        1,
        int(h * percent),
    )

    return cv2.copyMakeBorder(
        crop,
        py,
        py,
        px,
        px,
        cv2.BORDER_REPLICATE,
    )


# ---------------------------------------------------------------------------
# PLATE PREPROCESSING
# ---------------------------------------------------------------------------

def _plate_variants(
    crop: np.ndarray,
) -> list[
    tuple[str, np.ndarray]
]:
    """Generate multiple preprocessing variants for OCR."""

    if (
        crop is None
        or crop.size == 0
    ):
        return []

    variants: list[
        tuple[str, np.ndarray]
    ] = []

    for padding in (
        0.00,
        0.10,
        0.20,
        0.30,
    ):

        padded = _add_padding(
            crop,
            padding,
        )

        gray = cv2.cvtColor(
            padded,
            cv2.COLOR_BGR2GRAY,
        )

        enlarged = cv2.resize(
            gray,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC,
        )

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8),
        ).apply(enlarged)

        denoised = cv2.GaussianBlur(
            clahe,
            (3, 3),
            0,
        )

        otsu = cv2.threshold(
            denoised,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU,
        )[1]

        adaptive = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )

        variants.extend(
            [
                (
                    f"pad{padding}_gray",
                    enlarged,
                ),
                (
                    f"pad{padding}_clahe",
                    clahe,
                ),
                (
                    f"pad{padding}_otsu",
                    otsu,
                ),
                (
                    f"pad{padding}_otsu_inv",
                    cv2.bitwise_not(
                        otsu
                    ),
                ),
                (
                    f"pad{padding}_adaptive",
                    adaptive,
                ),
            ]
        )

        _, dark_on_yellow = (
            _yellow_dark_text_mask(
                padded,
                gray,
            )
        )

        yellow_dark = cv2.resize(
            dark_on_yellow,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC,
        )

        variants.extend(
            [
                (
                    f"pad{padding}_yellow_dark",
                    yellow_dark,
                ),
                (
                    f"pad{padding}_yellow_dark_inv",
                    cv2.bitwise_not(
                        yellow_dark
                    ),
                ),
            ]
        )

    return variants


# ---------------------------------------------------------------------------
# DEBUG IMAGE
# ---------------------------------------------------------------------------

def _save_debug_image(
    path: Path,
    image: np.ndarray,
) -> None:
    """Save debug image without breaking the pipeline."""

    try:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(path),
            image,
        )

    except Exception as exc:

        print(
            f"[DEBUG] Could not save "
            f"{path}: {exc}"
        )


# ---------------------------------------------------------------------------
# VEHICLE NUMBER DETECTION
# ---------------------------------------------------------------------------

def detect_vehicle_number(
    image: np.ndarray,
    full_image_ocr: dict,
) -> dict:
    """Detect and validate a likely Indian vehicle registration number."""

    _configure_tesseract()

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # CLEAR OLD DEBUG FILES
    # ------------------------------------------------------------------

    for pattern in (
        "region_*.jpg",
        "ocr_*.txt",
    ):

        for old_file in DEBUG_DIR.glob(
            pattern
        ):

            try:
                old_file.unlink()
            except Exception:
                pass

    full_text = (
        full_image_ocr.get(
            "raw_text",
            "",
        )
        .strip()
    )

    attempts: list[dict] = []

    # ------------------------------------------------------------------
    # FULL IMAGE OCR
    # ------------------------------------------------------------------

    if full_text:

        attempts.append(
            {
                "raw_text": full_text,
                "ocr_confidence": (
                    full_image_ocr.get(
                        "confidence"
                    )
                ),
                "source": "full_image",
                "region_score": 0.0,
                "support_key": (
                    "full_image",
                    "default",
                ),
            }
        )

    # ------------------------------------------------------------------
    # DETECT PLATE REGIONS
    # ------------------------------------------------------------------

    regions = detect_plate_regions(
        image
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PLATE DETECTION DEBUG"
    )

    print(
        "=" * 70
    )

    print(
        f"Image size: "
        f"{image.shape[1]}x"
        f"{image.shape[0]}"
    )

    print(
        f"Regions found: "
        f"{len(regions)}"
    )

    print(
        f"Debug directory: "
        f"{DEBUG_DIR}"
    )

    print(
        "=" * 70
    )

    # ------------------------------------------------------------------
    # SAVE ALL CANDIDATE REGIONS
    # ------------------------------------------------------------------

    for index, region in enumerate(
        regions
    ):

        print(
            f"REGION {index}: "
            f"bbox={region['bbox']} "
            f"score={region['score']} "
            f"signals={region.get('signals', {})}"
        )

        _save_debug_image(
            DEBUG_DIR
            / f"region_{index}.jpg",
            region["crop"],
        )

    print(
        "=" * 70
    )

    # ------------------------------------------------------------------
    # ONLY PROCESS STRONGEST REGIONS
    #
    # This prevents hundreds/thousands of OCR calls on a single image.
    # ------------------------------------------------------------------

    ocr_regions = regions[
        :OCR_REGIONS_TO_PROCESS
    ]

    # ------------------------------------------------------------------
    # OCR EACH REGION
    # ------------------------------------------------------------------

    for region_index, region in enumerate(
        ocr_regions
    ):

        variants = _plate_variants(
            region["crop"]
        )

        print(
            f"\nRegion {region_index}: "
            f"{len(variants)} "
            f"preprocessing variants"
        )

        # --------------------------------------------------------------
        # TEST COMPATIBILITY
        # --------------------------------------------------------------

        for variant_index, item in enumerate(
            variants
        ):

            if (
                isinstance(item, tuple)
                and len(item) == 2
            ):

                variant_name, variant = item

            else:

                variant_name = (
                    f"variant_{variant_index}"
                )

                variant = item

            # ----------------------------------------------------------
            # RAPIDOCR FIRST
            # ----------------------------------------------------------

            try:

                rapid_text, rapid_confidence = (
                    _rapidocr(
                        variant
                    )
                )

            except Exception as exc:

                print(
                    f"[RapidOCR ERROR] "
                    f"region="
                    f"{region_index} "
                    f"variant="
                    f"{variant_name}: "
                    f"{exc}"
                )

                rapid_text = ""
                rapid_confidence = None

            if rapid_text:

                rapid_candidates = (
                    plate_candidates_from_text(
                        rapid_text
                    )
                )

                print(
                    f"RapidOCR "
                    f"region="
                    f"{region_index} "
                    f"variant="
                    f"{variant_name} "
                    f"confidence="
                    f"{rapid_confidence} "
                    f"text="
                    f"{rapid_text!r} "
                    f"candidates="
                    f"{rapid_candidates}"
                )

                attempts.append(
                    {
                        "raw_text": rapid_text,
                        "ocr_confidence": (
                            rapid_confidence
                        ),
                        "source": (
                            f"rapidocr_region_"
                            f"{region_index}"
                        ),
                        "region_score": (
                            region["score"]
                        ),
                        "support_key": (
                            "rapidocr",
                            region_index,
                            variant_index,
                        ),
                    }
                )

            # ----------------------------------------------------------
            # TESSERACT FALLBACK / SECOND OPINION
            # ----------------------------------------------------------

            for config in PLATE_CONFIGS:

                try:

                    ocr_config = (
                        f"{config} "
                        f"-c "
                        f"tessedit_char_whitelist="
                        f"{PLATE_WHITELIST}"
                    )

                    text, confidence = _ocr(
                        variant,
                        ocr_config,
                    )

                except (
                    pytesseract.TesseractNotFoundError,
                    pytesseract.TesseractError,
                    RuntimeError,
                ) as exc:

                    print(
                        f"[Tesseract ERROR] "
                        f"region="
                        f"{region_index} "
                        f"variant="
                        f"{variant_name} "
                        f"config="
                        f"{config}: "
                        f"{exc}"
                    )

                    continue

                if not text:
                    continue

                candidates = (
                    plate_candidates_from_text(
                        text
                    )
                )

                print(
                    f"Tesseract "
                    f"region="
                    f"{region_index} "
                    f"variant="
                    f"{variant_name} "
                    f"config="
                    f"{config} "
                    f"confidence="
                    f"{confidence} "
                    f"text="
                    f"{text!r} "
                    f"candidates="
                    f"{candidates}"
                )

                attempts.append(
                    {
                        "raw_text": text,
                        "ocr_confidence": confidence,
                        "source": (
                            f"plate_region_"
                            f"{region_index}"
                        ),
                        "region_score": (
                            region["score"]
                        ),
                        "support_key": (
                            region_index,
                            variant_index,
                            config,
                        ),
                    }
                )

    # ------------------------------------------------------------------
    # SAVE OCR OUTPUT
    # ------------------------------------------------------------------

    try:

        with open(
            DEBUG_DIR
            / "ocr_results.txt",
            "w",
            encoding="utf-8",
        ) as file:

            for attempt in attempts:

                file.write(
                    f"{attempt}\n"
                )

    except Exception as exc:

        print(
            f"[DEBUG] Could not save "
            f"OCR results: {exc}"
        )

    # ------------------------------------------------------------------
    # GROUP CANDIDATES
    # ------------------------------------------------------------------

    grouped = defaultdict(list)

    for attempt in attempts:

        for candidate in (
            plate_candidates_from_text(
                attempt["raw_text"]
            )
        ):

            # Full-image OCR is noisy.
            # Only accept strict candidates from it.

            if (
                attempt["source"]
                == "full_image"
                and not candidate["strict"]
            ):
                continue

            grouped[
                candidate["candidate"]
            ].append(
                {
                    **attempt,
                    **candidate,
                }
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PLATE CANDIDATE SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Total OCR attempts: "
        f"{len(attempts)}"
    )

    print(
        f"Candidates found: "
        f"{len(grouped)}"
    )

    for candidate, entries in (
        grouped.items()
    ):

        print(
            f"{candidate}: "
            f"{len(entries)} "
            f"supporting OCR reads"
        )

    print(
        "=" * 70
    )

    # ------------------------------------------------------------------
    # NO CANDIDATE
    # ------------------------------------------------------------------

    if not grouped:

        return result(
            "warning",
            (
                "No broad Indian registration "
                "number was reliably detected."
            ),
            classification=(
                "invalid_or_not_detected"
            ),
            raw_ocr_text=full_text,
            normalized_candidate="",
            confidence=0.2,
            evidence={
                "plate_regions_considered": (
                    len(regions)
                ),
                "ocr_regions_processed": (
                    len(ocr_regions)
                ),
                "ocr_attempts": (
                    len(attempts)
                ),
                "matching_attempts": 0,
                "debug_directory": str(
                    DEBUG_DIR
                ),
            },
        )

    # ------------------------------------------------------------------
    # SCORE CANDIDATE GROUPS
    # ------------------------------------------------------------------

    def score(
        group: tuple[
            str,
            list[dict],
        ],
    ) -> tuple[
        float,
        int,
    ]:

        _, entries = group

        supports = {
            entry["support_key"]
            for entry in entries
        }

        direct = sum(
            bool(
                entry["strict"]
            )
            for entry in entries
        )

        region_strength = max(
            (
                entry.get(
                    "region_score",
                    0.0,
                )
                for entry in entries
            ),
            default=0.0,
        )

        corrections = min(
            entry[
                "correction_count"
            ]
            for entry in entries
        )

        return (
            direct * 2.0
            + len(supports) * 0.35
            + region_strength * 0.3
            - corrections * 0.25,
            direct,
        )

    candidate, entries = max(
        grouped.items(),
        key=score,
    )

    # ------------------------------------------------------------------
    # CONSENSUS
    # ------------------------------------------------------------------

    support_keys = {
        entry["support_key"]
        for entry in entries
    }

    direct = any(
        entry["strict"]
        for entry in entries
    )

    from_region = any(
        entry["source"].startswith(
            (
                "plate_region",
                "rapidocr_region",
            )
        )
        for entry in entries
    )

    best_entry = max(
        entries,
        key=lambda entry: (
            entry["strict"],
            entry.get(
                "region_score",
                0,
            ),
            entry[
                "ocr_confidence"
            ]
            or 0,
        ),
    )

    valid_confidences = [
        entry[
            "ocr_confidence"
        ]
        for entry in entries
        if entry[
            "ocr_confidence"
        ]
        is not None
    ]

    average_ocr = (
        sum(valid_confidences)
        / len(valid_confidences)
        if valid_confidences
        else 0.0
    )

    # ------------------------------------------------------------------
    # ENGINE AGREEMENT BONUS
    # ------------------------------------------------------------------

    engines = set()

    for entry in entries:

        source = entry[
            "source"
        ]

        if source.startswith(
            "rapidocr_region"
        ):
            engines.add(
                "rapidocr"
            )

        elif source.startswith(
            "plate_region"
        ):
            engines.add(
                "tesseract"
            )

    engine_agreement_bonus = (
        0.06
        if len(engines) >= 2
        else 0.0
    )

    # ------------------------------------------------------------------
    # FINAL CONFIDENCE
    # ------------------------------------------------------------------

    confidence = min(
        0.95,
        0.35
        + (
            0.32
            if direct
            else 0.0
        )
        + (
            0.12
            if from_region
            else 0.0
        )
        + engine_agreement_bonus
        + min(
            0.12,
            0.04
            * (
                len(support_keys)
                - 1
            ),
        )
        + 0.12
        * average_ocr
        - 0.08
        * min(
            entry[
                "correction_count"
            ]
            for entry in entries
        ),
    )

    # ------------------------------------------------------------------
    # FINAL DECISION
    # ------------------------------------------------------------------

    likely = (
        direct
        and (
            from_region
            or (
                len(entries) == 1
                and best_entry[
                    "source"
                ]
                == "full_image"
            )
        )
        and confidence >= 0.65
    )

    return result(
        "pass"
        if likely
        else "warning",
        (
            "Likely Indian vehicle "
            "registration number detected."
            if likely
            else (
                "Possible vehicle registration "
                "number detected, but OCR "
                "confidence is low."
            )
        ),
        classification=(
            "likely_valid"
            if likely
            else "uncertain"
        ),
        raw_ocr_text=best_entry[
            "raw_text"
        ],
        normalized_candidate=candidate,
        confidence=round(
            confidence,
            3,
        ),
        evidence={
            "plate_regions_considered": (
                len(regions)
            ),
            "ocr_regions_processed": (
                len(ocr_regions)
            ),
            "ocr_attempts": (
                len(attempts)
            ),
            "matching_attempts": (
                len(support_keys)
            ),
            "ocr_engines": sorted(
                engines
            ),
            "engine_agreement": (
                len(engines) >= 2
            ),
            "sources": sorted(
                {
                    entry["source"]
                    for entry in entries
                }
            ),
            "debug_directory": str(
                DEBUG_DIR
            ),
        },
    )