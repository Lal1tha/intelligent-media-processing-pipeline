"""Image quality, metadata, duplicate, tampering, and vehicle-number checks."""

from __future__ import annotations

import hashlib
import re
from itertools import product

import cv2
import numpy as np
from PIL import ExifTags, Image

from app.analyzers.common import gray, result
from app.core.config import settings


# ============================================================
# INDIAN VEHICLE REGISTRATION VALIDATION
# ============================================================

INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}

# Typical normalized registration lengths.
PLATE_LENGTHS = range(8, 11)

# Examples:
# KA01AB1234
# DL8CAF1234
# MH12DE1234
STRICT_PLATE_PATTERN = re.compile(
    r"(?<![A-Z0-9])"
    r"([A-Z]{2})"
    r"[\s\-_.:/]*"
    r"([0-9]{1,2})"
    r"[\s\-_.:/]*"
    r"([A-Z]{1,3})"
    r"[\s\-_.:/]*"
    r"([0-9]{1,4})"
    r"(?![A-Z0-9])"
)

# Bharat Series:
# 22BH1234AB
BH_SERIES_PATTERN = re.compile(
    r"(?<![A-Z0-9])"
    r"([0-9]{2})"
    r"[\s\-_.:/]*BH"
    r"[\s\-_.:/]*([0-9]{4})"
    r"[\s\-_.:/]*([A-Z]{1,2})"
    r"(?![A-Z0-9])"
)

# Common OCR mistakes where a character can look like a digit.
DIGIT_CORRECTIONS = {
    "O": "0",
    "I": "1",
    "S": "5",
    "B": "8",
    "Q": "0",
}

# Common OCR mistakes inside alphabetic series.
SERIES_CORRECTIONS = {
    "0": ("O",),
    "1": ("I", "W"),
    "5": ("S",),
    "8": ("B",),
    "6": ("G",),
}


# ============================================================
# BASIC IMAGE CHECKS
# ============================================================

def blur(image: np.ndarray) -> dict:
    """Estimate blur using Laplacian variance."""

    variance = float(
        cv2.Laplacian(gray(image), cv2.CV_64F).var()
    )

    warning = variance < settings.blur_threshold

    return result(
        "warning" if warning else "pass",
        (
            "Image appears potentially blurry based on Laplacian variance."
            if warning
            else "Sufficient edge detail detected."
        ),
        score=round(
            min(1, variance / settings.blur_threshold),
            3,
        ),
        confidence=0.82,
        measurement={
            "laplacian_variance": round(variance, 2)
        },
        threshold=settings.blur_threshold,
    )


def brightness(image: np.ndarray) -> dict:
    """Check overall brightness and extreme dark/bright pixels."""

    pixels = gray(image)

    mean = float(pixels.mean())
    dark = float((pixels < 40).mean() * 100)
    bright = float((pixels > 220).mean() * 100)

    warning = (
        mean < 55
        or dark > 65
        or mean > 210
        or bright > 65
    )

    message = (
        "Lighting is within a usable range."
        if not warning
        else (
            "Image is likely underexposed."
            if mean < 55 or dark > 65
            else "Image may be overexposed."
        )
    )

    return result(
        "warning" if warning else "pass",
        message,
        confidence=0.78,
        measurement={
            "mean_brightness": round(mean, 2),
            "dark_pixel_percent": round(dark, 2),
            "bright_pixel_percent": round(bright, 2),
            "contrast": round(float(pixels.std()), 2),
        },
        thresholds={
            "dark_mean": 55,
            "bright_mean": 210,
            "extreme_pixel_percent": 65,
        },
    )


def dimensions(image: np.ndarray) -> dict:
    """Check whether image dimensions meet minimum requirements."""

    height, width = image.shape[:2]
    pixels = width * height

    warning = (
        width < settings.min_image_width
        or height < settings.min_image_height
        or pixels < settings.min_image_pixels
    )

    return result(
        "warning" if warning else "pass",
        (
            "Image resolution may be too small for reliable analysis."
            if warning
            else "Image dimensions meet configured minimums."
        ),
        measurement={
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 3),
            "pixel_count": pixels,
        },
        thresholds={
            "min_width": settings.min_image_width,
            "min_height": settings.min_image_height,
            "min_pixels": settings.min_image_pixels,
        },
    )


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def perceptual_hash(image: np.ndarray) -> str:
    """Generate an average perceptual hash."""

    sample = cv2.resize(
        gray(image),
        (8, 8),
        interpolation=cv2.INTER_AREA,
    )

    avg = sample.mean()
    bits = (sample > avg).flatten()

    return f"{int(''.join('1' if x else '0' for x in bits), 2):016x}"


def hamming_distance(left: str, right: str) -> int:
    """Calculate Hamming distance between two hexadecimal hashes."""

    return (
        int(left, 16) ^ int(right, 16)
    ).bit_count()


def duplicate(
    sha256: str,
    phash: str,
    candidates: list[tuple[str, str]],
) -> dict:
    """Check exact and near duplicates."""

    if any(
        sha256 == other_sha
        for other_sha, _ in candidates
    ):
        return result(
            "warning",
            "Exact duplicate found using SHA-256.",
            classification="exact_duplicate",
            confidence=1.0,
        )

    distances = [
        hamming_distance(phash, other_phash)
        for _, other_phash in candidates
        if other_phash
    ]

    nearest = min(distances) if distances else None

    if nearest is not None and nearest <= 8:
        return result(
            "warning",
            "Near-duplicate signal found using average-hash distance.",
            classification="near_duplicate",
            confidence=round(1 - nearest / 64, 2),
            measurement={
                "nearest_hamming_distance": nearest
            },
            threshold=8,
        )

    return result(
        "pass",
        "No exact or near duplicate found among existing uploads.",
        classification="not_duplicate",
        measurement={
            "nearest_hamming_distance": nearest
        },
        threshold=8,
    )


# ============================================================
# OCR NORMALIZATION
# ============================================================

def normalize_ocr(text: str) -> str:
    """Normalize OCR text without changing characters."""

    return re.sub(
        r"\s+",
        " ",
        text.upper(),
    ).strip()


def normalize_plate(candidate: str) -> str:
    """Remove spaces and punctuation from a plate."""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        candidate.upper(),
    )


def plate_candidates_from_text(
    raw_text: str,
) -> list[dict]:
    """
    Extract Indian registration candidates from OCR.

    Uses:
    1. Strict matching first.
    2. BH-series matching.
    3. Conservative OCR correction fallback.
    """

    text = normalize_ocr(raw_text)

    candidates: list[dict] = []

    # --------------------------------------------------------
    # STRICT NORMAL INDIAN PLATE
    # --------------------------------------------------------

    for match in STRICT_PLATE_PATTERN.finditer(text):

        state, district, series, serial = match.groups()

        candidate = (
            f"{state}"
            f"{district}"
            f"{series}"
            f"{serial}"
        )

        if (
            state in INDIAN_STATES
            and len(candidate) in PLATE_LENGTHS
        ):
            candidates.append(
                {
                    "candidate": candidate,
                    "strict": True,
                    "correction_count": 0,
                }
            )

    # --------------------------------------------------------
    # BH SERIES
    # --------------------------------------------------------

    for match in BH_SERIES_PATTERN.finditer(text):

        year, serial, suffix = match.groups()

        candidate = (
            f"{year}"
            f"BH"
            f"{serial}"
            f"{suffix}"
        )

        if len(candidate) in PLATE_LENGTHS:
            candidates.append(
                {
                    "candidate": candidate,
                    "strict": True,
                    "correction_count": 0,
                }
            )

    # Strict result exists — don't correction-mine it.
    if candidates:
        return candidates

    # --------------------------------------------------------
    # SAFETY BOUND
    # --------------------------------------------------------

    # Relaxed from the earlier 32-character restriction.
    # This allows OCR output containing a little more surrounding
    # text while still preventing massive OCR garbage from being
    # brute-force searched.
    if len(text) > 32 or len(text.split()) > 4:
        return []

    compact = re.sub(
        r"[^A-Z0-9]",
        "",
        text,
    )

    # --------------------------------------------------------
    # CONSERVATIVE OCR CORRECTION
    # --------------------------------------------------------

    for start in range(
        max(0, len(compact) - 7)
    ):

        state = compact[start:start + 2]

        if state not in INDIAN_STATES:
            continue

        for district_size in (1, 2):

            for series_size in (1, 2, 3):

                for serial_size in (1, 2, 3, 4):

                    end = (
                        start
                        + 2
                        + district_size
                        + series_size
                        + serial_size
                    )

                    token = compact[start:end]

                    if len(token) not in PLATE_LENGTHS:
                        continue

                    district_raw = token[
                        2:2 + district_size
                    ]

                    series_start = (
                        2 + district_size
                    )

                    series_raw = token[
                        series_start:
                        series_start + series_size
                    ]

                    serial_raw = token[
                        series_start + series_size:
                    ]

                    district = "".join(
                        DIGIT_CORRECTIONS.get(
                            char,
                            char,
                        )
                        for char in district_raw
                    )

                    serial = "".join(
                        DIGIT_CORRECTIONS.get(
                            char,
                            char,
                        )
                        for char in serial_raw
                    )

                    if (
                        not district.isdigit()
                        or not serial.isdigit()
                    ):
                        continue

                    for parts in product(
                        *(
                            SERIES_CORRECTIONS.get(
                                char,
                                (char,),
                            )
                            for char in series_raw
                        )
                    ):

                        series = "".join(parts)

                        if not series.isalpha():
                            continue

                        original = (
                            district_raw
                            + series_raw
                            + serial_raw
                        )

                        corrected = (
                            district
                            + series
                            + serial
                        )

                        corrections = sum(
                            before != after
                            for before, after in zip(
                                original,
                                corrected,
                            )
                        )

                        # Maximum two OCR corrections.
                        if corrections <= 2:
                            candidates.append(
                                {
                                    "candidate": (
                                        f"{state}"
                                        f"{district}"
                                        f"{series}"
                                        f"{serial}"
                                    ),
                                    "strict": False,
                                    "correction_count": corrections,
                                }
                            )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    unique: dict[str, dict] = {}

    for candidate in candidates:

        previous = unique.get(
            candidate["candidate"]
        )

        if (
            previous is None
            or candidate["correction_count"]
            < previous["correction_count"]
        ):
            unique[
                candidate["candidate"]
            ] = candidate

    return list(unique.values())


def validate_indian_plate(
    raw_text: str,
) -> dict:
    """Validate a broad Indian or BH-series registration."""

    candidates = plate_candidates_from_text(
        raw_text
    )

    if not candidates:
        return result(
            "warning",
            "No broad Indian registration number was reliably detected.",
            classification="invalid_or_not_detected",
            raw_ocr_text=raw_text or "",
            normalized_candidate="",
            confidence=0.2,
            correction_applied=False,
        )

    best = min(
        candidates,
        key=lambda candidate: (
            candidate["correction_count"],
            not candidate["strict"],
        ),
    )

    # Strong strict match.
    if best["strict"]:
        return result(
            "pass",
            "Likely Indian vehicle registration number detected; OCR errors remain possible.",
            classification="likely_valid",
            raw_ocr_text=raw_text,
            normalized_candidate=best["candidate"],
            confidence=0.75,
            correction_applied=False,
        )

    # Corrected OCR candidate.
    return result(
        "warning",
        "Possible registration number detected after conservative OCR correction.",
        classification="uncertain",
        raw_ocr_text=raw_text,
        normalized_candidate=best["candidate"],
        confidence=0.45,
        correction_applied=True,
    )


# ============================================================
# SCREENSHOT CHECK
# ============================================================

def screenshot(image: np.ndarray) -> dict:
    """Heuristic screenshot detection."""

    h, w = image.shape[:2]

    common = (
        (w, h)
        in {
            (1080, 1920),
            (1920, 1080),
            (1170, 2532),
            (1280, 720),
            (1366, 768),
        }
    )

    edges = cv2.Canny(
        gray(image),
        100,
        200,
    )

    edge_density = float(
        (edges > 0).mean()
    )

    uniform = float(
        gray(image).std()
    ) < 35

    score = (
        (0.45 if common else 0)
        + min(edge_density * 3, 0.35)
        + (0.2 if uniform else 0)
    )

    return result(
        "warning" if score >= 0.55 else "pass",
        (
            "Possible screenshot signal; this is not a definitive classifier."
            if score >= 0.55
            else "No strong screenshot signal."
        ),
        classification=(
            "possible"
            if score >= 0.55
            else "unlikely"
        ),
        heuristic_score=round(score, 2),
        signals={
            "common_screen_dimensions": common,
            "edge_density": round(
                edge_density,
                3,
            ),
            "low_global_variance": uniform,
        },
    )


# ============================================================
# PHOTO-OF-PHOTO CHECK
# ============================================================

def photo_of_photo(image: np.ndarray) -> dict:
    """Look for strong rectangular/border signals."""

    h, w = image.shape[:2]

    edges = cv2.Canny(
        gray(image),
        80,
        180,
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(
            30,
            min(w, h) // 5,
        ),
        minLineLength=min(w, h) // 3,
        maxLineGap=15,
    )

    rectangles = (
        0
        if lines is None
        else len(lines)
    )

    border = (
        float(
            np.mean(
                edges[
                    :max(2, h // 20),
                    :
                ]
            )
            + np.mean(
                edges[
                    -max(2, h // 20):,
                    :
                ]
            )
            + np.mean(
                edges[
                    :,
                    :max(2, w // 20)
                ]
            )
            + np.mean(
                edges[
                    :,
                    -max(2, w // 20):
                ]
            )
        )
        / 4
    )

    score = min(
        1.0,
        rectangles / 20 * 0.6
        + (0.4 if border > 30 else 0),
    )

    return result(
        "warning" if score > 0.6 else "pass",
        (
            "Possible photographed-screen/print signal; manual review is needed."
            if score > 0.6
            else "No strong photo-of-photo signal."
        ),
        heuristic_score=round(score, 2),
        signals={
            "long_line_count": rectangles,
            "border_edge_strength": round(
                border,
                2,
            ),
        },
    )


# ============================================================
# METADATA CHECK
# ============================================================

def metadata(path: str) -> dict:
    """Inspect EXIF metadata for editing software."""

    with Image.open(path) as im:

        raw = im.getexif()

        data = {
            ExifTags.TAGS.get(
                key,
                str(key),
            ): str(value)[:200]
            for key, value in raw.items()
        }

    software = data.get("Software")

    return result(
        "warning" if software else "pass",
        (
            "Editing software metadata is present; this alone does not prove editing."
            if software
            else "No editing software metadata was found."
        ),
        exif_present=bool(data),
        camera_make=data.get("Make"),
        camera_model=data.get("Model"),
        capture_timestamp=(
            data.get("DateTimeOriginal")
            or data.get("DateTime")
        ),
        orientation=data.get("Orientation"),
        gps_present="GPSInfo" in data,
        software=software,
    )


# ============================================================
# TAMPERING CHECK
# ============================================================

def tampering(
    image: np.ndarray,
    metadata_check: dict,
) -> dict:
    """Lightweight image-tampering heuristic."""

    g = gray(image)

    blocks = [
        g[y:y + 32, x:x + 32].std()
        for y in range(
            0,
            g.shape[0] - 31,
            32,
        )
        for x in range(
            0,
            g.shape[1] - 31,
            32,
        )
    ]

    uniform_ratio = (
        float(
            np.mean(
                np.array(blocks) < 4
            )
        )
        if blocks
        else 0.0
    )

    score = min(
        1.0,
        uniform_ratio * 0.7
        + (
            0.3
            if metadata_check.get("software")
            else 0
        ),
    )

    return result(
        "warning" if score >= 0.55 else "pass",
        (
            "Possible editing/recompression signals detected; this is not forensic proof."
            if score >= 0.55
            else "No strong tampering signal from lightweight heuristics."
        ),
        classification=(
            "possible_tampering"
            if score >= 0.55
            else "no_strong_tampering_signal"
        ),
        heuristic_score=round(
            score,
            2,
        ),
        signals={
            "uniform_block_ratio": round(
                uniform_ratio,
                3,
            ),
            "editing_software_metadata": bool(
                metadata_check.get("software")
            ),
        },
    )