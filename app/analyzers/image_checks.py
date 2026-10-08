"""Image quality, metadata, duplicate, tampering, and vehicle-number checks."""



from __future__ import annotations
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
# Normalized plate lengths.

#

# Examples:

#   KA01AB1234 -> 10

#   DL8CAF1234 -> 10

#   MH12DE1234 -> 10

#

# Some valid registrations can be shorter, so we allow 8-10.

PLATE_LENGTHS = range(8, 11)





# ------------------------------------------------------------

# Standard Indian registration

# ------------------------------------------------------------

STRICT_PLATE_PATTERN = re.compile(

    r"(?<![A-Z0-9])"

    r"([A-Z]{2})"

    r"[\s\\-_.:/]*"

    r"([0-9]{1,2})"

    r"[\s\\-_.:/]*"

    r"([A-Z]{1,3})"

    r"[\s\\-_.:/]*"

    r"([0-9]{1,4})"

    r"(?![A-Z0-9])"

)





# ------------------------------------------------------------

# Bharat Series

#

# Examples:

#   22BH1234AB

#   24BH1234AA

# ------------------------------------------------------------

BH_SERIES_PATTERN = re.compile(

    r"(?<![A-Z0-9])"

    r"([0-9]{2})"

    r"[\s\\-_.:/]*BH"

    r"[\s\\-_.:/]*"

    r"([0-9]{4})"

    r"[\s\\-_.:/]*"

    r"([A-Z]{1,2})"

    r"(?![A-Z0-9])"

)





# ============================================================

# OCR CORRECTIONS

# ============================================================



# Corrections for positions that should contain digits.

DIGIT_CORRECTIONS = {

    "O": "0",

    "I": "1",

    "L": "1",

    "S": "5",

    "B": "8",

    "Q": "0",

    "G": "6",

}



# Corrections for positions that should contain letters.

SERIES_CORRECTIONS = {

    "0": ("O",),

    "1": ("I", "W"),

    "5": ("S",),

    "8": ("B",),

    "6": ("G",),

}

# ============================================================
# ENHANCED PLATE NORMALIZATION
# ============================================================

STATE_CODES = INDIAN_STATES


# Supported Indian registration formats.
SPECIFIC_PLATE_PATTERNS = (
    # Normal registration: KA01AB1234, DL8CAF1234, MH12DE1234
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"),

    # Bharat Series: 22BH1234AB / 24BH1234AA
    re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"),
)


# Vehicle-brand text that can appear around a plate.
VEHICLE_BRAND_KEYWORDS = {
    "HONDA", "TOYOTA", "HYUNDAI", "SUZUKI", "YAMAHA",
    "KAWASAKI", "DUCATI", "HARLEY", "NISSAN", "CHEVROLET",
    "VOLKSWAGEN", "BMW", "AUDI", "MERCEDES", "FORD", "TESLA",
    "MAZDA", "RENAULT", "PEUGEOT", "JEEP", "PORSCHE", "FERRARI",
    "ROYALENFIELD", "ENFIELD", "HERO", "TVS", "KTM", "BAJAJ",
    "VESPA", "PIAGGIO", "MAHINDRA", "TATA", "EICHER",
}


# Common OCR garbage / surrounding text.
NOISE_WORDS = {
    "IMD", "AGE", "THE", "NEW", "ALL", "EVE", "OUT", "SUP",
    "LIVE", "HARD", "SOFT", "MUSIC", "RACE", "SAFE", "DARE",
    "PACE", "LOGO",
}


NOISE_PREFIXES = (
    "HTTP", "HTTPS", "WWW", "IMEI", "LAT", "LONG", "TASK", "TEL",
)


# OCR substitutions when a position should contain a digit.
CHAR_SUB_TO_DIGIT = {
    "O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2",
    "S": "5", "G": "6", "T": "7", "B": "8",
}


# OCR substitutions when a position should contain a letter.
CHAR_SUB_TO_ALPHA = {
    "0": "O", "1": "I", "2": "Z", "5": "S",
    "6": "G", "7": "T", "8": "B",
}


def _is_noise_token(token: str) -> bool:
    """Reject obvious OCR noise, metadata and vehicle-brand text."""

    token = normalize_plate(token)

    if not token:
        return True

    if token in NOISE_WORDS:
        return True

    if any(token.startswith(prefix) for prefix in NOISE_PREFIXES):
        return True

    # Reject exact brand names. Avoid substring matching here so that
    # a legitimate plate is not accidentally discarded.
    if token in VEHICLE_BRAND_KEYWORDS:
        return True

    # Typical Indian mobile-number-like OCR output.
    if len(token) == 10 and token[0] in "6789" and token.isdigit():
        return True

    return False


def _matches_specific_plate_pattern(candidate: str) -> bool:
    """Check whether a normalized candidate has a supported format."""

    candidate = normalize_plate(candidate)

    if candidate[:2] in STATE_CODES:
        return any(
            pattern.fullmatch(candidate)
            for pattern in SPECIFIC_PLATE_PATTERNS
            if "^[A-Z]" in pattern.pattern
        )

    return bool(
        re.fullmatch(
            r"[0-9]{2}BH[0-9]{4}[A-Z]{1,2}",
            candidate,
        )
    )


def normalize_indian_plate_candidate(token: str) -> list[dict]:
    """Normalize one OCR token into conservative Indian plate candidates."""

    token = normalize_plate(token)

    if _is_noise_token(token):
        return []

    candidates: list[dict] = []

    # --------------------------------------------------------
    # 1. Exact standard/Bharat registration.
    # --------------------------------------------------------

    exact = _strict_candidates(token)
    if exact:
        return exact

    # --------------------------------------------------------
    # 2. Bharat Series OCR correction.
    # --------------------------------------------------------

    if len(token) in (9, 10) and token[2:4] == "BH":
        year = token[:2]
        serial_raw = token[4:8]
        suffix_raw = token[8:]

        if year.isdigit() and len(serial_raw) == 4:
            serial = "".join(
                CHAR_SUB_TO_DIGIT.get(char, char)
                for char in serial_raw
            )

            if serial.isdigit() and 1 <= len(suffix_raw) <= 2:
                suffix_parts = []
                corrections = 0

                for char in suffix_raw:
                    if char.isalpha():
                        suffix_parts.append(char)
                    elif char in CHAR_SUB_TO_ALPHA:
                        suffix_parts.append(CHAR_SUB_TO_ALPHA[char])
                        corrections += 1
                    else:
                        suffix_parts.append(char)

                suffix = "".join(suffix_parts)

                candidate = year + "BH" + serial + suffix

                if (
                    suffix.isalpha()
                    and len(candidate) in PLATE_LENGTHS
                    and corrections <= 2
                    and _matches_specific_plate_pattern(candidate)
                ):
                    _add_candidate(
                        candidates,
                        candidate,
                        strict=False,
                        correction_count=corrections,
                    )

    # --------------------------------------------------------
    # 3. Standard Indian registration OCR correction.
    # --------------------------------------------------------

    if len(token) < 8 or len(token) > 10:
        return candidates

    # The state itself should be a valid state code. We do not blindly
    # convert the first two characters because that can create false states.
    state = token[:2]
    if state not in STATE_CODES:
        return candidates

    for district_size in (1, 2):
        for series_size in (1, 2, 3):
            for serial_size in (1, 2, 3, 4):
                total_size = (
                    2 + district_size + series_size + serial_size
                )

                if total_size != len(token):
                    continue

                district_start = 2
                district_end = district_start + district_size
                series_start = district_end
                series_end = series_start + series_size

                district_raw = token[district_start:district_end]
                series_raw = token[series_start:series_end]
                serial_raw = token[series_end:]

                district = "".join(
                    CHAR_SUB_TO_DIGIT.get(char, char)
                    for char in district_raw
                )
                serial = "".join(
                    CHAR_SUB_TO_DIGIT.get(char, char)
                    for char in serial_raw
                )

                if not district.isdigit() or not serial.isdigit():
                    continue

                # Every series character must become a letter. Generate
                # only one conservative substitution per character.
                possibilities = []
                for char in series_raw:
                    if char.isalpha():
                        possibilities.append((char,))
                    elif char in CHAR_SUB_TO_ALPHA:
                        possibilities.append((CHAR_SUB_TO_ALPHA[char],))
                    else:
                        possibilities.append((char,))

                for parts in product(*possibilities):
                    series = "".join(parts)
                    if not series.isalpha():
                        continue

                    original = district_raw + series_raw + serial_raw
                    corrected = district + series + serial

                    corrections = sum(
                        before != after
                        for before, after in zip(original, corrected)
                    )

                    if corrections > 2:
                        continue

                    candidate = state + district + series + serial

                    if not _matches_specific_plate_pattern(candidate):
                        continue

                    _add_candidate(
                        candidates,
                        candidate,
                        strict=False,
                        correction_count=corrections,
                    )

    # Deduplicate within this helper.
    unique: dict[str, dict] = {}
    for candidate in candidates:
        key = candidate["candidate"]
        previous = unique.get(key)
        if (
            previous is None
            or candidate["correction_count"] < previous["correction_count"]
        ):
            unique[key] = candidate

    return list(unique.values())






# ============================================================

# BASIC IMAGE CHECKS

# ============================================================



def blur(image: np.ndarray) -> dict:

    """Estimate blur using Laplacian variance."""



    variance = float(

        cv2.Laplacian(

            gray(image),

            cv2.CV_64F,

        ).var()

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

            min(1.0, variance / settings.blur_threshold),

            3,

        ),

        confidence=0.82,

        measurement={

            "laplacian_variance": round(variance, 2),

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

    """Generate a simple 64-bit average perceptual hash."""



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

            confidence=round(

                1 - nearest / 64,

                2,

            ),

            measurement={

                "nearest_hamming_distance": nearest,

            },

            threshold=8,

        )



    return result(

        "pass",

        "No exact or near duplicate found among existing uploads.",

        classification="not_duplicate",

        measurement={

            "nearest_hamming_distance": nearest,

        },

        threshold=8,

    )





# ============================================================

# OCR NORMALIZATION

# ============================================================



def normalize_ocr(text: str) -> str:

    """Normalize OCR text while preserving useful word boundaries."""



    if not text:

        return ""



    text = text.upper()



    # Normalize common whitespace.

    text = re.sub(r"\s+", " ", text)



    return text.strip()





def normalize_plate(candidate: str) -> str:

    """Remove spaces and punctuation from a plate."""



    return re.sub(

        r"[^A-Z0-9]",

        "",

        candidate.upper(),

    )





# ============================================================

# PLATE CANDIDATE HELPERS

# ============================================================



def _add_candidate(

    candidates: list[dict],

    candidate: str,

    *,

    strict: bool,

    correction_count: int,

) -> None:

    """Add a valid-looking plate candidate."""



    candidate = normalize_plate(candidate)



    if len(candidate) not in PLATE_LENGTHS:

        return



    if correction_count > 2:

        return



    candidates.append(

        {

            "candidate": candidate,

            "strict": strict,

            "correction_count": correction_count,

        }

    )





def _strict_candidates(text: str) -> list[dict]:

    """Find exact-format registration candidates."""



    candidates: list[dict] = []



    # --------------------------------------------------------

    # Normal Indian registrations

    # --------------------------------------------------------



    for match in STRICT_PLATE_PATTERN.finditer(text):



        state, district, series, serial = match.groups()



        if state not in INDIAN_STATES:

            continue



        candidate = (

            f"{state}"

            f"{district}"

            f"{series}"

            f"{serial}"

        )



        _add_candidate(

            candidates,

            candidate,

            strict=True,

            correction_count=0,

        )



    # --------------------------------------------------------

    # Bharat Series

    # --------------------------------------------------------



    for match in BH_SERIES_PATTERN.finditer(text):



        year, serial, suffix = match.groups()



        candidate = (

            f"{year}"

            f"BH"

            f"{serial}"

            f"{suffix}"

        )



        _add_candidate(

            candidates,

            candidate,

            strict=True,

            correction_count=0,

        )



    return candidates





def _corrected_candidates(text: str) -> list[dict]:

    """

    Find possible registration numbers using conservative OCR

    character correction.



    The correction fallback is intentionally strict so that

    arbitrary OCR garbage cannot be converted into a vehicle

    registration number.

    """



    candidates: list[dict] = []



    # --------------------------------------------------------

    # SAFETY GUARD

    # --------------------------------------------------------

    #

    # OCR output containing too many separate words is more

    # likely to be surrounding text / garbage than a plate.

    #

    # Examples:

    #

    #   "KA 01 AB 1234"       -> 4 tokens -> allowed

    #   "22 BH 1234 AA"       -> 4 tokens -> allowed

    #   "PS Y Y G A S 2 BS RS AS" -> 9 tokens -> reject

    #

    if len(text.split()) > 4:

        return candidates



    compact = re.sub(

        r"[^A-Z0-9]",

        "",

        text,

    )



    # Don't process empty OCR.

    if not compact:

        return candidates



    # Don't brute-force very large OCR output.

    if len(compact) > 32:

        return candidates



    # --------------------------------------------------------

    # Search for state-code candidates.

    # --------------------------------------------------------



    for start in range(

        0,

        max(0, len(compact) - 7),

    ):



        state = compact[start:start + 2]



        if state not in INDIAN_STATES:

            continue



        # Typical Indian plate:

        #

        # STATE + 1/2 digit district

        #       + 1/2/3 letter series

        #       + 1/2/3/4 digit serial

        #

        for district_size in (1, 2):



            for series_size in (1, 2, 3):



                for serial_size in (1, 2, 3, 4):



                    total_size = (

                        2

                        + district_size

                        + series_size

                        + serial_size

                    )



                    end = start + total_size



                    if end > len(compact):

                        continue



                    token = compact[start:end]



                    if len(token) not in PLATE_LENGTHS:

                        continue



                    district_start = 2



                    district_end = (

                        district_start

                        + district_size

                    )



                    series_start = district_end



                    series_end = (

                        series_start

                        + series_size

                    )



                    district_raw = token[

                        district_start:district_end

                    ]



                    series_raw = token[

                        series_start:series_end

                    ]



                    serial_raw = token[

                        series_end:

                    ]



                    # ------------------------------------------------

                    # Correct digit positions

                    # ------------------------------------------------



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



                    if not district.isdigit():

                        continue



                    if not serial.isdigit():

                        continue



                    # ------------------------------------------------

                    # Correct letter-series positions

                    # ------------------------------------------------



                    possibilities = []



                    for char in series_raw:



                        options = SERIES_CORRECTIONS.get(

                            char,

                            (char,),

                        )



                        possibilities.append(options)



                    for parts in product(*possibilities):



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

                        if corrections > 2:

                            continue



                        candidate = (

                            state

                            + district

                            + series

                            + serial

                        )



                        _add_candidate(

                            candidates,

                            candidate,

                            strict=False,

                            correction_count=corrections,

                        )



    return candidates





def plate_candidates_from_text(
    raw_text: str,
) -> list[dict]:
    """
    Extract Indian registration candidates from OCR.

    Strategy:
      1. Strict complete-plate matching.
      2. Noise/brand filtering.
      3. Individual-token normalization.
      4. Two-, three-, and four-token joining.
      5. Existing conservative OCR-correction fallback.
      6. Deduplication and ranking.
    """

    text = normalize_ocr(raw_text)

    if not text:
        return []

    all_candidates: list[dict] = []

    # --------------------------------------------------------
    # 1. Strict matching
    # --------------------------------------------------------

    all_candidates.extend(_strict_candidates(text))

    # --------------------------------------------------------
    # 2. Tokenize and remove obvious OCR noise
    # --------------------------------------------------------

    tokens = re.findall(r"[A-Z0-9]+", text.upper())

    useful_tokens = [
        token
        for token in tokens
        if not _is_noise_token(token)
    ]

    # --------------------------------------------------------
    # 3. Normalize individual OCR tokens
    # --------------------------------------------------------

    for token in useful_tokens:
        all_candidates.extend(
            normalize_indian_plate_candidate(token)
        )

    # --------------------------------------------------------
    # 4. Join adjacent OCR tokens
    # --------------------------------------------------------
    # Handles:
    #   KA 01 AB 1234
    #   KA01 AB 1234
    #   22 BH 1234 AA
    # --------------------------------------------------------

    max_join = min(4, len(useful_tokens))

    for size in range(2, max_join + 1):
        for start in range(
            0,
            len(useful_tokens) - size + 1,
        ):
            joined = "".join(
                useful_tokens[start:start + size]
            )

            if len(joined) < 8 or len(joined) > 10:
                continue

            all_candidates.extend(
                normalize_indian_plate_candidate(joined)
            )

    # --------------------------------------------------------
    # 5. Existing conservative correction fallback
    # --------------------------------------------------------

    all_candidates.extend(
        _corrected_candidates(text)
    )

    # --------------------------------------------------------
    # 6. Deduplicate and prefer better evidence
    # --------------------------------------------------------

    unique: dict[str, dict] = {}

    for candidate in all_candidates:
        key = candidate["candidate"]
        previous = unique.get(key)

        if previous is None:
            unique[key] = candidate
            continue

        previous_rank = (
            0 if previous["strict"] else 1,
            previous["correction_count"],
        )

        current_rank = (
            0 if candidate["strict"] else 1,
            candidate["correction_count"],
        )

        if current_rank < previous_rank:
            unique[key] = candidate

    # --------------------------------------------------------
    # 7. Final ranking
    # --------------------------------------------------------

    return sorted(
        unique.values(),
        key=lambda item: (
            0 if item["strict"] else 1,
            item["correction_count"],
            -len(item["candidate"]),
            item["candidate"],
        ),
    )


def validate_indian_plate(

    raw_text: str,

) -> dict:

    """

    Validate a broad Indian or Bharat Series registration.



    IMPORTANT:

    This is probabilistic OCR validation. It does not prove that

    the registration number actually belongs to a vehicle.

    """



    raw_text = raw_text or ""



    candidates = plate_candidates_from_text(

        raw_text

    )



    if not candidates:

        return result(

            "warning",

            "No broad Indian registration number was reliably detected.",

            classification="invalid_or_not_detected",

            raw_ocr_text=raw_text,

            normalized_candidate="",

            confidence=0.2,

            correction_applied=False,

        )



    # Prefer:

    #   1. Strict candidates

    #   2. Fewer OCR corrections

    best = min(

        candidates,

        key=lambda candidate: (

            candidate["correction_count"],

            not candidate["strict"],

        ),

    )



    # --------------------------------------------------------

    # Strict match

    # --------------------------------------------------------



    if best["strict"]:



        return result(

            "pass",

            (

                "Likely Indian vehicle registration number detected; "

                "OCR errors remain possible."

            ),

            classification="likely_valid",

            raw_ocr_text=raw_text,

            normalized_candidate=best["candidate"],

            confidence=0.75,

            correction_applied=False,

        )



    # --------------------------------------------------------

    # Corrected OCR match

    # --------------------------------------------------------



    return result(

        "warning",

        (

            "Possible registration number detected after "

            "conservative OCR correction."

        ),

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



    line_count = (

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

        line_count / 20 * 0.6

        + (0.4 if border > 30 else 0),

    )



    return result(

        "warning" if score > 0.6 else "pass",

        (

            "Possible photographed-screen/print signal; manual review is needed."

            if score > 0.6

            else "No strong photo-of-photo signal."

        ),

        heuristic_score=round(

            score,

            2,

        ),

        signals={

            "long_line_count": line_count,

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