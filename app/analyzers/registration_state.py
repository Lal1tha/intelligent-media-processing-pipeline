
import csv
import re
from pathlib import Path
from functools import lru_cache


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "india_rto_codes.csv"


def normalize_rto_code(code: str) -> str | None:
    """Normalize an RTO code such as KL45 or KL-45 to KL45."""
    match = re.fullmatch(
        r"\s*([A-Z]{2})\s*[- ]?\s*(\d{1,2})\s*",
        code.upper(),
    )
    if not match:
        return None

    state_prefix, digits = match.groups()
    return f"{state_prefix}{int(digits):02d}"


@lru_cache(maxsize=1)
def load_rto_dataset() -> dict:
    """Load the CSV once and build an RTO lookup dictionary."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"RTO dataset not found: {DATASET_PATH}"
        )

    lookup = {}

    with DATASET_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {"RegNo", "Place", "State"}
        if not reader.fieldnames or not required_columns.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError(
                "CSV must contain RegNo, Place, and State columns."
            )

        for row in reader:
            raw_codes = (row.get("RegNo") or "").upper()
            state = (row.get("State") or "").strip()
            place = (row.get("Place") or "").strip()

            # Handles entries such as "AP16 & AP17".
            codes = re.findall(
                r"[A-Z]{2}\s*0*\d{1,2}",
                raw_codes,
            )

            for raw_code in codes:
                code = normalize_rto_code(raw_code)
                if not code or not state or not place:
                    continue

                # Keep the first entry if the dataset has duplicates.
                lookup.setdefault(
                    code,
                    {
                        "state": state,
                        "region": place,
                    },
                )

    if not lookup:
        raise ValueError("The RTO CSV loaded but contained no usable rows.")

    return lookup


def identify_registration_location(
    registration_number: str | None,
) -> dict:
    """Identify state and RTO region from a standard private plate."""

    if not registration_number:
        return {
            "status": "unavailable",
            "registration_number": None,
            "registration_state_code": None,
            "registration_state": None,
            "rto_code": None,
            "registration_region": None,
            "message": "No registration number was extracted.",
        }

    number = re.sub(
        r"[\s-]+",
        "",
        registration_number.upper(),
    )

    match = re.fullmatch(
        r"([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})",
        number,
    )

    if not match:
        return {
            "status": "unresolved",
            "registration_number": number,
            "registration_state_code": None,
            "registration_state": None,
            "rto_code": None,
            "registration_region": None,
            "message": (
                "Registration format is unsupported or OCR needs review."
            ),
        }

    state_code, rto_digits, series, serial = match.groups()
    rto_key = f"{state_code}{rto_digits}"
    dataset = load_rto_dataset()
    office = dataset.get(rto_key)

    if office:
        return {
            "status": "identified",
            "registration_number": number,
            "registration_state_code": state_code,
            "registration_state": office["state"],
            "rto_code": f"{state_code}-{rto_digits}",
            "registration_region": office["region"],
            "registration_series": series,
            "serial_number": serial,
            "message": "State and RTO found in the CSV dataset.",
        }

    return {
        "status": "state_identified",
        "registration_number": number,
        "registration_state_code": state_code,
        "registration_state": None,
        "rto_code": f"{state_code}-{rto_digits}",
        "registration_region": None,
        "registration_series": series,
        "serial_number": serial,
        "message": (
            "RTO code was not found in the dataset. "
            "The state prefix still requires a separate state-code lookup."
        ),
    }
