import cv2
import numpy as np
import app.analyzers.ocr as plate_ocr
from app.analyzers.image_checks import blur, brightness, dimensions, duplicate, perceptual_hash, validate_indian_plate
from app.db.models import ProcessingStatus, Upload, transition
from app.services.analysis_service import aggregate
#from app.analyzers.ocr import PLATE_CONFIGS


def test_blur_distinguishes_sharp_and_soft():
    sharp = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(sharp, (70, 70), (230, 230), (255, 255, 255), 5)
    soft = cv2.GaussianBlur(sharp, (31, 31), 0)
    assert blur(sharp)["measurement"]["laplacian_variance"] > blur(soft)["measurement"]["laplacian_variance"]


def test_brightness_warns_for_dark_image():
    assert brightness(np.zeros((100, 100, 3), dtype=np.uint8))["status"] == "warning"


def test_dimensions_exposes_measurements():
    report = dimensions(np.zeros((480, 640, 3), dtype=np.uint8))
    assert report["measurement"]["width"] == 640


def test_duplicate_exact_and_near():
    image = np.full((100, 100, 3), 140, dtype=np.uint8)
    phash = perceptual_hash(image)
    assert duplicate("a", phash, [("a", phash)])["classification"] == "exact_duplicate"
    assert duplicate("a", phash, [("b", phash)])["classification"] == "near_duplicate"


def test_vehicle_number_normalization_and_validation():
    expected = {
        "MH12KR1145": "MH12KR1145",
        "MH 12 KR 1145": "MH12KR1145",
        "MH-12-KR-1145": "MH12KR1145",
        "KA01AB1234": "KA01AB1234",
        "DL01C1234": "DL01C1234",
    }
    for source, candidate in expected.items():
        report = validate_indian_plate(source)
        assert report["classification"] == "likely_valid"
        assert report["normalized_candidate"] == candidate


def test_vehicle_number_is_extracted_from_surrounding_ocr_text():
    report = validate_indian_plate("ARENA MH 12 KR 1145 VEHICLE")
    assert report["classification"] == "likely_valid"
    assert report["normalized_candidate"] == "MH12KR1145"


def test_vehicle_number_ocr_correction_is_uncertain():
    ambiguous = validate_indian_plate("KAO1A81234")
    assert ambiguous["classification"] == "uncertain"
    assert ambiguous["raw_ocr_text"] == "KAO1A81234"
    assert validate_indian_plate("nonsense")["classification"] == "invalid_or_not_detected"


def test_vehicle_number_supports_bh_series_and_rejects_garbage_ocr():
    bh = validate_indian_plate("22 BH 1234 AA")
    assert bh["classification"] == "likely_valid"
    assert bh["normalized_candidate"] == "22BH1234AA"
    garbage = validate_indian_plate("PS Y Y G A S 2 BS RS AS")
    assert garbage["classification"] == "invalid_or_not_detected"


def test_consensus_prefers_a_strict_read_over_a_single_character_series_error(monkeypatch):
    image = np.zeros((160, 320, 3), dtype=np.uint8)
    monkeypatch.setattr(plate_ocr, "detect_plate_regions", lambda _: [{"crop": image, "bbox": [0, 0, 320, 160], "score": 0.8}])
    monkeypatch.setattr(plate_ocr, "_plate_variants", lambda _: [image])
    monkeypatch.setattr(plate_ocr, "_ocr", lambda _, config: ("MH12NW8556" if "--psm 6" in config else "MH12N18556", 0.8))
    report = plate_ocr.detect_vehicle_number(image, {"raw_text": "noisy unrelated full image text", "confidence": 0.1})
    assert report["classification"] == "likely_valid"
    assert report["normalized_candidate"] == "MH12NW8556"


def test_yellow_plate_text_region_is_proposed_without_location_assumption():
    image = np.full((420, 720, 3), 35, dtype=np.uint8)
    cv2.rectangle(image, (180, 110), (520, 190), (0, 220, 255), thickness=-1)
    cv2.putText(image, "MH12NW8556", (195, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    regions = plate_ocr.detect_plate_regions(image)
    assert regions
    assert any(region["signals"]["dark_on_yellow_support"] > 0 for region in regions)


def test_plate_ocr_consensus_increases_vehicle_number_confidence(monkeypatch):
    image = np.zeros((160, 320, 3), dtype=np.uint8)
    monkeypatch.setattr(plate_ocr, "detect_plate_regions", lambda _: [{"crop": image, "bbox": [0, 0, 320, 160], "score": 0.8}])
    monkeypatch.setattr(plate_ocr, "_plate_variants", lambda _: [image])
    monkeypatch.setattr(plate_ocr, "_ocr", lambda *_: ("MH 12 KR 1145", 0.8))
    report = plate_ocr.detect_vehicle_number(image, {"raw_text": "", "confidence": None})
    assert report["classification"] == "likely_valid"
    assert report["normalized_candidate"] == "MH12KR1145"
    assert report["evidence"]["matching_attempts"] == 5
    #assert report["evidence"]["matching_attempts"] == len(PLATE_CONFIGS)


def test_status_transitions_are_constrained():
    upload = Upload(status=ProcessingStatus.pending, original_filename="x.jpg", stored_filename="x.jpg", file_path="x", mime_type="image/jpeg", file_size=1, sha256="0" * 64, perceptual_hash="0" * 16)
    transition(upload, ProcessingStatus.processing)
    transition(upload, ProcessingStatus.completed)
    try:
        transition(upload, ProcessingStatus.failed)
        assert False, "completed jobs must be terminal"
    except ValueError:
        pass


def test_aggregation_is_conservative():
    checks = {"dimensions": {"status": "pass"}, "blur": {"status": "pass"}, "ocr": {"status": "warning"}}
    assert aggregate(checks)["overall_status"] == "review_recommended"
    checks["dimensions"]["status"] = "warning"
    checks["blur"]["status"] = "warning"
    checks["duplicate"] = {"status": "warning"}
    assert aggregate(checks)["overall_status"] == "rejected"
