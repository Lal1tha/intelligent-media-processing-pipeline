import cv2
import numpy as np


def result(status: str, message: str, **extra) -> dict:
    return {"status": status, "message": message, **extra}


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
