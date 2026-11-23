
import cv2
from pathlib import Path


# Lokasi model YuNet
BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    BASE_DIR
    / "ml_assets"
    / "classification"
    / "face_detection_yunet_2026may.onnx"
)


# Memuat model YuNet
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model YuNet tidak ditemukan: {MODEL_PATH}"
    )

FACE_DETECTOR = cv2.FaceDetectorYN.create(
    str(MODEL_PATH),
    "",
    (320, 320),
    0.5,
    0.3,
    5000
)


def detect_face(image):
    """
    Mendeteksi wajah menggunakan YuNet
    dan mengambil area wajah terbesar.
    """

    height, width = image.shape[:2]

    FACE_DETECTOR.setInputSize((width, height))

    _, faces = FACE_DETECTOR.detect(image)

    if faces is None or len(faces) == 0:
        raise ValueError("NO_FACE_DETECTED")

    # Mengambil wajah terbesar
    face = max(
        faces,
        key=lambda item: item[2] * item[3]
    )

    x, y, w, h = face[:4]

    x, y, w, h = map(int, (x, y, w, h))

    # Crop area wajah
    face_crop = image[y:y + h, x:x + w]

    return face_crop


def apply_clahe(image):
    """
    Meningkatkan kontras gambar menggunakan CLAHE.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    result = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return result


def preprocess_image(image):
    """
    Melakukan seluruh tahapan preprocessing gambar wajah.
    """

    # 1. Deteksi dan crop wajah
    face = detect_face(image)

    # 2. Peningkatan kontras
    face = apply_clahe(face)

    # 3. Resize menjadi 224 x 224
    face = cv2.resize(
        face,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    # 4. Mengubah BGR menjadi RGB
    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    return face
