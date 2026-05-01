import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================= CONFIG =================
RAW_IMG_DIR = r"C:\Users\AFS\Desktop\Deepfake_Detction\data\raw\images"
OUT_FACE_DIR = r"C:\Users\AFS\Desktop\Deepfake_Detction\data\processed\images\faces"
MODEL_PATH = r"C:\Users\AFS\Desktop\models\blaze_face_short_range.tflite"

IMG_SIZE = (224, 224)
MIN_CONF = 0.6
# =========================================

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=MIN_CONF
)
detector = vision.FaceDetector.create_from_options(options)

def extract_all_faces(image):
    h, w, _ = image.shape
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = detector.detect(mp_img)

    faces = []
    if not result.detections:
        return faces

    for det in result.detections:
        box = det.bounding_box
        x1, y1 = max(0, box.origin_x), max(0, box.origin_y)
        x2 = min(w, x1 + box.width)
        y2 = min(h, y1 + box.height)

        face = image[y1:y2, x1:x2]
        if face.size > 0:
            faces.append(face)

    return faces

for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        in_dir = os.path.join(RAW_IMG_DIR, split, label)
        out_dir = os.path.join(OUT_FACE_DIR, split, label)
        os.makedirs(out_dir, exist_ok=True)

        for img_name in os.listdir(in_dir):
            img_path = os.path.join(in_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = extract_all_faces(img)
            if not faces:
                continue

            img_id = os.path.splitext(img_name)[0]
            img_face_dir = os.path.join(out_dir, img_id)
            os.makedirs(img_face_dir, exist_ok=True)

            for i, face in enumerate(faces):
                face = cv2.resize(face, IMG_SIZE)
                cv2.imwrite(
                    os.path.join(img_face_dir, f"face_{i}.jpg"),
                    face
                )

print("[✓] IMAGE MULTI-FACE EXTRACTION COMPLETED")
