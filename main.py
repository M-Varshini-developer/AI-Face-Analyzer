
import cv2
import time
from deepface import DeepFace

# Open laptop camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera not found.")
    raise SystemExit

print("AI Face Analyzer started.")
print("Press Q to close.")

results = []
last_analysis = 0
analysis_interval = 2.0

while True:
    success, frame = camera.read()

    if not success:
        print("Unable to read camera frame.")
        break

    now = time.time()

    # Analyze every 2 seconds
    if now - last_analysis >= analysis_interval:
        try:
            detected = DeepFace.analyze(
                img_path=frame,
                actions=["age", "gender", "emotion"],
                detector_backend="opencv",
                enforce_detection=True,
                silent=True
            )

            if isinstance(detected, dict):
                detected = [detected]

            results = detected

        except Exception:
            results = []

        last_analysis = now

    # Draw face boxes and predictions
    for face in results:
        region = face.get("region", {})

        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("w", 0))
        h = int(region.get("h", 0))

        if w <= 0 or h <= 0:
            continue

        age = face.get("age", "Unknown")
        gender = str(face.get("dominant_gender", "Unknown")).strip().capitalize()
        gender = face.get("dominant_gender", "Unknown")
        emotion = face.get("dominant_emotion", "Unknown")

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        labels = [
            f"Age: ~{age}",
            f"Gender: {gender}",
            f"Expression: {emotion}"
        ]

        for i, label in enumerate(labels):
            text_y = max(
                25,
                y - 10 - (len(labels) - 1 - i) * 25
            )

            cv2.putText(
                frame,
                label,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    cv2.imshow("AI Face Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
