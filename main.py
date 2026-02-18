import cv2
import numpy as np
import os
from deepface import DeepFace

# Load emoji images into dictionary
emotion_images = {}
emotion_list = ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]
emotion_history = []

for emotion in emotion_list:
    path = f"emotions/{emotion}.png"
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is not None:
        # If image has transparency (4 channels)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        emotion_images[emotion] = img

cap = cv2.VideoCapture(0)

while True:
    # This grabs one image from the webcam, ret -> true if camera works, frame -> image from webcam.
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False # dont crash if face not detected
        )

        emotion = result[0]["dominant_emotion"]
        emotion_history.append(emotion)

        # Keep only last 10 frames
        if len(emotion_history) > 10:
            emotion_history.pop(0)

        # Pick most common emotion in last 10 frames
        emotion = max(set(emotion_history), key=emotion_history.count)
    except:
        emotion = "neutral"

    # Resize camera to fixed size
    frame = cv2.resize(frame, (640, 480))

    # Create right panel, black background
    right_panel = np.zeros((480, 640, 3), dtype=np.uint8)

    emoji = emotion_images[emotion]
    emoji = cv2.resize(emoji, (640, 480))
    right_panel = emoji

    # Combine camera + emoji panel
    combined = np.hstack((frame, right_panel))

    text = emotion.upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    color = (0, 0, 0)

    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Get center position
    x = (combined.shape[1] - text_width) // 2
    y = 50  # distance from top

    # Draw text
    cv2.putText(combined, text, (x, y), font, font_scale, color, thickness)

    cv2.imshow("press q to quit", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()