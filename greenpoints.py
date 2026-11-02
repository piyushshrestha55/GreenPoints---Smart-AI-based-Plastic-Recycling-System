
import cv2
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# ==========================================
# ECOPOINTS CONFIGURATION
# ==========================================

MODEL_PATH = "yolo26n.pt"

SESSION_DURATION = 120  # 2 minutes

BOTTLE_CONFIDENCE = 0.80
DETECTOR_CONFIDENCE = 0.25

STABLE_FRAMES_REQUIRED = 5
EMPTY_FRAMES_REQUIRED = 5

DOOR_OPEN_TIME = 1.0
DOOR_CLOSING_TIME = 0.8

# Camera
CAMERA_INDEX = 0

# Platform scanning area.
# Coordinates are percentages of the camera frame:
# (left, top), (right, top), (right, bottom), (left, bottom)
#
# IMPORTANT:
# Adjust these four points to match your actual platform.

PLATFORM_POLYGON = [
    (0.15, 0.30),
    (0.85, 0.30),
    (0.95, 0.95),
    (0.05, 0.95)
]

# ==========================================
# LOAD AI MODELS
# ==========================================

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

print("Loading MediaPipe hand detection...")

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# CAMERA SETUP
# ==========================================

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("ERROR: Cannot open webcam.")
    raise SystemExit

print("Camera connected.")

# ==========================================
# SESSION VARIABLES
# ==========================================

points = 0

state = "SCANNING"

session_start = time.time()

valid_frames = 0
empty_frames = 0

door_state_start = 0

message = "Place one bottle on the platform"

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_platform_polygon(frame):
    height, width = frame.shape[:2]

    polygon = np.array([
        [int(x * width), int(y * height)]
        for x, y in PLATFORM_POLYGON
    ], dtype=np.int32)

    return polygon


def is_inside_platform(x, y, polygon):
    result = cv2.pointPolygonTest(
        polygon,
        (float(x), float(y)),
        False
    )

    return result >= 0


def detect_hand_in_platform(frame, polygon):

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    hand_results = hands_detector.process(rgb_frame)

    hand_present = False

    if hand_results.multi_hand_landmarks:

        height, width = frame.shape[:2]

        for hand_landmarks in hand_results.multi_hand_landmarks:

            hand_points = []

            for landmark in hand_landmarks.landmark:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                hand_points.append((x, y))

            # If any detected hand landmark enters
            # the platform area, pause scanning.

            for x, y in hand_points:

                if is_inside_platform(x, y, polygon):
                    hand_present = True
                    break

            # Draw hand landmarks for visualization.

            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            if hand_present:
                break

    return hand_present


def detect_objects(frame, polygon):

    results = model.predict(
        frame,
        conf=DETECTOR_CONFIDENCE,
        verbose=False
    )

    bottles = []
    other_objects = []

    result = results[0]

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id].lower()

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Only consider objects whose center
            # falls inside the scanning platform.

            if not is_inside_platform(
                center_x,
                center_y,
                polygon
            ):
                continue

            if class_name == "person":
                continue

            if class_name == "bottle":

                bottles.append({
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2)
                })

            else:

                other_objects.append({
                    "name": class_name,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2)
                })

    return bottles, other_objects


def draw_box(frame, box, color, label):

    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


# ==========================================
# MAIN PROGRAM
# ==========================================

print("\nEcoPoints started!")
print("Press Q to quit.\n")

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Cannot read webcam.")
            break

        current_time = time.time()

        elapsed_time = current_time - session_start

        remaining_time = max(
            0,
            SESSION_DURATION - int(elapsed_time)
        )

        polygon = get_platform_polygon(frame)

        # Draw scanning area.

        cv2.polylines(
            frame,
            [polygon],
            True,
            (255, 255, 0),
            2
        )

        # ----------------------------------
        # SESSION TIME CHECK
        # ----------------------------------

        if elapsed_time >= SESSION_DURATION:

            state = "SESSION_ENDED"
            message = "SESSION ENDED"

        # ----------------------------------
        # DETECTION
        # ----------------------------------

        if state != "SESSION_ENDED":

            hand_present = detect_hand_in_platform(
                frame,
                polygon
            )

            bottles, other_objects = detect_objects(
                frame,
                polygon
            )

        else:

            hand_present = False
            bottles = []
            other_objects = []

        # ----------------------------------
        # DRAW DETECTIONS
        # ----------------------------------

        for bottle in bottles:

            confidence = bottle["confidence"]

            draw_box(
                frame,
                bottle["box"],
                (0, 255, 0) if confidence >= BOTTLE_CONFIDENCE
                else (0, 165, 255),
                f"Bottle {confidence:.0%}"
            )

        for obj in other_objects:

            draw_box(
                frame,
                obj["box"],
                (0, 0, 255),
                f"Rejected: {obj['name']}"
            )

        # ----------------------------------
        # VALIDATION
        # ----------------------------------

        exactly_one_bottle = len(bottles) == 1

        bottle_confident = (
            exactly_one_bottle
            and bottles[0]["confidence"] >= BOTTLE_CONFIDENCE
        )

        no_other_objects = len(other_objects) == 0

        valid_bottle = (
            bottle_confident
            and no_other_objects
        )

        platform_empty = (
            len(bottles) == 0
            and len(other_objects) == 0
            and not hand_present
        )

        # ----------------------------------
        # STATE MACHINE
        # ----------------------------------

        if state == "SCANNING":

            if hand_present:

                valid_frames = 0
                message = "HAND DETECTED - SCANNING PAUSED"

            elif valid_bottle:

                valid_frames += 1

                message = (
                    f"Valid bottle - checking "
                    f"{valid_frames}/{STABLE_FRAMES_REQUIRED}"
                )

                if valid_frames >= STABLE_FRAMES_REQUIRED:

                    points += 1

                    print(
                        f"Point awarded! Total points: {points}"
                    )

                    state = "DOOR_OPEN"

                    door_state_start = current_time

                    valid_frames = 0

                    message = "1 POINT AWARDED - DOOR OPENING"

            else:

                valid_frames = 0

                if len(bottles) > 1:

                    message = "REJECTED - MULTIPLE BOTTLES"

                elif len(bottles) == 1:

                    message = "REJECTED - CONFIDENCE BELOW 80%"

                elif len(other_objects) > 0:

                    message = "REJECTED - INVALID OBJECT"

                else:

                    message = "PLACE ONE BOTTLE ON PLATFORM"

        # ----------------------------------
        # SIMULATED TRAPDOOR
        # ----------------------------------

        elif state == "DOOR_OPEN":

            message = "TRAPDOOR OPEN"

            if current_time - door_state_start >= DOOR_OPEN_TIME:

                state = "DOOR_CLOSING"

                door_state_start = current_time

                message = "TRAPDOOR CLOSING"

        elif state == "DOOR_CLOSING":

            message = "TRAPDOOR CLOSING"

            if current_time - door_state_start >= DOOR_CLOSING_TIME:

                state = "WAIT_EMPTY"

                empty_frames = 0

                message = "WAITING FOR PLATFORM TO EMPTY"

        # ----------------------------------
        # WAIT FOR NEXT BOTTLE
        # ----------------------------------

        elif state == "WAIT_EMPTY":

            if platform_empty:

                empty_frames += 1

                message = (
                    f"Platform empty - "
                    f"{empty_frames}/{EMPTY_FRAMES_REQUIRED}"
                )

                if empty_frames >= EMPTY_FRAMES_REQUIRED:

                    state = "SCANNING"

                    empty_frames = 0

                    message = "READY FOR NEXT BOTTLE"

            else:

                empty_frames = 0

                if hand_present:

                    message = "REMOVE HAND FROM PLATFORM"

                else:

                    message = "REMOVE PREVIOUS OBJECT FIRST"

        elif state == "SESSION_ENDED":

            message = "SESSION ENDED - TRAPDOOR CLOSED"

        # ----------------------------------
        # DISPLAY INFORMATION
        # ----------------------------------

        cv2.putText(
            frame,
            f"Points: {points}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Time: {remaining_time}s",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"State: {state}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            message,
            (20, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "EcoPoints - Bottle Recycling System",
            frame
        )

        # Press Q to quit.

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.release()

    cv2.destroyAllWindows()

    hands_detector.close()

    print("\nEcoPoints stopped.")
    print(f"Final points: {points}")
