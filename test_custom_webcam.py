import os
import sys
import time
import cv2
import torch
import mediapipe as mp
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION & THRESHOLDS
# ==============================================================================

# Video source: DroidCam or Web Camera
# - If using DroidCam over Wi-Fi, enter your phone's IP address shown in the app:
DROIDCAM_IP = "192.168.1.100"   # <-- Replace with your phone's Wi-Fi IP shown in DroidCam
DROIDCAM_PORT = 4747
# Stream URL for DroidCam (mjpeg): "http://<IP>:4747/video"
CAMERA_SOURCE = 1
# Notes:
# - To use DroidCam PC Virtual Camera via USB/client instead: set CAMERA_SOURCE = 1 (or 2)
# - To revert to laptop/PC built-in webcam: set CAMERA_SOURCE = 0

# Model weights:
# - BASE_WEIGHTS: Pretrained COCO model used to detect general containers & people
# - CUSTOM_WEIGHTS: Fine-tuned model specifically trained to classify container materials
BASE_WEIGHTS = "weights/yolo26n.pt" if os.path.exists("weights/yolo26n.pt") else "yolo26n.pt"
CUSTOM_WEIGHTS = "weights/best.pt" if os.path.exists("weights/best.pt") else r"runs/detect/greenpoints_v2_finetuned-3/weights/best.pt" 

# Anti-flicker / stability requirements:
# - STABLE_FRAMES: How many consecutive frames the bottle must be seen before accepting
# - MISS_LIMIT: How many consecutive empty frames are needed to consider the platform cleared
STABLE_FRAMES = 8
MISS_LIMIT = 5

# Confidence thresholds:
# - BASE_CONF: Minimum confidence for base YOLO to identify a container or person (25%)
# - CUSTOM_CONF: Minimum confidence for custom model to identify the material (50%)
BASE_CONF = 0.25
CUSTOM_CONF = 0.50

# Generic container classes recognized by the base COCO model
CONTAINER_LABELS = {"bottle", "can", "cup", "wine glass"}

# Recycling policy:
# - Only plastic bottles earn points and open the trapdoor
# - Exactly 1 point awarded per valid bottle deposited
ACCEPTED_MATERIAL = "plastic_bottle"
POINTS_PER_ITEM = 1

# Trapdoor simulation timings (in seconds):
# - How long the door stays OPEN for the bottle to fall into the bin
# - How long the door takes to CLOSE before resetting
DOOR_OPEN_DURATION = 1.5
DOOR_CLOSE_DURATION = 0.8

# Visual color coding for different materials (BGR format for OpenCV)
CLASS_COLORS = {
    "aluminium_can": (255, 140, 0),    # Blue/Orange
    "glass_bottle": (0, 200, 255),     # Yellow
    "plastic_bottle": (80, 220, 80),   # Green (Accepted material)
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def overlaps_person(box, person_boxes, threshold=0.5):
    """
    Anti-false-positive check:
    Determines if a detected container overlaps significantly with a person's body or face.
    This prevents false detections (e.g. recognizing someone's shirt or body part as a bottle)
    and ensures we only evaluate objects placed clearly on the scanning area.

    Args:
        box: (x1, y1, x2, y2) coordinates of the container candidate
        person_boxes: list of (px1, py1, px2, py2) coordinates of detected people
        threshold: fraction of container box area that must overlap to trigger rejection (default: 50%)

    Returns:
        True if the container overlaps with a person beyond the threshold, False otherwise.
    """
    x1, y1, x2, y2 = box
    box_area = max(1.0, (x2 - x1) * (y2 - y1))

    for px1, py1, px2, py2 in person_boxes:
        # Calculate intersection rectangle
        ix1 = max(x1, px1)
        iy1 = max(y1, py1)
        ix2 = min(x2, px2)
        iy2 = min(y2, py2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih

        # If more than 50% of the candidate box is inside a person, reject it
        if (intersection / box_area) > threshold:
            return True
    return False


def get_padded_crop(frame, box, pad_ratio=0.05):
    """
    Crops the detected container from the camera frame and adds a small margin (padding).
    Padding helps the custom classifier see the full outer edges and contours of the bottle.

    Args:
        frame: The full camera image frame (numpy array)
        box: (x1, y1, x2, y2) bounding box coordinates
        pad_ratio: Percentage of width/height to add as padding (default: 5%)

    Returns:
        Cropped image slice clamped within image dimensions.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    pw = int((x2 - x1) * pad_ratio)
    ph = int((y2 - y1) * pad_ratio)

    cx1 = max(0, x1 - pw)
    cy1 = max(0, y1 - ph)
    cx2 = min(w, x2 + pw)
    cy2 = min(h, y2 + ph)

    return frame[cy1:cy2, cx1:cx2]


def draw_label(frame, text, origin, bg_color, text_color=(0, 0, 0)):
    """
    Draws a clean, readable text badge with a solid background rectangle behind it.

    Args:
        frame: OpenCV image frame to draw onto
        text: String label to display
        origin: (x, y) bottom-left coordinate for the text
        bg_color: BGR tuple for the badge background
        text_color: BGR tuple for the text font (default: black)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    y = max(th + 10, y)
    # Draw filled background pill
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y + baseline), bg_color, -1)
    # Draw text on top
    cv2.putText(frame, text, (x + 4, y - 4), font, scale, text_color, thickness, cv2.LINE_AA)


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    # Hardware acceleration: Use NVIDIA GPU (CUDA) if available, otherwise use CPU
    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = f"CUDA:0 ({torch.cuda.get_device_name(0)})" if device == 0 else "CPU"

    print("==================================================")
    print("GreenPoints AI: Two-Stage Pipeline + Anti-Fraud")
    print(f"Device: {device_name}")
    print("Stage 1 (Gatekeeper): Base YOLO (yolo26n.pt)")
    print("Stage 2 (Specialist): Custom YOLO (best.pt)")
    print("Stage 3 (Anti-Fraud): MediaPipe Hands Detection")
    print(f"Policy: Zero Hands + Exactly 1 Plastic Bottle -> {POINTS_PER_ITEM} Point")
    print("==================================================")

    # 1. Load the two YOLO neural network models
    model_base = YOLO(BASE_WEIGHTS)
    model_custom = YOLO(CUSTOM_WEIGHTS)

    # 2. Initialize MediaPipe Hand detector for fraud prevention
    # Tracks up to 2 hands in real time using 21 3D hand landmarks
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 3. Connect to video source (DroidCam / Webcam)
    # Check if a video source or IP was passed as a command-line argument:
    # e.g.: python test_custom_webcam.py 192.168.1.50
    # e.g.: python test_custom_webcam.py http://192.168.1.50:4747/video
    # e.g.: python test_custom_webcam.py 1
    source = sys.argv[1] if len(sys.argv) > 1 else CAMERA_SOURCE
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    elif isinstance(source, str) and not source.startswith(("http://", "https://", "rtsp://")):
        if ":" not in source:
            source = f"http://{source}:{DROIDCAM_PORT}/video"
        else:
            source = f"http://{source}/video"

    print(f"Connecting to video source: {source} ...")
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"\n[ERROR] Cannot open video source: {source}")
        print("DroidCam Troubleshooting Guide:")
        print("  1. Launch the DroidCam app on your phone.")
        print(f"  2. Check your phone's Wi-Fi IP and update 'DROIDCAM_IP' in test_custom_webcam.py (currently: {DROIDCAM_IP}).")
        print("     Or pass it directly in terminal: python test_custom_webcam.py <YOUR_PHONE_IP>")
        print("  3. Make sure your phone and PC are connected to the SAME Wi-Fi network.")
        print("  4. If using DroidCam PC Virtual Camera via USB/client, set CAMERA_SOURCE = 1 (or 2).")
        return

    print("Video stream connected successfully.")
    print("Hold container inside view. Keep hands clear for acceptance.")
    print("Press Q to quit.\n")

    # --------------------------------------------------------------------------
    # SESSION & STATE TRACKING VARIABLES
    # --------------------------------------------------------------------------
    points = 0                           # User's accumulated EcoPoints
    trapdoor_state = "CLOSED"            # State machine: "CLOSED", "OPEN", "CLOSING"
    door_timer_start = 0.0               # Timestamp when door opened or started closing
    item_processed = False               # Latch flag: prevents awarding multiple points for the same bottle

    streak_cls = None                    # Class ID of current stable item being held
    streak = 0                           # Consecutive frames this same item class was detected
    misses = 0                           # Consecutive frames where no containers were detected (empty platform)

    # --------------------------------------------------------------------------
    # MAIN VIDEO PROCESSING LOOP
    # --------------------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read webcam frame.")
            break

        h, w = frame.shape[:2]
        display = frame.copy()
        current_time = time.time()

        # ----------------------------------------------------------------------
        # STAGE 1: GATEKEEPER (Base YOLO on full frame)
        # Identifies any people and general containers in the scene.
        # ----------------------------------------------------------------------
        base_results = model_base.predict(
            source=frame,
            conf=BASE_CONF,
            imgsz=640,
            device=device,
            verbose=False,
        )[0]

        person_boxes = []       # Stores coordinates of detected people
        candidate_boxes = []    # Stores coordinates of detected containers

        if base_results.boxes is not None and len(base_results.boxes) > 0:
            for b in base_results.boxes:
                cls_id = int(b.cls[0])
                label = model_base.names[cls_id]
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

                if label == "person":
                    # Keep track of person locations to filter out overlapping containers
                    person_boxes.append((x1, y1, x2, y2))
                elif label in CONTAINER_LABELS:
                    # Generic container candidate found
                    candidate_boxes.append(((x1, y1, x2, y2), label, float(b.conf[0])))

        # ----------------------------------------------------------------------
        # STAGE 2: SPECIALIST (Custom model on cropped container)
        # Classifies each container into plastic_bottle, glass_bottle, or aluminium_can.
        # ----------------------------------------------------------------------
        classified_candidates = []

        for (box, base_label, base_score) in candidate_boxes:
            # Reject container if it overlaps significantly with a person (e.g. held against body)
            if overlaps_person(box, person_boxes):
                continue

            # Extract the cropped region of the container
            crop = get_padded_crop(frame, box)
            if crop.size == 0:
                continue

            # Run custom specialist model on the cropped region
            custom_crop_res = model_custom.predict(
                source=crop,
                conf=CUSTOM_CONF,
                imgsz=640,
                device=device,
                verbose=False,
            )[0]

            # If the custom model detected a specific material class, record it
            if custom_crop_res.boxes is not None and len(custom_crop_res.boxes) > 0:
                best_box = max(custom_crop_res.boxes, key=lambda b: float(b.conf[0]))
                cls_id = int(best_box.cls[0])
                mat_name = model_custom.names[cls_id]
                mat_conf = float(best_box.conf[0])

                x1, y1, x2, y2 = box
                classified_candidates.append({
                    "box": box,
                    "cls_id": cls_id,
                    "name": mat_name,
                    "conf": mat_conf,
                    "area": (x2 - x1) * (y2 - y1),
                    "source": f"Crop ({base_label}->{mat_name})",
                })

        # Fallback for aluminium cans:
        # Since standard COCO does not have an "aluminium can" class, if Stage 1 found no containers,
        # run custom model directly on the full frame at higher confidence (0.60) to look for cans.
        if len(classified_candidates) == 0:
            custom_full_res = model_custom.predict(
                source=frame,
                conf=0.60,
                imgsz=640,
                device=device,
                verbose=False,
            )[0]

            if custom_full_res.boxes is not None and len(custom_full_res.boxes) > 0:
                for cb in custom_full_res.boxes:
                    cls_id = int(cb.cls[0])
                    mat_name = model_custom.names[cls_id]
                    coords = tuple(map(int, cb.xyxy[0].tolist()))

                    if mat_name == "aluminium_can" and not overlaps_person(coords, person_boxes):
                        x1, y1, x2, y2 = coords
                        classified_candidates.append({
                            "box": coords,
                            "cls_id": cls_id,
                            "name": mat_name,
                            "conf": float(cb.conf[0]),
                            "area": (x2 - x1) * (y2 - y1),
                            "source": "Stage 2 Direct (AluCan)",
                        })

        # ----------------------------------------------------------------------
        # STAGE 3: HAND DETECTION (MediaPipe Anti-Fraud)
        # Checks if user hands are present in the frame.
        # ----------------------------------------------------------------------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands_detector.process(rgb_frame)
        hand_detected = bool(hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) > 0)

        # Visual feedback: Highlight detected hands in red so user knows hands are not allowed
        if hand_detected:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw skeleton landmarks with prominent red theme
                mp_drawing.draw_landmarks(
                    display,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 140, 255), thickness=2, circle_radius=2),
                )

                # Compute hand bounding box and label it
                hx_min = max(0, min(int(lm.x * w) for lm in hand_landmarks.landmark) - 10)
                hy_min = max(0, min(int(lm.y * h) for lm in hand_landmarks.landmark) - 10)
                hx_max = min(w, max(int(lm.x * w) for lm in hand_landmarks.landmark) + 10)
                hy_max = min(h, max(int(lm.y * h) for lm in hand_landmarks.landmark) + 10)

                cv2.rectangle(display, (hx_min, hy_min), (hx_max, hy_max), (0, 0, 255), 2)
                draw_label(display, "HAND DETECTED (REJECT)", (hx_min, hy_min - 6), (0, 0, 255), (255, 255, 255))

        # Visual feedback: Draw bounding boxes around classified containers
        for item in classified_candidates:
            x1, y1, x2, y2 = item["box"]
            c_name = item["name"]
            c_conf = item["conf"]
            c_color = CLASS_COLORS.get(c_name, (0, 255, 0))

            cv2.rectangle(display, (x1, y1), (x2, y2), c_color, 2)
            draw_label(display, f"{c_name} {c_conf:.2f}", (x1, y1 - 6), c_color)

        # ----------------------------------------------------------------------
        # STAGE 4: VERIFICATION & ANTI-FRAUD DECISION LOGIC
        # Evaluates recycling rules and manages simulated trapdoor state.
        # ----------------------------------------------------------------------
        container_count = len(classified_candidates)

        # A. TRAPDOOR IS OPEN: Waiting for bottle to drop through
        if trapdoor_state == "OPEN":
            status_text = "TRAPDOOR OPEN - DISPENSING POINTS..."
            status_color = (0, 255, 0)
            if current_time - door_timer_start >= DOOR_OPEN_DURATION:
                trapdoor_state = "CLOSING"
                door_timer_start = current_time
                print("[ESP32] Sending signal: CLOSE_TRAPDOOR")

        # B. TRAPDOOR IS CLOSING: Motor closing animation/timer
        elif trapdoor_state == "CLOSING":
            status_text = "TRAPDOOR CLOSING..."
            status_color = (0, 200, 255)
            if current_time - door_timer_start >= DOOR_CLOSE_DURATION:
                trapdoor_state = "CLOSED"
                item_processed = True   # Mark as processed; requires empty platform to reset

        # C. TRAPDOOR IS CLOSED: Active scanning and validation rules
        else:
            if hand_detected:
                # --------------------------------------------------------------
                # RULE 1: HAND DETECTED -> REJECT
                # Hands must be completely removed from the scanning area
                # --------------------------------------------------------------
                streak = 0
                streak_cls = None
                status_text = "HAND DETECTED - NO POINTS GIVEN! (Keep hands clear)"
                status_color = (0, 0, 255)  # Red
                trapdoor_state = "CLOSED"
                # TODO: Send REJECT signal to ESP32

            elif container_count > 1:
                # --------------------------------------------------------------
                # RULE 2: MULTIPLE CONTAINERS -> REJECT
                # Only one container allowed at a time for accurate verification
                # --------------------------------------------------------------
                streak = 0
                streak_cls = None
                status_text = f"MULTIPLE ITEMS ({container_count}) - Please insert one at a time!"
                status_color = (0, 140, 255)  # Orange
                trapdoor_state = "CLOSED"
                # TODO: Send REJECT signal to ESP32

            elif container_count == 1:
                # Exactly one container detected
                target = classified_candidates[0]
                mat_name = target["name"]
                mat_conf = target["conf"]

                # Stability counter: Ensure the container is resting steadily and class isn't flickering
                if target["cls_id"] == streak_cls:
                    streak += 1
                else:
                    streak_cls = target["cls_id"]
                    streak = 1

                # --------------------------------------------------------------
                # RULE 3: CHECK MATERIAL TYPE
                # --------------------------------------------------------------
                if mat_name == ACCEPTED_MATERIAL:
                    # Accepted plastic bottle
                    if streak >= STABLE_FRAMES:
                        if not item_processed:
                            # --------------------------------------------------
                            # ACCEPT: 0 hands + 1 stable plastic bottle
                            # Award 1 point and trigger trapdoor open
                            # --------------------------------------------------
                            points += POINTS_PER_ITEM
                            trapdoor_state = "OPEN"
                            door_timer_start = current_time
                            point_word = "Point" if POINTS_PER_ITEM == 1 else "Points"
                            status_text = f"VALID: {mat_name.upper()}! +{POINTS_PER_ITEM} {point_word} Awarded"
                            status_color = (0, 255, 0)
                            print(f"[ESP32] ACCEPT SIGNAL -> OPEN_TRAPDOOR | Points: +{POINTS_PER_ITEM} (Total: {points})")
                            # TODO: Send ACCEPT signal to ESP32 to open trapdoor
                        else:
                            status_text = "BOTTLE ACCEPTED - Platform reset"
                            status_color = (0, 200, 100)
                    else:
                        # Bottle is valid, but waiting to verify it is held steady
                        status_text = f"HOLD STEADY {streak}/{STABLE_FRAMES} ({mat_name})"
                        status_color = (0, 255, 255)  # Yellow
                else:
                    # REJECT: Invalid material (e.g. glass bottle or aluminium can)
                    streak = 0
                    streak_cls = None
                    status_text = f"REJECT: Material Not Accepted ({mat_name}) - Trapdoor Locked"
                    status_color = (0, 0, 255)  # Red
                    # TODO: Send REJECT signal to ESP32

            else:
                # container_count == 0 (Platform is empty)
                misses += 1
                # When platform stays empty for MISS_LIMIT frames, reset latch for next bottle
                if misses >= MISS_LIMIT:
                    streak = 0
                    streak_cls = None
                    item_processed = False  # Reset latch: ready for next bottle insertion

                status_text = "WAITING: Place 1 plastic bottle (keep hands clear)"
                status_color = (180, 180, 180)

        # ----------------------------------------------------------------------
        # TOP HUD DASHBOARD OVERLAY
        # Displays points, trapdoor state, hand detection status, and live decision banner.
        # ----------------------------------------------------------------------
        # 1. Semi-transparent dark banner background at top of screen
        hud_bg = display.copy()
        cv2.rectangle(hud_bg, (0, 0), (w, 95), (20, 20, 20), -1)
        cv2.addWeighted(hud_bg, 0.75, display, 0.25, 0, display)

        # 2. Header Line 1: Statistics (Points, Trapdoor, Hands)
        door_color = (0, 255, 0) if trapdoor_state == "OPEN" else ((0, 200, 255) if trapdoor_state == "CLOSING" else (200, 200, 200))
        hand_status_color = (0, 0, 255) if hand_detected else (0, 255, 0)
        hand_status_text = "DETECTED (REJECT)" if hand_detected else "CLEAR (OK)"

        cv2.putText(display, f"POINTS: {points}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, f"TRAPDOOR: {trapdoor_state}", (230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, door_color, 2, cv2.LINE_AA)
        cv2.putText(display, f"HANDS: {hand_status_text}", (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_status_color, 2, cv2.LINE_AA)

        # 3. Header Line 2: Real-time Decision & Guidance Banner
        cv2.rectangle(display, (15, 48), (w - 15, 88), status_color, 2)
        cv2.putText(display, status_text, (25, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

        # Show live video window
        cv2.imshow("GreenPoints AI - Anti-Fraud & Two-Stage Pipeline (Q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------------------------------------------------
    # CLEANUP & SHUTDOWN
    # --------------------------------------------------------------------------
    hands_detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
