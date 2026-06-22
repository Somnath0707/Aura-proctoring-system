import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque

# Initialize MediaPipe Face Mesh
# refine_landmarks=True is REQUIRED for iris tracking (landmarks 468, 473)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# A queue to hold the last 5 gaze ratios to smooth out webcam jitter
ratio_history = deque(maxlen=5)


def get_3d_distance(p1, p2):
    """Calculates 3D Euclidean distance between two MediaPipe landmarks."""
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2 +
        (p1.z - p2.z) ** 2
    )


def is_head_turned(landmarks):
    """
    Checks if the head is rotated too far left or right.
    Compares the Z-depth of the left edge of the face vs the right edge.
    """
    left_edge = landmarks[234]   # Left cheek area
    right_edge = landmarks[454]  # Right cheek area

    # Calculate the depth difference between the two sides of the face
    depth_diff = abs(left_edge.z - right_edge.z)

    # If the difference is too large, the head is turned.
    # (0.05 is a standard threshold, adjust if needed)
    return depth_diff > 0.05


def gaze_tracking(frame):
    """
    Robust gaze tracking using MediaPipe iris landmarks + 3D distance +
    head turn detection + temporal smoothing.
    
    Returns:
        dict with keys:
            - "gaze": one of "CENTERED", "LOOKING LEFT", "LOOKING RIGHT",
                      "HEAD TURNED", "NO FACE"
            - "ratio": smoothed gaze ratio float or None
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # --- 1. CHECK HEAD POSTURE FIRST ---
        if is_head_turned(landmarks):
            # Clear history so we don't carry over bad data when they turn back
            ratio_history.clear()
            return {"gaze": "HEAD TURNED", "ratio": None}

        # --- Viewer's Left Eye ---
        p_left_outer = landmarks[33]
        p_left_inner = landmarks[133]
        p_left_iris = landmarks[468]

        # --- Viewer's Right Eye ---
        p_right_inner = landmarks[362]
        p_right_outer = landmarks[263]
        p_right_iris = landmarks[473]

        # --- 2. CALCULATE 3D GAZE RATIO ---
        # Left Eye
        left_eye_width = get_3d_distance(p_left_outer, p_left_inner)
        left_iris_pos = get_3d_distance(p_left_outer, p_left_iris)
        left_ratio = left_iris_pos / left_eye_width if left_eye_width != 0 else 0.5

        # Right Eye
        right_eye_width = get_3d_distance(p_right_inner, p_right_outer)
        right_iris_pos = get_3d_distance(p_right_inner, p_right_iris)
        right_ratio = right_iris_pos / right_eye_width if right_eye_width != 0 else 0.5

        # Average both eyes
        current_gaze_ratio = (left_ratio + right_ratio) / 2

        # --- 3. APPLY TEMPORAL SMOOTHING ---
        ratio_history.append(current_gaze_ratio)
        smooth_gaze_ratio = sum(ratio_history) / len(ratio_history)

        # --- 4. EVALUATE DIRECTION ---
        # Widened the threshold (0.40 to 0.60) to prevent false positives
        if smooth_gaze_ratio < 0.40:
            gaze_direction = "LOOKING RIGHT"
        elif smooth_gaze_ratio > 0.60:
            gaze_direction = "LOOKING LEFT"
        else:
            gaze_direction = "CENTERED"

        return {"gaze": gaze_direction, "ratio": round(smooth_gaze_ratio, 2)}

    return {"gaze": "NO FACE", "ratio": None}