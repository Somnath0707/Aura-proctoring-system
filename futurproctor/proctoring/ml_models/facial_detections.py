import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
# Use max_num_faces=1 for just the candidate. 
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)

def gaze_tracking_debug(frame):
    """Detect gaze direction based on relative iris position and draw visual debug points."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    annotated_frame = frame.copy()
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # --- Viewer's Left Eye ---
        # Outer corner: 33, Inner corner: 133, Iris center: 468
        p_left_outer = landmarks[33]
        p_left_inner = landmarks[133]
        p_left_iris = landmarks[468]

        # --- Viewer's Right Eye ---
        # Inner corner: 362, Outer corner: 263, Iris center: 473
        p_right_inner = landmarks[362]
        p_right_outer = landmarks[263]
        p_right_iris = landmarks[473]

        # --- DRAW DEBUGGING POINTS ---
        # Draw Left Eye (Red outer, Blue inner, Green iris)
        cv2.circle(annotated_frame, (int(p_left_outer.x * w), int(p_left_outer.y * h)), 3, (0, 0, 255), -1)
        cv2.circle(annotated_frame, (int(p_left_inner.x * w), int(p_left_inner.y * h)), 3, (255, 0, 0), -1)
        cv2.circle(annotated_frame, (int(p_left_iris.x * w), int(p_left_iris.y * h)), 3, (0, 255, 0), -1)

        # Draw Right Eye
        cv2.circle(annotated_frame, (int(p_right_inner.x * w), int(p_right_inner.y * h)), 3, (255, 0, 0), -1)
        cv2.circle(annotated_frame, (int(p_right_outer.x * w), int(p_right_outer.y * h)), 3, (0, 0, 255), -1)
        cv2.circle(annotated_frame, (int(p_right_iris.x * w), int(p_right_iris.y * h)), 3, (0, 255, 0), -1)

        # --- CALCULATE GAZE RATIO ---
        # Ratio = Where is the iris located between the inner and outer corners?
        left_eye_width = p_left_inner.x - p_left_outer.x
        left_iris_pos = p_left_iris.x - p_left_outer.x
        left_ratio = left_iris_pos / left_eye_width if left_eye_width != 0 else 0.5

        right_eye_width = p_right_outer.x - p_right_inner.x
        right_iris_pos = p_right_iris.x - p_right_inner.x
        right_ratio = right_iris_pos / right_eye_width if right_eye_width != 0 else 0.5

        gaze_ratio = (left_ratio + right_ratio) / 2

        # --- EVALUATE DIRECTION ---
        # 0.5 is perfectly centered. 
        if gaze_ratio < 0.42:
            gaze_direction = "RIGHT"
            color = (0, 0, 255) # Red warning
        elif gaze_ratio > 0.58:
            gaze_direction = "LEFT"
            color = (0, 0, 255) # Red warning
        else:
            gaze_direction = "CENTER"
            color = (0, 255, 0) # Green OK

        # Print the status on the screen
        cv2.putText(annotated_frame, f'Gaze: {gaze_direction} ({gaze_ratio:.2f})', (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return {"gaze": gaze_direction, "ratio": round(gaze_ratio, 2)}, annotated_frame

    return {"gaze": "NO FACE", "ratio": None}, annotated_frame