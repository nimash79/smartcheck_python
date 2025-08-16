import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture("rtsp://admin:@192.168.0.197:554/live/ch00_0")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ==== NUMERIC RESULTS EXAMPLES ====

        # Face landmarks
        if results.face_landmarks:
            for id, lm in enumerate(results.face_landmarks.landmark[:5]):  # just first 5 points as example
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f"Face LM {id}: x={cx}, y={cy}, z={lm.z:.4f}")

        # Pose landmarks
        if results.pose_landmarks:
            left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            h, w, _ = frame.shape
            ls_x, ls_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
            rs_x, rs_y = int(right_shoulder.x * w), int(right_shoulder.y * h)
            print(f"Shoulders: Left=({ls_x}, {ls_y}), Right=({rs_x}, {rs_y})")

        # Hand landmarks
        if results.right_hand_landmarks:
            for id, lm in enumerate(results.right_hand_landmarks.landmark[:3]):  # just first 3 points
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f"Right hand LM {id}: x={cx}, y={cy}")

        # ==== DRAWING (optional) ====
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('Full Body Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
