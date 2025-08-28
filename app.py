import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading

# Initialize pygame mixer
pygame.mixer.init()
try:
    pygame.mixer.music.load("alarm.mp3")
except:
    print("Warning: Could not load alarm.mp3. Using default system sound.")
    # You might want to use a default system sound here

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# status counters
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
alarm_playing = False
alarm_lock = threading.Lock()

def play_alarm():
    global alarm_playing
    with alarm_lock:
        if not alarm_playing:
            alarm_playing = True
            try:
                pygame.mixer.music.play(-1)  # loop alarm
            except:
                print("Error playing alarm")

def stop_alarm():
    global alarm_playing
    with alarm_lock:
        if alarm_playing:
            try:
                pygame.mixer.music.stop()
            except:
                print("Error stopping alarm")
            alarm_playing = False

def euclidean_dist(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def blink_ratio(eye_points, landmarks, w, h):
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points]
    # Calculate vertical distances
    v1 = euclidean_dist(coords[1], coords[5])
    v2 = euclidean_dist(coords[2], coords[4])
    # Calculate horizontal distance
    h_dist = euclidean_dist(coords[0], coords[3])
    ratio = (v1 + v2) / (2.0 * h_dist)
    return ratio

def draw_eye_region(frame, eye_points, landmarks, w, h, color):
    points = []
    for i in eye_points:
        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)
        points.append([x, y])
    
    # Convert to numpy array and reshape for convexHull
    points = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(points)
    
    # Draw filled polygon for the eye region
    cv2.fillPoly(frame, [hull], color)

# Define eye landmarks (MediaPipe Face Mesh has 468 landmarks)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Threshold values (adjusted for correct behavior)
# Higher ratio means eyes are more closed
SLEEP_THRESHOLD = 0.20  # Eyes are closed
DROWSY_THRESHOLD = 0.25  # Eyes are partially closed
COUNTER_THRESHOLD = 6

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ratio = blink_ratio(LEFT_EYE, landmarks, w, h)
            right_ratio = blink_ratio(RIGHT_EYE, landmarks, w, h)
            ratio = (left_ratio + right_ratio) / 2

            # Draw eye regions with color based on state
            if ratio < SLEEP_THRESHOLD:
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > COUNTER_THRESHOLD:
                    status = "SLEEPING !!!"
                    color = (0, 0, 255)  # Red for sleeping
                    # Draw red eyes
                    draw_eye_region(frame, LEFT_EYE, landmarks, w, h, (0, 0, 255))
                    draw_eye_region(frame, RIGHT_EYE, landmarks, w, h, (0, 0, 255))
                    # Run alarm in a separate thread
                    threading.Thread(target=play_alarm, daemon=True).start()
                else:
                    status = "Blinking"
                    color = (255, 255, 0)  # Yellow for blinking
            elif ratio < DROWSY_THRESHOLD:
                drowsy += 1
                sleep = 0
                active = 0
                if drowsy > COUNTER_THRESHOLD:
                    status = "Drowsy !"
                    color = (0, 165, 255)  # Orange for drowsy
                    # Draw orange eyes
                    draw_eye_region(frame, LEFT_EYE, landmarks, w, h, (0, 165, 255))
                    draw_eye_region(frame, RIGHT_EYE, landmarks, w, h, (0, 165, 255))
                    threading.Thread(target=play_alarm, daemon=True).start()
                else:
                    status = "Getting drowsy"
                    color = (255, 255, 0)  # Yellow for getting drowsy
            else:
                active += 1
                sleep = 0
                drowsy = 0
                if active > COUNTER_THRESHOLD:
                    status = "Active :)"
                    color = (0, 255, 0)  # Green for active
                    # Draw green eyes
                    draw_eye_region(frame, LEFT_EYE, landmarks, w, h, (0, 255, 0))
                    draw_eye_region(frame, RIGHT_EYE, landmarks, w, h, (0, 255, 0))
                    threading.Thread(target=stop_alarm, daemon=True).start()
                else:
                    status = "Active"
                    color = (0, 255, 0)  # Green for active
        else:
            # No face detected
            status = "No face detected"
            color = (255, 255, 255)
            sleep = 0
            drowsy = 0
            active = 0
            threading.Thread(target=stop_alarm, daemon=True).start()

        # Display status and ratio
        cv2.putText(frame, status, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Display blink ratio for debugging
        if results.multi_face_landmarks:
            cv2.putText(frame, f"Eye Ratio: {ratio:.3f}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

finally:
    cap.release()
    stop_alarm()
    cv2.destroyAllWindows()
