import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

# status counters
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
alarm_playing = False

def play_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        pygame.mixer.music.play(-1)  # loop alarm

def stop_alarm():
    global alarm_playing
    if alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False

def euclidean_dist(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def blink_ratio(eye_points, landmarks, w, h):
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points]
    up = euclidean_dist(coords[1], coords[5]) + euclidean_dist(coords[2], coords[4])
    down = euclidean_dist(coords[0], coords[3])
    ratio = up / (2.0 * down)
    return ratio

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        left_ratio = blink_ratio(left_eye, landmarks, w, h)
        right_ratio = blink_ratio(right_eye, landmarks, w, h)

        ratio = (left_ratio + right_ratio) / 2

        if ratio > 0.25:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                play_alarm()
        elif 0.21 < ratio <= 0.25:
            drowsy += 1
            sleep = 0
            active = 0
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
                play_alarm()
        else:
            active += 1
            sleep = 0
            drowsy = 0
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)
                stop_alarm()

        cv2.putText(frame, status, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Sleepiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
stop_alarm()
cv2.destroyAllWindows()
