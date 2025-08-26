💤 Drowsiness Detection System (OpenCV + ML + Alarm)

A real-time driver drowsiness detection system that uses OpenCV for video processing and machine learning models (mediapipe) for facial landmark detection.
The system tracks eye blinks and classifies whether the person is Active, Drowsy, or Sleeping.
If sleepiness is detected, an alarm sound is played to alert the user.

📌 Features

🎥 Real-time video capture using OpenCV

👀 Eye-blink and facial landmark detection using ML models

🟢 Classifies states:

Active 🙂

Drowsy 😵

Sleeping 😴

🔔 Plays an alarm when the user is detected as Sleeping

⚡ Works with a webcam or mobile camera stream

🛠️ Tech Stack

Python 3.x

OpenCV → video processing & visualization

NumPy → mathematical operations

Mediapipe → facial landmark detection


pygame → for alarm sound

📂 Project Structure
sleepiness-detector/
│── app.py                          # Main application
│── requirements.txt                # Dependencies
│── alarm.mp3                       # Alarm sound file
│── README.md                       # Documentation


📊 How It Works

Capture frames from webcam using OpenCV

Convert frame to grayscale for processing

Detect face landmarks (eyes, mouth) using ML model

Compute Eye Aspect Ratio (EAR) to check blink rate

Decide the state:

Active 🙂 → eyes open

Drowsy 😵 → eyes partially closed

Sleeping 😴 → eyes closed for too long → 🔔 Alarm plays
