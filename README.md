# 🖐️ Hand Gesture Controlled Mouse and Scroll System

Control your system's mouse and scrolling behavior using hand gestures through your webcam. This Python-based application uses **MediaPipe**, **OpenCV**, and **PyAutoGUI** to recognize hand gestures and map them to system actions like cursor movement, clicks, drag-drop, and scrolling.

## 📌 Features

- 🖱️ **Move Cursor** — Using hand movement
- 👊 **Left Click / Drag & Drop** — Using Fist gesture
- ✌️ **Activate Pointer** — Using V Sign gesture
- ☝️ **Right Click** — Using Index finger
- ✌️ (closed) **Double Click** — With two fingers close together
- 🤏 **Scroll Horizontally/Vertically** — Using Pinch gestures

## 🛠️ Technologies Used

- [MediaPipe](https://google.github.io/mediapipe/) — for real-time hand landmark tracking
- [OpenCV](https://opencv.org/) — for video capture and processing
- [PyAutoGUI](https://pyautogui.readthedocs.io/) — for controlling mouse
- [PyCAW](https://github.com/AndreMiras/pycaw) — for system audio control (extension capability)

🧠 Gesture Guide
Gesture	Action
✌️ V Sign	Activate Pointer
👊 Fist	Drag (Hold & Move)
☝️ Index Finger Only	Right Click
✌️ (Closed Together)	Double Click
🤏 Pinch with Minor Hand	Scroll Content
🖐️ Palm (default state)	Idle / No Action

📷 How It Works
Uses webcam to track hand positions.

Processes hand landmarks with MediaPipe.

Classifies gestures based on finger positions and distances.

Maps gestures to mouse/scroll events using PyAutoGUI.

💡 Notes
Make sure your hand is visible and well-lit for accurate detection.

Use gestures steadily for consistent recognition.

Press Enter (↵) key to exit the app.
