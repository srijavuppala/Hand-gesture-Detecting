<div align="center">

# ✋ Hand Gesture Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand_Tracking-orange?style=for-the-badge)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

<br/>

> **Real-time hand gesture recognition using MediaPipe — detects 21 landmarks and classifies 9 gesture classes with up to 1.00 confidence. Built towards a touchless presentation controller.**

<br/>

<p align="center">
  <img src="assets/gesture_demo.jpg" alt="Hand Gesture Recognition — 9 gesture classes detected with confidence scores" width="700"/>
</p>

*Real-time hand gesture classification with landmark overlay — 9 gesture classes detected with up to 1.00 confidence*

</div>

---

## 🤖 What is this?

**Hand Gesture Detection** is a real-time computer vision system that detects and classifies hand gestures through a webcam feed. Using Google's MediaPipe framework to track 21 hand landmarks, it applies custom angle-and-distance logic to recognize 9 distinct gestures — each labeled with a confidence score overlaid directly on the video frame.

The end goal is a **touchless presentation controller**: navigate slides, mute/unmute, and control your screen with nothing but your hand. No clicker needed.

---

## 🎯 Detected Gesture Classes

| Gesture | Label | Confidence |
|---|---|---|
| ✊ Closed fist | `fist` | 1.00 |
| ✌️ Two fingers up | `two_up` | 0.95 |
| ☮️ Peace sign | `peace` | 0.85 |
| 🖖 Three fingers | `three` | 1.00 |
| ☝️ Index finger | `one` | 0.99 |
| 🤫 Finger to lips | `mute` | 0.94 |
| 🤙 Call me | `call` | 0.94 |
| 👍 Thumbs up | `like` | 1.00 |
| 🖐️ Four fingers | `four` | 0.98 |

---

## ✨ Features

- 🖐️ **21-Landmark Hand Tracking** — Full skeletal overlay on detected hand in real time
- 🧠 **9 Gesture Classes** — Custom classifier using joint angles and finger distances
- 📊 **Confidence Scores** — Every prediction displayed with its confidence level
- 🎯 **High Accuracy** — Multiple gestures classified at 1.00 confidence
- 🔌 **Extensible Pipeline** — Easily add new gestures or map them to keyboard/system actions
- 🖥️ **Presentation Ready** — Designed to evolve into a full slide controller app

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **Hand Tracking** | MediaPipe |
| **Computer Vision** | OpenCV |
| **Gesture Logic** | Custom angle & distance classifier |
| **Environment** | Conda (`environment.yml`) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [Anaconda](https://www.anaconda.com/) or Miniconda
- Webcam

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/srijavuppala/Hand-gesture-Detecting.git
cd Hand-gesture-Detecting

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate hand-recognition

# 3. Run the gesture detector
python v1.py
```

### Or use the Jupyter Notebook

```bash
jupyter notebook playground/
```

---

## 📁 Project Structure

```
Hand-gesture-Detecting/
├── v1.py                # Main gesture detection script
├── config/              # Gesture class configuration
├── playground/          # Jupyter notebooks for experimentation
├── debug_frames/        # Saved debug frames
├── exports/             # Exported model/data outputs
├── environment.yml      # Conda environment file
├── requirement.txt      # Pip requirements
└── gesture_log.txt      # Detection logs
```

---

## 🗺️ Roadmap

- [x] Real-time hand landmark detection (21 points)
- [x] 9 gesture classes with confidence scoring
- [x] Visual overlay with skeleton and labels
- [ ] Map gestures to keyboard shortcuts
- [ ] Presentation controller GUI (Next / Previous / Mute / Pointer)
- [ ] Zoom & custom gesture action mapping
- [ ] Packaged desktop app

---

## 👩‍💻 Author

**Srija Vuppala**
M.S. Computer Engineering @ UT Dallas | Systems Engineer | VLSI & Full-Stack Developer

[![GitHub](https://img.shields.io/badge/GitHub-srijavuppala-black?style=flat&logo=github)](https://github.com/srijavuppala)

---

<div align="center">

*Made with 🤚 using Python, OpenCV, and MediaPipe*

</div>
