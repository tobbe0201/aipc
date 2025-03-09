@echo off
echo 🚀 Installerar AI Desktop Controller...
mkdir ai_desktop_controller
cd ai_desktop_controller

echo 📂 Skapar mappar och filer...
copy nul main.py
copy nul config.py
copy nul event_listener.py
copy nul ai_engine.py
copy nul ui.py
copy nul recorder.py
copy nul executor.py

echo 📦 Installerar beroenden...
pip install pyautogui pynput opencv-python pillow pytesseract pygetwindow customtkinter

echo ✅ Installation klar! Kör 'python main.py' för att starta.
