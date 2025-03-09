import os
import sys
import subprocess

# Lista över alla Python-paket som behövs
REQUIRED_PACKAGES = [
    "pyautogui",
    "pynput",
    "opencv-python",
    "pillow",
    "pytesseract",
    "pygetwindow",
    "customtkinter"
]

# 🔹 1. Installera Python-bibliotek
def install_packages():
    print("🚀 Installerar nödvändiga Python-bibliotek...")
    for package in REQUIRED_PACKAGES:
        subprocess.call([sys.executable, "-m", "pip", "install", package])

# 🔹 2. Kontrollera och installera Tesseract OCR
def check_tesseract():
    print("🔍 Kontrollerar om Tesseract OCR är installerat...")
    if sys.platform.startswith("win"):
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if not os.path.exists(tesseract_path):
            print("⚠️ Tesseract OCR hittades inte! Ladda ner från:")
            print("🔗 https://github.com/UB-Mannheim/tesseract/wiki")
    else:
        result = subprocess.run(["which", "tesseract"], capture_output=True, text=True)
        if not result.stdout.strip():
            print("⚠️ Tesseract OCR saknas! Installera med:")
            print("🔹 Ubuntu: sudo apt install tesseract-ocr")
            print("🔹 macOS: brew install tesseract")

# 🔹 3. Skapa filsystemet om det saknas
def create_project_structure():
    print("📂 Skapar projektstruktur...")
    folders = ["ai_desktop_controller"]
    files = [
        "main.py",
        "config.py",
        "event_listener.py",
        "ai_engine.py",
        "ui.py",
        "recorder.py",
        "executor.py"
    ]
    
    os.makedirs(folders[0], exist_ok=True)
    
    for file in files:
        file_path = os.path.join(folders[0], file)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("# " + file + "\n")
            print(f"✅ Skapade: {file_path}")
    
    print("🎉 Projektet är redo!")

# 🔹 4. Kör installationsstegen
def main():
    install_packages()
    check_tesseract()
    create_project_structure()
    print("\n✅ Installation klar! Starta programmet med:")
    print("   cd ai_desktop_controller && python main.py")

if __name__ == "__main__":
    main()
