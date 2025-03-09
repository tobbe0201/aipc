import pyautogui
import pytesseract
import logging
import os
import time
import random
import cv2
import numpy as np
from PIL import Image, ImageGrab
import win32gui
import win32con
import win32api
import subprocess
from config import control_mode

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("executor.log"),
        logging.StreamHandler()
    ]
)

# Konfigurera pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class CommandExecutor:
    """Utför kommandon på datorn genom att interagera med skrivbordet"""
    
    def __init__(self, safety_level="high"):
        """
        Initiera exekveraren
        
        Parameters:
        - safety_level: "high", "medium" eller "low" - bestämmer vilka kommandon som tillåts
        """
        self.safety_level = safety_level
        self.last_action_time = 0
        self.min_action_interval = 0.5  # Minsta tid mellan handlingar (sekunder)
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Skapa kataloger för att spara skärmdumpar
        os.makedirs("data/screenshots", exist_ok=True)
        
        logging.info(f"🔧 CommandExecutor initierad med säkerhetsnivå: {safety_level}")
        
        # Kontrollera om kontrollägen är aktiverade
        self._check_control_modes()
        
    def _check_control_modes(self):
        """Kontrollera vilka kontrollägen som är aktiverade"""
        if not any([control_mode.get("mouse", False), 
                   control_mode.get("keyboard", False),
                   control_mode.get("system", False)]):
            logging.warning("⚠️ Inga kontrollägen är aktiverade! Aktivera i config.py")
            
        logging.info(f"🔍 Kontrollägen - Mus: {control_mode.get('mouse', False)}, "
                     f"Tangentbord: {control_mode.get('keyboard', False)}, "
                     f"System: {control_mode.get('system', False)}")
        
    def execute_command(self, command_type, params=None):
        """
        Utför ett kommando baserat på typ och parametrar
        
        Parameters:
        - command_type: Typen av kommando (click, move, type, etc.)
        - params: Parametrar för kommandot
        
        Returns:
        - dict: Resultat av kommandot
        """
        # Kontrollera intervall mellan handlingar för att undvika flödning
        current_time = time.time()
        if current_time - self.last_action_time < self.min_action_interval:
            time.sleep(self.min_action_interval - (current_time - self.last_action_time))
            
        # Uppdatera tidsstämpel
        self.last_action_time = time.time()
        
        # Standardresultat
        result = {
            "success": False,
            "command_type": command_type,
            "params": params,
            "timestamp": time.time()
        }
        
        try:
            # Utför olika kommandon baserat på typ
            if command_type == "click" and control_mode.get("mouse", False):
                result = self._execute_click(params)
                
            elif command_type == "move" and control_mode.get("mouse", False):
                result = self._execute_move(params)
                
            elif command_type == "type" and control_mode.get("keyboard", False):
                result = self._execute_type(params)
                
            elif command_type == "hotkey" and control_mode.get("keyboard", False):
                result = self._execute_hotkey(params)
                
            elif command_type == "press" and control_mode.get("keyboard", False):
                result = self._execute_press(params)
                
            elif command_type == "find_and_click" and control_mode.get("mouse", False):
                result = self._execute_find_and_click(params)
                
            elif command_type == "run_app" and control_mode.get("system", False):
                result = self._execute_run_app(params)
                
            else:
                result["error"] = f"Kommandot {command_type} stöds inte eller är inte aktiverat"
                
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            logging.error(f"Fel vid utförande av {command_type}: {e}")
            
        return result
        
    def _execute_click(self, params):
        """Utför ett musklick"""
        result = {
            "success": False,
            "command_type": "click"
        }
        
        try:
            # Hämta position
            x = params.get("x")
            y = params.get("y")
            button = params.get("button", "left")
            
            # Kontrollera att position är inom skärmen
            if x is None or y is None:
                # Om ingen position anges, använd aktuell position
                current_pos = pyautogui.position()
                x = current_pos[0]
                y = current_pos[1]
            else:
                # Kontrollera gränser
                x = max(0, min(x, self.screen_width))
                y = max(0, min(y, self.screen_height))
                
            # Utför klick
            pyautogui.click(x=x, y=y, button=button)
            
            result["success"] = True
            result["x"] = x
            result["y"] = y
            result["button"] = button
            
            logging.info(f"🖱️ Klick på ({x}, {y}) med {button}-knappen")
            
        except Exception as e:
            result["error"] = str(e)
            logging.error(f"Fel vid musklick: {e}")
            
        return result
        
    def _execute_move(self, params):
        """Flytta muspekaren"""
        result = {
            "success": False,
            "command_type": "move"
        }
        
        try:
            # Hämta position
            x = params.get("x")
            y = params.get("y")
            duration = params.get("duration", 0.5)
            
            # Kontrollera att position är inom skärmen
            if x is not None and y is not None:
                x = max(0, min(x, self.screen_width))
                y = max(0, min(y, self.screen_height))
                
                # Flytta muspekaren
                pyautogui.moveTo(x=x, y=y, duration=duration)
                
                result["success"] = True
                result["x"] = x
                result["y"] = y
                
                logging.info(f"🖱️ Flyttade muspekare till ({x}, {y})")
            else:
                result["error"] = "X och Y måste anges"
                
        except Exception as e:
            result["error"] = str(e)
            logging.error(f"Fel vid musrörelse: {e}")
            
        return result
        
    def _execute_type(self, params):
        """Skriv text"""
        result = {
            "success": False,
            "command_type": "type"
        }
        
        try:
            # Hämta text
            text = params.get("text", "")
            interval = params.get("interval", 0.05)
            
            if text:
                # Skriv text
                pyautogui.write(text, interval=interval)
                
                result["success"] = True
                result["text"] = text
                
                # För loggning - begränsa lång text
                log_text = text
                if len(log_text) > 30:
                    log_text = log_text[:27] + "..."
                logging.info(f"⌨️ Skrev text: {log_text}")
            else:
                result["error"] = "Ingen text angiven"
                
        except Exception as e:
            result["error"] = str(e)
            logging.error(f"Fel vid textskrivning: {e}")
            
        return result
        
    def _execute_hotkey(self, params):
        """Utför tangentkombination"""
        result = {
            "success": False,
            "command_type": "hotkey"
        }
        
        try:
            # Hämta tangenterna
            keys = params.get("keys", [])
            
            if keys and isinstance(keys, list) and len(keys) > 0:
                # Utför tangentkombination
                pyautogui.hotkey(*keys)
                
                result["success"] = True
                result["keys"] = keys
                
                logging.info(f"⌨️ Utförde tangentkombination: {'+'.join(keys)}")
            else:
                result["error"] = "Inga tangenter angivna"
                
        except Exception as e:
            result["error"] = str(e)
            logging.error(f"Fel vid tangentkombination: {e}")
            
        return result
        
    def _execute_press(self, params):
        """Tryck på en tangent"""
        result = {
            "success": False,
            "command_type": "press"
        }
        
        try:
            # Hämta tangent
            key = params.get("key")
            presses = params.get("presses", 1)
            interval = params.get("interval", 0.1)
            
            if key:
                # Tryck på tangent
                pyautogui.press(key, presses=presses, interval=interval)
                
                result["success"] = True
                result["key"] = key
                result["presses"] = presses
                
                logging.info(f"⌨️ Tryckte på {key} ({presses}x)")
            else:
                result["error"] = "Ingen tangent angiven"
                
        except Exception as e:
            result["error"] = str(e)
            logging.error(f"Fel vid tangenttryckning: {e}")
            
        return result
        
    def _execute_find_and_click(self, params):
        """Hitta och klicka på text eller element"""
        result = {
            "success": False,
            "command_type": "find_and_click"
        }
        
        try:
            # Hämta text att söka efter
            target_text = params.get("text")
            
            if not target_text:
                result["error"] = "Ingen måltext angiven"
                return result
                
            # Ta en skärmdump
            screenshot = self._capture_screenshot()
            
            # Använd OCR för att hitta text
            text_data = pytesseract.image_to_data(screenshot, output_type=pytesseract.Output.DICT)
            
            # Sök efter texten
            found = False
            for i, text in enumerate(text_data["text"]):
                if target_text.lower() in text.lower():
                    # Hitta position för texten
                    x = text_data["left"][i]
                    y = text_data["top"][i]
                    w = text_data["width"][i]
                    h = text_data["height"][i]
                    
                    # Beräkna mittpunkt
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Klicka
                    pyautogui.click(center_x, center_y)
                    
                    result["success"] = True
                    result["text"] = target_text
                    result["x"] = center_x
                    result["y"] = center_y
                    
                    logging.info(f"🔍 Hittade och klickade på '{target_text}' vid ({center_x}, {center_y})")
                    found = True
                    break
                    
            if not found:
                result["error"] = f"Kunde inte hitta texten '{target_text}' på skärmen"
                
        except Exception as e:
            result["error"] = str(e)
            logging.error(f"Fel vid hitta-och-klicka: {e}")
            
        return result
        
    def _execute_run_app(self, params):
        """Kör en applikation"""
        result = {
            "success": False,
            "command_type": "run_app"
        }
        
        # Endast tillåtet om systemkontroll är aktiverad och säkerhetsnivån är låg eller medel
        if not control_mode.get("system", False) or self.safety_level == "high":
            result["error"] = "Att köra applikationer är inte tillåtet med nuvarande inställningar"
            return result
            
        try:
            # Hämta appnamn
            app_name = params.get("app_name")
            
            if not app_name:
                result["error"] = "Inget applikationsnamn angivet"
                return result
                
            # Tillåtna appar (för säkerhet, utöka efter behov)
            allowed_apps = {
                "chrome": "chrome.exe",
                "firefox": "firefox.exe",
                "edge": "msedge.exe",
                "notepad": "notepad.exe",
                "calculator": "calc.exe",
                "explorer": "explorer.exe"
            }
            
            # Kontrollera om appen är tillåten
            app_exe = allowed_apps.get(app_name.lower())
            if not app_exe:
                result["error"] = f"Applikationen '{app_name}' är inte i listan över tillåtna appar"
                return result
                
            # Kör appen
            try:
                subprocess.Popen(app_exe)
                result["success"] = True
                result["app"] = app_name
                logging.info(f"🚀 Startade applikation: {app_name}")
            except FileNotFoundError:
                result["error"] = f"Kunde inte hitta applikationen '{app_name}'"
                
        except Exception as e:
            result["error"] = str(e)
            logging.error(f"Fel vid applikationsstart: {e}")
            
        return result
        
    def _capture_screenshot(self):
        """Ta en skärmdump och returnera som PIL Image"""
        screenshot = ImageGrab.grab()
        timestamp = int(time.time())
        
        # Spara skärmdumpen för framtida analys (kan vara användbart för felsökning)
        screenshot_path = f"data/screenshots/screen_{timestamp}.png"
        screenshot.save(screenshot_path)
        
        return screenshot
        
    def get_screen_info(self):
        """Analysera skärmen och returnera information"""
        try:
            # Ta en skärmdump
            screenshot = self._capture_screenshot()
            
            # Konvertera till OpenCV-format för analys
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Kör OCR för att identifiera text
            ocr_text = pytesseract.image_to_string(screenshot)
            
            # Få information om det aktiva fönstret
            active_window = win32gui.GetForegroundWindow()
            window_title = win32gui.GetWindowText(active_window)
            
            # Samla information
            screen_info = {
                "screen_width": self.screen_width,
                "screen_height": self.screen_height,
                "active_window": window_title,
                "ocr_text": ocr_text,
                "timestamp": time.time()
            }
            
            return screen_info
            
        except Exception as e:
            logging.error(f"Fel vid analys av skärmen: {e}")
            return {"error": str(e)}
            
    def find_elements(self, element_type="all"):
        """
        Hitta element på skärmen
        
        Parameters:
        - element_type: Typ av element att hitta ("all", "text", "button", "input")
        
        Returns:
        - list: Lista med hittade element
        """
        try:
            # Ta en skärmdump
            screenshot = self._capture_screenshot()
            
            elements = []
            
            # Använd OCR för att hitta text
            if element_type in ["all", "text"]:
                text_data = pytesseract.image_to_data(screenshot, output_type=pytesseract.Output.DICT)
                
                for i, text in enumerate(text_data["text"]):
                    if text.strip():  # Ignorera tom text
                        element = {
                            "type": "text",
                            "text": text,
                            "x": text_data["left"][i],
                            "y": text_data["top"][i],
                            "width": text_data["width"][i],
                            "height": text_data["height"][i],
                            "confidence": text_data["conf"][i]
                        }
                        elements.append(element)
                        
            # För knappar och input-fält skulle vi behöva mer avancerad bildanalys
            # eller använda UI-automation-bibliotek som t.ex. pywinauto
            # För denna demo använder vi en förenklad metod
                    
            return elements
            
        except Exception as e:
            logging.error(f"Fel vid sökning efter element: {e}")
            return []
            
    def execute_random_action(self, action_type="any"):
        """
        Utför en slumpmässig handling
        
        Parameters:
        - action_type: Typ av handling ("any", "click", "type", "press")
        
        Returns:
        - dict: Resultat av handlingen
        """
        if action_type == "any":
            # Välj en slumpmässig handlingstyp
            available_actions = []
            if control_mode.get("mouse", False):
                available_actions.extend(["click", "move"])
            if control_mode.get("keyboard", False):
                available_actions.extend(["type", "press"])
                
            if not available_actions:
                return {"success": False, "error": "Inga handlingstyper är aktiverade"}
                
            action_type = random.choice(available_actions)
            
        if action_type == "click" and control_mode.get("mouse", False):
            # Slumpmässigt klick
            x = random.randint(100, self.screen_width - 100)
            y = random.randint(100, self.screen_height - 100)
            
            return self.execute_command("click", {"x": x, "y": y})
            
        elif action_type == "move" and control_mode.get("mouse", False):
            # Slumpmässig musrörelse
            x = random.randint(100, self.screen_width - 100)
            y = random.randint(100, self.screen_height - 100)
            
            return self.execute_command("move", {"x": x, "y": y, "duration": 0.5})
            
        elif action_type == "type" and control_mode.get("keyboard", False):
            # Slumpmässig text
            sample_texts = ["Hej!", "Test", "AI Desktop Controller"]
            text = random.choice(sample_texts)
            
            return self.execute_command("type", {"text": text})
            
        elif action_type == "press" and control_mode.get("keyboard", False):
            # Slumpmässig tangenttryckning
            keys = ["enter", "tab", "esc", "space"]
            key = random.choice(keys)
            
            return self.execute_command("press", {"key": key})
            
        else:
            return {"success": False, "error": f"Handlingstypen {action_type} stöds inte eller är inte aktiverad"}