"""
Perception Integration - Integrerar alla perceptionsförmågor för AI Desktop Controller

Detta modul kombinerar UI Vision, Advanced OCR och Audio Recognition
för att skapa en fullständig perceptionsförmåga för AI-systemet.
"""

import os
import time
import logging
import threading
import json
import numpy as np
from PIL import Image
import pyautogui

# Importera perceptionskomponenter
import ui_vision
import advanced_ocr
import audio_recognition

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("perception.log"),
        logging.StreamHandler()
    ]
)

class PerceptionSystem:
    """Huvudklass för den kombinerade perceptionsförmågan"""
    
    def __init__(self, system_controller=None, use_gpu=True):
        """
        Initiera perceptionssystemet
        
        Args:
            system_controller: AI-systemkontroller
            use_gpu: Om GPU ska användas för perceptionsförmågorna
        """
        self.system_controller = system_controller
        self.use_gpu = use_gpu
        
        # Initiera perceptionskomponenter
        self.ui_vision = ui_vision.get_ui_vision(use_gpu=use_gpu)
        self.ocr_engine = advanced_ocr.get_ocr_engine(use_gpu=use_gpu)
        self.audio_manager = audio_recognition.get_audio_manager(system_controller)
        
        # Status
        self.running = False
        self.perception_thread = None
        
        # Perceptionsresultat
        self.last_screen_elements = []
        self.last_text_regions = []
        self.last_audio_events = []
        
        # Historik
        self.perception_history = []
        self.max_history = 100
        
        # Skapa mappar
        os.makedirs("data/perception", exist_ok=True)
        
        logging.info("✅ PerceptionSystem initierat")
    
    def start(self):
        """Starta alla perceptionsförmågor"""
        if self.running:
            logging.warning("⚠️ Perceptionssystemet körs redan")
            return
        
        try:
            # Starta ljudigenkänning
            self.audio_manager.start()
            
            # Starta perceptionsloop i en separat tråd
            self.running = True
            self.perception_thread = threading.Thread(target=self._perception_loop)
            self.perception_thread.daemon = True
            self.perception_thread.start()
            
            logging.info("🚀 Perceptionssystem startat")
            
        except Exception as e:
            logging.error(f"Fel vid start av perceptionssystem: {e}")
            self.stop()
    
    def stop(self):
        """Stoppa alla perceptionsförmågor"""
        if not self.running:
            return
        
        try:
            # Stoppa ljudigenkänning
            self.audio_manager.stop()
            
            # Stoppa perceptionstråd
            self.running = False
            if self.perception_thread and self.perception_thread.is_alive():
                self.perception_thread.join(timeout=2.0)
            
            logging.info("⏹️ Perceptionssystem stoppat")
            
        except Exception as e:
            logging.error(f"Fel vid stopp av perceptionssystem: {e}")
    
    def _perception_loop(self):
        """Huvudloop för kontinuerlig perception"""
        logging.info("🔄 Perceptionsloop startad")
        
        while self.running:
            try:
                # Ta skärmdump
                screenshot = pyautogui.screenshot()
                
                # Analysera UI-element
                elements = self.ui_vision.detect_elements(screenshot)
                self.last_screen_elements = elements
                
                # Kör OCR på skärmdumpen
                ocr_result = self.ocr_engine.extract_text(screenshot)
                self.last_text_regions = ocr_result.lines
                
                # Skapa perceptionsresultat
                perception_result = {
                    "timestamp": time.time(),
                    "ui_elements": {
                        "count": len(elements),
                        "types": {}
                    },
                    "text": {
                        "full_text": ocr_result.text[:1000],  # Begränsa för att spara minne
                        "count": len(ocr_result.lines)
                    },
                    "audio": {
                        "events": [e.to_dict() for e in self.last_audio_events]
                    }
                }
                
                # Räkna elementtyper
                for element in elements:
                    element_type = element.element_type
                    if element_type not in perception_result["ui_elements"]["types"]:
                        perception_result["ui_elements"]["types"][element_type] = 0
                    perception_result["ui_elements"]["types"][element_type] += 1
                
                # Lägg till i historik
                self.perception_history.append(perception_result)
                if len(self.perception_history) > self.max_history:
                    self.perception_history = self.perception_history[-self.max_history:]
                
                # Uppdatera med information från audio-hanteraren
                self.last_audio_events = self.audio_manager.audio_listener.detection_history[-5:] if self.audio_manager.audio_listener.detection_history else []
                
                # Rapportera till systemkontrollern om den finns
                if self.system_controller:
                    self._report_to_system_controller(perception_result)
                
                # Vänta innan nästa cykel
                time.sleep(1.0)
                
            except Exception as e:
                logging.error(f"Fel i perceptionsloop: {e}")
                time.sleep(1.0)
    
    def _report_to_system_controller(self, perception_result):
        """Rapportera perceptionsresultat till systemkontrollern"""
        try:
            # Anropa en metod i systemkontrollern (anpassa efter din faktiska implementation)
            if hasattr(self.system_controller, "update_perception"):
                self.system_controller.update_perception(perception_result)
            
            # Eller använd en generell loggmetod
            if hasattr(self.system_controller, "add_log_entry"):
                # Skapa ett loggmeddelande med de viktigaste observationerna
                elements_summary = ", ".join([f"{count} {type}" for type, count in 
                                           perception_result["ui_elements"]["types"].items()])
                
                text_sample = perception_result["text"]["full_text"][:100].replace('\n', ' ')
                if text_sample:
                    text_sample = f'"{text_sample}..."'
                
                audio_events = len(perception_result["audio"]["events"])
                
                log_message = (f"👁️ Observation: {len(self.last_screen_elements)} UI-element ({elements_summary}), "
                             f"{len(self.last_text_regions)} textregioner {text_sample}, "
                             f"{audio_events} ljudhändelser")
                
                self.system_controller.add_log_entry(log_message)
                
        except Exception as e:
            logging.error(f"Fel vid rapportering till systemkontrollern: {e}")
    
    def get_interactive_elements(self):
        """
        Hämta alla interaktiva element på skärmen
        
        Returns:
            list: Lista med interaktiva UI-element
        """
        return self.ui_vision.get_interactive_elements()
    
    def find_ui_element_by_text(self, text, min_confidence=0.6):
        """
        Hitta ett UI-element som innehåller specifik text
        
        Args:
            text: Text att söka efter
            min_confidence: Lägsta konfidens för matchning
            
        Returns:
            UIElement: Matchande element eller None
        """
        # Först, använd OCR för att hitta textens position
        text_bbox = self.ocr_engine.find_text_in_image(pyautogui.screenshot(), text, min_confidence)
        
        if text_bbox:
            # Beräkna mittpunkten för texten
            center_x = (text_bbox[0] + text_bbox[2]) // 2
            center_y = (text_bbox[1] + text_bbox[3]) // 2
            
            # Hitta UI-element på denna position
            element = self.ui_vision.find_element_by_position(center_x, center_y)
            
            if element:
                return element
            else:
                # Om inget element hittades på positionen, skapa ett "virtuellt" element
                from ui_vision import UIElement
                return UIElement("text_region", min_confidence, text_bbox)
        
        return None
    
    def find_clickable_by_text(self, text, min_confidence=0.6):
        """
        Hitta en klickbar kontroll med specifik text
        
        Args:
            text: Text att söka efter
            min_confidence: Lägsta konfidens för matchning
            
        Returns:
            tuple: (element, x, y) för klick eller None
        """
        # Sök efter ett passande element
        element = self.find_ui_element_by_text(text, min_confidence)
        
        if element:
            # Returnera element och dess mittposition för klick
            center_x, center_y = element.center
            return (element, center_x, center_y)
        
        return None
    
    def get_text_input_fields(self):
        """
        Hitta alla textinmatningsfält på skärmen
        
        Returns:
            list: Lista med textinmatningsfält
        """
        return self.ui_vision.find_text_input_elements()
    
    def get_all_visible_text(self):
        """
        Hämta all synlig text på skärmen
        
        Returns:
            str: All text
        """
        # Ta skärmdump
        screenshot = pyautogui.screenshot()
        
        # Kör OCR på skärmdumpen
        ocr_result = self.ocr_engine.extract_text(screenshot)
        
        return ocr_result.text
    
    def wait_for_audio_event(self, duration=10.0, pattern_type=None):
        """
        Vänta på ett ljudevent av en specifik typ
        
        Args:
            duration: Maximal väntetid i sekunder
            pattern_type: Typ av ljudmönster att vänta på (eller None för alla)
            
        Returns:
            AudioDetectionResult: Detekterat ljud eller None vid timeout
        """
        start_time = time.time()
        initial_count = len(self.audio_manager.audio_listener.detection_history)
        
        while time.time() - start_time < duration:
            current_history = self.audio_manager.audio_listener.detection_history
            
            # Om vi har nya detekteringar
            if len(current_history) > initial_count:
                # Kontrollera de nya detekteringarna
                new_detections = current_history[initial_count:]
                
                for detection in new_detections:
                    # Om ingen typ specifieras eller typen matchar
                    if pattern_type is None or detection.pattern_type == pattern_type:
                        return detection
            
            # Vänta lite innan nästa kontroll
            time.sleep(0.1)
        
        return None
    
    def analyze_screen(self, include_ocr=True, include_ui=True):
        """
        Utför en fullständig analys av den aktuella skärmen
        
        Args:
            include_ocr: Om OCR ska ingå
            include_ui: Om UI-elementdetektering ska ingå
            
        Returns:
            dict: Analysresultat
        """
        # Ta skärmdump
        screenshot = pyautogui.screenshot()
        
        result = {
            "timestamp": time.time(),
            "resolution": screenshot.size
        }
        
        # UI-elementanalys
        if include_ui:
            elements = self.ui_vision.detect_elements(screenshot)
            
            # Gruppera element efter typ
            element_types = {}
            clickable = []
            text_inputs = []
            
            for element in elements:
                # Räkna elementtyper
                element_type = element.element_type
                if element_type not in element_types:
                    element_types[element_type] = []
                element_types[element_type].append(element)
                
                # Identifiera klickbara element
                if element_type in ["button", "checkbox", "radio_button", "link", "tab"]:
                    clickable.append(element)
                
                # Identifiera textinmatningsfält
                if element_type in ["textbox", "input"]:
                    text_inputs.append(element)
            
            result["ui_elements"] = {
                "count": len(elements),
                "types": {t: len(e) for t, e in element_types.items()},
                "clickable_count": len(clickable),
                "text_input_count": len(text_inputs)
            }
        
        # OCR-analys
        if include_ocr:
            ocr_result = self.ocr_engine.extract_text(screenshot)
            
            # Hitta textrubriker, meningar, etc.
            text_lines = [line[0] for line in ocr_result.lines]
            
            # Hitta potentiella knapptextetiketter 
            button_texts = [line for line in text_lines 
                         if len(line.split()) <= 3 and len(line) < 20]
            
            result["text"] = {
                "full_text": ocr_result.text,
                "line_count": len(ocr_result.lines),
                "potential_button_labels": button_texts[:10],  # Begränsa antalet
                "confidence": ocr_result.confidence
            }
        
        # Lägg till ljudhändelser
        result["audio"] = {
            "recent_events": [e.to_dict() for e in self.last_audio_events]
        }
        
        return result
    
    def save_perception_history(self, filename=None):
        """
        Spara perceptionshistorik till fil
        
        Args:
            filename: Filnamn att spara till (eller None för automatgenererat)
            
        Returns:
            str: Filnamn
        """
        if filename is None:
            filename = f"data/perception/history_{int(time.time())}.json"
        
        # Spara till fil
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "history": self.perception_history
            }, f, indent=2)
        
        logging.info(f"💾 Perceptionshistorik sparad till {filename}")
        return filename
    
    def load_perception_history(self, filename):
        """
        Ladda perceptionshistorik från fil
        
        Args:
            filename: Filnamn att ladda från
            
        Returns:
            bool: True om laddning lyckades
        """
        if not os.path.exists(filename):
            logging.warning(f"⚠️ Filen {filename} existerar inte")
            return False
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            if "history" in data:
                self.perception_history = data["history"]
                logging.info(f"📂 Laddade {len(self.perception_history)} perceptionsresultat från {filename}")
                return True
            else:
                logging.warning(f"⚠️ Ingen historik hittades i {filename}")
                return False
                
        except Exception as e:
            logging.error(f"Fel vid laddning av perceptionshistorik: {e}")
            return False
    
    def visualize_perception(self, output_file=None):
        """
        Skapa en visuell representation av det perceptionssystemet "ser"
        
        Args:
            output_file: Filnamn att spara till (eller None för automatgenererat)
            
        Returns:
            str: Filnamn för sparad visualisering
        """
        if output_file is None:
            output_file = f"data/perception/visualization_{int(time.time())}.png"
        
        # Ta skärmdump
        screenshot = pyautogui.screenshot()
        
        # Detektera UI-element
        elements = self.ui_vision.detect_elements(screenshot)
        
        # Kör OCR
        ocr_result = self.ocr_engine.extract_text(screenshot)
        
        # Skapa visualisering för UI-element
        ui_visualization = self.ui_vision.visualize_elements(screenshot, elements, show_labels=True)
        
        # Skapa OCR-visualisering
        ocr_visualization = self.ocr_engine.visualize_ocr_result(ui_visualization, ocr_result)
        
        # Spara resultatet
        if isinstance(ocr_visualization, Image.Image):
            ocr_visualization.save(output_file)
        else:
            import cv2
            cv2.imwrite(output_file, ocr_visualization)
        
        logging.info(f"💾 Perceptionsvisualisering sparad till {output_file}")
        return output_file

# Singleton-instans
_perception_system = None

def get_perception_system(system_controller=None, use_gpu=True):
    """Hämta den globala perceptionssysteminstansen"""
    global _perception_system
    if _perception_system is None:
        _perception_system = PerceptionSystem(system_controller, use_gpu)
    return _perception_system

def start_perception():
    """Starta perceptionssystemet"""
    get_perception_system().start()

def stop_perception():
    """Stoppa perceptionssystemet"""
    if _perception_system:
        _perception_system.stop()

def analyze_current_screen():
    """Analysera aktuell skärm"""
    return get_perception_system().analyze_screen()

def find_by_text(text):
    """Convenience-funktion för att hitta element baserat på text"""
    return get_perception_system().find_ui_element_by_text(text)

def shutdown_perception():
    """Stäng ner perceptionssystemet"""
    global _perception_system
    if _perception_system:
        _perception_system.stop()
        _perception_system = None

# Om vi kör direkt, utför en demonstration
if __name__ == "__main__":
    import sys
    
    # Om det finns argument
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "analyze":
            # Analysera skärmen
            perception = get_perception_system()
            result = perception.analyze_screen()
            
            print(f"Skärmanalys:")
            print(f"- {result['ui_elements']['count']} UI-element detekterade")
            print(f"- {result['ui_elements']['clickable_count']} klickbara element")
            print(f"- {len(result['text']['full_text'])} tecken av text")
            
            # Spara visualisering
            output_file = perception.visualize_perception()
            print(f"Visualisering sparad till {output_file}")
            
        elif command == "listen":
            # Lyssna efter ljudhändelser
            duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
            
            perception = get_perception_system()
            perception.start()
            
            print(f"Lyssnar i {duration} sekunder...")
            time.sleep(duration)
            
            perception.stop()
            
            audio_events = perception.last_audio_events
            print(f"Detekterade {len(audio_events)} ljudhändelser")
            for event in audio_events:
                print(f"- {event}")
                
        elif command == "find":
            # Hitta text på skärmen
            if len(sys.argv) < 3:
                print("Användning: python perception_integration.py find \"text att söka efter\"")
                sys.exit(1)
                
            text = sys.argv[2]
            
            perception = get_perception_system()
            result = perception.find_clickable_by_text(text)
            
            if result:
                element, x, y = result
                print(f"Hittade '{text}' i element: {element.element_type} på position ({x}, {y})")
            else:
                print(f"Kunde inte hitta '{text}' på skärmen")
                
        else:
            print(f"Okänt kommando: {command}")
            print("Användning: python perception_integration.py [analyze|listen|find]")
    else:
        # Kör standarddemonstration
        print("Startar perceptionsdemonstration...")
        
        perception = get_perception_system()
        perception.start()
        
        try:
            print("Analyserar skärmen...")
            result = perception.analyze_screen()
            
            print(f"Skärmanalys:")
            if "ui_elements" in result:
                print(f"- {result['ui_elements']['count']} UI-element detekterade")
                print(f"- {result['ui_elements']['clickable_count']} klickbara element")
            
            if "text" in result:
                print(f"- Textinnehåll ({len(result['text']['full_text'])} tecken):")
                print(f"  \"{result['text']['full_text'][:100]}...\"")
            
            # Spara visualisering
            output_file = perception.visualize_perception()
            print(f"Visualisering sparad till {output_file}")
            
            print("\nLyssnar efter ljud i 5 sekunder...")
            audio_event = perception.wait_for_audio_event(5.0)
            
            if audio_event:
                print(f"Detekterat ljud: {audio_event}")
            else:
                print("Inga ljud detekterade inom tidsgränsen")
                
        except KeyboardInterrupt:
            print("\nAvbruten av användare")
        finally:
            perception.stop()
            
        print("Demonstration avslutad")