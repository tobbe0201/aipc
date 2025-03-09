"""
Advanced OCR - Förbättrad textidentifiering för AI Desktop Controller

Detta modul implementerar avancerad OCR med förbehandling,
flera OCR-motorer och postprocessing för förbättrad textidentifiering.
"""

import os
import time
import logging
import json
import numpy as np
import cv2
from PIL import Image
import threading
from queue import Queue
import re
import difflib

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("advanced_ocr.log"),
        logging.StreamHandler()
    ]
)

# Globala variabler
OCR_ENGINES = {}
DEFAULT_LANGUAGE = "swe+eng"
AVAILABLE_ENGINES = ["tesseract", "easyocr", "paddleocr", "mmocr"]

class OCRResult:
    """Klass för att representera OCR-resultat"""
    
    def __init__(self, text="", confidence=0.0, bbox=None, engine="unknown"):
        """
        Initiera OCR-resultat
        
        Args:
            text: Identifierad text
            confidence: Konfidens (0-1)
            bbox: Bounding box (x1, y1, x2, y2) eller None
            engine: OCR-motor som användes
        """
        self.text = text
        self.confidence = confidence
        self.bbox = bbox if bbox is not None else (0, 0, 0, 0)
        self.engine = engine
        self.timestamp = time.time()
        
        # Ytterligare attribut för detaljanalys
        self.words = []  # Lista med (word, conf, bbox)
        self.lines = []  # Lista med (line, conf, bbox)
        self.preprocessing = None  # Vilket förbehandlingssteg som användes
        self.processing_time = 0.0  # Tid det tog att utföra OCR
    
    def to_dict(self):
        """Konvertera till dictionary för JSON-serialisering"""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "engine": self.engine,
            "timestamp": self.timestamp,
            "words": self.words,
            "lines": self.lines,
            "preprocessing": self.preprocessing,
            "processing_time": self.processing_time
        }
    
    def __str__(self):
        return f"OCR Result: '{self.text[:30]}{'...' if len(self.text) > 30 else ''}' ({self.confidence:.2f}, {self.engine})"

class TesseractEngine:
    """Motor för Tesseract OCR"""
    
    def __init__(self):
        """Initiera Tesseract OCR-motorn"""
        try:
            import pytesseract
            self.engine = pytesseract
            self.available = True
            
            # Kontrollera Tesseract-sökväg
            if os.name == 'nt':  # Windows
                tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                if os.path.exists(tesseract_path):
                    self.engine.pytesseract.tesseract_cmd = tesseract_path
                    logging.info(f"✅ Tesseract sökväg satt till {tesseract_path}")
            
            # Hämta tillgängliga språk
            try:
                self.available_languages = self.engine.get_languages()
                logging.info(f"✅ Tesseract OCR initialiserad med {len(self.available_languages)} språk")
            except:
                self.available_languages = ["eng"]
                logging.warning("⚠️ Kunde inte hämta språk från Tesseract")
                
        except ImportError:
            self.available = False
            self.engine = None
            logging.warning("⚠️ pytesseract är inte installerat, Tesseract OCR kommer inte att vara tillgängligt")
    
    def is_available(self):
        """Kontrollera om motorn är tillgänglig"""
        return self.available
    
    def extract_text(self, image, lang=DEFAULT_LANGUAGE, config=""):
        """
        Extrahera text från en bild med Tesseract
        
        Args:
            image: PIL.Image, numpy-array eller filsökväg
            lang: Språkkod(er) för OCR
            config: Extra Tesseract-konfiguration
        
        Returns:
            OCRResult: Identifierad text med metadata
        """
        if not self.available:
            return OCRResult(text="OCR engine unavailable", confidence=0.0, engine="tesseract")
        
        # Konvertera till PIL Image om det är en numpy-array eller filsökväg
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)
        elif isinstance(image, str) and os.path.exists(image):
            image = Image.open(image)
        
        start_time = time.time()
        
        try:
            # Kör OCR
            custom_config = f"-l {lang} --oem 1 --psm 3 {config}"
            
            # Hämta text
            text = self.engine.image_to_string(image, config=custom_config)
            
            # Hämta detaljdata med konfidenser
            data = self.engine.image_to_data(image, config=custom_config, output_type=self.engine.Output.DICT)
            
            # Beräkna genomsnittlig konfidens för text
            valid_confidences = [conf for conf in data['conf'] if conf != -1]
            avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0
            avg_confidence = avg_confidence / 100.0  # Normalisera till [0, 1]
            
            # Skapa resultat
            result = OCRResult(
                text=text.strip(),
                confidence=avg_confidence,
                engine="tesseract"
            )
            
            # Lägg till ordnivåinformation
            for i in range(len(data['text'])):
                if data['text'][i].strip() and data['conf'][i] != -1:
                    word_conf = data['conf'][i] / 100.0
                    word_bbox = (
                        data['left'][i],
                        data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    )
                    result.words.append((data['text'][i], word_conf, word_bbox))
            
            # Gruppera ord till rader
            lines = {}
            for i in range(len(data['text'])):
                if data['text'][i].strip() and data['conf'][i] != -1:
                    line_num = data['block_num'][i], data['par_num'][i], data['line_num'][i]
                    if line_num not in lines:
                        lines[line_num] = {
                            'text': [],
                            'conf': [],
                            'left': data['left'][i],
                            'top': data['top'][i],
                            'right': data['left'][i] + data['width'][i],
                            'bottom': data['top'][i] + data['height'][i]
                        }
                    else:
                        lines[line_num]['right'] = max(lines[line_num]['right'], data['left'][i] + data['width'][i])
                        lines[line_num]['bottom'] = max(lines[line_num]['bottom'], data['top'][i] + data['height'][i])
                    
                    lines[line_num]['text'].append(data['text'][i])
                    lines[line_num]['conf'].append(data['conf'][i])
            
            # Lägg till radnivåinformation
            for line_num, line_data in lines.items():
                line_text = ' '.join(line_data['text'])
                line_conf = sum(line_data['conf']) / len(line_data['conf']) if line_data['conf'] else 0
                line_conf = line_conf / 100.0
                line_bbox = (line_data['left'], line_data['top'], line_data['right'], line_data['bottom'])
                
                result.lines.append((line_text, line_conf, line_bbox))
            
            # Lägg till processningstid
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            logging.error(f"Fel vid Tesseract OCR: {e}")
            return OCRResult(text=f"Error: {str(e)}", confidence=0.0, engine="tesseract")

class EasyOCREngine:
    """Motor för EasyOCR"""
    
    def __init__(self):
        """Initiera EasyOCR-motorn"""
        try:
            import easyocr
            self.engine_class = easyocr
            self.engine = None  # Lazy loading
            self.available = True
            self.available_languages = [
                'abq', 'ady', 'af', 'ar', 'as', 'ava', 'az', 'be', 'bg', 'bh', 'bho', 'bn', 'bs', 
                'ch', 'che', 'cs', 'cy', 'da', 'dar', 'de', 'en', 'es', 'et', 'fa', 'fr', 'ga', 
                'gom', 'hi', 'hr', 'hu', 'id', 'inh', 'is', 'it', 'ja', 'kbd', 'kn', 'ko', 'ku', 
                'la', 'lbe', 'lez', 'lt', 'lv', 'mhr', 'mi', 'mn', 'mr', 'ms', 'mt', 'ne', 'new', 
                'nl', 'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'ru', 'sck', 'si', 'sk', 'sl', 'sq', 
                'sv', 'sw', 'ta', 'tab', 'te', 'th', 'tjk', 'tl', 'tr', 'ug', 'uk', 'ur', 'uz', 'vi'
            ]
            logging.info("✅ EasyOCR importerad, kommer att initieras vid första användning")
        except ImportError:
            self.available = False
            self.engine_class = None
            self.engine = None
            logging.warning("⚠️ easyocr är inte installerat, EasyOCR kommer inte att vara tillgängligt")
    
    def is_available(self):
        """Kontrollera om motorn är tillgänglig"""
        return self.available
    
    def _initialize_engine(self, lang):
        """Lazy initialization av EasyOCR-motorn"""
        if not self.available:
            return False
        
        if self.engine is None:
            try:
                # Konvertera språkkombination till lista
                lang_list = [l.strip() for l in lang.split('+')]
                
                # Kontrollera att alla språk finns tillgängliga
                for l in lang_list:
                    if l not in self.available_languages:
                        logging.warning(f"⚠️ Språk {l} stöds inte av EasyOCR, använder 'en' istället")
                        lang_list = ['en']
                        break
                
                # Initiera motorn
                self.engine = self.engine_class.Reader(lang_list, gpu=True)
                logging.info(f"✅ EasyOCR initialiserad med språk: {lang_list}")
                return True
            except Exception as e:
                logging.error(f"Fel vid initiering av EasyOCR: {e}")
                self.available = False
                return False
        return True
    
    def extract_text(self, image, lang=DEFAULT_LANGUAGE, **kwargs):
        """
        Extrahera text från en bild med EasyOCR
        
        Args:
            image: PIL.Image, numpy-array eller filsökväg
            lang: Språkkod(er) för OCR
            **kwargs: Extra konfiguration
        
        Returns:
            OCRResult: Identifierad text med metadata
        """
        if not self.available:
            return OCRResult(text="OCR engine unavailable", confidence=0.0, engine="easyocr")
        
        # Initiera motorn om den inte redan är initierad
        if not self._initialize_engine(lang):
            return OCRResult(text="Failed to initialize OCR engine", confidence=0.0, engine="easyocr")
        
        # Konvertera till numpy-array om det är en PIL-bild
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        elif isinstance(image, str) and os.path.exists(image):
            image_np = np.array(Image.open(image))
        else:
            image_np = image
        
        start_time = time.time()
        
        try:
            # Kör OCR
            detail = kwargs.get('detail', 0)
            paragraph = kwargs.get('paragraph', False)
            min_size = kwargs.get('min_size', 10)
            
            # Hämta resultat
            results = self.engine.readtext(
                image_np,
                detail=detail,
                paragraph=paragraph,
                min_size=min_size
            )
            
            # Bearbeta resultatet
            full_text = []
            avg_confidence = 0.0
            word_count = 0
            
            words = []
            lines = []
            
            for r in results:
                bbox, text, confidence = r
                full_text.append(text)
                avg_confidence += confidence
                word_count += 1
                
                # Konvertera bbox från 4 punkter till (x1, y1, x2, y2)
                x_min = min(pt[0] for pt in bbox)
                y_min = min(pt[1] for pt in bbox)
                x_max = max(pt[0] for pt in bbox)
                y_max = max(pt[1] for pt in bbox)
                bbox_rect = (x_min, y_min, x_max, y_max)
                
                # Lägg till rad
                lines.append((text, confidence, bbox_rect))
                
                # Dela upp i ord
                for word in text.split():
                    words.append((word, confidence, bbox_rect))  # Använder samma bbox för alla ord, inte optimalt
            
            # Beräkna genomsnittlig konfidens
            avg_confidence = avg_confidence / word_count if word_count > 0 else 0.0
            
            # Skapa resultat
            result = OCRResult(
                text=' '.join(full_text),
                confidence=avg_confidence,
                engine="easyocr"
            )
            
            result.words = words
            result.lines = lines
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            logging.error(f"Fel vid EasyOCR: {e}")
            return OCRResult(text=f"Error: {str(e)}", confidence=0.0, engine="easyocr")

class PaddleOCREngine:
    """Motor för PaddleOCR"""
    
    def __init__(self):
        """Initiera PaddleOCR-motorn"""
        try:
            import paddleocr
            self.engine_class = paddleocr
            self.engine = None  # Lazy loading
            self.available = True
            self.available_languages = [
                'ch', 'en', 'fr', 'german', 'korean', 'japan', 'chinese_cht', 
                'ta', 'te', 'ka', 'latin', 'arabic', 'cyrillic', 'devanagari'
            ]
            logging.info("✅ PaddleOCR importerad, kommer att initieras vid första användning")
        except ImportError:
            self.available = False
            self.engine_class = None
            self.engine = None
            logging.warning("⚠️ paddleocr är inte installerat, PaddleOCR kommer inte att vara tillgängligt")
    
    def is_available(self):
        """Kontrollera om motorn är tillgänglig"""
        return self.available
    
    def _initialize_engine(self, lang):
        """Lazy initialization av PaddleOCR-motorn"""
        if not self.available:
            return False
        
        if self.engine is None:
            try:
                # Konvertera språk till PaddleOCR-format
                paddle_lang = 'en'
                if 'ch' in lang or 'zh' in lang:
                    paddle_lang = 'ch'
                elif 'en' in lang:
                    paddle_lang = 'en'
                elif 'fr' in lang:
                    paddle_lang = 'fr'
                elif 'de' in lang or 'german' in lang:
                    paddle_lang = 'german'
                elif 'ko' in lang or 'korean' in lang:
                    paddle_lang = 'korean'
                elif 'ja' in lang or 'jp' in lang or 'japan' in lang:
                    paddle_lang = 'japan'
                
                # Initiera motorn
                use_gpu = True
                self.engine = self.engine_class.PaddleOCR(use_gpu=use_gpu, lang=paddle_lang)
                logging.info(f"✅ PaddleOCR initialiserad med språk: {paddle_lang}")
                return True
            except Exception as e:
                logging.error(f"Fel vid initiering av PaddleOCR: {e}")
                self.available = False
                return False
        return True
    
    def extract_text(self, image, lang=DEFAULT_LANGUAGE, **kwargs):
        """
        Extrahera text från en bild med PaddleOCR
        
        Args:
            image: PIL.Image, numpy-array eller filsökväg
            lang: Språkkod(er) för OCR
            **kwargs: Extra konfiguration
        
        Returns:
            OCRResult: Identifierad text med metadata
        """
        if not self.available:
            return OCRResult(text="OCR engine unavailable", confidence=0.0, engine="paddleocr")
        
        # Initiera motorn om den inte redan är initierad
        if not self._initialize_engine(lang):
            return OCRResult(text="Failed to initialize OCR engine", confidence=0.0, engine="paddleocr")
        
        # Konvertera till rätt format
        if isinstance(image, Image.Image):
            image_path = None
            # Spara tillfälligt till fil eftersom PaddleOCR fungerar bäst med filsökvägar
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                image.save(temp.name)
                image_path = temp.name
        elif isinstance(image, str) and os.path.exists(image):
            image_path = image
        else:
            # För numpy-array, spara tillfälligt till fil
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                if len(image.shape) == 3 and image.shape[2] == 3:
                    cv2.imwrite(temp.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(temp.name, image)
                image_path = temp.name
        
        start_time = time.time()
        
        try:
            # Kör OCR
            results = self.engine.ocr(image_path, cls=True)
            
            # Ta bort temporär fil om vi skapade en
            if image_path is not None and image_path.startswith(tempfile.gettempdir()):
                try:
                    os.remove(image_path)
                except:
                    pass
            
            # Bearbeta resultatet
            full_text = []
            avg_confidence = 0.0
            line_count = 0
            
            words = []
            lines = []
            
            for result in results:
                for line in result:
                    bbox, (text, confidence) = line
                    
                    full_text.append(text)
                    avg_confidence += confidence
                    line_count += 1
                    
                    # Konvertera bbox från 4 punkter till (x1, y1, x2, y2)
                    x_min = min(pt[0] for pt in bbox)
                    y_min = min(pt[1] for pt in bbox)
                    x_max = max(pt[0] for pt in bbox)
                    y_max = max(pt[1] for pt in bbox)
                    bbox_rect = (int(x_min), int(y_min), int(x_max), int(y_max))
                    
                    # Lägg till rad
                    lines.append((text, confidence, bbox_rect))
                    
                    # Dela upp i ord
                    for word in text.split():
                        words.append((word, confidence, bbox_rect))  # Använder samma bbox för alla ord
            
            # Beräkna genomsnittlig konfidens
            avg_confidence = avg_confidence / line_count if line_count > 0 else 0.0
            
            # Skapa resultat
            result = OCRResult(
                text=' '.join(full_text),
                confidence=avg_confidence,
                engine="paddleocr"
            )
            
            result.words = words
            result.lines = lines
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            logging.error(f"Fel vid PaddleOCR: {e}")
            return OCRResult(text=f"Error: {str(e)}", confidence=0.0, engine="paddleocr")

class AdvancedOCR:
    """Huvudklass för avancerad OCR med förbehandling och flera motorer"""
    
    def __init__(self, preferred_engine=None, use_gpu=True):
        """
        Initiera AdvancedOCR
        
        Args:
            preferred_engine: Föredragen OCR-motor ('tesseract', 'easyocr', 'paddleocr')
            use_gpu: Om GPU ska användas för OCR
        """
        self.use_gpu = use_gpu
        
        # Initiera tillgängliga OCR-motorer
        self.engines = {}
        self._initialize_engines()
        
        # Sätt föredragen motor
        self.preferred_engine = preferred_engine
        if preferred_engine is None:
            # Använd första tillgängliga motor i prioritetsordning
            for engine in AVAILABLE_ENGINES:
                if engine in self.engines and self.engines[engine].is_available():
                    self.preferred_engine = engine
                    break
        
        # Skapa mappar för att spara OCR-resultat
        os.makedirs("data/ocr_results", exist_ok=True)
        
        # Preprocessing-metoder
        self.preprocessing_methods = [
            "none",
            "threshold",
            "adaptive_threshold",
            "bilateral_filter",
            "contrast_enhancement",
            "morphology",
            "denoising"
        ]
        
        # Historik för OCR-resultat
        self.result_history = []
        self.max_history = 100
        
        # Arbetsköer för parallell OCR
        self.ocr_queue = Queue(maxsize=10)
        self.result_queue = Queue()
        
        # Starta arbetartrådar
        self.workers = []
        self.running = True
        self._start_workers()
        
        logging.info(f"✅ AdvancedOCR initierad med föredragen motor: {self.preferred_engine}")
    
    def _initialize_engines(self):
        """Initiera alla tillgängliga OCR-motorer"""
        # Tesseract
        self.engines["tesseract"] = TesseractEngine()
        
        # EasyOCR
        self.engines["easyocr"] = EasyOCREngine()
        
        # PaddleOCR
        self.engines["paddleocr"] = PaddleOCREngine()
        
        # Logga tillgängliga motorer
        available_engines = [e for e, engine in self.engines.items() if engine.is_available()]
        logging.info(f"✅ Tillgängliga OCR-motorer: {', '.join(available_engines)}")
    
    def _start_workers(self):
        """Starta arbetartrådar för parallell OCR"""
        num_workers = 2
        
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        logging.info(f"🚀 {len(self.workers)} OCR-arbetare startade")
    
    def _worker_loop(self, worker_id):
        """Arbetarloop för parallell OCR"""
        logging.info(f"OCR-arbetare {worker_id} startad")
        
        while self.running:
            try:
                # Hämta nästa uppgift från kön
                task_id, image, params, callback = self.ocr_queue.get(timeout=0.1)
                
                # Utför OCR
                result = self.extract_text(image, **params)
                
                # Anropa callback eller lägg resultatet i svarskön
                if callback:
                    callback(result)
                else:
                    self.result_queue.put((task_id, result))
                
                # Markera uppgiften som klar
                self.ocr_queue.task_done()
                
            except Queue.Empty:
                # Ingen uppgift i kön
                pass
            except Exception as e:
                logging.error(f"Fel i OCR-arbetare {worker_id}: {e}")
        
        logging.info(f"OCR-arbetare {worker_id} avslutad")
    
    def stop_workers(self):
        """Stoppa alla arbetartrådar"""
        self.running = False
        
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1.0)
        
        self.workers = []
        logging.info("⏹️ OCR-arbetare stoppade")
    
    def extract_text_async(self, image, engine=None, lang=DEFAULT_LANGUAGE, 
                         preprocessing=None, **kwargs):
        """
        Asynkron textextrahering från bild
        
        Args:
            image: PIL.Image, numpy-array eller filsökväg
            engine: OCR-motor att använda (None för att använda föredragen)
            lang: Språkkod(er) för OCR
            preprocessing: Förbehandlingsmetod
            **kwargs: Extra parametrar till OCR-motorn
        
        Returns:
            str: Uppgifts-ID för den asynkrona OCR-operationen
        """
        # Generera uppgifts-ID
        task_id = f"ocr_task_{int(time.time())}_{id(image)}"
        
        # Paketear parametrarna
        params = {
            "engine": engine,
            "lang": lang,
            "preprocessing": preprocessing,
            **kwargs
        }
        
        # Lägg till i OCR-kön
        self.ocr_queue.put((task_id, image, params, None))
        
        return task_id
    
    def get_result(self, task_id, timeout=10.0):
        """
        Hämta resultatet av en asynkron OCR-operation
        
        Args:
            task_id: Uppgifts-ID från extract_text_async
            timeout: Timeout i sekunder
        
        Returns:
            OCRResult: OCR-resultat eller None vid timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Kontrollera om resultatet finns i kön
            if not self.result_queue.empty():
                result_task_id, result = self.result_queue.get()
                
                if result_task_id == task_id:
                    return result
            
            # Vänta lite innan nästa kontroll
            time.sleep(0.1)
        
        # Timeout
        logging.warning(f"Timeout vid väntande på OCR-resultat för {task_id}")
        return None
    
    def extract_text(self, image, engine=None, lang=DEFAULT_LANGUAGE, 
                    preprocessing=None, try_all_engines=False, 
                    postprocess=True, **kwargs):
        """
        Extrahera text från en bild med avancerad OCR
        
        Args:
            image: PIL.Image, numpy-array eller filsökväg
            engine: OCR-motor att använda (None för att använda föredragen)
            lang: Språkkod(er) för OCR
            preprocessing: Förbehandlingsmetod (None för att prova några)
            try_all_engines: Om alla tillgängliga motorer ska provas
            postprocess: Om postprocessing ska göras på resultatet
            **kwargs: Extra parametrar till OCR-motorn
        
        Returns:
            OCRResult: OCR-resultat
        """
        # Hantera bildformat
        if isinstance(image, str) and os.path.exists(image):
            image = Image.open(image)
        
        # Använd föredragen motor om ingen anges
        if engine is None:
            engine = self.preferred_engine
        
        # Kontrollera att motorn är tillgänglig
        if engine not in self.engines or not self.engines[engine].is_available():
            logging.warning(f"⚠️ OCR-motorn {engine} är inte tillgänglig, använder {self.preferred_engine} istället")
            engine = self.preferred_engine
        
        # Om vi ska prova flera förbehandlingsmetoder
        if preprocessing is None:
            # Prova olika förbehandlingsmetoder och välj den bästa
            best_result = None
            best_score = -1
            
            # Prova varje metod
            for method in ["none", "threshold", "contrast_enhancement"]:
                processed_image = self._preprocess_image(image, method)
                result = self._extract_with_engine(processed_image, engine, lang, **kwargs)
                result.preprocessing = method
                
                # Beräkna "poäng" för resultatet (längd, konfidens, etc.)
                score = len(result.text) * result.confidence
                
                if score > best_score:
                    best_score = score
                    best_result = result
            
            result = best_result
        else:
            # Använd angiven förbehandlingsmetod
            processed_image = self._preprocess_image(image, preprocessing)
            result = self._extract_with_engine(processed_image, engine, lang, **kwargs)
            result.preprocessing = preprocessing
        
        # Om vi ska prova alla motorer
        if try_all_engines:
            all_results = [result]
            
            # Prova varje tillgänglig motor
            for eng_name, eng in self.engines.items():
                if eng_name != engine and eng.is_available():
                    eng_result = self._extract_with_engine(processed_image, eng_name, lang, **kwargs)
                    eng_result.preprocessing = result.preprocessing
                    all_results.append(eng_result)
            
            # Välj det bästa resultatet baserat på konfidens och längd
            result = max(all_results, key=lambda r: len(r.text) * r.confidence)
        
        # Postprocessing om aktiverat
        if postprocess:
            result = self._postprocess_result(result)
        
        # Lägg till i historiken
        self.result_history.append(result)
        if len(self.result_history) > self.max_history:
            self.result_history = self.result_history[-self.max_history:]
        
        return result
    
    def _extract_with_engine(self, image, engine, lang, **kwargs):
        """
        Extrahera text med en specifik motor
        
        Args:
            image: Förbehandlad bild
            engine: OCR-motor att använda
            lang: Språkkod(er)
            **kwargs: Extra parametrar
        
        Returns:
            OCRResult: OCR-resultat
        """
        if engine in self.engines and self.engines[engine].is_available():
            return self.engines[engine].extract_text(image, lang, **kwargs)
        else:
            # Fallback till föredragen motor
            logging.warning(f"⚠️ Motorn {engine} är inte tillgänglig, använder {self.preferred_engine}")
            return self.engines[self.preferred_engine].extract_text(image, lang, **kwargs)
    
    def _preprocess_image(self, image, method):
        """
        Förbehandla bild för bättre OCR
        
        Args:
            image: PIL.Image eller numpy-array
            method: Förbehandlingsmetod
        
        Returns:
            PIL.Image: Förbehandlad bild
        """
        # Konvertera till numpy-array om det är en PIL-bild
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            pil_input = True
        else:
            image_np = image
            pil_input = False
        
        # Konvertera till gråskala om det är en färgbild
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Använd vald förbehandlingsmetod
        if method == "none" or method is None:
            processed = gray
        elif method == "threshold":
            # Enkel binarisering med Otsu's metod
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "adaptive_threshold":
            # Adaptiv binarisering för varierande ljusförhållanden
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
        elif method == "bilateral_filter":
            # Bilateral filtrering bevarar kanter medan brus reduceras
            processed = cv2.bilateralFilter(gray, 9, 75, 75)
        elif method == "contrast_enhancement":
            # CLAHE-baserad kontrastförbättring
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)
        elif method == "morphology":
            # Morfologiska operationer för att förbättra text
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        elif method == "denoising":
            # Non-local means denoising
            processed = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        else:
            # Okänd metod, använd original
            processed = gray
            logging.warning(f"⚠️ Okänd förbehandlingsmetod: {method}, använder ingen förbehandling")
        
        # Konvertera tillbaka till PIL om indata var PIL
        if pil_input:
            return Image.fromarray(processed)
        else:
            return processed
    
    def _postprocess_result(self, result):
        """
        Förbättra OCR-resultat genom postprocessing
        
        Args:
            result: OCRResult
        
        Returns:
            OCRResult: Förbättrat resultat
        """
        if not result.text:
            return result
        
        # Skapa en kopia av resultatet
        new_result = OCRResult(
            text=result.text,
            confidence=result.confidence,
            bbox=result.bbox,
            engine=result.engine
        )
        new_result.words = result.words.copy()
        new_result.lines = result.lines.copy()
        new_result.preprocessing = result.preprocessing
        new_result.processing_time = result.processing_time
        
        # 1. Ta bort extra blanksteg
        new_result.text = ' '.join(new_result.text.split())
        
        # 2. Korrigera vanliga OCR-fel
        new_result.text = self._fix_common_ocr_errors(new_result.text)
        
        # 3. Fixa radbrytningar
        new_result.text = new_result.text.replace('\n\n', '\n')
        
        # 4. Ta bort icke-alfanumeriska tecken i början/slutet
        new_result.text = new_result.text.strip('.,;:!?-_"\'')
        
        return new_result
    
    def _fix_common_ocr_errors(self, text):
        """
        Fixa vanliga OCR-fel
        
        Args:
            text: Text att fixa
        
        Returns:
            str: Fixad text
        """
        # Lista över vanliga OCR-fel och deras korrekta form
        common_errors = {
            # Bokstavsförväxlingar
            'c0': 'co', 'cl': 'd', 'rn': 'm', 'vv': 'w', 
            '0': 'o', '1': 'l', '5': 's', '8': 'B',
            # Vanliga svenska fel
            'å': 'ä', 'ä': 'å',
            # Specialtecken
            '|': 'I', '[': '(', ']': ')', '{': '(', '}': ')',
            # Skräptecken
            ''': "'", '"': '"', '"': '"', '„': '"',
            '–': '-', '—': '-', '−': '-'
        }
        
        # Ersätt kända fel
        for error, correction in common_errors.items():
            # Ersätt bara om det är hela ord eller delar av ord, inte inom ord
            text = re.sub(r'\b' + error + r'\b', correction, text)
            
        # Ta bort enstaka icke-alfanumeriska tecken som är omgivna av blanksteg
        text = re.sub(r' [^\w\s] ', ' ', text)
        
        return text
    
    def find_text_in_image(self, image, text_to_find, min_confidence=0.6):
        """
        Hitta specifik text i en bild och returnera dess position
        
        Args:
            image: PIL.Image, numpy-array eller filsökväg
            text_to_find: Text att hitta
            min_confidence: Lägsta konfidens för matchning
        
        Returns:
            tuple: (x1, y1, x2, y2) bbox för den hittade texten, eller None
        """
        # Kör OCR på bilden
        result = self.extract_text(image, postprocess=True)
        
        # Sök igenom resultatets rader och ord
        text_to_find = text_to_find.lower()
        
        # Exakt matchning på rader
        for line_text, confidence, bbox in result.lines:
            if text_to_find in line_text.lower() and confidence >= min_confidence:
                return bbox
        
        # Partial matchning med difflib
        best_match = None
        best_ratio = 0
        
        for line_text, confidence, bbox in result.lines:
            if confidence < min_confidence:
                continue
                
            # Använd difflib för fuzzy matching
            ratio = difflib.SequenceMatcher(None, line_text.lower(), text_to_find).ratio()
            
            if ratio > 0.7 and ratio > best_ratio:
                best_ratio = ratio
                best_match = bbox
        
        return best_match
    
    def get_all_text_regions(self, image, min_confidence=0.5):
        """
        Hitta alla textregioner i en bild
        
        Args:
            image: PIL.Image, numpy-array eller filsökväg
            min_confidence: Lägsta konfidens för inkludering
        
        Returns:
            list: Lista med (text, bbox, confidence) för alla textregioner
        """
        # Kör OCR på bilden
        result = self.extract_text(image, postprocess=True)
        
        # Filtrera rader baserat på konfidens
        text_regions = [(text, bbox, conf) for text, conf, bbox in result.lines 
                      if conf >= min_confidence]
        
        return text_regions
    
    def save_result_to_file(self, result, filename=None):
        """
        Spara OCR-resultat till fil
        
        Args:
            result: OCRResult
            filename: Filnamn att spara till (eller None för auto-genererat)
        
        Returns:
            str: Filnamn
        """
        if filename is None:
            filename = f"data/ocr_results/ocr_result_{int(time.time())}.json"
            
        # Säkerställ att mappen finns
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Spara resultatet
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2)
            
        logging.info(f"💾 OCR-resultat sparat till {filename}")
        return filename
    
    def load_result_from_file(self, filename):
        """
        Ladda OCR-resultat från fil
        
        Args:
            filename: Filnamn att ladda från
        
        Returns:
            OCRResult: Laddat resultat
        """
        if not os.path.exists(filename):
            logging.warning(f"⚠️ Filen {filename} hittades inte")
            return None
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Skapa ett nytt OCRResult från data
            result = OCRResult(
                text=data.get("text", ""),
                confidence=data.get("confidence", 0.0),
                bbox=data.get("bbox"),
                engine=data.get("engine", "unknown")
            )
            
            # Fyll i extra attribut
            result.words = data.get("words", [])
            result.lines = data.get("lines", [])
            result.preprocessing = data.get("preprocessing")
            result.processing_time = data.get("processing_time", 0.0)
            result.timestamp = data.get("timestamp", time.time())
            
            logging.info(f"📂 OCR-resultat laddat från {filename}")
            return result
            
        except Exception as e:
            logging.error(f"Fel vid laddning av OCR-resultat från {filename}: {e}")
            return None
    
    def visualize_ocr_result(self, image, result, show_words=True, 
                           show_lines=True, output_file=None):
        """
        Visualisera OCR-resultat på en bild
        
        Args:
            image: PIL.Image, numpy-array eller filsökväg
            result: OCRResult
            show_words: Om individuella ord ska visas
            show_lines: Om textrader ska visas
            output_file: Filnamn att spara resultatet till (eller None)
        
        Returns:
            PIL.Image: Bild med visualiserat OCR-resultat
        """
        # Konvertera till numpy-array om det är en PIL-bild eller filsökväg
        if isinstance(image, str):
            if os.path.exists(image):
                image = Image.open(image)
            else:
                raise FileNotFoundError(f"Bildfilen {image} hittades inte")
        
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        # Konvertera till BGR för OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_np.copy()
        
        # Rita textrader
        if show_lines and result.lines:
            for line_text, confidence, bbox in result.lines:
                if isinstance(bbox, tuple) and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    
                    # Skapa en färg baserat på konfidens (röd -> grön)
                    color = (
                        int(255 * (1 - confidence)),  # B
                        int(255 * confidence),       # G
                        0                            # R
                    )
                    
                    # Rita en rektangel
                    cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
                    
                    # Rita text med konfidens
                    conf_text = f"{confidence:.2f}"
                    cv2.putText(image_cv, conf_text, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Rita enskilda ord
        if show_words and result.words:
            for word_text, confidence, bbox in result.words:
                if isinstance(bbox, tuple) and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    
                    # Skapa en färg baserat på konfidens (röd -> grön)
                    color = (
                        0,                           # B
                        int(255 * confidence),       # G
                        int(255 * (1 - confidence))  # R
                    )
                    
                    # Rita en tunnare rektangel
                    cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 1)
        
        # Konvertera tillbaka till PIL om det var PIL-format ursprungligen
        if isinstance(image, Image.Image):
            result_img = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        else:
            result_img = image_cv
        
        # Spara till fil om angivet
        if output_file:
            if isinstance(result_img, Image.Image):
                result_img.save(output_file)
            else:
                cv2.imwrite(output_file, result_img)
            
            logging.info(f"💾 OCR-visualisering sparad till {output_file}")
        
        return result_img

 Convenience-funktion för att få en global OCR-instans
_advanced_ocr = None

def get_ocr_engine(preferred_engine=None, use_gpu=True):
    """Hämta en global OCR-instans"""
    global _advanced_ocr
    if _advanced_ocr is None:
        _advanced_ocr = AdvancedOCR(preferred_engine, use_gpu)
    return _advanced_ocr

def extract_text(image, engine=None, lang=DEFAULT_LANGUAGE, 
                preprocessing=None, try_all_engines=False, 
                postprocess=True, **kwargs):
    """Convenience-funktion för att extrahera text från en bild"""
    return get_ocr_engine().extract_text(
        image, engine, lang, preprocessing, try_all_engines, postprocess, **kwargs)

def find_text_in_image(image, text_to_find, min_confidence=0.6):
    """Convenience-funktion för att hitta text i en bild"""
    return get_ocr_engine().find_text_in_image(image, text_to_find, min_confidence)

def get_all_text_regions(image, min_confidence=0.5):
    """Convenience-funktion för att hitta alla textregioner i en bild"""
    return get_ocr_engine().get_all_text_regions(image, min_confidence)

def shutdown_ocr_engine():
    """Stäng ner OCR-motorn"""
    global _advanced_ocr
    if _advanced_ocr is not None:
        _advanced_ocr.stop_workers()
        _advanced_ocr = None

# Om vi kör direkt, utför ett enkelt test
if __name__ == "__main__":
    # Kolla om argumentet är en bildsökväg
    import sys
    
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        input_image = sys.argv[1]
    else:
        # Ta en skärmdump
        from PIL import ImageGrab
        input_image = ImageGrab.grab()
    
    # Initiera OCR
    ocr = AdvancedOCR(use_gpu=True)
    
    # Extrahera text
    result = ocr.extract_text(input_image, try_all_engines=True, postprocess=True)
    
    print(f"OCR Resultat ({result.engine}, konfidens: {result.confidence:.2f}):")
    print(result.text)
    
    # Visualisera resultat
    visualized = ocr.visualize_ocr_result(input_image, result)
    output_path = "ocr_result.png"
    
    if isinstance(visualized, Image.Image):
        visualized.save(output_path)
    else:
        cv2.imwrite(output_path, visualized)
    
    print(f"Visualisering sparad till {output_path}")