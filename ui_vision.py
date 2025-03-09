"""
UI Vision - Avancerad datorseende för AI Desktop Controller

Detta modul implementerar objektdetektering med YOLO för att identifiera
UI-element i skärmdumpar med hög precision.
"""

import os
import time
import logging
import numpy as np
import cv2
from PIL import Image
import tempfile
import threading
import json
from queue import Queue

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ui_vision.log"),
        logging.StreamHandler()
    ]
)

# Globala variabler för delning av yolov5-modellen
MODEL = None
DEVICE = "cpu"  # Standard, uppdateras om GPU finns tillgänglig
MODEL_LOCK = threading.Lock()
MODEL_NAME = "yolov5s"  # Standard-modell
MODEL_INITIALIZED = False

def initialize_model(model_name=None, use_gpu=True):
    """
    Initialisera YOLOv5-modellen för UI-elementdetektering
    
    Args:
        model_name: YOLOv5-modellnamn ('yolov5s', 'yolov5m', etc.)
        use_gpu: Om GPU ska användas om tillgänglig
    """
    global MODEL, DEVICE, MODEL_NAME, MODEL_INITIALIZED
    
    if MODEL_INITIALIZED:
        return MODEL
    
    with MODEL_LOCK:
        if MODEL_INITIALIZED:
            return MODEL
        
        if model_name:
            MODEL_NAME = model_name
        
        # Kontrollera om torch finns tillgängligt för modellimport
        try:
            import torch
            DEVICE = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
            logging.info(f"PyTorch använder enhet: {DEVICE}")
            
            try:
                # Försök importera YOLOv5 via torch hub
                MODEL = torch.hub.load('ultralytics/yolov5', MODEL_NAME)
                MODEL.to(DEVICE)
                MODEL.conf = 0.25  # Konfidenströskel
                MODEL.classes = None  # Alla klasser
                MODEL.multi_label = False
                MODEL.max_det = 100  # Max antal detektioner
                MODEL_INITIALIZED = True
                logging.info(f"✅ YOLOv5 ({MODEL_NAME}) modell laddad på {DEVICE}")
                
                # Om importering via torch hub lyckas, är vi klara
                return MODEL
            except Exception as e:
                logging.warning(f"Kunde inte ladda YOLOv5 via torch hub: {e}")
                
                # Fallback: Försök lokal import om YOLOv5 är installerat
                try:
                    import yolov5
                    MODEL = yolov5.load(MODEL_NAME)
                    MODEL.to(DEVICE)
                    MODEL.conf = 0.25  # Konfidenströskel
                    MODEL_INITIALIZED = True
                    logging.info(f"✅ YOLOv5 ({MODEL_NAME}) modell laddad lokalt på {DEVICE}")
                    return MODEL
                except ImportError:
                    logging.warning("YOLOv5 är inte installerat lokalt")
                    
        except ImportError:
            logging.warning("PyTorch är inte installerat")
        
        # Om vi når hit har både online och lokala importer misslyckats
        # Fallback till OpenCV DNN-baserad YOLO om möjligt
        try:
            # Kontrollera om vi kan hitta viktfilen i en förväntad plats
            weights_path = os.path.join("models", "yolov3-ui.weights")
            config_path = os.path.join("models", "yolov3-ui.cfg")
            
            if os.path.exists(weights_path) and os.path.exists(config_path):
                MODEL = cv2.dnn.readNetFromDarknet(config_path, weights_path)
                
                # Använd OpenCV GPU (CUDA) om tillgängligt
                if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    MODEL.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    MODEL.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    DEVICE = "opencv-cuda"
                else:
                    MODEL.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    MODEL.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    DEVICE = "opencv-cpu"
                
                MODEL_INITIALIZED = True
                logging.info(f"✅ OpenCV DNN YOLO modell laddad på {DEVICE}")
                return MODEL
            else:
                logging.warning(f"YOLO-viktfiler saknas i {weights_path}")
        except Exception as e:
            logging.error(f"Fel vid laddning av OpenCV DNN YOLO-modell: {e}")
        
        # Om inget av ovanstående fungerar, använd OpenCV-baserade metoder
        logging.warning("❌ Ingen YOLO-modell kunde laddas. Använder fallback till enklare CV-metoder.")
        MODEL = "opencv-fallback"
        MODEL_INITIALIZED = True
        DEVICE = "opencv-cpu"
    
    return MODEL

class UIElement:
    """Representation av ett detekterat UI-element"""
    
    def __init__(self, element_type, confidence, bbox, screenshot=None):
        """
        Initiera ett UI-element
        
        Args:
            element_type: Typ av element ('button', 'textbox', etc.)
            confidence: Detekteringskonfidens (0-1)
            bbox: Bounding box (x1, y1, x2, y2)
            screenshot: Skärmdump av elementet (valfritt)
        """
        self.element_type = element_type
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.screenshot = screenshot
        self.text = None  # Fylls i av OCR senare
        self.id = f"{element_type}_{int(time.time())}_{id(self)}"
        self.attributes = {}  # Extra attribut
    
    @property
    def center(self):
        """Beräkna elementets mittpunkt"""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    @property
    def width(self):
        """Elementets bredd"""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self):
        """Elementets höjd"""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self):
        """Elementets area"""
        return self.width * self.height
    
    def to_dict(self):
        """Konvertera elementet till en dictionary"""
        return {
            "id": self.id,
            "type": self.element_type,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "center": self.center,
            "width": self.width,
            "height": self.height,
            "text": self.text,
            "attributes": self.attributes
        }
    
    def __str__(self):
        return f"{self.element_type} ({self.confidence:.2f}): {self.bbox} | Text: {self.text}"

class UIVision:
    """
    Huvudklass för datorseende-baserad UI-analys med YOLO
    """
    
    def __init__(self, model_name=None, use_gpu=True):
        """
        Initiera UI Vision
        
        Args:
            model_name: YOLO-modellnamn (standard 'yolov5s')
            use_gpu: Om GPU ska användas för inference
        """
        self.use_gpu = use_gpu
        self.model = initialize_model(model_name, use_gpu)
        
        # Kategori-mappning för UI-element
        self.ui_classes = {
            0: "button",
            1: "textbox",
            2: "checkbox",
            3: "radio_button",
            4: "dropdown",
            5: "menu",
            6: "icon",
            7: "tab",
            8: "scrollbar",
            9: "slider",
            10: "link",
            11: "image",
            12: "panel",
            13: "toolbar",
            14: "dialog"
        }
        
        # Standardparametrar för detektering
        self.conf_threshold = 0.25
        self.nms_threshold = 0.45
        self.num_workers = 2
        
        # UI-element-historik
        self.detected_elements = {}  # element_id -> UIElement
        
        # Analysparametrar
        self.temporal_smoothing = True
        self.max_history_elements = 1000
        
        # Detektionsmetadata
        self.last_detection_time = 0
        self.detection_count = 0
        
        # Arbetsköer för parallell bearbetning
        self.detection_queue = Queue(maxsize=10)
        self.result_queue = Queue()
        
        # Starta arbetartrådar
        self._start_workers()
        
        logging.info(f"✅ UIVision initialiserad med {'GPU' if use_gpu else 'CPU'}")
    
    def _start_workers(self):
        """Starta arbetartrådar för parallell detektering"""
        self.workers = []
        self.running = True
        
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        logging.info(f"🚀 {len(self.workers)} UI-detekteringsarbetare startade")
    
    def _worker_loop(self, worker_id):
        """Arbetarloop för parallell detektering"""
        logging.info(f"Arbetare {worker_id} startad")
        
        while self.running:
            try:
                # Hämta nästa bild från kön
                task_id, image, callback = self.detection_queue.get(timeout=0.1)
                
                # Detektera element
                elements = self._detect_elements(image)
                
                # Notifiera via callback eller lägg till i resultatkön
                if callback:
                    callback(elements)
                else:
                    self.result_queue.put((task_id, elements))
                
                # Markera uppgiften som klar
                self.detection_queue.task_done()
                
            except Queue.Empty:
                # Ingen bild i kön, fortsätt loopen
                pass
            except Exception as e:
                logging.error(f"Fel i arbetare {worker_id}: {e}")
        
        logging.info(f"Arbetare {worker_id} avslutad")
    
    def stop_workers(self):
        """Stoppa alla arbetartrådar"""
        self.running = False
        
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1.0)
        
        self.workers = []
        logging.info("⏹️ UI-detekteringsarbetare stoppade")
    
    def detect_elements_async(self, image, callback=None):
        """
        Asynkron detektering av UI-element
        
        Args:
            image: PIL.Image, numpy-array eller filsökväg
            callback: Funktion att anropa med resultatet
        
        Returns:
            task_id: ID för uppgiften (om callback är None)
        """
        # Generera ett uppgifts-ID
        task_id = f"task_{int(time.time())}_{self.detection_count}"
        self.detection_count += 1
        
        # Lägg till i kön
        self.detection_queue.put((task_id, image, callback))
        
        return task_id
    
    def get_detection_result(self, task_id, timeout=10.0):
        """
        Hämta resultatet från en asynkron detektering
        
        Args:
            task_id: Uppgifts-ID från detect_elements_async
            timeout: Timeout i sekunder
        
        Returns:
            List[UIElement]: Detekterade element eller None vid timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Kontrollera om resultatet finns i kön
            if not self.result_queue.empty():
                result_task_id, elements = self.result_queue.get()
                
                if result_task_id == task_id:
                    return elements
            
            # Vänta lite innan nästa kontroll
            time.sleep(0.1)
        
        # Timeout
        logging.warning(f"Timeout vid väntande på resultat för {task_id}")
        return None
    
    def detect_elements(self, image):
        """
        Synkron detektering av UI-element
        
        Args:
            image: PIL.Image, numpy-array eller filsökväg
        
        Returns:
            List[UIElement]: Detekterade element
        """
        # Kontrollera och konvertera bilden
        if isinstance(image, str):
            # Filsökväg
            if os.path.exists(image):
                image = Image.open(image)
            else:
                raise FileNotFoundError(f"Bildfilen {image} hittades inte")
        
        # Detektera UI-element
        elements = self._detect_elements(image)
        
        # Uppdatera detektionsmetadata
        self.last_detection_time = time.time()
        
        # Spara detekterade element i historik
        for element in elements:
            self.detected_elements[element.id] = element
            
            # Begränsa historiken till max_history_elements
            if len(self.detected_elements) > self.max_history_elements:
                oldest_key = next(iter(self.detected_elements))
                del self.detected_elements[oldest_key]
        
        return elements
    
    def _detect_elements(self, image):
        """
        Intern metod för att detektera UI-element
        
        Args:
            image: PIL.Image eller numpy-array
        
        Returns:
            List[UIElement]: Detekterade element
        """
        # Konvertera till numpy-array om det är en PIL-bild
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Konvertera till BGR för OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_np
        
        elements = []
        
        # Använd YOLO-modellen beroende på typ
        if isinstance(self.model, str) and self.model == "opencv-fallback":
            # Fallback till OpenCV-metoder om ingen YOLO-modell finns
            elements = self._detect_elements_opencv(image_cv)
        elif "torch" in str(type(self.model)).lower():
            # YOLOv5 PyTorch-modell
            elements = self._detect_elements_yolov5(image)
        elif "cv2" in str(type(self.model)).lower() or "net" in str(type(self.model)).lower():
            # OpenCV DNN YOLO
            elements = self._detect_elements_opencv_dnn(image_cv)
        
        return elements
    
    def _detect_elements_yolov5(self, image):
        """
        Detektera UI-element med YOLOv5-modell
        
        Args:
            image: PIL.Image eller numpy-array
        
        Returns:
            List[UIElement]: Detekterade element
        """
        elements = []
        
        try:
            # Kör inference med YOLOv5
            results = self.model(image)
            
            # Konvertera resultat till UIElement-objekt
            predictions = results.pandas().xyxy[0]
            
            for _, row in predictions.iterrows():
                class_id = int(row['class'])
                confidence = float(row['confidence'])
                
                # Hämta klassen 
                if class_id in self.ui_classes:
                    element_type = self.ui_classes[class_id]
                else:
                    element_type = f"unknown_{class_id}"
                
                # Skapa bounding box
                x1, y1, x2, y2 = (
                    int(row['xmin']),
                    int(row['ymin']),
                    int(row['xmax']),
                    int(row['ymax'])
                )
                
                # Klippa ut elementets skärmdump
                if isinstance(image, Image.Image):
                    element_img = image.crop((x1, y1, x2, y2))
                else:
                    # Om det är en numpy-array
                    element_img = image[y1:y2, x1:x2]
                
                # Skapa UI-element
                element = UIElement(element_type, confidence, (x1, y1, x2, y2), element_img)
                elements.append(element)
        
        except Exception as e:
            logging.error(f"Fel vid YOLOv5-detektering: {e}")
        
        return elements
    
    def _detect_elements_opencv_dnn(self, image):
        """
        Detektera UI-element med OpenCV DNN YOLO
        
        Args:
            image: OpenCV-format (BGR) numpy-array
        
        Returns:
            List[UIElement]: Detekterade element
        """
        elements = []
        
        try:
            # Förbered bilden
            height, width = image.shape[:2]
            
            # Skapa blob för input till YOLO
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.model.setInput(blob)
            
            # Hämta nätverkets output-lagernamn
            output_layers = self.model.getUnconnectedOutLayersNames()
            
            # Kör inference
            outputs = self.model.forward(output_layers)
            
            # Bearbeta output
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.conf_threshold:
                        # YOLO ger ut (center_x, center_y, width, height)
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Beräkna övre vänstra hörnet
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Använd non-maximum suppression för att ta bort överlappande boxar
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
            
            # Skapa UIElement-objekt från detektioner
            for i in indices:
                if isinstance(i, tuple):  # OpenCV 3.x
                    i = i[0]
                    
                box = boxes[i]
                x, y, w, h = box
                x1, y1, x2, y2 = x, y, x + w, y + h
                
                # Kontrollera att boxen är inom bildgränserna
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                class_id = class_ids[i]
                confidence = confidences[i]
                
                # Hämta klassen 
                if class_id in self.ui_classes:
                    element_type = self.ui_classes[class_id]
                else:
                    element_type = f"unknown_{class_id}"
                
                # Klippa ut elementets skärmdump
                element_img = image[y1:y2, x1:x2]
                
                # Skapa UI-element
                element = UIElement(element_type, confidence, (x1, y1, x2, y2), element_img)
                elements.append(element)
        
        except Exception as e:
            logging.error(f"Fel vid OpenCV DNN YOLO-detektering: {e}")
        
        return elements
    
    def _detect_elements_opencv(self, image):
        """
        Detektera UI-element med grundläggande OpenCV-metoder (fallback)
        
        Args:
            image: OpenCV-format (BGR) numpy-array
        
        Returns:
            List[UIElement]: Detekterade element
        """
        elements = []
        
        try:
            # Konvertera till gråskala
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Förbehandling
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Hitta konturer
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Bearbeta konturerna
            for contour in contours:
                # Ignorera små konturer
                if cv2.contourArea(contour) < 50:
                    continue
                
                # Beräkna bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Klassificera elementet baserat på form
                element_type = self._classify_element_shape(x, y, w, h, contour)
                
                # Beräkna en pseudo-konfidens baserat på formkriterier
                confidence = self._calculate_shape_confidence(contour, element_type)
                
                # Skapa UI-element-objekt
                element = UIElement(element_type, confidence, (x, y, x+w, y+h), image[y:y+h, x:x+w])
                elements.append(element)
        
        except Exception as e:
            logging.error(f"Fel vid OpenCV element-detektering: {e}")
        
        return elements
    
    def _classify_element_shape(self, x, y, w, h, contour):
        """
        Klassificera UI-element baserat på form
        
        Args:
            x, y, w, h: Bounding box-koordinater
            contour: Kontur från OpenCV
        
        Returns:
            str: Element-typ
        """
        # Beräkna aspektkvot
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Beräkna area och konturomkrets
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Approximera konturen för att förenkla formen
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        
        # Räkna antalet hörn
        num_vertices = len(approx)
        
        # Klassificera baserat på egenskaper
        if 0.9 <= aspect_ratio <= 1.1 and num_vertices in [3, 4, 5]:
            # Nästan kvadratisk/rund - troligtvis checkbox, radiobutton eller icon
            if area < 500:
                return "checkbox" if num_vertices == 4 else "radio_button"
            else:
                return "icon"
        elif 2.0 <= aspect_ratio <= 6.0:
            # Horisontell rektangel - troligtvis knapp eller textbox
            if h < 40:
                return "button"
            else:
                return "textbox"
        elif 0.16 <= aspect_ratio <= 0.5:
            # Vertikal rektangel - troligtvis scrollbar
            return "scrollbar"
        elif aspect_ratio > 6.0:
            # Mycket bred - troligtvis toolbar eller menu
            return "menu" if y < 100 else "toolbar"
        elif w > 100 and h > 100:
            # Stor rektangel - troligtvis panel eller dialog
            return "panel"
        else:
            # Standardvärde för okända former
            return "unknown"
    
    def _calculate_shape_confidence(self, contour, element_type):
        """
        Beräkna en pseudo-konfidens för elementklassificering baserat på form
        
        Args:
            contour: Kontur från OpenCV
            element_type: Klassificerad elementtyp
        
        Returns:
            float: Pseudo-konfidensvärde (0.0-1.0)
        """
        # Beräkna formens egenskaper
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Approximera konturen
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        num_vertices = len(approx)
        
        # Beräkna konvext hölje och konvexitetsdefekter
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull) if hull.size > 3 else None
        
        # Grundläggande baskonfidens
        base_confidence = 0.5
        
        # Justera baserat på elementtyp och formegenskaper
        if element_type == "button":
            # Knappar är ofta rektangulära med 4 hörn
            if num_vertices == 4:
                base_confidence += 0.2
            
            # Knappar har ofta relativt liten area
            if 500 < area < 5000:
                base_confidence += 0.1
                
        elif element_type == "checkbox":
            # Checkboxar är ofta små och fyrkantiga
            if num_vertices == 4 and area < 500:
                base_confidence += 0.3
                
        elif element_type == "radio_button":
            # Radioknappar är ofta runda (många hörn i approximationen)
            if num_vertices > 4 and area < 500:
                base_confidence += 0.3
                
        elif element_type == "textbox":
            # Textboxar är ofta rektangulära med större area
            if num_vertices == 4 and area > 1000:
                base_confidence += 0.2
        
        # Begränsa konfidens till intervallet [0.1, 0.9]
        return max(0.1, min(0.9, base_confidence))
    
    def find_element_by_type(self, element_type, min_confidence=0.5, sort_by="confidence"):
        """
        Hitta UI-element av en specifik typ bland tidigare detekterade element
        
        Args:
            element_type: Typ av element att hitta
            min_confidence: Lägsta konfidens att acceptera
            sort_by: Sortera resultat efter ('confidence', 'area', 'x', 'y')
        
        Returns:
            List[UIElement]: Matchande element
        """
        matching_elements = [
            element for element in self.detected_elements.values()
            if element.element_type == element_type and element.confidence >= min_confidence
        ]
        
        # Sortera element
        if sort_by == "confidence":
            matching_elements.sort(key=lambda e: e.confidence, reverse=True)
        elif sort_by == "area":
            matching_elements.sort(key=lambda e: e.area, reverse=True)
        elif sort_by == "x":
            matching_elements.sort(key=lambda e: e.bbox[0])
        elif sort_by == "y":
            matching_elements.sort(key=lambda e: e.bbox[1])
        
        return matching_elements
    
    def find_element_by_position(self, x, y, element_types=None):
        """
        Hitta ett element på en specifik position
        
        Args:
            x, y: Koordinater att söka på
            element_types: Lista med elementtyper att begränsa till (valfritt)
        
        Returns:
            UIElement eller None
        """
        candidates = []
        
        for element in self.detected_elements.values():
            x1, y1, x2, y2 = element.bbox
            
            # Kontrollera om punkten är inom elementets bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Om elementtyper är specificerade, filtrera på dem
                if element_types is None or element.element_type in element_types:
                    candidates.append(element)
        
        # Om flera element matchar, välj det med högst konfidens
        if candidates:
            return max(candidates, key=lambda e: e.confidence)
        
        return None
    
    def get_all_elements(self, min_confidence=0.0):
        """
        Hämta alla tidigare detekterade element
        
        Args:
            min_confidence: Lägsta konfidens att inkludera
        
        Returns:
            List[UIElement]: Alla element
        """
        return [e for e in self.detected_elements.values() if e.confidence >= min_confidence]
    
    def clear_history(self):
        """Rensa elementhistorik"""
        self.detected_elements.clear()
        logging.info("🧹 Elementhistorik rensad")
    
    def save_elements_to_file(self, filename=None):
        """
        Spara detekterade element till fil
        
        Args:
            filename: Filnamn att spara till (eller None för auto-genererat)
        
        Returns:
            str: Filnamn
        """
        if filename is None:
            filename = f"data/ui_elements/detection_{int(time.time())}.json"
            
        # Säkerställ att mappen finns
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Konvertera element till dict för JSON-serialisering
        elements_dict = {eid: e.to_dict() for eid, e in self.detected_elements.items()}
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "device": DEVICE,
                "model": MODEL_NAME,
                "elements": elements_dict,
                "metadata": {
                    "count": len(self.detected_elements),
                    "types": {etype: len([e for e in self.detected_elements.values() if e.element_type == etype])
                             for etype in set(e.element_type for e in self.detected_elements.values())}
                }
            }, f, indent=2)
        
        logging.info(f"💾 Detekterade element sparade till {filename}")
        return filename
    
    def load_elements_from_file(self, filename):
        """
        Ladda detekterade element från fil
        
        Args:
            filename: Filnamn att ladda från
        
        Returns:
            bool: True om laddning lyckades
        """
        if not os.path.exists(filename):
            logging.warning(f"⚠️ Filen {filename} hittades inte")
            return False
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Rensa befintliga element
            self.detected_elements.clear()
            
            # Ladda elementen
            if "elements" in data:
                for eid, e_dict in data["elements"].items():
                    element = UIElement(
                        e_dict["type"], 
                        e_dict["confidence"], 
                        e_dict["bbox"]
                    )
                    element.id = e_dict["id"]
                    element.text = e_dict.get("text")
                    element.attributes = e_dict.get("attributes", {})
                    
                    self.detected_elements[eid] = element
                
                logging.info(f"📂 Laddade {len(self.detected_elements)} element från {filename}")
                return True
            else:
                logging.warning(f"⚠️ Inga element hittades i {filename}")
                return False
        
        except Exception as e:
            logging.error(f"Fel vid laddning av element från {filename}: {e}")
            return False
    
    def visualize_elements(self, image, elements=None, show_labels=True, output_file=None):
        """
        Visualisera detekterade element på en bild
        
        Args:
            image: PIL.Image, numpy-array eller filsökväg
            elements: Lista med element att visualisera (eller None för alla)
            show_labels: Om etiketter ska visas
            output_file: Filnamn att spara resultatet till (eller None)
        
        Returns:
            PIL.Image: Bild med visualiserade element
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
        
        # Om inga element anges, använd alla
        if elements is None:
            elements = list(self.detected_elements.values())
        
        # Skapa färgmappning för elementtyper
        element_types = set(e.element_type for e in elements)
        colors = {}
        
        for i, etype in enumerate(element_types):
            # Använd HSV för att skapa jämnt fördelade färger
            hue = i * 180 / len(element_types) if len(element_types) > 0 else 0
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
            colors[etype] = (int(color[0]), int(color[1]), int(color[2]))
        
        # Rita element
        for element in elements:
            x1, y1, x2, y2 = element.bbox
            color = colors.get(element.element_type, (0, 255, 0))
            
            # Rita rektangel
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
            
            # Rita etikett om önskat
            if show_labels:
                label = f"{element.element_type} {element.confidence:.2f}"
                
                # Rita bakgrund för text
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(image_cv, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
                
                # Rita text
                cv2.putText(image_cv, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
            
            logging.info(f"💾 Visualisering sparad till {output_file}")
        
        return result_img
    
    def get_interactive_elements(self, min_confidence=0.5):
        """
        Hämta element som troligen är interaktiva (knappar, textboxar, etc.)
        
        Args:
            min_confidence: Lägsta konfidens att acceptera
        
        Returns:
            List[UIElement]: Interaktiva element
        """
        interactive_types = ["button", "textbox", "checkbox", "radio_button", 
                           "dropdown", "link", "slider", "menu"]
        
        return [e for e in self.detected_elements.values() 
               if e.element_type in interactive_types and e.confidence >= min_confidence]
    
    def find_clickable_elements(self, min_confidence=0.5):
        """
        Hitta element som kan klickas på
        
        Args:
            min_confidence: Lägsta konfidens att acceptera
        
        Returns:
            List[UIElement]: Klickbara element
        """
        clickable_types = ["button", "checkbox", "radio_button", "link", "icon", "tab", "menu"]
        
        return [e for e in self.detected_elements.values() 
               if e.element_type in clickable_types and e.confidence >= min_confidence]
    
    def find_text_input_elements(self, min_confidence=0.5):
        """
        Hitta element där text kan matas in
        
        Args:
            min_confidence: Lägsta konfidens att acceptera
        
        Returns:
            List[UIElement]: Textinmatningselement
        """
        text_input_types = ["textbox", "dropdown"]
        
        return [e for e in self.detected_elements.values() 
               if e.element_type in text_input_types and e.confidence >= min_confidence]

# Convenience-funktion för att få en global instans
_ui_vision = None

def get_ui_vision(model_name=None, use_gpu=True):
    """Hämta en global UIVision-instans"""
    global _ui_vision
    if _ui_vision is None:
        _ui_vision = UIVision(model_name, use_gpu)
    return _ui_vision

def shutdown_ui_vision():
    """Stäng ner UIVision-instansen"""
    global _ui_vision
    if _ui_vision is not None:
        _ui_vision.stop_workers()
        _ui_vision = None

def detect_ui_elements(image, min_confidence=0.25):
    """Convenience-funktion för att detektera UI-element"""
    return get_ui_vision().detect_elements(image)

def visualize_ui_elements(image, elements=None, show_labels=True, output_file=None):
    """Convenience-funktion för att visualisera UI-element"""
    return get_ui_vision().visualize_elements(image, elements, show_labels, output_file)

# Om vi kör direkt, utför ett enkelt test
if __name__ == "__main__":
    # Kolla om argumentet är en skärmdumpssökväg
    import sys
    
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        input_image = sys.argv[1]
    else:
        # Ta en skärmdump
        from PIL import ImageGrab
        input_image = ImageGrab.grab()
    
    # Initialisera UI Vision
    vision = UIVision(use_gpu=True)
    
    # Detektera UI-element
    elements = vision.detect_elements(input_image)
    
    print(f"Detekterade {len(elements)} UI-element:")
    for element in elements:
        print(f"  {element}")
    
    # Visa resultat
    visualized = vision.visualize_elements(input_image, elements)
    output_path = "ui_detection_result.png"
    
    if isinstance(visualized, Image.Image):
        visualized.save(output_path)
    else:
        cv2.imwrite(output_path, visualized)
    
    print(f"Resultat sparat till {output_path}")