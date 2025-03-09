"""
GPU Acceleration för AI Desktop Controller

Detta modul tillhandahåller GPU-accelererad bearbetning för
bildanalys och maskinlärning för att öka prestandan.
"""

import os
import time
import logging
import numpy as np
from PIL import Image
import threading
from queue import Queue

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("gpu_accelerator.log"),
        logging.StreamHandler()
    ]
)

# Försök att importera GPU-bibliotek
# Fallback på CPU-versioner om inte tillgängliga
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        logging.info(f"🎮 GPU acceleration aktiverad: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("⚠️ GPU acceleration inte tillgänglig via PyTorch. Använder CPU.")
except ImportError:
    USE_CUDA = False
    logging.warning("⚠️ PyTorch är inte installerat. GPU-acceleration kommer inte att vara tillgänglig.")

# Försök att importera OpenCV med CUDA-stöd
try:
    import cv2
    # Kontrollera om OpenCV har CUDA-stöd
    cv2_cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if cv2_cuda_available:
        logging.info(f"🎮 OpenCV CUDA acceleration aktiverad: {cv2.cuda.getCudaEnabledDeviceCount()} enheter")
    else:
        logging.info("⚠️ OpenCV CUDA acceleration inte tillgänglig. Använder CPU.")
except (ImportError, AttributeError):
    cv2_cuda_available = False
    logging.warning("⚠️ OpenCV med CUDA-stöd är inte tillgängligt.")

class GPUAccelerator:
    """Hantera GPU-accelererade operationer"""
    
    def __init__(self, use_gpu=True, batch_size=4, num_workers=2):
        self.use_gpu = use_gpu and (USE_CUDA or cv2_cuda_available)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Processköer för parallell bearbetning
        self.image_queue = Queue(maxsize=batch_size * 2)
        self.result_queue = Queue()
        
        # Worker-trådar
        self.workers = []
        self.running = False
        
        if self.use_gpu:
            # För PyTorch
            if USE_CUDA:
                self.device = torch.device('cuda')
                # Visa tillgängligt GPU-minne
                memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
                memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
                logging.info(f"GPU minne: {memory_allocated:.1f}MB allokerat, {memory_reserved:.1f}MB reserverat")
            
            # För OpenCV CUDA
            if cv2_cuda_available:
                # Initiera CUDA-strömmar för OpenCV
                self.cv2_streams = [cv2.cuda_Stream() for _ in range(self.num_workers)]
        
        logging.info(f"✅ GPUAccelerator initialized (use_gpu={self.use_gpu}, batch_size={batch_size}, num_workers={num_workers})")
    
    def start(self):
        """Starta worker-trådar för parallell bearbetning"""
        if self.running:
            return
        
        self.running = True
        
        # Starta worker-trådar
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        logging.info(f"🚀 {len(self.workers)} GPU worker-trådar startade")
    
    def stop(self):
        """Stoppa worker-trådar"""
        self.running = False
        
        # Vänta på att alla worker-trådar avslutas
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1.0)
        
        self.workers = []
        logging.info("⏹️ GPU worker-trådar stoppade")
    
    def _worker_loop(self, worker_id):
        """Huvudloop för worker-trådar"""
        logging.info(f"Worker {worker_id} startad")
        
        while self.running:
            try:
                # Hämta nästa uppgift från kön
                task_type, data, callback, args, kwargs = self.image_queue.get(timeout=0.1)
                
                # Utför uppgiften
                if task_type == 'process_image':
                    result = self._process_image_gpu(data, *args, **kwargs)
                elif task_type == 'detect_objects':
                    result = self._detect_objects_gpu(data, *args, **kwargs)
                elif task_type == 'ocr':
                    result = self._ocr_gpu(data, *args, **kwargs)
                else:
                    result = None
                
                # Lägg till resultatet i resultatskön eller anropa callback
                if callback:
                    callback(result)
                else:
                    self.result_queue.put((task_type, result))
                
                # Markera uppgiften som klar
                self.image_queue.task_done()
                
            except Exception as e:
                logging.error(f"Fel i worker {worker_id}: {e}")
        
        logging.info(f"Worker {worker_id} avslutad")
    
    def process_image(self, image, operations=None, callback=None):
        """
        Asynkront bildbearbetning
        
        Args:
            image: PIL.Image, numpy array eller sökväg till bildfil
            operations: Lista med bearbetningsoperationer att utföra
            callback: Funktion att anropa med resultatet
        """
        if isinstance(image, str) and os.path.exists(image):
            # Ladda bilden om det är en sökväg
            image = Image.open(image)
        
        # Standardoperationer om inga anges
        if operations is None:
            operations = ['normalize', 'enhance']
        
        # Lägg till i kön för bearbetning
        self.image_queue.put(('process_image', image, callback, (operations,), {}))
        
        if callback is None:
            # Om ingen callback anges, vänta på resultatet och returnera
            while True:
                if not self.result_queue.empty():
                    task_type, result = self.result_queue.get()
                    if task_type == 'process_image':
                        return result
                time.sleep(0.01)
    
    def detect_objects(self, image, conf_threshold=0.5, nms_threshold=0.4, callback=None):
        """
        Asynkron objektdetektering i en bild
        
        Args:
            image: PIL.Image, numpy array eller sökväg till bildfil
            conf_threshold: Konfidenströskel
            nms_threshold: Non-maximum suppression threshold
            callback: Funktion att anropa med resultatet
        """
        if isinstance(image, str) and os.path.exists(image):
            # Ladda bilden om det är en sökväg
            image = Image.open(image)
        
        # Lägg till i kön för bearbetning
        self.image_queue.put(('detect_objects', image, callback, (), 
                             {'conf_threshold': conf_threshold, 'nms_threshold': nms_threshold}))
        
        if callback is None:
            # Om ingen callback anges, vänta på resultatet och returnera
            while True:
                if not self.result_queue.empty():
                    task_type, result = self.result_queue.get()
                    if task_type == 'detect_objects':
                        return result
                time.sleep(0.01)
    
    def ocr(self, image, lang='swe+eng', callback=None):
        """
        Asynkron OCR på en bild
        
        Args:
            image: PIL.Image, numpy array eller sökväg till bildfil
            lang: Språk för OCR
            callback: Funktion att anropa med resultatet
        """
        if isinstance(image, str) and os.path.exists(image):
            # Ladda bilden om det är en sökväg
            image = Image.open(image)
        
        # Lägg till i kön för bearbetning
        self.image_queue.put(('ocr', image, callback, (), {'lang': lang}))
        
        if callback is None:
            # Om ingen callback anges, vänta på resultatet och returnera
            while True:
                if not self.result_queue.empty():
                    task_type, result = self.result_queue.get()
                    if task_type == 'ocr':
                        return result
                time.sleep(0.01)
    
    def parallel_process_images(self, images, operation='process_image', **kwargs):
        """
        Bearbeta flera bilder parallellt
        
        Args:
            images: Lista med bilder
            operation: Typ av operation att utföra
            **kwargs: Extra argument till operationen
        
        Returns:
            Lista med resultat
        """
        results = [None] * len(images)
        completed = [False] * len(images)
        
        def callback(idx, result):
            results[idx] = result
            completed[idx] = True
        
        # Lägg till alla bilder i kön
        for i, image in enumerate(images):
            if operation == 'process_image':
                self.process_image(image, callback=lambda r, idx=i: callback(idx, r), **kwargs)
            elif operation == 'detect_objects':
                self.detect_objects(image, callback=lambda r, idx=i: callback(idx, r), **kwargs)
            elif operation == 'ocr':
                self.ocr(image, callback=lambda r, idx=i: callback(idx, r), **kwargs)
        
        # Vänta på att alla bilder bearbetas
        while not all(completed):
            time.sleep(0.01)
        
        return results
    
    def _process_image_gpu(self, image, operations):
        """
        Bearbeta en bild med GPU-acceleration
        
        Args:
            image: PIL.Image eller numpy array
            operations: Lista med bearbetningsoperationer att utföra
            
        Returns:
            Bearbetad bild
        """
        # Konvertera till numpy array om det är en PIL-bild
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        # Konvertera till BGR för OpenCV
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_cv = img_np
        
        # Använd GPU-acceleration om tillgänglig
        if self.use_gpu and cv2_cuda_available:
            # Ladda upp bilden till GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img_cv)
            
            # Utför operationer på GPU
            for op in operations:
                if op == 'normalize':
                    # Normalisera bilden
                    gpu_img = cv2.cuda.normalize(gpu_img, None, 0, 255, cv2.NORM_MINMAX)
                elif op == 'enhance':
                    # Förbättra kontrast och skärpa
                    gpu_img = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gpu_img)
                elif op == 'blur':
                    # Gaussisk blur
                    gpu_img = cv2.cuda.createGaussianFilter(gpu_img.type(), gpu_img.type(), (5, 5), 1.0).apply(gpu_img)
                elif op == 'edge':
                    # Kantdetektering
                    gpu_img = cv2.cuda.createCannyEdgeDetector(50, 150).detect(gpu_img)
                
            # Ladda ner resultatet från GPU
            result = gpu_img.download()
        else:
            # Fallback på CPU-operationer
            result = img_cv
            
            for op in operations:
                if op == 'normalize':
                    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
                elif op == 'enhance':
                    if len(result.shape) == 3:
                        # Konvertera till gråskala för CLAHE
                        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        gray = clahe.apply(gray)
                        # Konvertera tillbaka till färg
                        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    else:
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        result = clahe.apply(result)
                elif op == 'blur':
                    result = cv2.GaussianBlur(result, (5, 5), 1.0)
                elif op == 'edge':
                    if len(result.shape) == 3:
                        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                        result = cv2.Canny(gray, 50, 150)
                    else:
                        result = cv2.Canny(result, 50, 150)
        
        # Konvertera tillbaka till PIL Image om det var det ursprungliga formatet
        if isinstance(image, Image.Image):
            if len(result.shape) == 3:
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            return Image.fromarray(result)
        
        return result
    
    def _detect_objects_gpu(self, image, conf_threshold=0.5, nms_threshold=0.4):
        """
        Detektera objekt i en bild med GPU-acceleration
        
        Denna implementation använder en förenklad modell. I en faktisk implementation
        skulle detta använda en förtränad YOLO/EfficientDet/SSD modell med CUDA-stöd.
        
        Args:
            image: PIL.Image eller numpy array
            conf_threshold: Konfidenströskel (0.0-1.0)
            nms_threshold: Non-maximum suppression threshold (0.0-1.0)
            
        Returns:
            Lista med detekterade objekt (klass, konfidenspoäng, bounding box)
        """
        # Konvertera till numpy array om det är en PIL-bild
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        # Konvertera till BGR för OpenCV
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_cv = img_np
        
        # Enkel hårdkodat demo-exempel för UI-element (i en faktisk implementation 
        # skulle detta använda ett riktigt objekt-detection nätverk)
        dummy_detection_results = []
        
        # Försök detektera knappar, textfält och fönsterkomponenter
        if self.use_gpu and USE_CUDA:
            # Här skulle en GPU-accelererad modell användas
            # För demo, skapa några dummy-detektioner
            
            # Gör en gråskalebild
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
            
            # Hitta kanter
            edges = cv2.Canny(gray, 50, 150)
            
            # Hitta konturer
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Hitta rektangulära områden som kan vara UI-element
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Ignorera mycket små områden
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Klassificera element baserat på form och area
                    if 2.5 < aspect_ratio < 8 and area < 10000:
                        # Troligtvis en knapp
                        dummy_detection_results.append({
                            'class': 'button',
                            'confidence': np.random.uniform(0.7, 0.9),
                            'bbox': (x, y, x+w, y+h)
                        })
                    elif 1.5 < aspect_ratio < 10 and area < 20000:
                        # Troligtvis ett textfält
                        dummy_detection_results.append({
                            'class': 'textfield',
                            'confidence': np.random.uniform(0.6, 0.8),
                            'bbox': (x, y, x+w, y+h)
                        })
                    elif aspect_ratio < 1.5 and area < 5000:
                        # Troligtvis en checkbox eller radiobutton
                        dummy_detection_results.append({
                            'class': 'checkbox',
                            'confidence': np.random.uniform(0.5, 0.7),
                            'bbox': (x, y, x+w, y+h)
                        })
            
            # Filtrera baserat på konfidenspoäng
            dummy_detection_results = [d for d in dummy_detection_results if d['confidence'] > conf_threshold]
        else:
            # CPU-baserad implementation, liknande men långsammare
            # Gör en gråskalebild
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
            
            # Hitta kanter
            edges = cv2.Canny(gray, 50, 150)
            
            # Hitta konturer
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Hitta rektangulära områden som kan vara UI-element
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Ignorera mycket små områden
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Klassificera element baserat på form och area
                    if 2.5 < aspect_ratio < 8 and area < 10000:
                        # Troligtvis en knapp
                        dummy_detection_results.append({
                            'class': 'button',
                            'confidence': np.random.uniform(0.7, 0.9),
                            'bbox': (x, y, x+w, y+h)
                        })
                    elif 1.5 < aspect_ratio < 10 and area < 20000:
                        # Troligtvis ett textfält
                        dummy_detection_results.append({
                            'class': 'textfield',
                            'confidence': np.random.uniform(0.6, 0.8),
                            'bbox': (x, y, x+w, y+h)
                        })
                    elif aspect_ratio < 1.5 and area < 5000:
                        # Troligtvis en checkbox eller radiobutton
                        dummy_detection_results.append({
                            'class': 'checkbox',
                            'confidence': np.random.uniform(0.5, 0.7),
                            'bbox': (x, y, x+w, y+h)
                        })
            
            # Filtrera baserat på konfidenspoäng
            dummy_detection_results = [d for d in dummy_detection_results if d['confidence'] > conf_threshold]
        
        return dummy_detection_results
    
    def _ocr_gpu(self, image, lang='swe+eng'):
        """
        Utför OCR på en bild med GPU-acceleration (om möjligt)
        
        I en faktisk implementation skulle detta använda GPU-accelererad tesseract
        eller en annan GPU-optimerad OCR-lösning.
        
        Args:
            image: PIL.Image eller numpy array
            lang: Språk för OCR
            
        Returns:
            Text från bilden
        """
        # Konvertera till numpy array om det är en PIL-bild
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        # För demo, använd standard tesseract via import
        try:
            import pytesseract
            
            # Förbearbeta bilden för bättre OCR-resultat
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
            
            # Förbättra kontrast för bättre OCR
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Använd tesseract för OCR
            try:
                text = pytesseract.image_to_string(enhanced, lang=lang)
                return text
            except Exception as e:
                logging.error(f"OCR error: {e}")
                return ""
        except ImportError:
            logging.error("pytesseract är inte installerat. Kan inte utföra OCR.")
            return "OCR är inte tillgängligt."

# Convenience-funktioner för global användning

_accelerator = None

def get_accelerator(use_gpu=True, batch_size=4, num_workers=2):
    """Hämta den globala acceleratorn eller skapa en ny om den inte finns"""
    global _accelerator
    if _accelerator is None:
        _accelerator = GPUAccelerator(use_gpu=use_gpu, batch_size=batch_size, num_workers=num_workers)
        _accelerator.start()
    return _accelerator

def process_image(image, operations=None, callback=None):
    """Bearbeta en bild med GPU-acceleration"""
    return get_accelerator().process_image(image, operations, callback)

def detect_objects(image, conf_threshold=0.5, nms_threshold=0.4, callback=None):
    """Detektera objekt i en bild med GPU-acceleration"""
    return get_accelerator().detect_objects(image, conf_threshold, nms_threshold, callback)

def ocr(image, lang='swe+eng', callback=None):
    """Utför OCR på en bild med GPU-acceleration"""
    return get_accelerator().ocr(image, lang, callback)

def parallel_process_images(images, operation='process_image', **kwargs):
    """Bearbeta flera bilder parallellt med GPU-acceleration"""
    return get_accelerator().parallel_process_images(images, operation, **kwargs)

def shutdown():
    """Stäng ner acceleratorn"""
    global _accelerator
    if _accelerator is not None:
        _accelerator.stop()
        _accelerator = None