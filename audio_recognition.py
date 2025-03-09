"""
Audio Recognition - Ljudigenkänning för AI Desktop Controller

Detta modul implementerar ljudigenkänning för att detektera och reagera
på ljudsignaler från applikationer, varningar och systemhändelser.
"""

import os
import time
import logging
import json
import numpy as np
import threading
import queue
from datetime import datetime
import pyaudio
import wave
import tempfile
from collections import deque
import audioop
import scipy.signal
import scipy.fft
import matplotlib.pyplot as plt

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("audio_recognition.log"),
        logging.StreamHandler()
    ]
)

# Globala konstanter
RATE = 44100  # Samplingsfrekvens i Hz
CHUNK = 1024  # Buffertstorlek för ljudsampling
FORMAT = pyaudio.paInt16  # Ljudformat
CHANNELS = 2  # Antal kanaler (stereo)
DETECTION_THRESHOLD = 0.65  # Tröskel för mönstermatchning
MIN_SOUND_LEVEL = 500  # Minsta ljudnivå för detektering
REFERENCE_PATTERNS_PATH = "data/audio_patterns"

class AudioPattern:
    """Representation av ett ljudmönster för igenkänning"""
    
    def __init__(self, name, data=None, file_path=None, pattern_type="alert"):
        """
        Initiera ett ljudmönster
        
        Args:
            name: Namn på mönstret
            data: Ljuddata (numpy array) eller None om file_path används
            file_path: Sökväg till ljudfil eller None om data används
            pattern_type: Typ av mönster ('alert', 'notification', 'success', 'error', etc.)
        """
        self.name = name
        self.pattern_type = pattern_type
        self.file_path = file_path
        self.data = data
        self.features = None
        self.duration = 0
        self.timestamp = time.time()
        self.fingerprint = None
        
        # Ladda data om filsökväg anges
        if file_path and not data:
            self._load_from_file()
        
        # Beräkna funktioner om data finns
        if self.data is not None:
            self._extract_features()
    
    def _load_from_file(self):
        """Ladda ljuddata från fil"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Ljudfil hittades inte: {self.file_path}")
        
        try:
            with wave.open(self.file_path, 'rb') as wf:
                # Läs fil-egenskaper
                self.channels = wf.getnchannels()
                self.sample_width = wf.getsampwidth()
                self.rate = wf.getframerate()
                self.n_frames = wf.getnframes()
                self.duration = self.n_frames / float(self.rate)
                
                # Läs alla frames
                frames = wf.readframes(self.n_frames)
                
                # Konvertera till numpy array
                self.data = np.frombuffer(frames, dtype=np.int16)
                
                # Om stereo, konvertera till mono
                if self.channels == 2:
                    self.data = np.mean(self.data.reshape(-1, 2), axis=1).astype(np.int16)
                
                logging.info(f"✅ Ljudmönster '{self.name}' laddat från {self.file_path} "
                           f"({self.duration:.2f}s, {self.rate}Hz)")
        except Exception as e:
            logging.error(f"Fel vid laddning av ljudfil {self.file_path}: {e}")
            raise
    
    def _extract_features(self):
        """Extrahera egenskaper från ljuddatan för igenkänning"""
        # Normalisera data
        normalized = self.data.astype(float) / np.max(np.abs(self.data))
        
        # Beräkna spektrogram
        f, t, spectro = scipy.signal.spectrogram(
            normalized, 
            fs=RATE, 
            nperseg=CHUNK,
            noverlap=CHUNK // 2,
            scaling='spectrum'
        )
        
        # Spara funktioner
        self.features = {
            "spectro": spectro,
            "frequency": f,
            "time": t,
            "energy": np.sum(np.abs(normalized) ** 2),
            "rms": np.sqrt(np.mean(normalized ** 2)),
            "zero_crossings": len(np.where(np.diff(np.signbit(normalized)))[0]),
            "peak_frequency": f[np.argmax(np.mean(spectro, axis=1))]
        }
        
        # Beräkna fingerprint
        self.fingerprint = self._create_fingerprint(spectro)
    
    def _create_fingerprint(self, spectro):
        """Skapa ett fingeravtryck från spektrogrammet för snabb matchning"""
        # Hitta toppar i spektrogrammet
        peaks = []
        
        # För varje tidsfönster
        for t in range(spectro.shape[1]):
            # Hitta de starkaste frekvenserna
            spectrum = spectro[:, t]
            top_indices = np.argsort(spectrum)[-5:]  # Topp 5 frekvenser
            
            for idx in top_indices:
                if spectrum[idx] > 0.01:  # Ignorera svaga frekvenser
                    peaks.append((idx, t))
        
        return peaks
    
    def match(self, other_pattern, threshold=DETECTION_THRESHOLD):
        """
        Jämför detta mönster med ett annat och returnera matchningspoäng
        
        Args:
            other_pattern: Annat AudioPattern att jämföra med
            threshold: Tröskel för att anses som en match (0-1)
            
        Returns:
            float: Matchningspoäng (0-1)
        """
        if not self.features or not other_pattern.features:
            return 0.0
        
        # Beräkna korrelation mellan spektrogrammen
        corr_score = self._correlation_score(other_pattern)
        
        # Jämför energi och RMS
        energy_match = 1.0 - min(1.0, abs(self.features["energy"] - other_pattern.features["energy"]) / 
                               max(self.features["energy"], other_pattern.features["energy"]))
        
        rms_match = 1.0 - min(1.0, abs(self.features["rms"] - other_pattern.features["rms"]) / 
                             max(self.features["rms"], other_pattern.features["rms"]))
        
        # Kombinera poäng
        match_score = 0.6 * corr_score + 0.2 * energy_match + 0.2 * rms_match
        
        return match_score
    
    def _correlation_score(self, other_pattern):
        """Beräkna korrelationspoäng mellan två mönster"""
        # Om spektrogrammen har olika form, ändra storlek på det kortare
        spectro1 = self.features["spectro"]
        spectro2 = other_pattern.features["spectro"]
        
        # Korta ner det längre spektrogrammet
        if spectro1.shape[1] > spectro2.shape[1]:
            spectro1 = spectro1[:, :spectro2.shape[1]]
        elif spectro2.shape[1] > spectro1.shape[1]:
            spectro2 = spectro2[:, :spectro1.shape[1]]
        
        # Kontrollera också frekvensupplösningen
        if spectro1.shape[0] > spectro2.shape[0]:
            spectro1 = spectro1[:spectro2.shape[0], :]
        elif spectro2.shape[0] > spectro1.shape[0]:
            spectro2 = spectro2[:spectro1.shape[0], :]
        
        # Normalisera spektrogrammen
        spectro1_norm = spectro1 / np.max(spectro1)
        spectro2_norm = spectro2 / np.max(spectro2)
        
        # Beräkna korrelation
        corr = np.corrcoef(spectro1_norm.flatten(), spectro2_norm.flatten())[0, 1]
        
        # Hantera NaN-värden (kan inträffa om ett spektrogram är konstant)
        if np.isnan(corr):
            return 0.0
        
        # Konvertera Pearsons korrelation (-1 till 1) till (0 till 1)
        return (corr + 1) / 2
    
    def visualize(self, output_file=None):
        """
        Visualisera ljudmönstret
        
        Args:
            output_file: Filnamn att spara visualisering till (eller None)
            
        Returns:
            matplotlib.figure eller None
        """
        if not self.features:
            logging.warning(f"⚠️ Inga funktioner att visualisera för {self.name}")
            return None
        
        # Skapa figur med tre panneler
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
        
        # Rita vågform
        time_axis = np.arange(len(self.data)) / RATE
        ax1.plot(time_axis, self.data)
        ax1.set_title(f"Vågform: {self.name}")
        ax1.set_xlabel("Tid (s)")
        ax1.set_ylabel("Amplitud")
        
        # Rita spektrogram
        spectro = self.features["spectro"]
        f = self.features["frequency"]
        t = self.features["time"]
        
        ax2.pcolormesh(t, f, spectro, shading='gouraud')
        ax2.set_title("Spektrogram")
        ax2.set_ylabel("Frekvens (Hz)")
        ax2.set_xlabel("Tid (s)")
        
        # Rita fingeravtryck
        if self.fingerprint:
            fingerprint_map = np.zeros_like(spectro)
            for freq_idx, time_idx in self.fingerprint:
                if freq_idx < fingerprint_map.shape[0] and time_idx < fingerprint_map.shape[1]:
                    fingerprint_map[freq_idx, time_idx] = 1
            
            ax3.pcolormesh(t, f, fingerprint_map, shading='gouraud', cmap='hot')
            ax3.set_title("Fingeravtryck")
            ax3.set_ylabel("Frekvens (Hz)")
            ax3.set_xlabel("Tid (s)")
        
        plt.tight_layout()
        
        # Spara till fil om angivet
        if output_file:
            plt.savefig(output_file)
            logging.info(f"💾 Visualisering sparad till {output_file}")
        
        return fig
    
    def save(self, filename=None):
        """
        Spara mönstret till fil
        
        Args:
            filename: Filnamn att spara till (eller None för automatgenererat)
            
        Returns:
            str: Filnamn
        """
        if filename is None:
            # Skapa katalog om den inte finns
            os.makedirs(REFERENCE_PATTERNS_PATH, exist_ok=True)
            
            # Skapa ett läsbart filnamn
            safe_name = ''.join(c if c.isalnum() else '_' for c in self.name)
            filename = os.path.join(REFERENCE_PATTERNS_PATH, f"{safe_name}_{int(time.time())}.json")
        
        # Konvertera data till beständig form
        if self.data is not None:
            audio_data = self.data.tolist()
        else:
            audio_data = None
        
        # Förbered spectogram för serialisering
        if self.features and "spectro" in self.features:
            spectro = self.features["spectro"].tolist() 
        else:
            spectro = None
        
        # Skapa dictionary för att spara
        pattern_dict = {
            "name": self.name,
            "pattern_type": self.pattern_type,
            "file_path": self.file_path,
            "timestamp": self.timestamp,
            "data": audio_data,
            "duration": self.duration,
            "fingerprint": self.fingerprint,
            "features": {
                "spectro": spectro,
                "frequency": self.features["frequency"].tolist() if self.features and "frequency" in self.features else None,
                "time": self.features["time"].tolist() if self.features and "time" in self.features else None,
                "energy": self.features["energy"] if self.features and "energy" in self.features else None,
                "rms": self.features["rms"] if self.features and "rms" in self.features else None,
                "zero_crossings": self.features["zero_crossings"] if self.features and "zero_crossings" in self.features else None,
                "peak_frequency": self.features["peak_frequency"] if self.features and "peak_frequency" in self.features else None
            }
        }
        
        # Spara till fil
        with open(filename, 'w') as f:
            json.dump(pattern_dict, f)
        
        logging.info(f"💾 Ljudmönster '{self.name}' sparat till {filename}")
        return filename
    
    @classmethod
    def load(cls, filename):
        """
        Ladda ett mönster från fil
        
        Args:
            filename: Filnamn att ladda från
            
        Returns:
            AudioPattern: Laddat mönster
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Mönsterfil hittades inte: {filename}")
        
        try:
            with open(filename, 'r') as f:
                pattern_dict = json.load(f)
            
            # Skapa ett nytt mönster
            pattern = cls(
                name=pattern_dict["name"],
                pattern_type=pattern_dict.get("pattern_type", "alert"),
                file_path=pattern_dict.get("file_path")
            )
            
            # Återställ data
            if pattern_dict.get("data") is not None:
                pattern.data = np.array(pattern_dict["data"], dtype=np.int16)
            
            # Återställ övriga attribut
            pattern.timestamp = pattern_dict.get("timestamp", time.time())
            pattern.duration = pattern_dict.get("duration", 0)
            pattern.fingerprint = pattern_dict.get("fingerprint")
            
            # Återställ funktioner
            if "features" in pattern_dict:
                features = pattern_dict["features"]
                pattern.features = {}
                
                if features.get("spectro") is not None:
                    pattern.features["spectro"] = np.array(features["spectro"])
                
                if features.get("frequency") is not None:
                    pattern.features["frequency"] = np.array(features["frequency"])
                
                if features.get("time") is not None:
                    pattern.features["time"] = np.array(features["time"])
                
                pattern.features["energy"] = features.get("energy")
                pattern.features["rms"] = features.get("rms")
                pattern.features["zero_crossings"] = features.get("zero_crossings")
                pattern.features["peak_frequency"] = features.get("peak_frequency")
            
            logging.info(f"📂 Ljudmönster '{pattern.name}' laddat från {filename}")
            return pattern
            
        except Exception as e:
            logging.error(f"Fel vid laddning av mönster från {filename}: {e}")
            raise

class AudioDetectionResult:
    """Resultat från ljudigenkänning"""
    
    def __init__(self, pattern_name=None, match_score=0.0, timestamp=None, duration=0.0):
        """
        Initiera ett detekteringsresultat
        
        Args:
            pattern_name: Namn på matchat mönster eller None
            match_score: Matchningspoäng (0-1)
            timestamp: Tidpunkt för detektering
            duration: Varaktighet i sekunder
        """
        self.pattern_name = pattern_name
        self.match_score = match_score
        self.timestamp = timestamp or time.time()
        self.duration = duration
        self.audio_data = None
        self.pattern_type = None
    
    def __str__(self):
        pattern_str = self.pattern_name if self.pattern_name else "Okänt"
        return (f"AudioDetectionResult: {pattern_str} "
               f"(Poäng: {self.match_score:.2f}, Tid: {self.duration:.2f}s)")
    
    def to_dict(self):
        """Konvertera till dictionary för JSON-serialisering"""
        return {
            "pattern_name": self.pattern_name,
            "match_score": self.match_score,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "pattern_type": self.pattern_type
        }

class AudioListener:
    """Lyssnar på ljud från datorn för att detektera mönster"""
    
    def __init__(self, chunk_size=CHUNK, rate=RATE, threshold=MIN_SOUND_LEVEL, 
                listen_device_index=None, reference_patterns=None):
        """
        Initiera ljudlyssnare
        
        Args:
            chunk_size: Buffertstorlek för ljudsampling
            rate: Samplingsfrekvens
            threshold: Ljudnivåtröskel för att börja spela in
            listen_device_index: Index för ljudingångsenhet (None för standard)
            reference_patterns: Lista med referensmönster
        """
        self.chunk_size = chunk_size
        self.rate = rate
        self.threshold = threshold
        self.device_index = listen_device_index
        
        # Audio streaming
        self.stream = None
        self.audio = None
        self.frames = []
        self.listening = False
        self.audio_thread = None
        
        # Referensmönster
        self.reference_patterns = reference_patterns or []
        
        # Circular buffer för kontinuerlig analys
        self.buffer = deque(maxlen=int(rate * 3 / chunk_size))  # 3 sekunder buffer
        
        # Callbacks
        self.detection_callbacks = []
        
        # Lyssningslås
        self.is_processing = False
        self.processing_lock = threading.Lock()
        
        # Upptäcktshistorik
        self.detection_history = []
        self.max_history = 100
        
        # Listor för detekteringen
        self.common_alert_sounds = self._load_common_alert_sounds()
        
        # Status
        self.status = "initialized"
        
        logging.info(f"✅ AudioListener initierad (rate={rate}Hz, chunk={chunk_size})")
    
    def _load_common_alert_sounds(self):
        """Ladda vanliga varningsljud som referens"""
        common_patterns = []
        
        # Titta i referenskatalogen
        if os.path.exists(REFERENCE_PATTERNS_PATH):
            for filename in os.listdir(REFERENCE_PATTERNS_PATH):
                if filename.endswith(".json"):
                    filepath = os.path.join(REFERENCE_PATTERNS_PATH, filename)
                    try:
                        pattern = AudioPattern.load(filepath)
                        common_patterns.append(pattern)
                        logging.info(f"✅ Laddat referensmönster: {pattern.name}")
                    except Exception as e:
                        logging.error(f"Fel vid laddning av mönster {filepath}: {e}")
        
        return common_patterns
    
    def add_reference_pattern(self, pattern):
        """
        Lägg till ett referensmönster
        
        Args:
            pattern: AudioPattern att lägga till
        """
        self.reference_patterns.append(pattern)
        logging.info(f"➕ Lade till referensmönster: {pattern.name}")
    
    def load_patterns_from_directory(self, directory=REFERENCE_PATTERNS_PATH):
        """
        Ladda alla mönster från en katalog
        
        Args:
            directory: Katalog att ladda från
            
        Returns:
            int: Antal laddade mönster
        """
        if not os.path.exists(directory):
            logging.warning(f"⚠️ Katalogen {directory} existerar inte")
            return 0
        
        count = 0
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                filepath = os.path.join(directory, filename)
                try:
                    pattern = AudioPattern.load(filepath)
                    self.add_reference_pattern(pattern)
                    count += 1
                except Exception as e:
                    logging.error(f"Fel vid laddning av mönster {filepath}: {e}")
        
        logging.info(f"📂 Laddade {count} mönster från {directory}")
        return count
    
    def register_detection_callback(self, callback):
        """
        Registrera en callback för ljuddetektering
        
        Args:
            callback: Funktion som anropas vid detektering
        """
        self.detection_callbacks.append(callback)
    
    def start_listening(self):
        """Starta lyssning på ljudströmmen"""
        if self.listening:
            logging.warning("⚠️ Redan lyssnar")
            return
        
        try:
            # Initiera PyAudio
            self.audio = pyaudio.PyAudio()
            
            # Få information om ljudenheter
            self._log_audio_devices()
            
            # Öppna ljudström
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self.rate,
                input=True,
                output=False,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.device_index,
                stream_callback=self._audio_callback
            )
            
            # Starta lyssning
            self.listening = True
            self.stream.start_stream()
            
            # Starta bearbetningstråd
            self.audio_thread = threading.Thread(target=self._processing_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            self.status = "listening"
            logging.info("🎧 Ljudlyssning startad")
            
        except Exception as e:
            logging.error(f"Fel vid start av ljudlyssning: {e}")
            self.stop_listening()
    
    def stop_listening(self):
        """Stoppa lyssning på ljudströmmen"""
        self.listening = False
        
        # Stoppa och stäng ljudströmmen
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        # Stäng PyAudio
        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass
            self.audio = None
        
        self.status = "stopped"
        logging.info("⏹️ Ljudlyssning stoppad")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback för PyAudio-strömmen"""
        if not self.listening:
            return None, pyaudio.paAbort
        
        # Lägg till data i buffern
        self.buffer.append(in_data)
        
        # Kontrollera ljudnivån
        try:
            # Konvertera bytes till numpy array
            data_np = np.frombuffer(in_data, dtype=np.int16)
            
            # Beräkna ljudnivå
            rms = audioop.rms(in_data, 2)  # 2 för 16-bit ljud
            
            # Om ljudnivån är över tröskeln, börja spela in
            if rms > self.threshold and not self.is_processing:
                with self.processing_lock:
                    self.is_processing = True
                    # Skapa en kopia av buffern för bearbetning
                    full_buffer = b''.join(self.buffer)
                    threading.Thread(target=self._process_audio, args=(full_buffer,)).start()
        except Exception as e:
            logging.error(f"Fel i ljudcallback: {e}")
        
        return None, pyaudio.paContinue
    
    def _processing_loop(self):
        """Huvudloop för ljudbearbetning"""
        logging.info("🔄 Ljudbearbetningsloop startad")
        
        while self.listening:
            # Sleep för att inte belasta CPU
            time.sleep(0.1)
    
    def _process_audio(self, audio_bytes):
        """
        Bearbeta ljudbuffer för detektering
        
        Args:
            audio_bytes: Ljuddata som bytes
        """
        try:
            # Konvertera bytes till numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Om stereo, konvertera till mono
            if CHANNELS == 2:
                audio_data = np.mean(audio_data.reshape(-1, 2), axis=1).astype(np.int16)
            
            # Skapa ett temporärt AudioPattern för matchning
            temp_pattern = AudioPattern("temp", data=audio_data)
            
            # Jämför med referensmönstren
            best_match = None
            best_score = 0
            
            for pattern in self.reference_patterns + self.common_alert_sounds:
                match_score = pattern.match(temp_pattern)
                
                if match_score > DETECTION_THRESHOLD and match_score > best_score:
                    best_score = match_score
                    best_match = pattern
            
            # Om vi hittar en match
            if best_match:
                # Skapa resultat
                result = AudioDetectionResult(
                    pattern_name=best_match.name,
                    match_score=best_score,
                    duration=temp_pattern.duration
                )
                result.pattern_type = best_match.pattern_type
                
                # Lägg till i historik
                self.detection_history.append(result)
                if len(self.detection_history) > self.max_history:
                    self.detection_history = self.detection_history[-self.max_history:]
                
                # Logga detektering
                logging.info(f"🔊 Detekterade ljudmönster: {result}")
                
                # Anropa callbacks
                for callback in self.detection_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logging.error(f"Fel i ljuddetektering-callback: {e}")
            
        except Exception as e:
            logging.error(f"Fel vid ljudbearbetning: {e}")
        finally:
            # Återställ bearbetningsflagg
            with self.processing_lock:
                self.is_processing = False
    
    def _log_audio_devices(self):
        """Logga tillgängliga ljudenheter"""
        if not self.audio:
            return
        
        try:
            info = []
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:
                    info.append(f"Index {i}: {device_info.get('name')}")
            
            logging.info(f"🎤 Tillgängliga ljudingångsenheter:\n" + "\n".join(info))
            
            # Logga vald enhet
            if self.device_index is not None:
                device_info = self.audio.get_device_info_by_index(self.device_index)
                logging.info(f"🎙️ Använder ljudingångsenhet: {device_info.get('name')}")
            else:
                default_device = self.audio.get_default_input_device_info()
                logging.info(f"🎙️ Använder standardingångsenhet: {default_device.get('name')}")
                
        except Exception as e:
            logging.error(f"Fel vid loggning av ljudenheter: {e}")
    
    def record_pattern(self, duration=3.0, name=None, pattern_type="alert"):
        """
        Spela in ett nytt ljudmönster
        
        Args:
            duration: Inspelningstid i sekunder
            name: Namn på mönstret (eller None för automatiskt)
            pattern_type: Typ av mönster
            
        Returns:
            AudioPattern: Inspelat mönster
        """
        if self.listening:
            # Pausa lyssning
            was_listening = True
            old_stream = self.stream
            old_audio = self.audio
            self.stream = None
        else:
            was_listening = False
        
        try:
            # Skapa nytt PyAudio-objekt för inspelning
            p = pyaudio.PyAudio()
            
            # Öppna ström för inspelning
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.device_index
            )
            
            # Logga inspelningsstart
            logging.info(f"🔴 Startar inspelning ({duration}s)")
            
            # Spela in
            frames = []
            for i in range(0, int(self.rate / self.chunk_size * duration)):
                data = stream.read(self.chunk_size)
                frames.append(data)
            
            # Stäng ström
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Konvertera till numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            
            # Om stereo, konvertera till mono
            if CHANNELS == 2:
                audio_data = np.mean(audio_data.reshape(-1, 2), axis=1).astype(np.int16)
            
            # Generera namn om inget anges
            if name is None:
                name = f"pattern_{int(time.time())}"
            
            # Skapa mönster
            pattern = AudioPattern(name, data=audio_data, pattern_type=pattern_type)
            
            # Lägg till i referensmönster
            self.add_reference_pattern(pattern)
            
            # Spara mönstret
            pattern.save()
            
            logging.info(f"✅ Inspelat nytt ljudmönster: {name}")
            
            return pattern
            
        except Exception as e:
            logging.error(f"Fel vid inspelning: {e}")
            return None
            
        finally:
            # Återställ lyssning om det var på innan
            if was_listening:
                self.audio = old_audio
                self.stream = old_stream
                if not self.stream.is_active():
                    self.stream.start_stream()
    
    def capture_application_sounds(self, app_name=None, duration=30.0, max_patterns=5):
        """
        Spela in ljud från en applikation för att bygga en referenssamling
        
        Args:
            app_name: Applikationsnamn
            duration: Total inspelningstid
            max_patterns: Maximalt antal mönster att spela in
            
        Returns:
            list: Inspelade mönster
        """
        if app_name is None:
            app_name = "application"
        
        # Informera användaren
        logging.info(f"🔊 Börjar lyssna efter ljud från {app_name} i {duration} sekunder")
        logging.info("Använd applikationen och generera dess varnings- och notifikationsljud")
        
        # Spara aktuellt status
        was_listening = self.listening
        if was_listening:
            self.stop_listening()
        
        # Starta lyssning med låg tröskel
        old_threshold = self.threshold
        self.threshold = MIN_SOUND_LEVEL // 2
        
        # Skapa en separat regulator för inspelningar
        recorded_patterns = []
        last_recording_time = 0
        
        try:
            # Starta lyssning
            self._capture_thread_running = True
            self.start_listening()
            
            # Definiera callback för detektering
            def sound_detected(result):
                nonlocal last_recording_time, recorded_patterns
                
                # Om vi har nått max eller inte fått något resultat, avsluta
                if len(recorded_patterns) >= max_patterns or not result:
                    return
                
                # Om det gått tillräckligt med tid sedan senaste inspelningen
                current_time = time.time()
                if current_time - last_recording_time > 2.0:
                    # Spela in ett nytt mönster
                    pattern_name = f"{app_name}_sound_{len(recorded_patterns) + 1}"
                    pattern = self.record_pattern(3.0, pattern_name, "notification")
                    
                    if pattern:
                        recorded_patterns.append(pattern)
                        last_recording_time = time.time()
            
            # Registrera temporär callback
            self.register_detection_callback(sound_detected)
            
            # Vänta angivet antal sekunder
            start_time = time.time()
            while time.time() - start_time < duration and len(recorded_patterns) < max_patterns:
                time.sleep(0.5)
                
            # Ta bort callback
            self.detection_callbacks.remove(sound_detected)
            
            logging.info(f"✅ Spelade in {len(recorded_patterns)} ljudmönster från {app_name}")
            
            return recorded_patterns
            
        except Exception as e:
            logging.error(f"Fel vid inspelning av applikationsljud: {e}")
            return []
            
        finally:
            # Stoppa lyssning
            self.stop_listening()
            
            # Återställ tröskel
            self.threshold = old_threshold
            
            # Återställ lyssning om det var på innan
            if was_listening:
                self.start_listening()
    
    def save_detection_history(self, filename=None):
        """
        Spara detekteringshistorik till fil
        
        Args:
            filename: Filnamn att spara till (eller None för automatgenererat)
            
        Returns:
            str: Filnamn
        """
        if filename is None:
            os.makedirs("data/audio_detections", exist_ok=True)
            filename = f"data/audio_detections/history_{int(time.time())}.json"
        
        # Konvertera till dictionary för JSON-serialisering
        history_dict = {
            "timestamp": time.time(),
            "detections": [r.to_dict() for r in self.detection_history]
        }
        
        # Spara till fil
        with open(filename, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        logging.info(f"💾 Detekteringshistorik sparad till {filename}")
        return filename
    
    def load_detection_history(self, filename):
        """
        Ladda detekteringshistorik från fil
        
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
                history_dict = json.load(f)
            
            # Konvertera till AudioDetectionResult-objekt
            self.detection_history = []
            
            for detection in history_dict.get("detections", []):
                result = AudioDetectionResult(
                    pattern_name=detection.get("pattern_name"),
                    match_score=detection.get("match_score", 0.0),
                    timestamp=detection.get("timestamp", time.time()),
                    duration=detection.get("duration", 0.0)
                )
                result.pattern_type = detection.get("pattern_type")
                self.detection_history.append(result)
            
            logging.info(f"📂 Laddade {len(self.detection_history)} detekteringar från {filename}")
            return True
            
        except Exception as e:
            logging.error(f"Fel vid laddning av detekteringshistorik: {e}")
            return False

class AudioRecognitionManager:
    """Hantera ljudigenkänning och integrera med AI-systemet"""
    
    def __init__(self, system_controller=None, device_index=None):
        """
        Initiera ljudigenkänningshanterare
        
        Args:
            system_controller: AI-systemkontroller
            device_index: Index för ljudingångsenhet
        """
        self.system_controller = system_controller
        self.audio_listener = AudioListener(listen_device_index=device_index)
        
        # Ladda referensljud
        self.audio_listener.load_patterns_from_directory()
        
        # Registrera callbacks
        self.audio_listener.register_detection_callback(self._audio_detected_callback)
        
        # Mappning av ljudtyper till handlingar
        self.action_mapping = {
            "alert": self._handle_alert,
            "notification": self._handle_notification,
            "error": self._handle_error,
            "success": self._handle_success,
            "warning": self._handle_warning
        }
        
        # Historik för mappade handlingar
        self.action_history = []
        self.max_history = 100
        
        logging.info("✅ AudioRecognitionManager initierad")
    
    def start(self):
        """Starta ljudigenkänning"""
        self.audio_listener.start_listening()
        logging.info("🎧 AudioRecognitionManager startad")
    
    def stop(self):
        """Stoppa ljudigenkänning"""
        self.audio_listener.stop_listening()
        logging.info("⏹️ AudioRecognitionManager stoppad")
    
    def _audio_detected_callback(self, result):
        """
        Callback för detekterade ljud
        
        Args:
            result: AudioDetectionResult
        """
        logging.info(f"🔔 Detekterat ljud: {result}")
        
        # Mappa ljud till handling
        pattern_type = result.pattern_type or "notification"
        
        # Hämta rätt hanterare
        handler = self.action_mapping.get(pattern_type, self._handle_default)
        
        # Utför handling
        action_result = handler(result)
        
        # Spara i historik
        if action_result:
            action_entry = {
                "timestamp": time.time(),
                "audio_result": result.to_dict(),
                "action": action_result
            }
            
            self.action_history.append(action_entry)
            
            # Begränsa historikens storlek
            if len(self.action_history) > self.max_history:
                self.action_history = self.action_history[-self.max_history:]
    
    def _handle_alert(self, result):
        """
        Hantera varningsljud
        
        Args:
            result: AudioDetectionResult
        
        Returns:
            dict: Handlingsresultat
        """
        action = {
            "type": "alert",
            "description": f"Detekterade varningsljud: {result.pattern_name}",
            "priority": "high",
            "actions_taken": []
        }
        
        # Om vi har en systemkontroller, meddela den
        if self.system_controller:
            # Logga händelsen i systemet
            self.system_controller.add_log_entry(f"🔔 Varningsljud detekterat: {result.pattern_name}")
            action["actions_taken"].append("system_log")
            
            # Utför lämpliga åtgärder baserat på ljud
            # T.ex. leta efter dialogrutor att svara på
            #self.system_controller.scan_for_dialogs()
            action["actions_taken"].append("scan_for_dialogs")
        
        logging.info(f"🚨 Hanterar varningsljud: {result.pattern_name}")
        return action
    
    def _handle_notification(self, result):
        """
        Hantera notifikationsljud
        
        Args:
            result: AudioDetectionResult
        
        Returns:
            dict: Handlingsresultat
        """
        action = {
            "type": "notification",
            "description": f"Detekterade notifikation: {result.pattern_name}",
            "priority": "medium",
            "actions_taken": []
        }
        
        # Om vi har en systemkontroller, meddela den
        if self.system_controller:
            # Logga händelsen i systemet
            self.system_controller.add_log_entry(f"🔔 Notifikation detekterad: {result.pattern_name}")
            action["actions_taken"].append("system_log")
            
            # Utför lämpliga åtgärder baserat på ljud
            # T.ex. leta efter notifikationsbara
            #self.system_controller.scan_for_notifications()
            action["actions_taken"].append("scan_for_notifications")
        
        logging.info(f"📢 Hanterar notifikation: {result.pattern_name}")
        return action
    
    def _handle_error(self, result):
        """
        Hantera felljud
        
        Args:
            result: AudioDetectionResult
        
        Returns:
            dict: Handlingsresultat
        """
        action = {
            "type": "error",
            "description": f"Detekterade felljud: {result.pattern_name}",
            "priority": "high",
            "actions_taken": []
        }
        
        # Om vi har en systemkontroller, meddela den
        if self.system_controller:
            # Logga händelsen i systemet
            self.system_controller.add_log_entry(f"❌ Fel detekterat: {result.pattern_name}")
            action["actions_taken"].append("system_log")
            
            # Utför lämpliga åtgärder baserat på ljud
            # T.ex. leta efter felmeddelanden
            #self.system_controller.scan_for_error_messages()
            action["actions_taken"].append("scan_for_error_messages")
            
            # Ta en skärmdump för att dokumentera felet
            #self.system_controller.capture_screen("error_screenshot")
            action["actions_taken"].append("capture_error_screenshot")
        
        logging.info(f"❌ Hanterar felljud: {result.pattern_name}")
        return action
    
    def _handle_success(self, result):
        """
        Hantera lyckade operationer
        
        Args:
            result: AudioDetectionResult
        
        Returns:
            dict: Handlingsresultat
        """
        action = {
            "type": "success",
            "description": f"Detekterade framgångsljud: {result.pattern_name}",
            "priority": "low",
            "actions_taken": []
        }
        
        # Om vi har en systemkontroller, meddela den
        if self.system_controller:
            # Logga händelsen i systemet
            self.system_controller.add_log_entry(f"✅ Framgång detekterad: {result.pattern_name}")
            action["actions_taken"].append("system_log")
        
        logging.info(f"✅ Hanterar framgångsljud: {result.pattern_name}")
        return action
    
    def _handle_warning(self, result):
        """
        Hantera varningsljud
        
        Args:
            result: AudioDetectionResult
        
        Returns:
            dict: Handlingsresultat
        """
        action = {
            "type": "warning",
            "description": f"Detekterade varning: {result.pattern_name}",
            "priority": "medium",
            "actions_taken": []
        }
        
        # Om vi har en systemkontroller, meddela den
        if self.system_controller:
            # Logga händelsen i systemet
            self.system_controller.add_log_entry(f"⚠️ Varning detekterad: {result.pattern_name}")
            action["actions_taken"].append("system_log")
            
            # Utför lämpliga åtgärder baserat på ljud
            # T.ex. leta efter varningsmeddelanden
            #self.system_controller.scan_for_warning_messages()
            action["actions_taken"].append("scan_for_warning_messages")
        
        logging.info(f"⚠️ Hanterar varning: {result.pattern_name}")
        return action
    
    def _handle_default(self, result):
        """
        Standardhantering för oidentifierade ljudmönster
        
        Args:
            result: AudioDetectionResult
        
        Returns:
            dict: Handlingsresultat
        """
        action = {
            "type": "unknown",
            "description": f"Detekterade okänt ljud: {result.pattern_name}",
            "priority": "low",
            "actions_taken": []
        }
        
        # Om vi har en systemkontroller, meddela den
        if self.system_controller:
            # Logga händelsen i systemet
            self.system_controller.add_log_entry(f"🔊 Okänt ljud detekterat: {result.pattern_name}")
            action["actions_taken"].append("system_log")
        
        logging.info(f"❓ Hanterar okänt ljud: {result.pattern_name}")
        return action
    
    def learn_application_sounds(self, app_name, duration=30.0):
        """
        Lär dig ljudmönster från en specifik applikation
        
        Args:
            app_name: Namn på applikationen
            duration: Tid att lyssna (sekunder)
            
        Returns:
            int: Antal inlärda ljudmönster
        """
        # Informera användaren
        logging.info(f"🎧 Lär in ljud från {app_name} i {duration} sekunder...")
        logging.info("Använd applikationen och generera dess olika ljud")
        
        # Spela in ljud
        patterns = self.audio_listener.capture_application_sounds(app_name, duration)
        
        # Byt namn på dem för klarhet
        for i, pattern in enumerate(patterns):
            pattern.name = f"{app_name}_{pattern.pattern_type}_{i+1}"
            pattern.save()
        
        return len(patterns)
    
    def save_state(self, filename=None):
        """
        Spara tillstånd till fil
        
        Args:
            filename: Filnamn att spara till (eller None för automatgenererat)
            
        Returns:
            str: Filnamn
        """
        if filename is None:
            os.makedirs("data/audio_recognition", exist_ok=True)
            filename = f"data/audio_recognition/state_{int(time.time())}.json"
        
        # Spara historik
        detection_history_file = self.audio_listener.save_detection_history()
        
        # Skapa tillståndsobjekt
        state = {
            "timestamp": time.time(),
            "detection_history_file": detection_history_file,
            "action_history": self.action_history,
            "status": self.audio_listener.status
        }
        
        # Spara till fil
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        
        logging.info(f"💾 AudioRecognitionManager-tillstånd sparat till {filename}")
        return filename
    
    def load_state(self, filename):
        """
        Ladda tillstånd från fil
        
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
                state = json.load(f)
            
            # Ladda detekteringshistorik
            if "detection_history_file" in state and os.path.exists(state["detection_history_file"]):
                self.audio_listener.load_detection_history(state["detection_history_file"])
            
            # Ladda handlingshistorik
            if "action_history" in state:
                self.action_history = state["action_history"]
            
            logging.info(f"📂 AudioRecognitionManager-tillstånd laddat från {filename}")
            return True
            
        except Exception as e:
            logging.error(f"Fel vid laddning av tillstånd: {e}")
            return False

# Globala funktioner för enkel användning
_audio_manager = None

def get_audio_manager(system_controller=None, device_index=None):
    """Hämta en global AudioRecognitionManager-instans"""
    global _audio_manager
    if _audio_manager is None:
        _audio_manager = AudioRecognitionManager(system_controller, device_index)
    return _audio_manager

def start_audio_recognition():
    """Starta ljudigenkänning"""
    get_audio_manager().start()

def stop_audio_recognition():
    """Stoppa ljudigenkänning"""
    if _audio_manager:
        _audio_manager.stop()

def learn_application_sounds(app_name, duration=30.0):
    """Lär in ljud från en applikation"""
    return get_audio_manager().learn_application_sounds(app_name, duration)

def shutdown_audio_recognition():
    """Stäng ner ljudigenkänningssystemet"""
    global _audio_manager
    if _audio_manager:
        _audio_manager.stop()
        _audio_manager = None

# Om vi kör direkt, utför ett enkelt test
if __name__ == "__main__":
    import sys
    
    # Skapa data-kataloger
    os.makedirs("data/audio_patterns", exist_ok=True)
    os.makedirs("data/audio_detections", exist_ok=True)
    
    # Hantera kommandradsargument
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "learn":
            # Lär in ljud från en applikation
            app_name = sys.argv[2] if len(sys.argv) > 2 else "default_app"
            duration = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
            
            audio_manager = get_audio_manager()
            patterns = audio_manager.learn_application_sounds(app_name, duration)
            
            print(f"Lärde in {len(patterns)} ljudmönster från {app_name}")
            
        elif command == "listen":
            # Lyssna efter ljud
            duration = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
            
            audio_manager = get_audio_manager()
            audio_manager.start()
            
            print(f"Lyssnar efter ljud i {duration} sekunder...")
            try:
                time.sleep(duration)
            except KeyboardInterrupt:
                print("Avbruten av användare")
            finally:
                audio_manager.stop()
                
            # Visa resultat
            history = audio_manager.audio_listener.detection_history
            if history:
                print(f"Detekterade {len(history)} ljud:")
                for result in history:
                    print(f"  {result}")
            else:
                print("Inga ljud detekterade")
                
        elif command == "record":
            # Spela in ett nytt ljudmönster
            name = sys.argv[2] if len(sys.argv) > 2 else None
            duration = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
            
            audio_listener = AudioListener()
            pattern = audio_listener.record_pattern(duration, name)
            
            if pattern:
                print(f"Ljudmönster sparat: {pattern.name}")
                
                # Visualisera
                output_file = f"data/audio_patterns/{pattern.name}.png"
                pattern.visualize(output_file)
                print(f"Visualisering sparad till {output_file}")
        else:
            print("Okänt kommando")
            print("Användning: python audio_recognition.py [learn|listen|record] [args...]")
    else:
        # Standardtest
        print("Starta ljudigenkänning...")
        audio_manager = get_audio_manager()
        audio_manager.start()
        
        print("Lyssnar efter ljud i 30 sekunder...")
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("Avbruten av användare")
        finally:
            audio_manager.stop()
            
        # Visa resultat
        history = audio_manager.audio_listener.detection_history
        if history:
            print(f"Detekterade {len(history)} ljud:")
            for result in history:
                print(f"  {result}")
        else:
            print("Inga ljud detekterades")