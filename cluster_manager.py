import threading
import time
import logging
import queue
import json
from datetime import datetime
import os

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("cluster_manager.log"),
        logging.StreamHandler()
    ]
)

class ClusterManager:
    """Hanterar alla kluster och deras koordination"""
    
    def __init__(self, ai_system):
        self.ai_system = ai_system
        self.clusters = {}
        self.cluster_status = {}
        self.event_queue = queue.Queue()
        self.running = False
        self.thread = None
        
        # Skapa datakataloger
        os.makedirs("data/clusters", exist_ok=True)
        
        logging.info("🔄 ClusterManager initierad")
        
    def start(self):
        """Starta klusterhanteraren"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        
        logging.info("🚀 ClusterManager startad")
        
    def stop(self):
        """Stoppa klusterhanteraren"""
        if not self.running:
            return
            
        self.running = False
        
        # Stoppa alla kluster
        for cluster_id in list(self.clusters.keys()):
            self.stop_cluster(cluster_id)
            
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        logging.info("⏹️ ClusterManager stoppad")
        
    def create_cluster(self, cluster_id, strategy_id, num_instances=5):
        """Skapa ett nytt kluster"""
        if cluster_id in self.clusters:
            logging.warning(f"Kluster {cluster_id} existerar redan!")
            return False
            
        # Hämta strategin från global router
        strategy = self.ai_system.global_router.strategies.get(strategy_id)
        if not strategy:
            logging.error(f"Strategi {strategy_id} finns inte!")
            return False
            
        # Skapa klustret
        from ai_engine import Cluster
        cluster = Cluster(cluster_id, strategy.get("name", "Okänd"), self.ai_system, num_instances)
        
        # Spara och starta klustret
        self.clusters[cluster_id] = cluster
        cluster.start()
        
        # Tilldela strategi
        cluster.set_strategy(strategy)
        
        # Uppdatera status
        self.cluster_status[cluster_id] = {
            "status": "running",
            "strategy_id": strategy_id,
            "num_instances": num_instances,
            "created_at": datetime.now().isoformat()
        }
        
        self._save_cluster_status()
        
        logging.info(f"✅ Kluster {cluster_id} skapat med {num_instances} instanser")
        return True
        
    def stop_cluster(self, cluster_id):
        """Stoppa ett specifikt kluster"""
        if cluster_id not in self.clusters:
            logging.warning(f"Kluster {cluster_id} finns inte!")
            return False
            
        # Stoppa klustret
        cluster = self.clusters[cluster_id]
        cluster.stop()
        
        # Ta bort från aktiva kluster
        del self.clusters[cluster_id]
        
        # Uppdatera status
        if cluster_id in self.cluster_status:
            self.cluster_status[cluster_id]["status"] = "stopped"
            self.cluster_status[cluster_id]["stopped_at"] = datetime.now().isoformat()
            
        self._save_cluster_status()
        
        logging.info(f"⏹️ Kluster {cluster_id} stoppad")
        return True
        
    def get_cluster_status(self, cluster_id=None):
        """Hämta status för ett eller alla kluster"""
        if cluster_id:
            return self.cluster_status.get(cluster_id)
        else:
            return self.cluster_status
            
    def scale_cluster(self, cluster_id, num_instances):
        """Skala upp eller ner ett kluster"""
        if cluster_id not in self.clusters:
            logging.warning(f"Kluster {cluster_id} finns inte!")
            return False
            
        # För att göra detta riktigt skulle vi behöva lägga till stöd i Cluster-klassen
        # för att dynamiskt lägga till/ta bort instanser
        # För nu, stoppa och återskapa med nytt antal
        
        # Hämta strategin
        strategy_id = self.cluster_status[cluster_id]["strategy_id"]
        
        # Stoppa klustret
        self.stop_cluster(cluster_id)
        
        # Återskapa med nytt antal
        return self.create_cluster(cluster_id, strategy_id, num_instances)
        
    def report_event(self, cluster_id, event_type, data):
        """Rapportera en händelse från ett kluster"""
        self.event_queue.put({
            "timestamp": datetime.now().isoformat(),
            "cluster_id": cluster_id,
            "type": event_type,
            "data": data
        })
        
    def _run(self):
        """Huvudloop för klusterhanteraren"""
        last_check_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Periodisk kontroll av klusterstatus
            if current_time - last_check_time > 30:  # Var 30:e sekund
                self._check_cluster_health()
                last_check_time = current_time
                
            # Hantera händelser i kön
            try:
                while not self.event_queue.empty():
                    event = self.event_queue.get(block=False)
                    self._handle_event(event)
            except queue.Empty:
                pass
                
            time.sleep(1)
            
    def _check_cluster_health(self):
        """Kontrollera hälsan för alla kluster"""
        for cluster_id, cluster in list(self.clusters.items()):
            # Kontrollera om klustret fortfarande kör
            if not getattr(cluster, "running", False):
                logging.warning(f"Kluster {cluster_id} verkar ha stannat oväntat!")
                
                # Uppdatera status
                if cluster_id in self.cluster_status:
                    self.cluster_status[cluster_id]["status"] = "crashed"
                    self.cluster_status[cluster_id]["crashed_at"] = datetime.now().isoformat()
                    
                # Ta bort från aktiva kluster
                del self.clusters[cluster_id]
                
                # Spara status
                self._save_cluster_status()
                
    def _handle_event(self, event):
        """Hantera en händelse från ett kluster"""
        # Logga händelsen
        logging.info(f"📣 Händelse från kluster {event['cluster_id']}: {event['type']}")
        
        # Skicka vidare till AI-systemet om det behövs
        if hasattr(self.ai_system, "add_event"):
            self.ai_system.add_event(f"cluster_{event['type']}", {
                "cluster_id": event["cluster_id"],
                "data": event["data"]
            })
            
    def _save_cluster_status(self):
        """Spara klusterstatus till disk"""
        try:
            with open("data/clusters/status.json", "w") as f:
                json.dump(self.cluster_status, f, indent=2)
        except Exception as e:
            logging.error(f"Kunde inte spara klusterstatus: {e}")
            
    def _load_cluster_status(self):
        """Ladda klusterstatus från disk"""
        try:
            if os.path.exists("data/clusters/status.json"):
                with open("data/clusters/status.json", "r") as f:
                    status = json.load(f)
                logging.info(f"📂 Laddade status för {len(status)} kluster")
                return status
        except Exception as e:
            logging.error(f"Kunde inte ladda klusterstatus: {e}")
            
        return {}