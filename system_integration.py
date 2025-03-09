"""
System Integration - Kopplar ihop alla komponenter i AI Desktop Controller

Denna modul fungerar som en brygga mellan det befintliga användargränssnittet (ui.py)
och det nya hierarkiska AI-systemet (ai_engine.py, cluster_manager.py, command_executor.py).
"""

import logging
import threading
import time
from datetime import datetime
import os

# Importera systemkomponenter
from ai_engine import AISystem, run_ai_system
from cluster_manager import ClusterManager
from command_executor import CommandExecutor
from config import control_mode

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("system_integration.log"),
        logging.StreamHandler()
    ]
)

class SystemController:
    """
    Huvudkontrollklass som integrerar alla systemkomponenter och 
    tillhandahåller ett enkelt API för ui.py
    """
    
    def __init__(self):
        self.ai_system = None
        self.cluster_manager = None
        self.command_executor = None
        self.running = False
        self.update_thread = None
        self.log_callback = None
        self.ui_update_callback = None
        
        # Skapa datakataloger
        os.makedirs("data/logs", exist_ok=True)
        os.makedirs("data/screenshots", exist_ok=True)
        
        logging.info("🔄 SystemController initierad")
        
    def set_callbacks(self, log_callback=None, ui_update_callback=None):
        """
        Sätt callbacks för att kommunicera med UI
        
        Parametrar:
        - log_callback: Funktion som anropas med loggmeddelanden
        - ui_update_callback: Funktion som anropas för UI-uppdateringar
        """
        self.log_callback = log_callback
        self.ui_update_callback = ui_update_callback
        
    def log(self, message):
        """Logga ett meddelande och skicka till UI om möjligt"""
        logging.info(message)
        if self.log_callback:
            self.log_callback(message)
        
    def start_system(self, safety_level="high"):
        """
        Starta hela AI-systemet
        
        Parametrar:
        - safety_level: "high", "medium" eller "low" - säkerhetsnivå för kommandokörning
        
        Returnerar:
        - bool: True om systemet startades framgångsrikt
        """
        if self.running:
            self.log("⚠️ Systemet körs redan!")
            return False
            
        try:
            self.log("🚀 Startar AI Desktop Controller systemet...")
            
            # Skapa och starta AI-systemet
            self.ai_system = AISystem()
            self.ai_system.initialize()
            
            # Skapa klusterhanterare
            self.cluster_manager = ClusterManager(self.ai_system)
            
            # Skapa kommandokörare
            self.command_executor = CommandExecutor(safety_level=safety_level)
            
            # Starta komponenterna
            self.ai_system.start()
            self.cluster_manager.start()
            
            # Skapa initial konfiguration
            self._setup_initial_configuration()
            
            # Starta uppdateringstråd
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            
            self.log("✅ AI-systemet har startats och körs")
            return True
            
        except Exception as e:
            self.log(f"❌ Fel vid systemstart: {str(e)}")
            logging.error(f"Fel vid systemstart: {e}")
            return False
        
    def stop_system(self):
        """
        Stoppa hela AI-systemet
        
        Returnerar:
        - bool: True om systemet stoppades framgångsrikt
        """
        if not self.running:
            self.log("⚠️ Systemet körs inte!")
            return False
            
        try:
            self.log("⏹️ Stoppar AI-systemet...")
            
            # Stoppa uppdateringstråd
            self.running = False
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=2.0)
            
            # Stoppa komponenterna
            if self.cluster_manager:
                self.cluster_manager.stop()
                
            if self.ai_system:
                self.ai_system.stop()
                
            self.log("✅ AI-systemet har stoppats")
            return True
            
        except Exception as e:
            self.log(f"❌ Fel vid systemstopp: {str(e)}")
            logging.error(f"Fel vid systemstopp: {e}")
            return False
        
    def add_cluster(self, num_instances=5):
        """
        Lägg till ett nytt kluster
        
        Parametrar:
        - num_instances: Antal instanser i klustret
        
        Returnerar:
        - str: ID för det nya klustret, eller None vid fel
        """
        if not self.running or not self.ai_system or not self.cluster_manager:
            self.log("❌ Systemet är inte igång!")
            return None
            
        try:
            # Generera kluster-ID
            cluster_id = f"Cluster_{int(time.time())}"
            
            # Välj en tillgänglig strategi
            available_strategies = list(self.ai_system.global_router.strategies.keys())
            if not available_strategies:
                self.log("❌ Inga strategier tillgängliga!")
                return None
                
            # Skapa klustret
            strategy_id = available_strategies[0]
            if self.cluster_manager.create_cluster(cluster_id, strategy_id, num_instances):
                self.log(f"➕ Lade till kluster: {cluster_id} med {num_instances} instanser")
                return cluster_id
            else:
                self.log(f"❌ Kunde inte skapa kluster!")
                return None
                
        except Exception as e:
            self.log(f"❌ Fel vid tillägg av kluster: {str(e)}")
            logging.error(f"Fel vid tillägg av kluster: {e}")
            return None
        
    def remove_cluster(self, cluster_id):
        """
        Ta bort ett specifikt kluster
        
        Parametrar:
        - cluster_id: ID för klustret som ska tas bort
        
        Returnerar:
        - bool: True om klustret togs bort framgångsrikt
        """
        if not self.running or not self.cluster_manager:
            self.log("❌ Systemet är inte igång!")
            return False
            
        try:
            if self.cluster_manager.stop_cluster(cluster_id):
                self.log(f"🗑️ Tog bort kluster: {cluster_id}")
                return True
            else:
                self.log(f"❌ Kunde inte hitta kluster: {cluster_id}")
                return False
                
        except Exception as e:
            self.log(f"❌ Fel vid borttagning av kluster: {str(e)}")
            logging.error(f"Fel vid borttagning av kluster: {e}")
            return False
        
    def get_clusters(self):
        """
        Hämta information om alla aktiva kluster
        
        Returnerar:
        - dict: Kluster-information eller tom dict vid fel
        """
        if not self.running or not self.cluster_manager:
            return {}
            
        try:
            clusters_info = {}
            
            # Samla information från klusterhanteraren
            for cluster_id, cluster in self.cluster_manager.clusters.items():
                # Hämta strategi
                strategy_name = "Okänd"
                if cluster.active_strategy and "name" in cluster.active_strategy:
                    strategy_name = cluster.active_strategy["name"]
                    
                # Beräkna framgångsfrekvens
                success_rate = 0
                if cluster.recent_results:
                    success_count = sum(1 for r in cluster.recent_results[-20:] 
                                      if r["data"].get("success", False) == True)
                    total_count = min(len(cluster.recent_results), 20)
                    if total_count > 0:
                        success_rate = success_count / total_count
                
                # Samla information
                clusters_info[cluster_id] = {
                    "id": cluster_id,
                    "strategy": strategy_name,
                    "num_instances": len(cluster.instances),
                    "success_rate": success_rate,
                    "active": cluster.running
                }
                
            return clusters_info
            
        except Exception as e:
            logging.error(f"Fel vid hämtning av klusterinfo: {e}")
            return {}
        
    def get_system_stats(self):
        """
        Hämta statistik om systemet
        
        Returnerar:
        - dict: Systemstatistik eller tom dict vid fel
        """
        if not self.running:
            return {}
            
        try:
            stats = {
                "running": self.running,
                "num_clusters": len(self.cluster_manager.clusters) if self.cluster_manager else 0,
                "num_strategies": len(self.ai_system.global_router.strategies) if self.ai_system and self.ai_system.global_router else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Räkna totala antalet instanser
            total_instances = 0
            if self.cluster_manager:
                for cluster in self.cluster_manager.clusters.values():
                    total_instances += len(cluster.instances)
                    
            stats["total_instances"] = total_instances
            
            return stats
            
        except Exception as e:
            logging.error(f"Fel vid hämtning av systemstatistik: {e}")
            return {}
        
    def execute_command(self, command_type, params=None):
        """
        Utför ett specifikt kommando direkt
        
        Parametrar:
        - command_type: Typ av kommando (click, move, type, osv.)
        - params: Parametrar för kommandot
        
        Returnerar:
        - dict: Resultat från kommandot
        """
        if not self.running or not self.command_executor:
            return {"success": False, "error": "Systemet är inte igång!"}
            
        try:
            return self.command_executor.execute_command(command_type, params)
        except Exception as e:
            logging.error(f"Fel vid exekvering av kommando: {e}")
            return {"success": False, "error": str(e)}
        
    def update_control_modes(self, mouse=None, keyboard=None, system=None):
        """
        Uppdatera kontrollägen för systemet
        
        Parametrar:
        - mouse: True/False för muskontroll
        - keyboard: True/False för tangentbordskontroll
        - system: True/False för systemkontroll
        
        Returnerar:
        - dict: Aktuella kontrollägen
        """
        # Uppdatera endast angivna värden
        if mouse is not None:
            control_mode["mouse"] = mouse
            
        if keyboard is not None:
            control_mode["keyboard"] = keyboard
            
        if system is not None:
            control_mode["system"] = system
            
        # Uppdatera command executor om den finns
        if self.command_executor:
            self.command_executor._check_control_modes()
            
        self.log(f"🔄 Uppdaterade kontrollägen: Mus={control_mode['mouse']}, "
                f"Tangentbord={control_mode['keyboard']}, System={control_mode['system']}")
                
        return control_mode.copy()
        
    def get_control_modes(self):
        """
        Hämta aktuella kontrollägen
        
        Returnerar:
        - dict: Aktuella kontrollägen
        """
        return control_mode.copy()
        
    def _update_loop(self):
        """Huvudloop för uppdatering och synkronisering mellan komponenter"""
        while self.running:
            try:
                # Hämta och hantera händelser från AI-systemet
                if self.ai_system:
                    event = self.ai_system.get_event(timeout=0.1)
                    if event:
                        self.log(f"🔔 Händelse: {event['type']} - {str(event['data'])[:50]}")
                        
                # Kör UI-uppdatering om callback finns
                if self.ui_update_callback:
                    clusters = self.get_clusters()
                    stats = self.get_system_stats()
                    self.ui_update_callback(clusters, stats)
                    
            except Exception as e:
                logging.error(f"Fel i uppdateringsloop: {e}")
                
            time.sleep(1)
            
    def _setup_initial_configuration(self):
        """Skapa initial konfiguration för systemet"""
        # Skapa första klustret för att komma igång
        self.add_cluster(num_instances=3)
        self.log("🏁 Skapade initialt kluster med 3 instanser")
        
# Singleton-instans för enkel åtkomst
system_controller = SystemController()

def get_controller():
    """Hämta controller-instansen"""
    return system_controller