"""
CLI Runner - Prestandaläge för AI Desktop Controller

Detta är ett terminalbaserat gränssnitt för att köra AI Desktop Controller
i prestandaläge, utan det grafiska gränssnittet. Detta är idealiskt
när många instanser ska köras samtidigt för maximal parallell exekvering.
"""

import os
import time
import logging
import argparse
import threading
import json
from datetime import datetime

# Importera system integration
from system_integration import get_controller

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("cli_runner.log"),
        logging.StreamHandler()
    ]
)

class CLIRunner:
    """CLI-baserad körning av AI Desktop Controller i prestandaläge"""
    
    def __init__(self, args):
        self.args = args
        self.system = get_controller()
        self.running = False
        self.stats_interval = args.stats_interval
        self.clusters = {}
        self.start_time = None
        
        # Konfigurera kontrollägen
        from config import control_mode
        control_mode["mouse"] = args.mouse
        control_mode["keyboard"] = args.keyboard
        control_mode["system"] = args.system
        
        # Skapa datamappar
        os.makedirs("data/cli_stats", exist_ok=True)
        
        # Registrera callback för loggning
        self.system.set_callbacks(log_callback=self.log)
        
    def log(self, message):
        """Logga ett meddelande"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        
    def start(self):
        """Starta systemet i CLI-läge"""
        if self.running:
            self.log("Systemet körs redan!")
            return False
            
        try:
            self.log(f"Startar AI Desktop Controller i prestandaläge...")
            self.log(f"Kontrollägen: Mus={self.args.mouse}, Tangentbord={self.args.keyboard}, System={self.args.system}")
            self.log(f"Säkerhetsnivå: {self.args.safety}")
            
            # Starta systemet
            if not self.system.start_system(safety_level=self.args.safety):
                self.log("Fel vid start av systemet!")
                return False
                
            self.running = True
            self.start_time = time.time()
            
            # Skapa initiala kluster
            if self.args.clusters > 0 and self.args.instances > 0:
                self.log(f"Skapar {self.args.clusters} kluster med {self.args.instances} instanser vardera...")
                
                for i in range(self.args.clusters):
                    cluster_id = self.system.add_cluster(self.args.instances)
                    if cluster_id:
                        self.clusters[cluster_id] = {
                            "instances": self.args.instances,
                            "created_at": datetime.now().isoformat()
                        }
                        self.log(f"Skapat kluster: {cluster_id}")
                    else:
                        self.log(f"Kunde inte skapa kluster {i+1}!")
                        
            # Starta statistik-tråd
            if self.stats_interval > 0:
                self.stats_thread = threading.Thread(target=self._stats_monitor)
                self.stats_thread.daemon = True
                self.stats_thread.start()
                
            return True
            
        except Exception as e:
            self.log(f"Fel vid start: {e}")
            logging.exception("Start failure")
            return False
            
    def stop(self):
        """Stoppa systemet"""
        if not self.running:
            self.log("Systemet körs inte!")
            return False
            
        try:
            self.log("Stoppar systemet...")
            
            if not self.system.stop_system():
                self.log("Fel vid stopp av systemet!")
                return False
                
            self.running = False
            
            # Spara slutstatistik
            self._save_final_stats()
            
            return True
            
        except Exception as e:
            self.log(f"Fel vid stopp: {e}")
            logging.exception("Stop failure")
            return False
            
    def run(self):
        """Kör systemet och vänta på avslutningssignal"""
        if not self.start():
            return
            
        self.log("Systemet körs. Tryck Ctrl+C för att avsluta.")
        
        try:
            # Kör tills avbrott
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.log("Avbryter...")
            
        finally:
            self.stop()
            
    def _stats_monitor(self):
        """Övervaka och rapportera systemstatistik med jämna mellanrum"""
        last_stats_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Dags att rapportera statistik?
            if current_time - last_stats_time >= self.stats_interval:
                self._report_stats()
                last_stats_time = current_time
                
            time.sleep(1)
            
    def _report_stats(self):
        """Rapportera aktuell systemstatistik"""
        try:
            # Hämta information
            clusters = self.system.get_clusters()
            stats = self.system.get_system_stats()
            uptime = time.time() - self.start_time
            
            # Beräkna totalt antal aktiva instanser
            total_instances = sum(cluster.get("num_instances", 0) for cluster in clusters.values())
            
            # Beräkna genomsnittlig framgångsfrekvens
            success_rates = [c.get("success_rate", 0) for c in clusters.values()]
            avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
            
            # Visa statistik
            self.log("\n--- Systemstatistik ---")
            self.log(f"Körtid: {int(uptime)} sekunder")
            self.log(f"Aktiva kluster: {len(clusters)}")
            self.log(f"Totalt antal instanser: {total_instances}")
            self.log(f"Genomsnittlig framgångsfrekvens: {avg_success_rate:.1%}")
            
            # Visa information om varje kluster
            self.log("\n--- Klusterstatistik ---")
            for cluster_id, info in clusters.items():
                self.log(f"Kluster: {cluster_id}")
                self.log(f"  Strategi: {info.get('strategy', 'Okänd')}")
                self.log(f"  Instanser: {info.get('num_instances', 0)}")
                self.log(f"  Framgång: {info.get('success_rate', 0):.1%}")
                
            self.log("----------------------\n")
            
            # Spara statistik till fil
            self._save_stats(clusters, stats, uptime, avg_success_rate)
            
        except Exception as e:
            logging.error(f"Fel vid statistikrapportering: {e}")
            
    def _save_stats(self, clusters, stats, uptime, avg_success_rate):
        """Spara statistik till fil"""
        try:
            timestamp = int(time.time())
            
            stats_data = {
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "uptime": uptime,
                "clusters": len(clusters),
                "total_instances": stats.get("total_instances", 0),
                "avg_success_rate": avg_success_rate,
                "clusters_info": clusters,
                "system_stats": stats
            }
            
            # Spara till fil
            with open(f"data/cli_stats/stats_{timestamp}.json", "w") as f:
                json.dump(stats_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Fel vid sparande av statistik: {e}")
            
    def _save_final_stats(self):
        """Spara slutstatistik när systemet stoppas"""
        try:
            uptime = time.time() - self.start_time if self.start_time else 0
            
            final_stats = {
                "timestamp": int(time.time()),
                "datetime": datetime.now().isoformat(),
                "total_uptime": uptime,
                "clusters_created": len(self.clusters),
                "clusters_info": self.clusters
            }
            
            # Spara till fil
            with open(f"data/cli_stats/final_stats_{int(time.time())}.json", "w") as f:
                json.dump(final_stats, f, indent=2)
                
            self.log(f"Slutstatistik sparad. Total körtid: {int(uptime)} sekunder.")
            
        except Exception as e:
            logging.error(f"Fel vid sparande av slutstatistik: {e}")

def parse_arguments():
    """Hantera kommandoradsargument"""
    parser = argparse.ArgumentParser(description="AI Desktop Controller - Prestandaläge")
    
    # Kluster och instanser
    parser.add_argument("-c", "--clusters", type=int, default=3,
                      help="Antal kluster att starta (standard: 3)")
    parser.add_argument("-i", "--instances", type=int, default=5,
                      help="Antal instanser per kluster (standard: 5)")
    
    # Kontrollägen
    parser.add_argument("--mouse", action="store_true", default=False,
                      help="Aktivera muskontroll")
    parser.add_argument("--keyboard", action="store_true", default=False,
                      help="Aktivera tangentbordskontroll")
    parser.add_argument("--system", action="store_true", default=False,
                      help="Aktivera systemkontroll")
    
    # Säkerhetsnivå
    parser.add_argument("--safety", choices=["high", "medium", "low"], default="high",
                      help="Säkerhetsnivå (standard: high)")
    
    # Övervakningsintervalll
    parser.add_argument("--stats-interval", type=int, default=30,
                      help="Intervall för statistikrapportering i sekunder (standard: 30)")
    
    return parser.parse_args()

def main():
    """Huvudfunktion"""
    print("🚀 AI Desktop Controller - Prestandaläge")
    
    # Hantera argument
    args = parse_arguments()
    
    # Skapa och kör CLI-köraren
    runner = CLIRunner(args)
    runner.run()

if __name__ == "__main__":
    main()