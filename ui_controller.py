import os
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import time
import threading
import logging
import json
import pyautogui

# Importera systemkomponenter
from ai_engine import AISystem, GlobalRouter
from cluster_manager import ClusterManager
from command_executor import CommandExecutor

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ui_controller.log"),
        logging.StreamHandler()
    ]
)

# Aktivera Dark Mode
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class UIController(ctk.CTk):
    """Huvudgränssnitt för AI Desktop Controller"""
    
    def __init__(self):
        super().__init__()
        
        # Konfigurera fönster
        self.title("AI Desktop Controller")
        self.geometry("1200x800")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Globala variabler
        self.ai_system = None
        self.cluster_manager = None
        self.command_executor = None
        self.running = False
        self.update_interval = 1000  # ms
        self.show_screen = True
        self.log_entries = []
        
        # Skapa datakataloger
        os.makedirs("data/logs", exist_ok=True)
        
        # Skapa gränssnitt
        self._create_ui()
        
        # Status
        self.status_text.set("❌ System inte startat")
        
    def _create_ui(self):
        """Skapa det grafiska gränssnittet"""
        # Huvudkolumner
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        
        # Kontrollpanel (vänster sida)
        self.control_panel = ctk.CTkFrame(self)
        self.control_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self._create_control_panel()
        
        # Huvudvy (höger sida)
        self.main_panel = ctk.CTkFrame(self)
        self.main_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self._create_main_panel()
        
    def _create_control_panel(self):
        """Skapa kontrollpanelen med knappar och inställningar"""
        self.control_panel.grid_columnconfigure(0, weight=1)
        
        # Titel
        ctrl_title = ctk.CTkLabel(self.control_panel, text="🧠 Systemkontroll", font=("Arial", 20, "bold"))
        ctrl_title.grid(row=0, column=0, padx=10, pady=(20, 10), sticky="w")
        
        # Status
        self.status_text = ctk.StringVar(value="Inte startat")
        status_label = ctk.CTkLabel(self.control_panel, textvariable=self.status_text, font=("Arial", 14))
        status_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        # Huvudkontroll-knappar
        ctrl_frame = ctk.CTkFrame(self.control_panel)
        ctrl_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        # Start-knapp
        self.start_btn = ctk.CTkButton(ctrl_frame, text="🚀 Starta system", 
                                      command=self.start_system,
                                      fg_color="#28a745", hover_color="#218838")
        self.start_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Stopp-knapp
        self.stop_btn = ctk.CTkButton(ctrl_frame, text="⏹️ Stoppa system", 
                                     command=self.stop_system,
                                     fg_color="#dc3545", hover_color="#c82333",
                                     state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Kluster-hantering
        cluster_title = ctk.CTkLabel(self.control_panel, text="📊 Klusterhantering", font=("Arial", 16, "bold"))
        cluster_title.grid(row=3, column=0, padx=10, pady=(20, 10), sticky="w")
        
        # Lägg till kluster
        add_cluster_frame = ctk.CTkFrame(self.control_panel)
        add_cluster_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        
        # Antal instanser för nytt kluster
        self.num_instances_label = ctk.CTkLabel(add_cluster_frame, text="Instanser:")
        self.num_instances_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.num_instances_var = ctk.StringVar(value="5")
        self.num_instances_entry = ctk.CTkEntry(add_cluster_frame, textvariable=self.num_instances_var, width=50)
        self.num_instances_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Lägg till kluster-knapp
        self.add_cluster_btn = ctk.CTkButton(add_cluster_frame, text="➕ Lägg till kluster", 
                                           command=self.add_cluster,
                                           state="disabled")
        self.add_cluster_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        # Lista med aktiva kluster
        cluster_frame = ctk.CTkFrame(self.control_panel)
        cluster_frame.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
        
        cluster_list_label = ctk.CTkLabel(cluster_frame, text="Aktiva kluster:")
        cluster_list_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.cluster_listbox = tk.Listbox(cluster_frame, bg="#333", fg="white", height=8)
        self.cluster_listbox.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # Ta bort kluster-knapp
        self.remove_cluster_btn = ctk.CTkButton(cluster_frame, text="🗑️ Ta bort kluster", 
                                              command=self.remove_selected_cluster,
                                              fg_color="#dc3545", hover_color="#c82333",
                                              state="disabled")
        self.remove_cluster_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        # Inställningar
        settings_title = ctk.CTkLabel(self.control_panel, text="⚙️ Inställningar", font=("Arial", 16, "bold"))
        settings_title.grid(row=6, column=0, padx=10, pady=(20, 10), sticky="w")
        
        # Säkerhetsnivå
        safety_frame = ctk.CTkFrame(self.control_panel)
        safety_frame.grid(row=7, column=0, padx=10, pady=5, sticky="ew")
        
        safety_label = ctk.CTkLabel(safety_frame, text="Säkerhetsnivå:")
        safety_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.safety_var = ctk.StringVar(value="high")
        safety_options = ["high", "medium", "low"]
        safety_dropdown = ctk.CTkOptionMenu(safety_frame, values=safety_options, variable=self.safety_var)
        safety_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Styr-lägen
        control_modes_frame = ctk.CTkFrame(self.control_panel)
        control_modes_frame.grid(row=8, column=0, padx=10, pady=5, sticky="ew")
        
        control_modes_label = ctk.CTkLabel(control_modes_frame, text="Kontrollägen:")
        control_modes_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Mus-kontroll
        self.mouse_control_var = ctk.BooleanVar(value=False)
        mouse_cb = ctk.CTkCheckBox(control_modes_frame, text="Mus", variable=self.mouse_control_var,
                                  command=self.update_control_modes)
        mouse_cb.grid(row=1, column=0, padx=5, pady=2, sticky="w")
        
        # Tangentbord-kontroll
        self.keyboard_control_var = ctk.BooleanVar(value=False)
        keyboard_cb = ctk.CTkCheckBox(control_modes_frame, text="Tangentbord", variable=self.keyboard_control_var,
                                     command=self.update_control_modes)
        keyboard_cb.grid(row=2, column=0, padx=5, pady=2, sticky="w")
        
        # System-kontroll
        self.system_control_var = ctk.BooleanVar(value=False)
        system_cb = ctk.CTkCheckBox(control_modes_frame, text="System", variable=self.system_control_var,
                                   command=self.update_control_modes)
        system_cb.grid(row=3, column=0, padx=5, pady=2, sticky="w")
        
    def _create_main_panel(self):
        """Skapa huvudpanelen med flikar"""
        self.main_panel.grid_rowconfigure(0, weight=1)
        self.main_panel.grid_columnconfigure(0, weight=1)
        
        # Skapa flikar
        self.tabview = ctk.CTkTabview(self.main_panel)
        self.tabview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Lägg till flikar
        self.screen_tab = self.tabview.add("Skärm")
        self.clusters_tab = self.tabview.add("Kluster")
        self.logs_tab = self.tabview.add("Loggar")
        self.stats_tab = self.tabview.add("Statistik")
        
        # Konfigurera flikar
        self._create_screen_tab()
        self._create_clusters_tab()
        self._create_logs_tab()
        self._create_stats_tab()
        
    def _create_screen_tab(self):
        """Skapa fliken för skärmvisning"""
        self.screen_tab.grid_rowconfigure(0, weight=1)
        self.screen_tab.grid_columnconfigure(0, weight=1)
        
        # Bild för skärmdump
        self.screen_frame = ctk.CTkFrame(self.screen_tab)
        self.screen_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.screen_label = ctk.CTkLabel(self.screen_frame, text="Skärmbild laddas...")
        self.screen_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Kontroller
        screen_controls = ctk.CTkFrame(self.screen_tab)
        screen_controls.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        # Toggle-knapp för att visa/dölja skärm
        self.toggle_screen_btn = ctk.CTkButton(screen_controls, text="🖥️ Visa/Dölj Skärm", 
                                             command=self.toggle_screen)
        self.toggle_screen_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Ta ny skärmdump-knapp
        self.capture_btn = ctk.CTkButton(screen_controls, text="📸 Ta skärmdump", 
                                       command=self.capture_screen)
        self.capture_btn.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
    def _create_clusters_tab(self):
        """Skapa fliken för klusterhantering"""
        self.clusters_tab.grid_rowconfigure(0, weight=1)
        self.clusters_tab.grid_columnconfigure(0, weight=1)
        
        # Container för klustervisning
        self.clusters_container = ctk.CTkScrollableFrame(self.clusters_tab)
        self.clusters_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Platshållare tills systemet startas
        placeholder = ctk.CTkLabel(self.clusters_container, text="Starta systemet för att se kluster")
        placeholder.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
    def _create_logs_tab(self):
        """Skapa fliken för loggvisning"""
        self.logs_tab.grid_rowconfigure(0, weight=1)
        self.logs_tab.grid_columnconfigure(0, weight=1)
        
        # Textområde för loggar
        self.log_text = ctk.CTkTextbox(self.logs_tab, height=500)
        self.log_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Kontroller
        log_controls = ctk.CTkFrame(self.logs_tab)
        log_controls.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        # Rensa loggar-knapp
        self.clear_logs_btn = ctk.CTkButton(log_controls, text="🧹 Rensa loggar", 
                                          command=self.clear_logs)
        self.clear_logs_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Spara loggar-knapp
        self.save_logs_btn = ctk.CTkButton(log_controls, text="💾 Spara loggar", 
                                         command=self.save_logs)
        self.save_logs_btn.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
    def _create_stats_tab(self):
        """Skapa fliken för statistik"""
        self.stats_tab.grid_rowconfigure(0, weight=1)
        self.stats_tab.grid_columnconfigure(0, weight=1)
        
        # Container för statistik
        self.stats_container = ctk.CTkScrollableFrame(self.stats_tab)
        self.stats_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Platshållare tills systemet startas
        placeholder = ctk.CTkLabel(self.stats_container, text="Starta systemet för att se statistik")
        placeholder.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
    def update_control_modes(self):
        """Uppdatera kontrollägen i config"""
        from config import control_mode
        
        # Uppdatera globala inställningar
        control_mode["mouse"] = self.mouse_control_var.get()
        control_mode["keyboard"] = self.keyboard_control_var.get()
        control_mode["system"] = self.system_control_var.get()
        
        # Uppdatera command executor om den finns
        if self.command_executor:
            self.command_executor._check_control_modes()
            
        # Logga ändringar
        self.add_log_entry(f"🔄 Uppdaterade kontrollägen: Mus={control_mode['mouse']}, "
                          f"Tangentbord={control_mode['keyboard']}, System={control_mode['system']}")
        
    def start_system(self):
        """Starta hela AI-systemet"""
        if self.running:
            return
            
        try:
            # Uppdatera UI
            self.status_text.set("⏳ Startar system...")
            self.update()
            
            # Skapa och starta komponenterna
            self.ai_system = AISystem()
            self.ai_system.initialize()
            
            self.cluster_manager = ClusterManager(self.ai_system)
            
            # Skapa command executor med vald säkerhetsnivå
            self.command_executor = CommandExecutor(safety_level=self.safety_var.get())
            
            # Starta systemen
            self.ai_system.start()
            self.cluster_manager.start()
            
            # Uppdatera UI-inställningar
            from config import control_mode
            self.mouse_control_var.set(control_mode.get("mouse", False))
            self.keyboard_control_var.set(control_mode.get("keyboard", False))
            self.system_control_var.set(control_mode.get("system", False))
            
            # Aktivera knappar
            self.running = True
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.add_cluster_btn.configure(state="normal")
            self.remove_cluster_btn.configure(state="normal")
            
            # Starta uppdateringsloop
            self._schedule_updates()
            
            # Uppdatera status
            self.status_text.set("✅ System igång")
            
            # Logga start
            self.add_log_entry("🚀 AI-systemet har startats")
            
        except Exception as e:
            self.status_text.set(f"❌ Startfel: {str(e)}")
            logging.error(f"Fel vid systemstart: {e}")
            self.add_log_entry(f"❌ Startfel: {str(e)}")
            
    def stop_system(self):
        """Stoppa hela AI-systemet"""
        if not self.running:
            return
            
        try:
            # Uppdatera UI
            self.status_text.set("⏳ Stoppar system...")
            self.update()
            
            # Stoppa systemen
            if self.ai_system:
                self.ai_system.stop()
                
            if self.cluster_manager:
                self.cluster_manager.stop()
                
            # Rensa klusterlistan
            self.cluster_listbox.delete(0, tk.END)
            
            # Inaktivera knappar
            self.running = False
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.add_cluster_btn.configure(state="disabled")
            self.remove_cluster_btn.configure(state="disabled")
            
            # Uppdatera status
            self.status_text.set("⏹️ System stoppat")
            
            # Logga stopp
            self.add_log_entry("⏹️ AI-systemet har stoppats")
            
        except Exception as e:
            self.status_text.set(f"❌ Stoppfel: {str(e)}")
            logging.error(f"Fel vid systemstopp: {e}")
            self.add_log_entry(f"❌ Stoppfel: {str(e)}")
            
    def add_cluster(self):
        """Lägg till ett nytt kluster"""
        if not self.running or not self.ai_system or not self.cluster_manager:
            return
            
        try:
            # Läs antal instanser
            try:
                num_instances = int(self.num_instances_var.get())
                if num_instances < 1:
                    raise ValueError("Antal instanser måste vara minst 1")
            except ValueError as e:
                self.add_log_entry(f"❌ Ogiltigt antal instanser: {str(e)}")
                return
                
            # Generera kluster-ID
            cluster_id = f"Cluster_{int(time.time())}"
            
            # Välj en tillgänglig strategi
            available_strategies = list(self.ai_system.global_router.strategies.keys())
            if not available_strategies:
                self.add_log_entry("❌ Inga strategier tillgängliga!")
                return
                
            # Skapa klustret
            strategy_id = available_strategies[0]
            self.cluster_manager.create_cluster(cluster_id, strategy_id, num_instances)
            
            # Uppdatera UI
            self.cluster_listbox.insert(tk.END, cluster_id)
            
            # Logga tillägg
            self.add_log_entry(f"➕ Lade till kluster: {cluster_id} med {num_instances} instanser")
            
            # Uppdatera klustervy
            self.update_clusters_view()
            
        except Exception as e:
            logging.error(f"Fel vid tillägg av kluster: {e}")
            self.add_log_entry(f"❌ Kunde inte lägga till kluster: {str(e)}")
            
    def remove_selected_cluster(self):
        """Ta bort det valda klustret"""
        if not self.running or not self.cluster_manager:
            return
            
        selected = self.cluster_listbox.curselection()
        if not selected:
            self.add_log_entry("❌ Inget kluster valt!")
            return
            
        cluster_id = self.cluster_listbox.get(selected[0])
        
        try:
            # Stoppa klustret
            self.cluster_manager.stop_cluster(cluster_id)
            
            # Uppdatera UI
            self.cluster_listbox.delete(selected[0])
            
            # Logga borttagning
            self.add_log_entry(f"🗑️ Tog bort kluster: {cluster_id}")
            
            # Uppdatera klustervy
            self.update_clusters_view()
            
        except Exception as e:
            logging.error(f"Fel vid borttagning av kluster: {e}")
            self.add_log_entry(f"❌ Kunde inte ta bort kluster: {str(e)}")
            
    def toggle_screen(self):
        """Växla mellan att visa och dölja skärm"""
        self.show_screen = not self.show_screen
        
        if not self.show_screen:
            self.screen_label.configure(text="🖥️ Skärm dold", image=None)
        else:
            self.capture_screen()
            
    def capture_screen(self):
        """Ta en ny skärmdump"""
        if not self.show_screen:
            return
            
        try:
            # Ta skärmdump
            screenshot = pyautogui.screenshot()
            
            # Anpassa storlek för visning
            max_width = 800
            max_height = 600
            
            width, height = screenshot.size
            ratio = min(max_width / width, max_height / height)
            new_size = (int(width * ratio), int(height * ratio))
            
            resized_img = screenshot.resize(new_size)
            
            # Konvertera för visning
            tk_img = ImageTk.PhotoImage(resized_img)
            
            # Uppdatera label
            self.screen_label.configure(image=tk_img, text="")
            self.screen_label.image = tk_img  # Behåll referens för att undvika garbage collection
            
        except Exception as e:
            logging.error(f"Fel vid skärmdump: {e}")
            self.screen_label.configure(text=f"❌ Fel vid skärmdump: {str(e)}", image=None)
            
    def clear_logs(self):
        """Rensa loggarna"""
        self.log_text.delete("1.0", tk.END)
        self.log_entries = []
        
    def save_logs(self):
        """Spara loggar till fil"""
        try:
            timestamp = int(time.time())
            filename = f"data/logs/log_{timestamp}.txt"
            
            with open(filename, "w") as f:
                f.write(self.log_text.get("1.0", tk.END))
                
            self.add_log_entry(f"💾 Sparade loggar till {filename}")
            
        except Exception as e:
            logging.error(f"Fel vid sparande av loggar: {e}")
            self.add_log_entry(f"❌ Kunde inte spara loggar: {str(e)}")
            
    def add_log_entry(self, entry):
        """Lägg till en ny loggpost"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {entry}\n"
        
        # Lägg till i textområdet
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)  # Scrolla till slutet
        
        # Spara i listan
        self.log_entries.append(log_entry)
        
        # Begränsa längden
        if len(self.log_entries) > 1000:
            self.log_entries = self.log_entries[-1000:]
            
    def update_clusters_view(self):
        """Uppdatera klustervyn"""
        if not self.running or not self.cluster_manager:
            return
            
        # Rensa befintlig vy
        for widget in self.clusters_container.winfo_children():
            widget.destroy()
            
        # Lägg till klusterinformation
        row = 0
        for cluster_id, cluster in self.cluster_manager.clusters.items():
            # Rubrik för kluster
            cluster_frame = ctk.CTkFrame(self.clusters_container)
            cluster_frame.grid(row=row, column=0, padx=10, pady=5, sticky="ew")
            
            # Kluster-titel
            title = ctk.CTkLabel(cluster_frame, text=f"🔷 {cluster_id}", font=("Arial", 16, "bold"))
            title.grid(row=0, column=0, padx=10, pady=5, sticky="w")
            
            # Strategi
            strategy_name = "Okänd"
            if cluster.active_strategy and "name" in cluster.active_strategy:
                strategy_name = cluster.active_strategy["name"]
                
            strategy_label = ctk.CTkLabel(cluster_frame, text=f"Strategi: {strategy_name}")
            strategy_label.grid(row=1, column=0, padx=10, pady=2, sticky="w")
            
            # Antal instanser
            instances_label = ctk.CTkLabel(cluster_frame, text=f"Instanser: {len(cluster.instances)}")
            instances_label.grid(row=2, column=0, padx=10, pady=2, sticky="w")
            
            # Framgångsfrekvens
            success_rate = 0
            if cluster.recent_results:
                success_count = sum(1 for r in cluster.recent_results[-20:] 
                                  if r["data"].get("success", False) == True)
                total_count = min(len(cluster.recent_results), 20)
                if total_count > 0:
                    success_rate = success_count / total_count
                    
            success_label = ctk.CTkLabel(cluster_frame, text=f"Framgångsfrekvens: {success_rate:.1%}")
            success_label.grid(row=3, column=0, padx=10, pady=2, sticky="w")
            
            row += 1
            
    def update_stats_view(self):
        """Uppdatera statistikvyn"""
        if not self.running or not self.ai_system:
            return
            
        # Rensa befintlig vy
        for widget in self.stats_container.winfo_children():
            widget.destroy()
            
        # Övergripande statistik
        stats_title = ctk.CTkLabel(self.stats_container, text="📊 Systemstatistik", font=("Arial", 16, "bold"))
        stats_title.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Antal kluster
        num_clusters = len(self.cluster_manager.clusters) if self.cluster_manager else 0
        clusters_label = ctk.CTkLabel(self.stats_container, text=f"Aktiva kluster: {num_clusters}")
        clusters_label.grid(row=1, column=0, padx=10, pady=2, sticky="w")
        
        # Antal strategier
        num_strategies = len(self.ai_system.global_router.strategies) if self.ai_system and self.ai_system.global_router else 0
        strategies_label = ctk.CTkLabel(self.stats_container, text=f"Tillgängliga strategier: {num_strategies}")
        strategies_label.grid(row=2, column=0, padx=10, pady=2, sticky="w")
        
        # Totalt antal instanser
        total_instances = 0
        if self.cluster_manager:
            for cluster in self.cluster_manager.clusters.values():
                total_instances += len(cluster.instances)
                
        instances_label = ctk.CTkLabel(self.stats_container, text=f"Totalt antal instanser: {total_instances}")
        instances_label.grid(row=3, column=0, padx=10, pady=2, sticky="w")
        
        # Logg-antal
        logs_label = ctk.CTkLabel(self.stats_container, text=f"Loggade händelser: {len(self.log_entries)}")
        logs_label.grid(row=4, column=0, padx=10, pady=2, sticky="w")
        
        # Körtid
        if hasattr(self, "start_time"):
            runtime = time.time() - self.start_time
            runtime_label = ctk.CTkLabel(self.stats_container, text=f"Körtid: {runtime:.1f} sekunder")
            runtime_label.grid(row=5, column=0, padx=10, pady=2, sticky="w")
            
    def _schedule_updates(self):
        """Schemalägg periodiska uppdateringar"""
        if not self.running:
            return
            
        try:
            # Uppdatera skärm
            if self.show_screen:
                self.capture_screen()
                
            # Uppdatera klustervyn
            self.update_clusters_view()
            
            # Uppdatera statistik
            self.update_stats_view()
            
            # Processhändelser från AI-systemet om det finns några
            if self.ai_system:
                event = self.ai_system.get_event(timeout=0.1)
                if event:
                    self.add_log_entry(f"🔔 Händelse: {event['type']} - {str(event['data'])[:50]}")
                    
        except Exception as e:
            logging.error(f"Fel vid uppdatering: {e}")
            
        # Schemalägg nästa uppdatering
        if self.running:
            self.after(self.update_interval, self._schedule_updates)
            
    def on_close(self):
        """Hantera stängning av fönstret"""
        if self.running:
            self.stop_system()
            
        self.destroy()
        
# Kör UI
def run_ui():
    try:
        app = UIController()
        app.mainloop()
    except Exception as e:
        logging.error(f"UI-fel: {e}")
        raise
        
if __name__ == "__main__":
    run_ui()