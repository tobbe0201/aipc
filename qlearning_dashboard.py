"""
Q-Learning Dashboard för AI Desktop Controller

Detta modulen implementerar en interaktiv dashboard för visualisering
och övervakning av Q-learning-processen.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import customtkinter as ctk
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Använd Agg backend för att undvika GUI-krav
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("qlearning_dashboard.log"),
        logging.StreamHandler()
    ]
)

class QLearningDashboard(ctk.CTk):
    """Interaktiv dashboard för visualisering av Q-learning"""
    
    def __init__(self):
        super().__init__()
        
        # Konfigurera fönster
        self.title("Q-Learning Dashboard")
        self.geometry("1200x800")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Data för visualisering
        self.q_values = {}  # state_id -> {action_id -> q_value}
        self.rewards_history = []
        self.loss_history = []
        self.epsilon_history = []
        self.action_counts = {}  # action_id -> count
        
        # Aktiva agenter och strategier
        self.active_agents = {}  # agent_id -> agent_metadata
        self.selected_agent_id = None
        
        # Sätt upp UI
        self._setup_ui()
        
        # Uppdateringstråd
        self.update_thread = None
        self.running = False
        
        # Uppdatera från fil om tillgängligt
        self._try_load_initial_data()
        
        logging.info("✅ Q-Learning Dashboard initialiserad")
    
    def _setup_ui(self):
        """Skapa dashboard-gränssnittet"""
        # Huvudcontainer med två paneler
        self.main_container = ctk.CTkFrame(self)
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(1, weight=3)
        self.main_container.grid_rowconfigure(0, weight=1)
        
        # Vänster panel för kontroller
        self.left_panel = ctk.CTkFrame(self.main_container)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self._setup_left_panel()
        
        # Höger panel för visualiseringar
        self.right_panel = ctk.CTkFrame(self.main_container)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self._setup_right_panel()
    
    def _setup_left_panel(self):
        """Skapa vänster kontrollpanel"""
        self.left_panel.grid_columnconfigure(0, weight=1)
        
        # Agent-sektion
        agent_frame = ctk.CTkFrame(self.left_panel)
        agent_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        agent_label = ctk.CTkLabel(agent_frame, text="Aktiva Agenter", font=("Arial", 14, "bold"))
        agent_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        self.agent_listbox = tk.Listbox(agent_frame, height=6, bg="#2b2b2b", fg="white", 
                                      selectbackground="#1f538d")
        self.agent_listbox.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.agent_listbox.bind("<<ListboxSelect>>", self._on_agent_selected)
        
        # Uppdateringsknapp
        refresh_button = ctk.CTkButton(agent_frame, text="🔄 Uppdatera", command=self.refresh_data)
        refresh_button.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Agent-statistik
        stats_frame = ctk.CTkFrame(self.left_panel)
        stats_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        
        stats_label = ctk.CTkLabel(stats_frame, text="Agent-statistik", font=("Arial", 14, "bold"))
        stats_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        self.stats_text = ctk.CTkTextbox(stats_frame, height=200)
        self.stats_text.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Kontrollknappar
        control_frame = ctk.CTkFrame(self.left_panel)
        control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        
        # Spara data-knapp
        save_button = ctk.CTkButton(control_frame, text="💾 Spara data", command=self._save_data_to_file)
        save_button.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Ladda data-knapp
        load_button = ctk.CTkButton(control_frame, text="📂 Ladda data", command=self._load_data_from_file)
        load_button.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Auto-uppdatering
        auto_update_var = tk.BooleanVar(value=False)
        auto_update_cb = ctk.CTkCheckBox(control_frame, text="Auto-uppdatering", 
                                       variable=auto_update_var, command=self._toggle_auto_update)
        auto_update_cb.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        
        # Strategier/Kurvdiagram-val
        view_label = ctk.CTkLabel(control_frame, text="Visa diagram", font=("Arial", 12))
        view_label.grid(row=3, column=0, sticky="w", padx=5, pady=(10, 5))
        
        self.view_var = tk.StringVar(value="rewards")
        views = [
            ("Belöningar", "rewards"),
            ("Förlust", "loss"),
            ("Epsilon", "epsilon"),
            ("Handlingsfördelning", "actions")
        ]
        
        for i, (text, value) in enumerate(views):
            rb = ctk.CTkRadioButton(control_frame, text=text, variable=self.view_var, value=value,
                                  command=self._update_view)
            rb.grid(row=4+i, column=0, sticky="w", padx=5, pady=2)
    
    def _setup_right_panel(self):
        """Skapa höger visualiseringspanel"""
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(1, weight=1)
        
        # Översta stapel för Q-matris
        q_matrix_frame = ctk.CTkFrame(self.right_panel)
        q_matrix_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        q_matrix_frame.grid_columnconfigure(0, weight=1)
        q_matrix_frame.grid_rowconfigure(0, weight=0)
        q_matrix_frame.grid_rowconfigure(1, weight=1)
        
        q_matrix_label = ctk.CTkLabel(q_matrix_frame, text="Q-värdesmatris", font=("Arial", 14, "bold"))
        q_matrix_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        # Skapa en matplotlib-figur för Q-matris
        self.q_matrix_fig = Figure(figsize=(6, 4), dpi=100)
        self.q_matrix_ax = self.q_matrix_fig.add_subplot(111)
        
        self.q_matrix_canvas = FigureCanvasTkAgg(self.q_matrix_fig, q_matrix_frame)
        self.q_matrix_canvas.draw()
        self.q_matrix_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Nedre stapel för belöningar, förlust och epsilon
        charts_frame = ctk.CTkFrame(self.right_panel)
        charts_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        charts_frame.grid_columnconfigure(0, weight=1)
        charts_frame.grid_rowconfigure(0, weight=0)
        charts_frame.grid_rowconfigure(1, weight=1)
        
        self.charts_label = ctk.CTkLabel(charts_frame, text="Belöningshistorik", font=("Arial", 14, "bold"))
        self.charts_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        # Skapa en matplotlib-figur för diagrammet
        self.charts_fig = Figure(figsize=(6, 4), dpi=100)
        self.charts_ax = self.charts_fig.add_subplot(111)
        
        self.charts_canvas = FigureCanvasTkAgg(self.charts_fig, charts_frame)
        self.charts_canvas.draw()
        self.charts_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    
    def _on_agent_selected(self, event):
        """Hantera val av agent i listbox"""
        selection = self.agent_listbox.curselection()
        if selection:
            index = selection[0]
            agent_id = list(self.active_agents.keys())[index]
            self.selected_agent_id = agent_id
            self._update_agent_stats()
            self._update_visualizations()
    
    def _update_agent_stats(self):
        """Uppdatera statistik för vald agent"""
        if not self.selected_agent_id or self.selected_agent_id not in self.active_agents:
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert(tk.END, "Ingen agent vald.")
            return
        
        agent = self.active_agents[self.selected_agent_id]
        
        self.stats_text.delete("1.0", tk.END)
        
        # Visa agent-information
        self.stats_text.insert(tk.END, f"Agent ID: {self.selected_agent_id}\n")
        if 'name' in agent:
            self.stats_text.insert(tk.END, f"Namn: {agent['name']}\n")
        if 'type' in agent:
            self.stats_text.insert(tk.END, f"Typ: {agent['type']}\n")
        
        # Visa statistik
        self.stats_text.insert(tk.END, "\n--- Träningsstatistik ---\n")
        
        if len(self.rewards_history) > 0:
            avg_reward = sum(self.rewards_history[-10:]) / min(10, len(self.rewards_history))
            self.stats_text.insert(tk.END, f"Genomsnittlig belöning (senaste 10): {avg_reward:.4f}\n")
        
        if len(self.loss_history) > 0:
            avg_loss = sum(self.loss_history[-10:]) / min(10, len(self.loss_history))
            self.stats_text.insert(tk.END, f"Genomsnittlig förlust (senaste 10): {avg_loss:.4f}\n")
        
        if len(self.epsilon_history) > 0:
            current_epsilon = self.epsilon_history[-1]
            self.stats_text.insert(tk.END, f"Aktuell epsilon: {current_epsilon:.4f}\n")
        
        # Visa handlingsstatistik
        if self.action_counts:
            self.stats_text.insert(tk.END, "\n--- Handlingsstatistik ---\n")
            total_actions = sum(self.action_counts.values())
            for action, count in sorted(self.action_counts.items()):
                percentage = (count / total_actions) * 100 if total_actions > 0 else 0
                self.stats_text.insert(tk.END, f"{action}: {count} ({percentage:.1f}%)\n")
    
    def _update_visualizations(self):
        """Uppdatera alla visualiseringar"""
        self._update_q_matrix()
        self._update_view()
    
    def _update_q_matrix(self):
        """Uppdatera Q-matrisvisualisering"""
        if not self.q_values:
            return
        
        # Rensa tidigare visualisering
        self.q_matrix_ax.clear()
        
        # Konvertera Q-värden till en matris
        states = list(self.q_values.keys())
        if not states:
            return
            
        # Ta de första 10 tillstånden för läsbarhet
        states = states[:10]
        
        # Hämta alla handlingar från Q-värde-dictionaryn
        all_actions = set()
        for state in states:
            all_actions.update(self.q_values[state].keys())
        actions = sorted(list(all_actions))
        
        # Skapa Q-matris
        q_matrix = np.zeros((len(states), len(actions)))
        for i, state in enumerate(states):
            for j, action in enumerate(actions):
                if action in self.q_values[state]:
                    q_matrix[i, j] = self.q_values[state][action]
        
        # Skapa visualisering
        cmap = plt.cm.viridis
        im = self.q_matrix_ax.imshow(q_matrix, cmap=cmap)
        
        # Lägg till färgstapel
        self.q_matrix_fig.colorbar(im, ax=self.q_matrix_ax)
        
        # Konfigurera axlar
        self.q_matrix_ax.set_xticks(np.arange(len(actions)))
        self.q_matrix_ax.set_yticks(np.arange(len(states)))
        
        # Förkorta långa tillstånds-ID för läsbarhet
        short_states = [f"S{i}" for i in range(len(states))]
        short_actions = [f"A{i}" for i in range(len(actions))]
        
        self.q_matrix_ax.set_xticklabels(short_actions)
        self.q_matrix_ax.set_yticklabels(short_states)
        
        # Rotera x-labels för bättre läsbarhet
        plt.setp(self.q_matrix_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Lägg till värden i cellerna
        for i in range(len(states)):
            for j in range(len(actions)):
                text_color = "black" if q_matrix[i, j] > np.max(q_matrix) / 2 else "white"
                self.q_matrix_ax.text(j, i, f"{q_matrix[i, j]:.2f}", ha="center", va="center", color=text_color)
        
        self.q_matrix_ax.set_title("Q-värdesmatris")
        self.q_matrix_fig.tight_layout()
        self.q_matrix_canvas.draw()
    
    def _update_view(self):
        """Uppdatera diagrammet baserat på vald vy"""
        view_type = self.view_var.get()
        
        # Uppdatera etikett
        view_titles = {
            "rewards": "Belöningshistorik",
            "loss": "Förlusthistorik",
            "epsilon": "Epsilon-utveckling",
            "actions": "Handlingsfördelning"
        }
        self.charts_label.configure(text=view_titles[view_type])
        
        # Rensa tidigare diagram
        self.charts_ax.clear()
        
        if view_type == "rewards":
            if self.rewards_history:
                self.charts_ax.plot(self.rewards_history, color='green')
                self.charts_ax.set_xlabel('Episode')
                self.charts_ax.set_ylabel('Belöning')
                self.charts_ax.set_title('Belöning över tid')
                
                # Beräkna glidande medelvärde för att visa trend
                if len(self.rewards_history) > 10:
                    rolling_avg = np.convolve(self.rewards_history, np.ones(10)/10, mode='valid')
                    self.charts_ax.plot(range(9, len(self.rewards_history)), rolling_avg, color='red', linewidth=2)
                    self.charts_ax.legend(['Belöning', '10-ep glidande medelvärde'])
                
        elif view_type == "loss":
            if self.loss_history:
                self.charts_ax.plot(self.loss_history, color='red')
                self.charts_ax.set_xlabel('Träningssteg')
                self.charts_ax.set_ylabel('Förlust')
                self.charts_ax.set_title('Träningsförlust över tid')
                
                # Beräkna glidande medelvärde för att visa trend
                if len(self.loss_history) > 10:
                    rolling_avg = np.convolve(self.loss_history, np.ones(10)/10, mode='valid')
                    self.charts_ax.plot(range(9, len(self.loss_history)), rolling_avg, color='blue', linewidth=2)
                    self.charts_ax.legend(['Förlust', '10-steg glidande medelvärde'])
                
        elif view_type == "epsilon":
            if self.epsilon_history:
                self.charts_ax.plot(self.epsilon_history, color='blue')
                self.charts_ax.set_xlabel('Episode')
                self.charts_ax.set_ylabel('Epsilon')
                self.charts_ax.set_title('Epsilon-utveckling (utforskningsgrad)')
                self.charts_ax.set_ylim(0, 1)
                
        elif view_type == "actions":
            if self.action_counts:
                actions = list(self.action_counts.keys())
                counts = list(self.action_counts.values())
                
                # Sortera efter antal
                sorted_indices = np.argsort(counts)[::-1]
                actions = [actions[i] for i in sorted_indices]
                counts = [counts[i] for i in sorted_indices]
                
                # Visa bara de 10 vanligaste handlingarna för läsbarhet
                if len(actions) > 10:
                    actions = actions[:10]
                    counts = counts[:10]
                
                # Använd en annan färgpalett för staplarna
                colors = plt.cm.viridis(np.linspace(0, 1, len(actions)))
                
                self.charts_ax.bar(actions, counts, color=colors)
                self.charts_ax.set_xlabel('Handling')
                self.charts_ax.set_ylabel('Antal')
                self.charts_ax.set_title('Handlingsfördelning')
                
                # Rotera x-labels för bättre läsbarhet
                plt.setp(self.charts_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        self.charts_fig.tight_layout()
        self.charts_canvas.draw()
    
    def _toggle_auto_update(self):
        """Slå på/av automatisk uppdatering"""
        self.running = not self.running
        
        if self.running:
            if self.update_thread is None or not self.update_thread.is_alive():
                self.update_thread = threading.Thread(target=self._auto_update_loop)
                self.update_thread.daemon = True
                self.update_thread.start()
                logging.info("🔄 Auto-uppdatering startad")
        else:
            logging.info("⏹️ Auto-uppdatering stoppad")
    
    def _auto_update_loop(self):
        """Loop för automatisk uppdatering"""
        while self.running:
            try:
                self.refresh_data()
                time.sleep(5)  # Uppdatera var 5:e sekund
            except Exception as e:
                logging.error(f"Fel i auto-uppdatering: {e}")
    
    def refresh_data(self):
        """Uppdatera data från aktiva agenter"""
        # I en faktisk implementation skulle detta hämta data från de aktiva agenterna
        # För demo, titta på senaste filen i data/stats-katalogen
        
        # Finn senaste stats-filen
        stats_dir = "data/stats"
        if os.path.exists(stats_dir):
            stats_files = [os.path.join(stats_dir, f) for f in os.listdir(stats_dir) 
                         if f.startswith("training_stats_") and f.endswith(".json")]
            
            if stats_files:
                # Sortera efter modifieringstid (senaste först)
                stats_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                latest_file = stats_files[0]
                
                self._load_data_from_file(latest_file)
                logging.info(f"📊 Data uppdaterad från {latest_file}")
            else:
                logging.warning("⚠️ Inga stats-filer hittades.")
        else:
            logging.warning(f"⚠️ Katalogen {stats_dir} existerar inte.")
            
        # Uppdatera UI
        self._update_agent_list()
        if self.selected_agent_id:
            self._update_agent_stats()
            self._update_visualizations()
    
    def _update_agent_list(self):
        """Uppdatera listan över aktiva agenter"""
        self.agent_listbox.delete(0, tk.END)
        
        for agent_id, agent in self.active_agents.items():
            name = agent.get('name', agent_id)
            self.agent_listbox.insert(tk.END, name)
            
        # Behåll markering om möjligt
        if self.selected_agent_id in self.active_agents:
            index = list(self.active_agents.keys()).index(self.selected_agent_id)
            self.agent_listbox.selection_set(index)
    
    def _save_data_to_file(self, filename=None):
        """Spara dashboard-data till fil"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"data/stats/dashboard_state_{timestamp}.json"
            
            # Säkerställ att katalogen finns
            os.makedirs("data/stats", exist_ok=True)
        
        data = {
            'active_agents': self.active_agents,
            'q_values': self.q_values,
            'rewards_history': self.rewards_history,
            'loss_history': self.loss_history,
            'epsilon_history': self.epsilon_history,
            'action_counts': self.action_counts,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        logging.info(f"💾 Dashboard-tillstånd sparat till {filename}")
        
        # Visa bekräftelse
        status_label = ctk.CTkLabel(self, text=f"✅ Data sparad till {filename}")
        status_label.place(relx=0.5, rely=0.95, anchor=tk.CENTER)
        
        # Ta bort bekräftelse efter 3 sekunder
        self.after(3000, status_label.destroy)
    
    def _load_data_from_file(self, filename=None):
        """Ladda dashboard-data från fil"""
        if filename is None:
            # Finns det någon tidigare fil?
            stats_dir = "data/stats"
            if os.path.exists(stats_dir):
                dashboard_files = [os.path.join(stats_dir, f) for f in os.listdir(stats_dir) 
                                 if f.startswith("dashboard_state_") and f.endswith(".json")]
                
                if dashboard_files:
                    # Sortera efter modifieringstid (senaste först)
                    dashboard_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    filename = dashboard_files[0]
        
        if filename and os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Uppdatera data
                if 'active_agents' in data:
                    self.active_agents = data['active_agents']
                if 'q_values' in data:
                    self.q_values = data['q_values']
                if 'rewards_history' in data:
                    self.rewards_history = data['rewards_history']
                if 'loss_history' in data:
                    self.loss_history = data['loss_history']
                if 'epsilon_history' in data:
                    self.epsilon_history = data['epsilon_history']
                if 'action_counts' in data:
                    self.action_counts = data['action_counts']
                
                logging.info(f"📂 Data laddad från {filename}")
                
                # Uppdatera UI
                self._update_agent_list()
                if self.selected_agent_id in self.active_agents:
                    self._update_agent_stats()
                self._update_visualizations()
                
                # Visa bekräftelse
                status_label = ctk.CTkLabel(self, text=f"✅ Data laddad från {filename}")
                status_label.place(relx=0.5, rely=0.95, anchor=tk.CENTER)
                
                # Ta bort bekräftelse efter 3 sekunder
                self.after(3000, status_label.destroy)
                
                return True
                
            except Exception as e:
                logging.error(f"Fel vid laddning av data: {e}")
                return False
        
        logging.warning(f"⚠️ Ingen fil att ladda data från.")
        return False
        
    def _try_load_initial_data(self):
        """Försök ladda initial data vid start"""
        self._load_data_from_file()
        
        # Om ingen data laddades, skapa dummy-data för demo
        if not self.active_agents:
            self._create_demo_data()
            logging.info("📊 Demo-data skapad")
    
    def _create_demo_data(self):
        """Skapa demo-data för visualisering"""
        # Skapa en dummy-agent
        self.active_agents = {
            'agent_001': {
                'name': 'Demo Q-Learning Agent',
                'type': 'DQN',
                'state_dim': 10,
                'action_dim': 6,
                'hidden_dim': 64
            }
        }
        
        # Skapa dummy Q-värden
        self.q_values = {}
        for i in range(10):
            state_id = f"state_{i}"
            self.q_values[state_id] = {}
            for j in range(6):
                action_id = f"action_{j}"
                # Skapa lite intressanta mönster i Q-värdena
                if i % 2 == 0 and j % 2 == 0:
                    self.q_values[state_id][action_id] = np.random.uniform(0.7, 1.0)
                elif i % 2 == 1 and j % 2 == 1:
                    self.q_values[state_id][action_id] = np.random.uniform(0.5, 0.8)
                else:
                    self.q_values[state_id][action_id] = np.random.uniform(0.0, 0.3)
        
        # Skapa dummy belöningshistorik
        # Simulera ökande trend med lite brus
        self.rewards_history = []
        baseline = -100
        for i in range(100):
            baseline += np.random.uniform(0.5, 1.5)
            reward = baseline + np.random.normal(0, 10)
            self.rewards_history.append(reward)
        
        # Skapa dummy förlusthistorik
        # Simulera minskande trend med lite brus
        self.loss_history = []
        baseline = 100
        for i in range(200):
            baseline *= 0.98
            loss = baseline + np.random.normal(0, baseline * 0.1)
            self.loss_history.append(max(0, loss))
        
        # Skapa dummy epsilon-historik
        # Simulera exponentiell minskning
        self.epsilon_history = []
        epsilon = 1.0
        for i in range(100):
            self.epsilon_history.append(epsilon)
            epsilon *= 0.97
        
        # Skapa handlingsstatistik
        self.action_counts = {
            'action_0': 245,
            'action_1': 189,
            'action_2': 312,
            'action_3': 156,
            'action_4': 88,
            'action_5': 45
        }
        
        # Välj första agenten
        self.selected_agent_id = 'agent_001'

# Funktion för att starta dashboarden
def run_dashboard():
    dashboard = QLearningDashboard()
    dashboard.mainloop()

if __name__ == "__main__":
    run_dashboard()