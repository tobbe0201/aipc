"""
Automation Builder - Visuellt gränssnitt för att bygga automationssekvenser med drag-and-drop

Detta modul tillhandahåller ett grafiskt gränssnitt där användaren kan:
- Dra och släppa aktionsblock för att skapa en automationssekvens
- Konfigurera varje block med specifika parametrar
- Spara, ladda och köra automationssekvenser
- Testa steg för steg
"""

import os
import json
import time
import logging
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import threading

# Importera automationsmotor
from automation_engine import AutomationEngine, Action, Condition, Sequence

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("automation_builder.log"),
        logging.StreamHandler()
    ]
)

# Standardsökväg för att spara automationer
DEFAULT_AUTOMATION_PATH = "data/automations"
os.makedirs(DEFAULT_AUTOMATION_PATH, exist_ok=True)

# Aktionstyper
ACTION_TYPES = {
    "click": {
        "name": "Klicka",
        "description": "Klicka på en specifik position eller UI-element",
        "icon": "🖱️",
        "parameters": [
            {"name": "target", "type": "element", "description": "Element att klicka på"},
            {"name": "button", "type": "choice", "options": ["left", "right", "middle"], "default": "left", "description": "Musknapp att använda"}
        ]
    },
    "type": {
        "name": "Skriv text",
        "description": "Skriv text i ett textfält",
        "icon": "⌨️",
        "parameters": [
            {"name": "text", "type": "text", "description": "Text att skriva"},
            {"name": "target", "type": "element", "description": "Element att skriva i (valfritt)"}
        ]
    },
    "wait": {
        "name": "Vänta",
        "description": "Vänta en specifik tid eller tills ett element visas",
        "icon": "⏱️",
        "parameters": [
            {"name": "time", "type": "number", "default": 1, "description": "Tid att vänta (sekunder)"},
            {"name": "element", "type": "element", "description": "Element att vänta på (valfritt)"}
        ]
    },
    "find": {
        "name": "Hitta element",
        "description": "Hitta ett UI-element på skärmen",
        "icon": "🔍",
        "parameters": [
            {"name": "text", "type": "text", "description": "Text att söka efter"},
            {"name": "element_type", "type": "choice", "options": ["button", "text", "any"], "default": "any", "description": "Typ av element att hitta"}
        ]
    },
    "if": {
        "name": "Om",
        "description": "Villkorlig förgrening",
        "icon": "🔀",
        "parameters": [
            {"name": "condition", "type": "condition", "description": "Villkor att kontrollera"},
            {"name": "then_actions", "type": "actions", "description": "Handlingar att utföra om villkoret är sant"}
        ]
    },
    "loop": {
        "name": "Upprepa",
        "description": "Upprepa en sekvens av handlingar",
        "icon": "🔄",
        "parameters": [
            {"name": "count", "type": "number", "default": 3, "description": "Antal upprepningar"},
            {"name": "actions", "type": "actions", "description": "Handlingar att upprepa"}
        ]
    },
    "capture": {
        "name": "Ta skärmdump",
        "description": "Ta en skärmdump och spara den",
        "icon": "📷",
        "parameters": [
            {"name": "filename", "type": "text", "default": "screenshot", "description": "Filnamn (utan filändelse)"}
        ]
    },
    "keyboard": {
        "name": "Tangentbordskommando",
        "description": "Utför ett tangentbordskommando",
        "icon": "🔣",
        "parameters": [
            {"name": "keys", "type": "text", "description": "Tangenter att trycka (t.ex. 'ctrl+c')"}
        ]
    },
    "ocr": {
        "name": "Läs text (OCR)",
        "description": "Extrahera text från skärmen med OCR",
        "icon": "👁️",
        "parameters": [
            {"name": "region", "type": "region", "description": "Region att läsa (valfritt, hela skärmen om tom)"},
            {"name": "variable", "type": "text", "default": "extracted_text", "description": "Variabelnamn att spara resultatet i"}
        ]
    },
    "listen": {
        "name": "Lyssna efter ljud",
        "description": "Vänta på ett specifikt ljudmönster",
        "icon": "🔊",
        "parameters": [
            {"name": "sound_type", "type": "choice", "options": ["alert", "notification", "any"], "default": "any", "description": "Typ av ljud att lyssna efter"},
            {"name": "timeout", "type": "number", "default": 10, "description": "Maximal väntetid (sekunder)"}
        ]
    }
}

# Färger för olika aktionstyper
ACTION_COLORS = {
    "click": "#4CAF50",     # Grön
    "type": "#2196F3",      # Blå
    "wait": "#FF9800",      # Orange
    "find": "#9C27B0",      # Lila
    "if": "#F44336",        # Röd
    "loop": "#607D8B",      # Blågrå
    "capture": "#00BCD4",   # Cyan
    "keyboard": "#795548",  # Brun
    "ocr": "#3F51B5",       # Indigo
    "listen": "#E91E63"     # Rosa
}

class AutomationActionBlock(ctk.CTkFrame):
    """Widget som representerar ett aktionsblock i automationsbyggaren"""
    
    def __init__(self, parent, action_type, config=None, **kwargs):
        """
        Initiera ett aktionsblock
        
        Args:
            parent: Förälderwidget
            action_type: Typ av aktion ("click", "type", etc.)
            config: Konfiguration för aktionen (dictionary)
        """
        # Bestäm bakgrundsfärg baserat på aktionstyp
        bg_color = ACTION_COLORS.get(action_type, "#CCCCCC")
        
        super().__init__(parent, fg_color=bg_color, corner_radius=10, **kwargs)
        
        self.parent = parent
        self.action_type = action_type
        self.config = config or {}
        self.selected = False
        
        # Hämta aktionsinformation
        self.action_info = ACTION_TYPES.get(action_type, {
            "name": f"Okänd ({action_type})",
            "description": "Okänd aktionstyp",
            "icon": "❓",
            "parameters": []
        })
        
        # Skapa UI
        self._create_widgets()
        
        # Konfigurera drag-and-drop
        self._setup_drag_drop()
        
        # Ställ in parametrar om konfiguration finns
        if config:
            self._set_parameters_from_config()
    
    def _create_widgets(self):
        """Skapa och placera widgets"""
        # Huvudram med padding
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Titelrad
        self.title_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.title_frame.pack(fill="x", pady=2)
        
        # Ikon och titel
        self.icon_label = ctk.CTkLabel(self.title_frame, text=self.action_info["icon"], font=("Arial", 14, "bold"))
        self.icon_label.pack(side="left", padx=5)
        
        self.title_label = ctk.CTkLabel(self.title_frame, text=self.action_info["name"], font=("Arial", 12, "bold"))
        self.title_label.pack(side="left", padx=5)
        
        # Handtagsikon för drag-and-drop
        self.handle_label = ctk.CTkLabel(self.title_frame, text="⣿", font=("Arial", 14))
        self.handle_label.pack(side="right", padx=5)
        
        # Parameterfram
        self.param_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.param_frame.pack(fill="x", pady=5)
        
        # Skapa parameterelement
        self.param_widgets = {}
        
        for i, param in enumerate(self.action_info["parameters"]):
            param_name = param["name"]
            param_type = param["type"]
            param_desc = param.get("description", "")
            
            # Etikett
            label = ctk.CTkLabel(self.param_frame, text=f"{param_desc}:")
            label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
            
            # Olika widgets beroende på parametertyp
            if param_type == "text":
                widget = ctk.CTkEntry(self.param_frame, placeholder_text=param.get("default", ""))
                widget.grid(row=i, column=1, sticky="ew", padx=5, pady=2)
                
            elif param_type == "number":
                widget = ctk.CTkEntry(self.param_frame, placeholder_text=str(param.get("default", 0)))
                widget.grid(row=i, column=1, sticky="ew", padx=5, pady=2)
                
            elif param_type == "choice":
                options = param.get("options", [])
                widget = ctk.CTkOptionMenu(self.param_frame, values=options)
                if "default" in param:
                    widget.set(param["default"])
                widget.grid(row=i, column=1, sticky="ew", padx=5, pady=2)
                
            elif param_type == "element":
                # För element, skapa en kombinerad widget
                frame = ctk.CTkFrame(self.param_frame)
                frame.grid(row=i, column=1, sticky="ew", padx=5, pady=2)
                
                entry = ctk.CTkEntry(frame, placeholder_text="Elementbeskrivning")
                entry.pack(side="left", fill="x", expand=True)
                
                pick_btn = ctk.CTkButton(frame, text="🔍", width=30, command=lambda p=param_name: self._pick_element(p))
                pick_btn.pack(side="right", padx=2)
                
                widget = {"frame": frame, "entry": entry, "button": pick_btn}
                
            elif param_type == "region":
                # För region, skapa en kombinerad widget
                frame = ctk.CTkFrame(self.param_frame)
                frame.grid(row=i, column=1, sticky="ew", padx=5, pady=2)
                
                entry = ctk.CTkEntry(frame, placeholder_text="x,y,bredd,höjd")
                entry.pack(side="left", fill="x", expand=True)
                
                pick_btn = ctk.CTkButton(frame, text="⬜", width=30, command=lambda p=param_name: self._pick_region(p))
                pick_btn.pack(side="right", padx=2)
                
                widget = {"frame": frame, "entry": entry, "button": pick_btn}
                
            elif param_type == "condition":
                # För villkor, skapa en redigeringsknapp
                widget = ctk.CTkButton(self.param_frame, text="Redigera villkor...", 
                                     command=lambda p=param_name: self._edit_condition(p))
                widget.grid(row=i, column=1, sticky="ew", padx=5, pady=2)
                
            elif param_type == "actions":
                # För underaktioner, skapa en redigeringsknapp
                widget = ctk.CTkButton(self.param_frame, text="Redigera handlingar...", 
                                     command=lambda p=param_name: self._edit_actions(p))
                widget.grid(row=i, column=1, sticky="ew", padx=5, pady=2)
                
            else:
                # Fallback för okända typer
                widget = ctk.CTkEntry(self.param_frame)
                widget.grid(row=i, column=1, sticky="ew", padx=5, pady=2)
            
            # Spara widget i dictionary
            self.param_widgets[param_name] = widget
        
        # Knapprad
        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.button_frame.pack(fill="x", pady=2)
        
        self.delete_button = ctk.CTkButton(self.button_frame, text="🗑️", width=30, 
                                         command=self._delete, fg_color="#F44336", hover_color="#D32F2F")
        self.delete_button.pack(side="right", padx=2)
        
        self.duplicate_button = ctk.CTkButton(self.button_frame, text="📋", width=30, 
                                            command=self._duplicate)
        self.duplicate_button.pack(side="right", padx=2)
    
    def _setup_drag_drop(self):
        """Konfigurera drag-and-drop-funktionalitet"""
        # Händelsehanterare för drag-and-drop
        self.bind("<Button-1>", self._on_press)
        self.title_frame.bind("<Button-1>", self._on_press)
        self.handle_label.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<B1-Motion>", self._on_motion)
        
        # Flaggor för drag-and-drop
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.dragging = False
    
    def _on_press(self, event):
        """Hantera musklick på blocket"""
        # Spara startposition för drag
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
        # Markera som vald
        self.select()
    
    def _on_motion(self, event):
        """Hantera musrörelse under drag"""
        if not self.dragging:
            # Kontrollera om vi har flyttat tillräckligt för att starta drag
            if abs(event.x - self.drag_start_x) > 5 or abs(event.y - self.drag_start_y) > 5:
                self.dragging = True
                
                # Skapa en klon för att visa under drag
                self.drag_image = ctk.CTkToplevel(self)
                self.drag_image.overrideredirect(True)
                self.drag_image.attributes("-alpha", 0.7)
                self.drag_image.attributes("-topmost", True)
                
                # Kopiera innehåll
                clone = ctk.CTkLabel(self.drag_image, text=f"{self.action_info['icon']} {self.action_info['name']}")
                clone.pack(padx=10, pady=5)
                
                # Placera vid muspekaren
                self.drag_image.geometry(f"+{event.x_root}+{event.y_root}")
        
        if self.dragging:
            # Uppdatera position för drag-bild
            self.drag_image.geometry(f"+{event.x_root}+{event.y_root}")
            
            # Låt föräldern veta att vi drar
            if hasattr(self.parent, "on_block_drag"):
                self.parent.on_block_drag(self, event)
    
    def _on_release(self, event):
        """Hantera musuppsläpp efter drag"""
        if self.dragging:
            # Stäng drag-bild
            self.drag_image.destroy()
            self.dragging = False
            
            # Meddela föräldern om drop
            if hasattr(self.parent, "on_block_drop"):
                self.parent.on_block_drop(self, event)
    
    def _set_parameters_from_config(self):
        """Ställ in parametervärden från konfigurationen"""
        for param_name, value in self.config.items():
            if param_name in self.param_widgets:
                widget = self.param_widgets[param_name]
                
                # Olika hantering beroende på widgettyp
                if isinstance(widget, ctk.CTkEntry):
                    widget.delete(0, "end")
                    widget.insert(0, str(value))
                elif isinstance(widget, ctk.CTkOptionMenu):
                    widget.set(value)
                elif isinstance(widget, dict) and "entry" in widget:
                    # För sammansatta widgets som element och region
                    widget["entry"].delete(0, "end")
                    widget["entry"].insert(0, str(value))
    
    def get_config(self):
        """
        Hämta aktionskonfiguration från UI
        
        Returns:
            dict: Konfiguration för aktionen
        """
        config = {"type": self.action_type}
        
        # Hämta parametervärden
        for param in self.action_info["parameters"]:
            param_name = param["name"]
            param_type = param["type"]
            
            if param_name in self.param_widgets:
                widget = self.param_widgets[param_name]
                
                # Olika hantering beroende på widgettyp
                if isinstance(widget, ctk.CTkEntry):
                    value = widget.get()
                    
                    # Konvertera till rätt typ
                    if param_type == "number":
                        try:
                            value = float(value)
                            # Om heltal, konvertera till int
                            if value.is_integer():
                                value = int(value)
                        except ValueError:
                            value = param.get("default", 0)
                            
                    config[param_name] = value
                    
                elif isinstance(widget, ctk.CTkOptionMenu):
                    config[param_name] = widget.get()
                    
                elif isinstance(widget, dict) and "entry" in widget:
                    # För sammansatta widgets som element och region
                    value = widget["entry"].get()
                    
                    if param_type == "region" and value:
                        # Försök tolka region som x,y,w,h
                        try:
                            parts = [int(p.strip()) for p in value.split(",")]
                            if len(parts) == 4:
                                config[param_name] = parts
                        except ValueError:
                            pass
                    else:
                        config[param_name] = value
                        
                elif param_type in ["condition", "actions"]:
                    # För villkor och underaktioner, använd befintliga värden
                    if param_name in self.config:
                        config[param_name] = self.config[param_name]
        
        return config
    
    def select(self):
        """Markera blocket som valt"""
        # Ändra utseende för att visa att det är valt
        self.selected = True
        self.configure(border_width=2, border_color="#FFD700")
        
        # Meddela föräldern
        if hasattr(self.parent, "on_block_select"):
            self.parent.on_block_select(self)
    
    def deselect(self):
        """Avmarkera blocket"""
        self.selected = False
        self.configure(border_width=0)
    
    def _delete(self):
        """Ta bort blocket"""
        # Meddela föräldern
        if hasattr(self.parent, "on_block_delete"):
            self.parent.on_block_delete(self)
        
        # Ta bort från UI
        self.destroy()
    
    def _duplicate(self):
        """Duplicera blocket"""
        # Hämta aktuell konfiguration
        config = self.get_config()
        
        # Meddela föräldern
        if hasattr(self.parent, "on_block_duplicate"):
            self.parent.on_block_duplicate(self, config)
    
    def _pick_element(self, param_name):
        """Öppna elementväljare"""
        # Detta skulle anropa en funktion för att välja ett element visuellt
        # För nu, simulera med en messagebox
        messagebox.showinfo("Elementväljare", 
                          "I en fullständig implementation skulle detta öppna "
                          "en elementväljare för att klicka på skärmen och välja ett element.")
                          
        # Exempel på hur det skulle kunna fungera
        if hasattr(self.parent, "on_pick_element"):
            element_id = self.parent.on_pick_element()
            if element_id:
                self.param_widgets[param_name]["entry"].delete(0, "end")
                self.param_widgets[param_name]["entry"].insert(0, element_id)
    
    def _pick_region(self, param_name):
        """Öppna regionväljare"""
        # Detta skulle anropa en funktion för att välja en region visuellt
        # För nu, simulera med en messagebox
        messagebox.showinfo("Regionväljare", 
                          "I en fullständig implementation skulle detta öppna "
                          "en regionväljare för att rita en rektangel på skärmen.")
                          
        # Exempel på hur det skulle kunna fungera
        if hasattr(self.parent, "on_pick_region"):
            region = self.parent.on_pick_region()
            if region:
                region_str = ",".join(str(x) for x in region)
                self.param_widgets[param_name]["entry"].delete(0, "end")
                self.param_widgets[param_name]["entry"].insert(0, region_str)
    
    def _edit_condition(self, param_name):
        """Öppna villkorsredigerare"""
        # Detta skulle öppna en dialogruta för att redigera ett villkor
        # För nu, simulera med en enkel dialogruta
        
        # Skapa en dialogruta
        dialog = ctk.CTkToplevel(self)
        dialog.title("Redigera villkor")
        dialog.geometry("400x300")
        dialog.grab_set()
        
        # Lägg till lite exempel på villkor
        ctk.CTkLabel(dialog, text="Välj villkor:").pack(pady=10)
        
        condition_var = tk.StringVar()
        conditions = [
            "Element finns på skärmen",
            "Text finns på skärmen",
            "Bild matchar",
            "Variabel == värde"
        ]
        
        for condition in conditions:
            ctk.CTkRadioButton(dialog, text=condition, variable=condition_var, value=condition).pack(anchor="w", padx=20, pady=5)
        
        # Om vi har ett befintligt villkor, ställ in det
        existing_condition = self.config.get(param_name, {})
        if existing_condition and "type" in existing_condition:
            condition_type = existing_condition["type"]
            if condition_type == "element_exists":
                condition_var.set("Element finns på skärmen")
            elif condition_type == "text_exists":
                condition_var.set("Text finns på skärmen")
        
        # OK-knapp
        ctk.CTkButton(dialog, text="OK", command=dialog.destroy).pack(pady=20)
        
        # Vänta tills dialogrutan stängs
        self.wait_window(dialog)
        
        # Uppdatera konfigurationen
        selected = condition_var.get()
        if selected:
            if selected == "Element finns på skärmen":
                self.config[param_name] = {"type": "element_exists", "element": "element_id"}
            elif selected == "Text finns på skärmen":
                self.config[param_name] = {"type": "text_exists", "text": "text att söka"}
            elif selected == "Bild matchar":
                self.config[param_name] = {"type": "image_match", "image": "image.png", "threshold": 0.8}
            elif selected == "Variabel == värde":
                self.config[param_name] = {"type": "variable_equals", "variable": "var_name", "value": "värde"}
    
    def _edit_actions(self, param_name):
        """Öppna underaktionsredigerare"""
        # Detta skulle öppna en dialogruta för att redigera underaktioner
        # För nu, simulera med en enkel dialogruta
        
        # Skapa en dialogruta
        dialog = ctk.CTkToplevel(self)
        dialog.title("Redigera underaktioner")
        dialog.geometry("600x400")
        dialog.grab_set()
        
        # Skapa en mini-version av byggaren
        ctk.CTkLabel(dialog, text="Lägg till underaktioner:").pack(pady=10)
        
        # Förenkla genom att bara visa en lista
        action_types = ["click", "type", "wait"]
        actions_frame = ctk.CTkFrame(dialog)
        actions_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Befintliga underaktioner
        existing_actions = self.config.get(param_name, [])
        action_vars = []
        
        for action_type in action_types:
            action_info = ACTION_TYPES[action_type]
            var = tk.BooleanVar(value=any(a.get("type") == action_type for a in existing_actions))
            action_vars.append((action_type, var))
            
            ctk.CTkCheckBox(actions_frame, text=f"{action_info['icon']} {action_info['name']}", 
                          variable=var).pack(anchor="w", padx=20, pady=5)
        
        # OK-knapp
        def on_ok():
            # Skapa lista med valda aktioner
            actions = []
            for action_type, var in action_vars:
                if var.get():
                    actions.append({"type": action_type})
            
            # Uppdatera konfigurationen
            self.config[param_name] = actions
            dialog.destroy()
        
        ctk.CTkButton(dialog, text="OK", command=on_ok).pack(pady=20)
        
        # Vänta tills dialogrutan stängs
        self.wait_window(dialog)

class ActionPalette(ctk.CTkFrame):
    """Widget som visar alla tillgängliga aktionstyper för drag-and-drop"""
    
    def __init__(self, parent, on_action_select=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.parent = parent
        self.on_action_select = on_action_select
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Skapa aktionspanelen"""
        # Titelrubrik
        title_label = ctk.CTkLabel(self, text="Aktioner", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Sökfält
        search_frame = ctk.CTkFrame(self)
        search_frame.pack(fill="x", padx=10, pady=5)
        
        search_icon = ctk.CTkLabel(search_frame, text="🔍")
        search_icon.pack(side="left", padx=5)
        
        self.search_entry = ctk.CTkEntry(search_frame, placeholder_text="Sök aktioner...")
        self.search_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.search_entry.bind("<KeyRelease>", self._on_search)
        
        # Aktionskategorierna
        self.categories = {
            "basic": {"name": "Grundläggande", "actions": ["click", "type", "wait"]},
            "advanced": {"name": "Avancerat", "actions": ["find", "if", "loop"]},
            "media": {"name": "Media", "actions": ["capture", "ocr"]},
            "input": {"name": "Inmatning", "actions": ["keyboard", "listen"]}
        }
        
        # Skapa ram för aktioner med scrollbar
        self.action_canvas = ctk.CTkScrollableFrame(self)
        self.action_canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Skapa aktionskategorier
        self.category_frames = {}
        for cat_id, category in self.categories.items():
            frame = self._create_category(cat_id, category)
            self.category_frames[cat_id] = frame
    
    def _create_category(self, category_id, category):
        """Skapa en kategori med aktioner"""
        # Skapa rubrikram
        category_frame = ctk.CTkFrame(self.action_canvas)
        category_frame.pack(fill="x", pady=5)
        
        # Kategorirubrik
        header_frame = ctk.CTkFrame(category_frame, fg_color=("#EEEEEE", "#333333"))
        header_frame.pack(fill="x")
        
        category_label = ctk.CTkLabel(header_frame, text=category["name"], font=("Arial", 12, "bold"))
        category_label.pack(side="left", padx=10, pady=5)
        
        # Aktionskontainer
        actions_frame = ctk.CTkFrame(category_frame)
        actions_frame.pack(fill="x", pady=2)
        
        # Skapa aktionsknappar
        action_buttons = []
        for action_type in category["actions"]:
            if action_type in ACTION_TYPES:
                action_info = ACTION_TYPES[action_type]
                
                btn = self._create_action_button(actions_frame, action_type, action_info)
                action_buttons.append((action_type, btn))
        
        # Ordna knappar i rutnät, 2 per rad
        for i, (action_type, btn) in enumerate(action_buttons):
            row = i // 2
            col = i % 2
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
        
        # Konfigurera kolumner för att vara lika breda
        actions_frame.grid_columnconfigure(0, weight=1)
        actions_frame.grid_columnconfigure(1, weight=1)
        
        return category_frame
    
    def _create_action_button(self, parent, action_type, action_info):
        """Skapa en aktionsknapp"""
        # Hämta färg för aktionstypen
        bg_color = ACTION_COLORS.get(action_type, "#CCCCCC")
        
        btn = ctk.CTkButton(
            parent,
            text=f"{action_info['icon']} {action_info['name']}",
            fg_color=bg_color,
            command=lambda t=action_type: self._on_action_click(t)
        )
        
        # Konfigurera drag-and-drop
        btn.bind("<Button-1>", lambda e, t=action_type: self._on_action_drag_start(e, t))
        btn.bind("<ButtonRelease-1>", lambda e, t=action_type: self._on_action_drag_end(e, t))
        btn.bind("<B1-Motion>", lambda e, t=action_type: self._on_action_drag(e, t))
        
        # Lägg till tooltip
        self._add_tooltip(btn, action_info["description"])
        
        return btn
    
    def _add_tooltip(self, widget, text):
        """Lägg till tooltip för en widget"""
        # Enkel tooltip-implementation
        def enter(event):
            x, y = event.x_root, event.y_root
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x+10}+{y+10}")
            tooltip.attributes("-topmost", True)
            
            label = tk.Label(tooltip, text=text, background="#FFFFCC", relief="solid", borderwidth=1)
            label.pack()
            
            widget.tooltip = tooltip
            
        def leave(event):
            if hasattr(widget, "tooltip"):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
    
    def _on_action_click(self, action_type):
        """Hantera klick på en aktionsknapp"""
        if self.on_action_select:
            self.on_action_select(action_type)
    
    def _on_action_drag_start(self, event, action_type):
        """Starta drag av en aktionsknapp"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.drag_action = action_type
        self.dragging = False
    
    def _on_action_drag(self, event, action_type):
        """Fortsätt drag av en aktionsknapp"""
        if not hasattr(self, "dragging") or not hasattr(self, "drag_start_x"):
            return
            
        if not self.dragging:
            # Kolla om vi har flyttat tillräckligt för att starta drag
            if abs(event.x - self.drag_start_x) > 5 or abs(event.y - self.drag_start_y) > 5:
                self.dragging = True
                
                # Skapa en ghostbild för drag
                self.drag_image = ctk.CTkToplevel(self)
                self.drag_image.overrideredirect(True)
                self.drag_image.attributes("-alpha", 0.7)
                self.drag_image.attributes("-topmost", True)
                
                # Hämta aktionsinformation
                action_info = ACTION_TYPES.get(action_type, {"name": action_type, "icon": "❓"})
                
                # Skapa en etikett för dragbilden
                ghost_label = ctk.CTkLabel(
                    self.drag_image, 
                    text=f"{action_info['icon']} {action_info['name']}",
                    fg_color=ACTION_COLORS.get(action_type, "#CCCCCC"),
                    corner_radius=5
                )
                ghost_label.pack(padx=10, pady=5)
                
                # Placera vid muspekaren
                self.drag_image.geometry(f"+{event.x_root}+{event.y_root}")
        
        if self.dragging:
            # Uppdatera position för drag-bild
            self.drag_image.geometry(f"+{event.x_root}+{event.y_root}")
            
            # Meddela föräldern
            if hasattr(self.parent, "on_action_drag"):
                self.parent.on_action_drag(action_type, event)
    
    def _on_action_drag_end(self, event, action_type):
        """Avsluta drag av en aktionsknapp"""
        if hasattr(self, "dragging") and self.dragging:
            # Stäng drag-bild
            self.drag_image.destroy()
            
            # Meddela föräldern
            if hasattr(self.parent, "on_action_drop"):
                self.parent.on_action_drop(action_type, event)
            
            self.dragging = False
    
    def _on_search(self, event):
        """Sökfunktion för att filtrera aktioner"""
        search_text = self.search_entry.get().lower()
        
        # Visa alla kategorier om söktexten är tom
        if not search_text:
            for frame in self.category_frames.values():
                frame.pack(fill="x", pady=5)
            return
        
        # Genomsök aktioner och visa endast matchande kategorier
        for cat_id, category in self.categories.items():
            frame = self.category_frames[cat_id]
            
            # Kolla om någon aktion i kategorin matchar sökningen
            match_found = False
            for action_type in category["actions"]:
                if action_type in ACTION_TYPES:
                    action_info = ACTION_TYPES[action_type]
                    
                    # Kolla om namn eller beskrivning matchar
                    if (search_text in action_info["name"].lower() or 
                        search_text in action_info["description"].lower()):
                        match_found = True
                        break
            
            # Visa eller dölj kategori baserat på sökresultat
            if match_found:
                frame.pack(fill="x", pady=5)
            else:
                frame.pack_forget()

class AutomationSequenceBuilder(ctk.CTkFrame):
    """Huvudwidget för att bygga automationssekvenser"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.parent = parent
        
        # Aktiva block
        self.blocks = []
        self.selected_block = None
        
        # Automationsmotor för att köra sekvenser
        self.engine = AutomationEngine()
        
        # Skapa UI
        self._create_widgets()
    
    def _create_widgets(self):
        """Skapa huvudwidgets för byggaren"""
        # Konfigurera kolumner och rader
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)
        
        # Aktionspanel (vänster sida)
        self.action_palette = ActionPalette(self, on_action_select=self.add_action)
        self.action_palette.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Huvudområde för sekvensbyggande (höger sida)
        self.sequence_frame = ctk.CTkFrame(self)
        self.sequence_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Rubrik
        header_frame = ctk.CTkFrame(self.sequence_frame)
        header_frame.pack(fill="x", pady=10)
        
        title_label = ctk.CTkLabel(header_frame, text="Automationssekvens", font=("Arial", 16, "bold"))
        title_label.pack(side="left", padx=20)
        
        # Knappar för sekvenshantering
        button_frame = ctk.CTkFrame(header_frame)
        button_frame.pack(side="right", padx=10)
        
        self.new_button = ctk.CTkButton(button_frame, text="Ny", command=self.new_sequence)
        self.new_button.pack(side="left", padx=5)
        
        self.open_button = ctk.CTkButton(button_frame, text="Öppna", command=self.open_sequence)
        self.open_button.pack(side="left", padx=5)
        
        self.save_button = ctk.CTkButton(button_frame, text="Spara", command=self.save_sequence)
        self.save_button.pack(side="left", padx=5)
        
        self.test_button = ctk.CTkButton(button_frame, text="Testa", command=self.test_sequence,
                                       fg_color="#4CAF50", hover_color="#45a049")
        self.test_button.pack(side="left", padx=5)
        
        # Scrollbar för sekvensblock
        self.sequence_canvas = ctk.CTkScrollableFrame(self.sequence_frame)
        self.sequence_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tom platshållare för drag-and-drop
        self.placeholder = ctk.CTkLabel(self.sequence_canvas, text="Dra aktioner hit för att bygga en sekvens",
                                      height=100, fg_color=("#EEEEEE", "#333333"))
        self.placeholder.pack(fill="x", pady=20)
    
    def add_action(self, action_type, config=None, index=None):
        """
        Lägg till en ny aktion i sekvensen
        
        Args:
            action_type: Typ av aktion att lägga till
            config: Konfiguration för aktionen (valfritt)
            index: Position att lägga till vid (valfritt, läggs till sist om None)
        
        Returns:
            AutomationActionBlock: Det skapade aktionsblocket
        """
        # Ta bort platshållare om det är första blocket
        if self.placeholder.winfo_ismapped():
            self.placeholder.pack_forget()
        
        # Skapa ett nytt aktionsblock
        block = AutomationActionBlock(self.sequence_canvas, action_type, config)
        
        # Lägg till i listan över block
        if index is not None and 0 <= index < len(self.blocks):
            self.blocks.insert(index, block)
            
            # Ta bort alla block från UI och lägg till dem igen i rätt ordning
            for b in self.blocks:
                b.pack_forget()
            
            for b in self.blocks:
                b.pack(fill="x", padx=10, pady=5)
        else:
            self.blocks.append(block)
            block.pack(fill="x", padx=10, pady=5)
        
        # Avmarkera eventuellt tidigare valda block
        if self.selected_block:
            self.selected_block.deselect()
        
        # Markera det nya blocket
        self.selected_block = block
        block.select()
        
        return block
    
    def on_block_select(self, block):
        """Hantera när ett block väljs"""
        # Avmarkera tidigare valt block
        if self.selected_block and self.selected_block != block:
            self.selected_block.deselect()
        
        # Uppdatera valt block
        self.selected_block = block
    
    def on_block_delete(self, block):
        """Hantera när ett block tas bort"""
        # Ta bort från listan
        if block in self.blocks:
            self.blocks.remove(block)
        
        # Nullställ valt block om det var detta som togs bort
        if self.selected_block == block:
            self.selected_block = None
        
        # Visa platshållare om inga block finns kvar
        if not self.blocks:
            self.placeholder.pack(fill="x", pady=20)
    
    def on_block_duplicate(self, block, config):
        """Hantera när ett block dupliceras"""
        # Hitta index för originalet
        if block in self.blocks:
            index = self.blocks.index(block)
            
            # Lägg till en kopia efter originalet
            self.add_action(config["type"], config, index + 1)
    
    def on_block_drag(self, block, event):
        """Hantera drag av ett block"""
        # Markera möjliga drop-positioner
        if block in self.blocks:
            pass  # Här skulle vi kunna visa en visuell markör för drop-position
    
    def on_block_drop(self, block, event):
        """Hantera drop av ett block (för att ändra ordning)"""
        if block in self.blocks:
            # Hitta positionen under drop-punkten
            drop_pos = None
            
            # Beräkna muspositionen relativt till sequence_canvas
            canvas_y = event.y_root - self.sequence_canvas.winfo_rooty()
            
            # Hitta block som musen är över
            for i, b in enumerate(self.blocks):
                if b == block:
                    continue
                    
                b_y1 = b.winfo_y()
                b_y2 = b_y1 + b.winfo_height()
                
                if b_y1 <= canvas_y <= b_y2:
                    # Avgör om det ska vara före eller efter detta block
                    if canvas_y < b_y1 + b.winfo_height() / 2:
                        drop_pos = i
                    else:
                        drop_pos = i + 1
                    break
            
            if drop_pos is not None:
                # Flytta blocket
                old_pos = self.blocks.index(block)
                
                # Justera position om blocket flyttas nedåt
                if old_pos < drop_pos:
                    drop_pos -= 1
                
                if old_pos != drop_pos:
                    # Ta bort från gamla positionen
                    self.blocks.pop(old_pos)
                    
                    # Lägg till vid nya positionen
                    self.blocks.insert(drop_pos, block)
                    
                    # Uppdatera UI
                    for b in self.blocks:
                        b.pack_forget()
                    
                    for b in self.blocks:
                        b.pack(fill="x", padx=10, pady=5)
    
    def on_action_drag(self, action_type, event):
        """Hantera drag av en ny aktion från paletten"""
        # Här skulle vi kunna visa en markör för var aktionen kommer att placeras
        pass
    
    def on_action_drop(self, action_type, event):
        """Hantera drop av en ny aktion från paletten"""
        # Lägg till en ny aktion vid drop-position
        self.add_action(action_type)
    
    def on_pick_element(self):
        """Välj ett element på skärmen"""
        # Detta skulle anropa någon form av elementväljarfunktion
        # För nu, returnera bara ett exempel
        return "example_element_id"
    
    def on_pick_region(self):
        """Välj en region på skärmen"""
        # Detta skulle anropa någon form av regionväljarfunktion
        # För nu, returnera bara ett exempel
        return [100, 100, 200, 150]  # x, y, width, height
    
    def get_sequence_config(self):
        """
        Hämta konfiguration för hela sekvensen
        
        Returns:
            list: Lista med aktionskonfigurationer
        """
        return [block.get_config() for block in self.blocks]
    
    def set_sequence_config(self, config):
        """
        Ställ in sekvensen från en konfiguration
        
        Args:
            config: Lista med aktionskonfigurationer
        """
        # Rensa befintliga block
        self.new_sequence()
        
        # Lägg till block från konfiguration
        for action_config in config:
            if "type" in action_config:
                self.add_action(action_config["type"], action_config)
    
    def new_sequence(self):
        """Skapa en ny tom sekvens"""
        # Fråga användaren om befintlig sekvens ska sparas
        if self.blocks:
            if messagebox.askyesno("Skapa ny sekvens", 
                                 "Vill du spara den aktuella sekvensen först?"):
                self.save_sequence()
        
        # Ta bort alla block
        for block in self.blocks:
            block.destroy()
        
        self.blocks = []
        self.selected_block = None
        
        # Visa platshållare
        self.placeholder.pack(fill="x", pady=20)
    
    def open_sequence(self):
        """Öppna en befintlig sekvens från fil"""
        # Fråga användaren om befintlig sekvens ska sparas
        if self.blocks:
            if messagebox.askyesno("Öppna sekvens", 
                                 "Vill du spara den aktuella sekvensen först?"):
                self.save_sequence()
        
        # Visa fil-dialog
        filetypes = [("Automationssekvenser", "*.json"), ("Alla filer", "*.*")]
        filename = filedialog.askopenfilename(
            title="Öppna automationssekvens",
            initialdir=DEFAULT_AUTOMATION_PATH,
            filetypes=filetypes
        )
        
        if not filename:
            return
        
        try:
            # Läs fil
            with open(filename, "r") as f:
                config = json.load(f)
            
            # Ställ in sekvens
            self.set_sequence_config(config)
            
            # Visa meddelande
            messagebox.showinfo("Sekvens öppnad", f"Sekvensen har laddats från {filename}")
            
        except Exception as e:
            messagebox.showerror("Fel", f"Kunde inte öppna filen: {e}")
    
    def save_sequence(self):
        """Spara sekvensen till fil"""
        # Kontrollera om det finns något att spara
        if not self.blocks:
            messagebox.showinfo("Tom sekvens", "Det finns ingen sekvens att spara.")
            return
        
        # Visa fil-dialog
        filetypes = [("Automationssekvenser", "*.json"), ("Alla filer", "*.*")]
        filename = filedialog.asksaveasfilename(
            title="Spara automationssekvens",
            initialdir=DEFAULT_AUTOMATION_PATH,
            filetypes=filetypes,
            defaultextension=".json"
        )
        
        if not filename:
            return
        
        try:
            # Hämta sekvenskonfiguration
            config = self.get_sequence_config()
            
            # Spara till fil
            with open(filename, "w") as f:
                json.dump(config, f, indent=2)
            
            # Visa meddelande
            messagebox.showinfo("Sekvens sparad", f"Sekvensen har sparats till {filename}")
            
        except Exception as e:
            messagebox.showerror("Fel", f"Kunde inte spara filen: {e}")
    
    def test_sequence(self):
        """Testa sekvensen"""
        # Kontrollera om det finns något att testa
        if not self.blocks:
            messagebox.showinfo("Tom sekvens", "Det finns ingen sekvens att testa.")
            return
        
        # Fråga användaren om de vill starta testet
        if not messagebox.askyesno("Testa sekvens", 
                                 "Vill du testa sekvensen? Programmet kommer att utföra "
                                 "handlingar på din dator. Se till att du inte använder "
                                 "datorn under tiden."):
            return
        
        # Hämta sekvenskonfiguration
        config = self.get_sequence_config()
        
        # Konvertera konfiguration till Sequence
        try:
            sequence = Sequence.from_config(config)
        except Exception as e:
            messagebox.showerror("Konfigurationsfel", f"Kunde inte skapa sekvensen: {e}")
            return
        
        # Skapa en statusdialog
        status_dialog = ctk.CTkToplevel(self)
        status_dialog.title("Kör automation")
        status_dialog.geometry("400x300")
        status_dialog.grab_set()
        
        # Statuslabel
        status_label = ctk.CTkLabel(status_dialog, text="Startar automation...", font=("Arial", 14))
        status_label.pack(pady=20)
        
        # Progressbar
        progress = tk.IntVar(value=0)
        progressbar = ctk.CTkProgressBar(status_dialog)
        progressbar.pack(fill="x", padx=40, pady=20)
        progressbar.set(0)
        
        # Detaljer
        details_text = ctk.CTkTextbox(status_dialog, height=150)
        details_text.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Stoppknapp
        stop_button = ctk.CTkButton(status_dialog, text="Avbryt", command=lambda: setattr(self, "stop_requested", True))
        stop_button.pack(pady=10)
        
        # Flagga för att avbryta
        self.stop_requested = False
        
        # Uppdateringsfunktion för status
        def update_status(step, total, message):
            if status_dialog.winfo_exists():
                status_label.configure(text=f"Steg {step}/{total}")
                progressbar.set(step / total)
                details_text.insert("end", f"{message}\n")
                details_text.see("end")
                status_dialog.update()
        
        # Callback för att kontrollera om vi ska avbryta
        def check_stop():
            return self.stop_requested
        
        # Starta testet i en separat tråd
        def run_test():
            try:
                self.engine.run_sequence(sequence, update_status, check_stop)
                
                if not self.stop_requested:
                    # Uppdatera UI när det är klart
                    status_label.configure(text="Automation slutförd!")
                    progressbar.set(1.0)
                    details_text.insert("end", "✅ Sekvensen har körts klart.\n")
                    
                    # Ändra stoppknappen till OK
                    stop_button.configure(text="OK", command=status_dialog.destroy)
                else:
                    # Användaren avbröt
                    status_label.configure(text="Automation avbruten")
                    details_text.insert("end", "⚠️ Sekvensen avbröts av användaren.\n")
                    
                    # Ändra stoppknappen till OK
                    stop_button.configure(text="OK", command=status_dialog.destroy)
                
            except Exception as e:
                # Hantera fel
                if status_dialog.winfo_exists():
                    status_label.configure(text="Fel inträffade!")
                    details_text.insert("end", f"❌ Fel: {str(e)}\n")
                    stop_button.configure(text="OK", command=status_dialog.destroy)
        
        # Starta tråden
        threading.Thread(target=run_test, daemon=True).start()

class AutomationBuilder(ctk.CTk):
    """Huvudfönster för automationsbyggaren"""
    
    def __init__(self):
        super().__init__()
        
        # Ställ in fönster
        self.title("AI Desktop Controller - Automationsbyggare")
        self.geometry("1200x800")
        
        # Skapa meny
        self._create_menu()
        
        # Skapa huvudinnehåll
        self.builder = AutomationSequenceBuilder(self)
        self.builder.pack(fill="both", expand=True, padx=10, pady=10)
    
    def _create_menu(self):
        """Skapa programmets meny"""
        # Skapa huvudmenyn
        menubar = tk.Menu(self)
        
        # Arkivmeny
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Ny sekvens", command=self._new_sequence)
        file_menu.add_command(label="Öppna sekvens...", command=self._open_sequence)
        file_menu.add_command(label="Spara sekvens...", command=self._save_sequence)
        file_menu.add_separator()
        file_menu.add_command(label="Avsluta", command=self.destroy)
        menubar.add_cascade(label="Arkiv", menu=file_menu)
        
        # Redigeringsmeny
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Klipp ut", command=lambda: None)
        edit_menu.add_command(label="Kopiera", command=lambda: None)
        edit_menu.add_command(label="Klistra in", command=lambda: None)
        edit_menu.add_separator()
        edit_menu.add_command(label="Inställningar...", command=self._show_settings)
        menubar.add_cascade(label="Redigera", menu=edit_menu)
        
        # Kör-meny
        run_menu = tk.Menu(menubar, tearoff=0)
        run_menu.add_command(label="Kör sekvens", command=self._run_sequence)
        run_menu.add_command(label="Kör steg för steg", command=self._run_step_by_step)
        run_menu.add_separator()
        run_menu.add_command(label="Schemalägg...", command=self._schedule_sequence)
        menubar.add_cascade(label="Kör", menu=run_menu)
        
        # Hjälpmeny
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Dokumentation", command=self._show_docs)
        help_menu.add_command(label="Exempel", command=self._show_examples)
        help_menu.add_separator()
        help_menu.add_command(label="Om", command=self._show_about)
        menubar.add_cascade(label="Hjälp", menu=help_menu)
        
        # Sätt menyn
        self.config(menu=menubar)
    
    def _new_sequence(self):
        """Skapa en ny sekvens"""
        self.builder.new_sequence()
    
    def _open_sequence(self):
        """Öppna en befintlig sekvens"""
        self.builder.open_sequence()
    
    def _save_sequence(self):
        """Spara sekvensen"""
        self.builder.save_sequence()
    
    def _run_sequence(self):
        """Kör sekvensen"""
        self.builder.test_sequence()
    
    def _run_step_by_step(self):
        """Kör sekvensen steg för steg"""
        # Simulera med en messagebox
        messagebox.showinfo("Steg-för-steg", 
                          "Denna funktion är inte implementerad än. "
                          "Den skulle köra sekvensen steg för steg och pausa mellan varje steg.")
    
    def _schedule_sequence(self):
        """Schemalägg sekvensen"""
        # Simulera med en messagebox
        messagebox.showinfo("Schemaläggning", 
                          "Denna funktion är inte implementerad än. "
                          "Den skulle låta dig schemalägga sekvensen att köras vid specifika tider.")
    
    def _show_settings(self):
        """Visa inställningsdialog"""
        # Simulera med en messagebox
        messagebox.showinfo("Inställningar", 
                          "Denna funktion är inte implementerad än. "
                          "Den skulle visa en dialog för att konfigurera automationsbyggaren.")
    
    def _show_docs(self):
        """Visa dokumentation"""
        # Simulera med en messagebox
        messagebox.showinfo("Dokumentation", 
                          "Denna funktion är inte implementerad än. "
                          "Den skulle visa dokumentation för automationsbyggaren.")
    
    def _show_examples(self):
        """Visa exempelsekvenser"""
        # Simulera med en messagebox
        messagebox.showinfo("Exempel", 
                          "Denna funktion är inte implementerad än. "
                          "Den skulle visa exempelsekvenser som kan laddas in.")
    
    def _show_about(self):
        """Visa om-dialog"""
        messagebox.showinfo("Om AI Desktop Controller - Automationsbyggare", 
                          "AI Desktop Controller - Automationsbyggare\n"
                          "Version 1.0\n\n"
                          "Ett verktyg för att bygga automatiserade sekvenser "
                          "med drag-and-drop-gränssnitt.")

def run_builder():
    """Starta automationsbyggaren"""
    app = AutomationBuilder()
    app.mainloop()

if __name__ == "__main__":
    run_builder()