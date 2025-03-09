"""
Web Server - Webbgränssnitt för AI Desktop Controller

Implementerar ett Flask-baserat webbgränssnitt för fjärrövervakning och styrning.
"""

import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta
import random
from functools import wraps

from flask import Flask, render_template, jsonify, request, redirect, url_for, session, Response
from flask_socketio import SocketIO, emit
import secrets

# Sätt upp loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("webserver.log"),
        logging.StreamHandler()
    ]
)

# Skapa mappar för data
os.makedirs("data/web", exist_ok=True)
os.makedirs("data/web/automations", exist_ok=True)

# Initiera Flask-app
app = Flask(__name__, 
           static_folder='web/static',
           template_folder='web/templates')
app.secret_key = secrets.token_hex(16)
socketio = SocketIO(app, cors_allowed_origins="*")

# Systemkontroller-referens (kommer att sättas vid start)
SYSTEM = None

# Autentiseringsinställningar
AUTH_ENABLED = False
USERS = {
    "admin": "password"  # Byte till säkra lösenord i produktion!
}

# Funktionen för att kräva inloggning
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if AUTH_ENABLED and 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Dummy-data för demo (tas bort när systemet integreras)
def generate_dummy_data():
    clusters = {}
    strategies = {}
    
    # Generera klusterdata
    for i in range(5):
        cluster_id = f"cluster_{i+1}"
        clusters[cluster_id] = {
            "id": cluster_id,
            "name": f"Cluster {i+1}",
            "strategy": f"strategy_{random.randint(1, 5)}",
            "instances": random.randint(3, 8),
            "success_rate": random.uniform(0.4, 0.95),
            "status": random.choice(["active", "active", "active", "paused", "error"]),
            "uptime": random.randint(10, 3600)
        }
    
    # Generera strategidata
    for i in range(8):
        strategy_id = f"strategy_{i+1}"
        strategies[strategy_id] = {
            "id": strategy_id,
            "name": f"Strategy {i+1}",
            "type": random.choice(["exploration", "optimization", "analysis"]),
            "success_rate": random.uniform(0.3, 0.98),
            "avg_reward": random.uniform(-10, 100),
            "usage_count": random.randint(0, 20)
        }
    
    # Generera prestationsdata över tid
    timestamps = []
    performance = []
    rewards = []
    base_time = datetime.now() - timedelta(hours=24)
    
    for i in range(288):  # 24 timmar med 5-minutersintervall
        timestamp = base_time + timedelta(minutes=5*i)
        timestamps.append(timestamp.isoformat())
        
        # Lägg till lite slumpmässig variation men med en uppåtgående trend
        perf_base = 0.5 + (i / 576)  # Ökar långsamt från 0.5 till 1.0
        perf_noise = random.uniform(-0.05, 0.05)
        performance.append(min(0.98, max(0.1, perf_base + perf_noise)))
        
        # Belöning som följer en liknande men inte identisk kurva
        reward_base = 20 + (i / 3)  # Ökar från 20 till ~116
        reward_noise = random.uniform(-5, 5)
        rewards.append(reward_base + reward_noise)
    
    return {
        "clusters": clusters,
        "strategies": strategies,
        "performance_history": {
            "timestamps": timestamps,
            "performance": performance,
            "rewards": rewards
        },
        "system_stats": {
            "uptime": random.randint(3600, 86400),
            "active_clusters": len(clusters),
            "total_instances": sum(c["instances"] for c in clusters.values()),
            "avg_success_rate": sum(c["success_rate"] for c in clusters.values()) / len(clusters),
            "total_strategies": len(strategies),
            "memory_usage": random.uniform(200, 800),
            "cpu_usage": random.uniform(10, 80)
        },
        "automations": [
            {
                "id": "auto_1",
                "name": "Textedi Automatisering",
                "description": "Öppnar Notepad och skriver en text",
                "actions": 5,
                "last_run": (datetime.now() - timedelta(hours=2)).isoformat(),
                "success_rate": 0.92
            },
            {
                "id": "auto_2",
                "name": "Webb Automatisering",
                "description": "Öppnar webbläsaren och navigerar till en sida",
                "actions": 8,
                "last_run": (datetime.now() - timedelta(minutes=45)).isoformat(),
                "success_rate": 0.78
            }
        ]
    }

# Autentiseringsrutter
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in USERS and USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Ogiltigt användarnamn eller lösenord")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

# Huvudrutter
@app.route('/')
@login_required
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/automations')
@login_required
def automations():
    return render_template('automations.html')

@app.route('/clusters')
@login_required
def clusters():
    return render_template('clusters.html')

@app.route('/strategies')
@login_required
def strategies():
    return render_template('strategies.html')

@app.route('/editor')
@login_required
def editor():
    automation_id = request.args.get('id', None)
    return render_template('automation_editor.html', automation_id=automation_id)

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

# API-rutter
@app.route('/api/status')
@login_required
def api_status():
    if SYSTEM is not None:
        # Hämta systemstatus från riktiga systemet
        try:
            status = {
                "running": True,
                "uptime": SYSTEM.get_uptime(),
                "clusters": len(SYSTEM.ai_system.clusters),
                "active_instances": SYSTEM.get_active_instances_count(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Fel vid hämtning av systemstatus: {e}")
            status = {
                "running": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    else:
        # Generera dummy-data om systemet inte är anslutet
        dummy_data = generate_dummy_data()
        status = {
            "running": True,
            "uptime": dummy_data["system_stats"]["uptime"],
            "clusters": dummy_data["system_stats"]["active_clusters"],
            "active_instances": dummy_data["system_stats"]["total_instances"],
            "timestamp": datetime.now().isoformat(),
            "demo_mode": True
        }
    
    return jsonify(status)

@app.route('/api/system_stats')
@login_required
def api_system_stats():
    if SYSTEM is not None:
        try:
            stats = SYSTEM.get_system_stats()
        except Exception as e:
            logging.error(f"Fel vid hämtning av systemstatistik: {e}")
            stats = {"error": str(e)}
    else:
        dummy_data = generate_dummy_data()
        stats = dummy_data["system_stats"]
        stats["demo_mode"] = True
        
    return jsonify(stats)

@app.route('/api/clusters')
@login_required
def api_clusters():
    if SYSTEM is not None:
        try:
            clusters = SYSTEM.get_clusters_info()
        except Exception as e:
            logging.error(f"Fel vid hämtning av klusterinformation: {e}")
            clusters = {"error": str(e)}
    else:
        dummy_data = generate_dummy_data()
        clusters = dummy_data["clusters"]
    
    return jsonify(clusters)

@app.route('/api/strategies')
@login_required
def api_strategies():
    if SYSTEM is not None:
        try:
            strategies = SYSTEM.get_strategies_info()
        except Exception as e:
            logging.error(f"Fel vid hämtning av strategiinformation: {e}")
            strategies = {"error": str(e)}
    else:
        dummy_data = generate_dummy_data()
        strategies = dummy_data["strategies"]
    
    return jsonify(strategies)

@app.route('/api/performance_history')
@login_required
def api_performance_history():
    if SYSTEM is not None:
        try:
            history = SYSTEM.get_performance_history()
        except Exception as e:
            logging.error(f"Fel vid hämtning av prestandahistorik: {e}")
            history = {"error": str(e)}
    else:
        dummy_data = generate_dummy_data()
        history = dummy_data["performance_history"]
        
    return jsonify(history)

@app.route('/api/automations')
@login_required
def api_automations():
    if SYSTEM is not None:
        try:
            automations = SYSTEM.get_automations()
        except Exception as e:
            logging.error(f"Fel vid hämtning av automationer: {e}")
            automations = {"error": str(e)}
    else:
        dummy_data = generate_dummy_data()
        automations = dummy_data["automations"]
    
    return jsonify(automations)

@app.route('/api/automation/<automation_id>')
@login_required
def api_automation(automation_id):
    # Sökväg för sparade automationer
    automation_path = os.path.join('data/web/automations', f"{automation_id}.json")
    
    if os.path.exists(automation_path):
        try:
            with open(automation_path, 'r') as f:
                automation = json.load(f)
            return jsonify(automation)
        except Exception as e:
            logging.error(f"Fel vid läsning av automation {automation_id}: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        # Om fil saknas, returnera standard tom automation
        if automation_id == "new":
            automation = {
                "id": "new",
                "name": "Ny Automation",
                "description": "Beskriv vad automationen ska göra",
                "nodes": [],
                "edges": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            return jsonify(automation)
        else:
            return jsonify({"error": "Automation not found"}), 404

@app.route('/api/automation/<automation_id>', methods=['POST'])
@login_required
def api_save_automation(automation_id):
    # Validera indata
    if not request.json:
        return jsonify({"error": "Invalid JSON data"}), 400
    
    automation = request.json
    
    # Om det är en ny automation, generera ett ID
    if automation_id == "new":
        automation_id = f"auto_{int(time.time())}"
        automation["id"] = automation_id
    
    # Uppdatera timestamp
    automation["updated_at"] = datetime.now().isoformat()
    
    # Spara automation till fil
    automation_path = os.path.join('data/web/automations', f"{automation_id}.json")
    try:
        with open(automation_path, 'w') as f:
            json.dump(automation, f, indent=2)
        return jsonify({"id": automation_id, "success": True})
    except Exception as e:
        logging.error(f"Fel vid sparande av automation {automation_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/automation/<automation_id>/run', methods=['POST'])
@login_required
def api_run_automation(automation_id):
    if SYSTEM is not None:
        try:
            result = SYSTEM.run_automation(automation_id)
            return jsonify(result)
        except Exception as e:
            logging.error(f"Fel vid körning av automation {automation_id}: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        # Simulera körning i demo-läge
        time.sleep(2)  # Simulera bearbetningstid
        success = random.random() > 0.2  # 80% chans för framgång
        
        result = {
            "success": success,
            "automation_id": automation_id,
            "timestamp": datetime.now().isoformat(),
            "execution_time": random.uniform(1.5, 4.0),
            "demo_mode": True
        }
        
        if not success:
            result["error"] = "Simulated execution failure"
            
        return jsonify(result)

@app.route('/api/screenshot')
@login_required
def api_screenshot():
    if SYSTEM is not None:
        try:
            # Ta en skärmdump och returnera
            screenshot_path = SYSTEM.capture_screen()
            
            with open(screenshot_path, 'rb') as f:
                screenshot_data = f.read()
            
            return Response(screenshot_data, mimetype='image/png')
        except Exception as e:
            logging.error(f"Fel vid hämtning av skärmdump: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        # I demo-läge, returnera en placeholder-bild
        return redirect(url_for('static', filename='img/demo_screenshot.png'))

@app.route('/api/recent_events')
@login_required
def api_recent_events():
    if SYSTEM is not None:
        try:
            events = SYSTEM.get_recent_events()
            return jsonify(events)
        except Exception as e:
            logging.error(f"Fel vid hämtning av händelser: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        # Generera dummy-händelser
        events = []
        event_types = ["info", "warning", "error", "success"]
        event_sources = ["system", "cluster", "instance", "ui", "audio"]
        event_messages = [
            "Kluster 2 upptäckte ny optimal strategi",
            "Instans 3 avslutades oväntat",
            "Kunde inte läsa text från dialogruta",
            "Automationssekvens slutfördes framgångsrikt",
            "Detekterade varningsljud från applikation",
            "Nya UI-element identifierade",
            "Systemresurser över tröskelvärde",
            "Strategi anpassad baserat på resultat",
            "Skärmdump sparad"
        ]
        
        base_time = datetime.now() - timedelta(minutes=30)
        
        for i in range(20):
            timestamp = base_time + timedelta(minutes=i*1.5)
            events.append({
                "id": f"event_{i}",
                "timestamp": timestamp.isoformat(),
                "type": random.choice(event_types),
                "source": random.choice(event_sources),
                "message": random.choice(event_messages)
            })
        
        return jsonify(events)

# WebSocket-händelser
@socketio.on('connect')
def handle_connect():
    emit('status', {'connected': True})

@socketio.on('request_update')
def handle_update_request(data):
    """Hantera begäran om uppdateringar i realtid"""
    if SYSTEM is not None:
        try:
            update_type = data.get('type', 'all')
            
            if update_type == 'system_stats':
                stats = SYSTEM.get_system_stats()
                emit('system_stats_update', stats)
            elif update_type == 'clusters':
                clusters = SYSTEM.get_clusters_info()
                emit('clusters_update', clusters)
            elif update_type == 'all':
                # Hämta all information
                stats = SYSTEM.get_system_stats()
                clusters = SYSTEM.get_clusters_info()
                strategies = SYSTEM.get_strategies_info()
                
                emit('full_update', {
                    'system_stats': stats,
                    'clusters': clusters,
                    'strategies': strategies,
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            logging.error(f"Fel vid hantering av uppdateringsbegäran: {e}")
            emit('error', {'message': str(e)})
    else:
        # Generera dummy-data i demo-läge
        dummy_data = generate_dummy_data()
        update_type = data.get('type', 'all')
        
        if update_type == 'system_stats':
            emit('system_stats_update', dummy_data['system_stats'])
        elif update_type == 'clusters':
            emit('clusters_update', dummy_data['clusters'])
        elif update_type == 'all':
            emit('full_update', {
                'system_stats': dummy_data['system_stats'],
                'clusters': dummy_data['clusters'],
                'strategies': dummy_data['strategies'],
                'timestamp': datetime.now().isoformat(),
                'demo_mode': True
            })

# Funktioner för att integreras med systemet
def init_webserver(system_controller=None, enable_auth=False, host="0.0.0.0", port=5000, debug=False):
    """
    Initiera och starta webbservern
    
    Args:
        system_controller: Systemkontroller för AI Desktop Controller
        enable_auth: Om autentisering ska aktiveras
        host: Värdadressen att binda till
        port: Porten att lyssna på
        debug: Om Flask ska köras i debug-läge
    """
    global SYSTEM, AUTH_ENABLED
    SYSTEM = system_controller
    AUTH_ENABLED = enable_auth
    
    # Kontrollera eller skapa mallmapp
    check_templates()
    
    # Starta webbservern i en separat tråd
    server_thread = threading.Thread(target=lambda: socketio.run(
        app, host=host, port=port, debug=debug, use_reloader=False
    ))
    server_thread.daemon = True
    server_thread.start()
    
    logging.info(f"🌐 Webbserver startad på http://{host}:{port}/")
    if enable_auth:
        logging.info("🔒 Autentisering aktiverad")
    
    return server_thread

def check_templates():
    """Kontrollera att mallmappar finns och skapa dem vid behov"""
    template_dir = os.path.join(os.path.dirname(__file__), 'web/templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'web/static')
    
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(os.path.join(static_dir, 'css'), exist_ok=True)
    os.makedirs(os.path.join(static_dir, 'js'), exist_ok=True)
    os.makedirs(os.path.join(static_dir, 'img'), exist_ok=True)
    
    # Kontrollera och skapa basmallarna
    create_basic_templates(template_dir, static_dir)

def create_basic_templates(template_dir, static_dir):
    """Skapa de nödvändiga mallfilerna om de inte finns"""
    # Lista över standardmallar och deras innehåll implementeras här
    # I en verklig implementation skulle vi skapa flera HTML, JS och CSS-filer
    pass

# Kör direkt för testning
if __name__ == "__main__":
    print("🌐 Startar webbserver för testning...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)