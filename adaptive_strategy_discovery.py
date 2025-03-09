"""
Adaptiv strategi-discovery för AI Desktop Controller

Denna modul implementerar mekanismer för att automatiskt upptäcka
och generera nya strategier baserat på tidigare resultat.
"""

import os
import json
import time
import random
import numpy as np
import logging
from datetime import datetime
import uuid

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("strategy_discovery.log"),
        logging.StreamHandler()
    ]
)

class StrategyPool:
    """Hantera en pool av strategier och deras prestationshistorik"""
    
    def __init__(self, capacity=100):
        self.strategies = {}  # id -> strategy
        self.performance = {}  # id -> [performance_history]
        self.capacity = capacity
        self.active_strategies = set()  # Set av aktiva strategi-ids
        
        # Skapa datamappar
        os.makedirs("data/strategies", exist_ok=True)
        
        logging.info(f"✅ StrategyPool initialiserad med kapacitet {capacity}")
    
    def add_strategy(self, strategy):
        """Lägg till en ny strategi i poolen"""
        if len(self.strategies) >= self.capacity and strategy['id'] not in self.strategies:
            # Ta bort den sämst presterande strategin om poolen är full
            self._remove_worst_strategy()
        
        # Lägg till eller uppdatera strategin
        self.strategies[strategy['id']] = strategy
        
        if strategy['id'] not in self.performance:
            self.performance[strategy['id']] = []
        
        # Markera som aktiv
        self.active_strategies.add(strategy['id'])
        
        logging.info(f"➕ Strategi '{strategy['name']}' ({strategy['id']}) tillagd i poolen")
        return True
    
    def remove_strategy(self, strategy_id):
        """Ta bort en strategi från poolen"""
        if strategy_id in self.strategies:
            strategy_name = self.strategies[strategy_id]['name']
            del self.strategies[strategy_id]
            if strategy_id in self.performance:
                del self.performance[strategy_id]
            self.active_strategies.discard(strategy_id)
            logging.info(f"➖ Strategi '{strategy_name}' ({strategy_id}) borttagen från poolen")
            return True
        return False
    
    def get_strategy(self, strategy_id):
        """Hämta en specifik strategi"""
        return self.strategies.get(strategy_id)
    
    def get_all_strategies(self):
        """Hämta alla strategier"""
        return self.strategies
    
    def get_active_strategies(self):
        """Hämta alla aktiva strategier"""
        active = {}
        for sid in self.active_strategies:
            if sid in self.strategies:
                active[sid] = self.strategies[sid]
        return active
    
    def update_performance(self, strategy_id, performance_score):
        """Uppdatera en strategis prestationshistorik"""
        if strategy_id in self.strategies:
            if strategy_id not in self.performance:
                self.performance[strategy_id] = []
            
            self.performance[strategy_id].append({
                'score': performance_score,
                'timestamp': datetime.now().isoformat()
            })
            
            # Behåll bara de senaste 100 prestationsmätningarna
            if len(self.performance[strategy_id]) > 100:
                self.performance[strategy_id] = self.performance[strategy_id][-100:]
            
            return True
        return False
    
    def get_performance(self, strategy_id):
        """Hämta prestationshistorik för en specifik strategi"""
        return self.performance.get(strategy_id, [])
    
    def get_average_performance(self, strategy_id, window=10):
        """Beräkna genomsnittlig prestanda för en strategi över en tidsfönster"""
        if strategy_id not in self.performance:
            return 0
        
        performance_history = self.performance[strategy_id]
        if not performance_history:
            return 0
        
        # Använd de senaste 'window' mätningarna
        recent = performance_history[-min(window, len(performance_history)):]
        return sum(p['score'] for p in recent) / len(recent)
    
    def get_best_strategy(self):
        """Hämta den bäst presterande strategin"""
        best_strategy_id = None
        best_performance = -float('inf')
        
        for sid in self.active_strategies:
            if sid in self.strategies:
                avg_perf = self.get_average_performance(sid)
                if avg_perf > best_performance:
                    best_performance = avg_perf
                    best_strategy_id = sid
        
        if best_strategy_id:
            return self.strategies[best_strategy_id]
        return None
    
    def get_worst_strategy(self):
        """Hämta den sämst presterande strategin"""
        worst_strategy_id = None
        worst_performance = float('inf')
        
        for sid in self.active_strategies:
            if sid in self.strategies:
                avg_perf = self.get_average_performance(sid)
                if avg_perf < worst_performance:
                    worst_performance = avg_perf
                    worst_strategy_id = sid
        
        if worst_strategy_id:
            return self.strategies[worst_strategy_id]
        return None
    
    def deactivate_strategy(self, strategy_id):
        """Deaktivera en strategi (utan att ta bort den)"""
        if strategy_id in self.active_strategies:
            self.active_strategies.remove(strategy_id)
            if strategy_id in self.strategies:
                strategy_name = self.strategies[strategy_id]['name']
                logging.info(f"🔄 Strategi '{strategy_name}' ({strategy_id}) deaktiverad")
            return True
        return False
    
    def activate_strategy(self, strategy_id):
        """Aktivera en tidigare deaktiverad strategi"""
        if strategy_id in self.strategies:
            self.active_strategies.add(strategy_id)
            strategy_name = self.strategies[strategy_id]['name']
            logging.info(f"🔄 Strategi '{strategy_name}' ({strategy_id}) aktiverad")
            return True
        return False
    
    def _remove_worst_strategy(self):
        """Ta bort den sämst presterande strategin från poolen"""
        worst_strategy = self.get_worst_strategy()
        if worst_strategy:
            return self.remove_strategy(worst_strategy['id'])
        return False
    
    def save(self, filename=None):
        """Spara strategipoolen till fil"""
        if filename is None:
            filename = f"data/strategies/strategy_pool_{int(time.time())}.json"
        
        data = {
            'strategies': self.strategies,
            'performance': self.performance,
            'active_strategies': list(self.active_strategies),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logging.info(f"💾 Strategipool sparad till {filename}")
        return filename
    
    def load(self, filename):
        """Ladda strategipoolen från fil"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.strategies = data['strategies']
            self.performance = data['performance']
            self.active_strategies = set(data['active_strategies'])
            
            logging.info(f"📂 Strategipool laddad från {filename} "
                        f"({len(self.strategies)} strategier, {len(self.active_strategies)} aktiva)")
            return True
        
        logging.warning(f"⚠️ Kunde inte ladda strategipool från {filename}")
        return False


class StrategyGenerator:
    """Generera nya strategier baserat på tidigare resultat"""
    
    def __init__(self, strategy_pool):
        self.strategy_pool = strategy_pool
        self.generation_techniques = [
            self._mutate_strategy,
            self._combine_strategies,
            self._specialized_strategy,
            self._random_strategy
        ]
        
        logging.info("✅ StrategyGenerator initialiserad")
    
    def generate_new_strategy(self, base_strategy=None):
        """Generera en ny strategi"""
        # Välj en genereringsteknik
        technique = random.choice(self.generation_techniques)
        
        # Generera en ny strategi
        if base_strategy:
            new_strategy = technique(base_strategy)
        else:
            # Om ingen basstrategi anges, välj en slumpmässig från poolen
            active_strategies = list(self.strategy_pool.get_active_strategies().values())
            if active_strategies:
                base_strategy = random.choice(active_strategies)
                new_strategy = technique(base_strategy)
            else:
                # Om inga strategier finns i poolen, skapa en slumpmässig
                new_strategy = self._random_strategy()
        
        # Generera unikt ID och tidsstämpel
        new_strategy['id'] = f"strategy_{uuid.uuid4().hex[:8]}"
        new_strategy['created_at'] = datetime.now().isoformat()
        
        logging.info(f"🧪 Ny strategi genererad: '{new_strategy['name']}' med metod {technique.__name__}")
        return new_strategy
    
    def _mutate_strategy(self, base_strategy=None):
        """Mutera en befintlig strategi med små förändringar"""
        if not base_strategy:
            active_strategies = list(self.strategy_pool.get_active_strategies().values())
            if not active_strategies:
                return self._random_strategy()
            base_strategy = random.choice(active_strategies)
        
        # Skapa en kopia av bastrategin
        new_strategy = base_strategy.copy()
        if 'params' in new_strategy:
            new_strategy['params'] = new_strategy['params'].copy()
        
        # Applicera slumpmässiga mutationer
        if 'params' in new_strategy:
            params = new_strategy['params']
            
            # Mutera parametervärdena
            for key in params:
                if isinstance(params[key], (int, float)):
                    # Mutera numeriska värden med ±10%
                    mutation_factor = 1 + (random.random() * 0.2 - 0.1)
                    params[key] = params[key] * mutation_factor
                    
                    # Avrunda heltal
                    if isinstance(params[key], int):
                        params[key] = int(params[key])
                        
                elif isinstance(params[key], str) and params[key] in ['low', 'medium', 'high']:
                    # Mutera enumvärden
                    options = ['low', 'medium', 'high']
                    options.remove(params[key])
                    params[key] = random.choice(options)
        
        # Uppdatera namn för att indikera att det är en mutation
        new_strategy['name'] = f"Muterad {base_strategy['name']}"
        new_strategy['description'] = f"Mutation av {base_strategy['name']} med små parameterjusteringar"
        
        return new_strategy
    
    def _combine_strategies(self, base_strategy=None):
        """Kombinera två strategier till en ny"""
        active_strategies = list(self.strategy_pool.get_active_strategies().values())
        if len(active_strategies) < 2:
            return self._mutate_strategy(base_strategy)
        
        # Välj två olika strategier att kombinera
        if base_strategy:
            strategy1 = base_strategy
            active_strategies = [s for s in active_strategies if s['id'] != base_strategy['id']]
            if not active_strategies:
                return self._mutate_strategy(base_strategy)
            strategy2 = random.choice(active_strategies)
        else:
            strategy1, strategy2 = random.sample(active_strategies, 2)
        
        # Skapa en ny strategi med egenskaper från båda
        new_strategy = {
            'name': f"{strategy1['name']} + {strategy2['name']}",
            'description': f"Kombination av {strategy1['name']} och {strategy2['name']}",
        }
        
        # Kombinera parametrar
        if 'params' in strategy1 and 'params' in strategy2:
            new_params = {}
            all_keys = set(strategy1['params'].keys()).union(set(strategy2['params'].keys()))
            
            for key in all_keys:
                if key in strategy1['params'] and key in strategy2['params']:
                    # Om båda har parametern, välj slumpmässigt eller beräkna genomsnittet
                    if isinstance(strategy1['params'][key], (int, float)) and isinstance(strategy2['params'][key], (int, float)):
                        # Beräkna genomsnittet för numeriska värden
                        new_params[key] = (strategy1['params'][key] + strategy2['params'][key]) / 2
                        
                        # Avrunda heltal
                        if isinstance(strategy1['params'][key], int) and isinstance(strategy2['params'][key], int):
                            new_params[key] = int(new_params[key])
                    else:
                        # Välj slumpmässigt för icke-numeriska värden
                        new_params[key] = random.choice([strategy1['params'][key], strategy2['params'][key]])
                elif key in strategy1['params']:
                    new_params[key] = strategy1['params'][key]
                else:
                    new_params[key] = strategy2['params'][key]
            
            new_strategy['params'] = new_params
        elif 'params' in strategy1:
            new_strategy['params'] = strategy1['params'].copy()
        elif 'params' in strategy2:
            new_strategy['params'] = strategy2['params'].copy()
        
        return new_strategy
    
    def _specialized_strategy(self, base_strategy=None):
        """Skapa en specialiserad version av en strategi"""
        if not base_strategy:
            active_strategies = list(self.strategy_pool.get_active_strategies().values())
            if not active_strategies:
                return self._random_strategy()
            base_strategy = random.choice(active_strategies)
        
        # Skapa en kopia av bastrategin
        new_strategy = base_strategy.copy()
        if 'params' in new_strategy:
            new_strategy['params'] = new_strategy['params'].copy()
        
        # Specialiseringsområden
        specializations = [
            "Utforskande", "Exploaterande", "Risktagande", "Konservativ", 
            "Snabb", "Noggrann", "Visuell", "Textbaserad"
        ]
        
        # Välj en specialisering
        specialization = random.choice(specializations)
        
        # Applicera specialiseringen
        if 'params' in new_strategy:
            params = new_strategy['params']
            
            if specialization == "Utforskande":
                # Öka utforskning
                if 'exploration_rate' in params:
                    params['exploration_rate'] = min(1.0, params['exploration_rate'] * 1.5)
                if 'risk_level' in params:
                    params['risk_level'] = 'high'
                    
            elif specialization == "Exploaterande":
                # Fokusera på att exploatera istället för att utforska
                if 'exploration_rate' in params:
                    params['exploration_rate'] = max(0.1, params['exploration_rate'] * 0.5)
                    
            elif specialization == "Risktagande":
                # Öka risktagande
                if 'risk_level' in params:
                    params['risk_level'] = 'high'
                
            elif specialization == "Konservativ":
                # Minska risktagande
                if 'risk_level' in params:
                    params['risk_level'] = 'low'
                    
            elif specialization == "Snabb":
                # Fokusera på snabbhet
                if 'speed' in params:
                    params['speed'] = min(10, params['speed'] * 1.5)
                if 'timeout' in params:
                    params['timeout'] = max(1, params['timeout'] * 0.7)
                    
            elif specialization == "Noggrann":
                # Fokusera på noggrannhet
                if 'accuracy' in params:
                    params['accuracy'] = min(1.0, params['accuracy'] * 1.2)
                if 'timeout' in params:
                    params['timeout'] = params['timeout'] * 1.3
                    
            elif specialization == "Visuell":
                # Fokusera på visuell igenkänning
                if 'focus_area' in params:
                    params['focus_area'] = 'visual'
                    
            elif specialization == "Textbaserad":
                # Fokusera på textanalys
                if 'focus_area' in params:
                    params['focus_area'] = 'text'
        
        # Uppdatera namn och beskrivning
        new_strategy['name'] = f"{specialization} {base_strategy['name']}"
        new_strategy['description'] = f"Specialiserad version av {base_strategy['name']} med fokus på {specialization.lower()}"
        
        return new_strategy
    
    def _random_strategy(self, base_strategy=None):
        """Skapa en helt ny slumpmässig strategi"""
        # Skapa en ny strategi från grunden
        strategy_types = [
            "Utforska", "Navigera", "Interagera", "Analysera", 
            "Söka", "Testa", "Optimera", "Kombinera"
        ]
        
        focus_areas = [
            "desktop", "browser", "text_fields", "buttons", 
            "menus", "images", "links", "forms"
        ]
        
        interaction_types = [
            "clicks_first", "keyboard_first", "balanced", 
            "visual_based", "text_based", "pattern_based"
        ]
        
        # Välj slumpmässiga egenskaper
        strategy_type = random.choice(strategy_types)
        focus_area = random.choice(focus_areas)
        interaction_type = random.choice(interaction_types)
        risk_level = random.choice(["low", "medium", "high"])
        
        # Skapa en unik strategi
        new_strategy = {
            'name': f"{strategy_type} {focus_area.capitalize()}",
            'description': f"Slumpmässigt genererad strategi som fokuserar på {focus_area} med {interaction_type} interaktion",
            'params': {
                'focus_area': focus_area,
                'interaction_type': interaction_type,
                'risk_level': risk_level,
                'exploration_rate': random.uniform(0.1, 0.9),
                'timeout': random.uniform(1, 10),
                'retry_attempts': random.randint(1, 5)
            }
        }
        
        return new_strategy


class AdaptiveStrategyDiscovery:
    """Huvudklass för adaptiv strategi-discovery"""
    
    def __init__(self, initial_strategies=None):
        self.strategy_pool = StrategyPool()
        self.strategy_generator = StrategyGenerator(self.strategy_pool)
        
        # Lägg till initiala strategier om några finns
        if initial_strategies:
            for strategy in initial_strategies:
                self.strategy_pool.add_strategy(strategy)
        else:
            # Lägg till några grundläggande strategier
            self._create_initial_strategies()
        
        # Statistik
        self.discovery_stats = {
            'strategies_generated': 0,
            'strategies_removed': 0,
            'discovery_iterations': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        logging.info("✅ AdaptiveStrategyDiscovery initialiserad")
    
    def _create_initial_strategies(self):
        """Skapa några initiala strategier att börja med"""
        initial_strategies = [
            {
                'id': 'basic_desktop_exploration',
                'name': 'Utforska Skrivbordet',
                'description': 'Grundläggande utforskning av skrivbordsmiljön',
                'params': {
                    'focus_area': 'desktop',
                    'interaction_type': 'balanced',
                    'risk_level': 'low',
                    'exploration_rate': 0.7,
                    'timeout': 5,
                    'retry_attempts': 3
                }
            },
            {
                'id': 'text_field_interaction',
                'name': 'Textfältsinteraktion',
                'description': 'Fokuserar på att hitta och interagera med textfält',
                'params': {
                    'focus_area': 'text_fields',
                    'interaction_type': 'keyboard_first',
                    'risk_level': 'low',
                    'exploration_rate': 0.5,
                    'timeout': 3,
                    'retry_attempts': 2
                }
            },
            {
                'id': 'button_navigation',
                'name': 'Knappnavigering',
                'description': 'Fokuserar på att hitta och klicka på knappar',
                'params': {
                    'focus_area': 'buttons',
                    'interaction_type': 'clicks_first',
                    'risk_level': 'medium',
                    'exploration_rate': 0.6,
                    'timeout': 4,
                    'retry_attempts': 3
                }
            }
        ]
        
        for strategy in initial_strategies:
            self.strategy_pool.add_strategy(strategy)
        
        logging.info(f"✅ {len(initial_strategies)} initiala strategier skapade")
    
    def get_strategy(self, strategy_id):
        """Hämta en specifik strategi"""
        return self.strategy_pool.get_strategy(strategy_id)
    
    def get_all_strategies(self):
        """Hämta alla strategier"""
        return self.strategy_pool.get_all_strategies()
    
    def get_active_strategies(self):
        """Hämta alla aktiva strategier"""
        return self.strategy_pool.get_active_strategies()
    
    def update_strategy_performance(self, strategy_id, performance_score):
        """Uppdatera en strategis prestationspoäng"""
        return self.strategy_pool.update_performance(strategy_id, performance_score)
    
    def get_best_strategy(self):
        """Hämta den bäst presterande strategin"""
        return self.strategy_pool.get_best_strategy()
    
    def discover_new_strategies(self, num_strategies=1, base_strategy=None):
        """
        Upptäck och generera nya strategier
        
        Args:
            num_strategies: Antal nya strategier att generera
            base_strategy: Basstrategi att utgå från (valfritt)
        
        Returns:
            List of new strategy IDs
        """
        new_strategy_ids = []
        
        for _ in range(num_strategies):
            # Generera en ny strategi
            new_strategy = self.strategy_generator.generate_new_strategy(base_strategy)
            
            # Lägg till i poolen
            if self.strategy_pool.add_strategy(new_strategy):
                new_strategy_ids.append(new_strategy['id'])
                self.discovery_stats['strategies_generated'] += 1
        
        self.discovery_stats['discovery_iterations'] += 1
        logging.info(f"🧪 Genererade {len(new_strategy_ids)} nya strategier")
        
        return new_strategy_ids
    
    def perform_discovery_iteration(self, performance_threshold=0.3):
        """
        Utför en fullständig discovery-iteration
        
        1. Utvärdera befintliga strategier
        2. Ta bort lågpresterande strategier
        3. Generera nya strategier baserat på de bästa
        
        Args:
            performance_threshold: Prestandatröskel för att behålla strategier
        
        Returns:
            dict med resultat från iterationen
        """
        iteration_results = {
            'timestamp': datetime.now().isoformat(),
            'removed_strategies': 0,
            'new_strategies': 0,
            'best_strategy': None
        }
        
        # 1. Hitta låg- och högpresterande strategier
        active_strategies = self.strategy_pool.get_active_strategies()
        
        low_performing = []
        high_performing = []
        
        for strategy_id, strategy in active_strategies.items():
            avg_performance = self.strategy_pool.get_average_performance(strategy_id)
            
            if avg_performance < performance_threshold:
                low_performing.append(strategy_id)
            elif avg_performance > 0.7:  # Godtycklig tröskel för "högt presterande"
                high_performing.append(strategy_id)
        
        # 2. Ta bort lågpresterande strategier
        for strategy_id in low_performing:
            # Deaktivera istället för att ta bort helt (så vi kan återaktivera senare om det behövs)
            if self.strategy_pool.deactivate_strategy(strategy_id):
                iteration_results['removed_strategies'] += 1
                self.discovery_stats['strategies_removed'] += 1
        
        # 3. Generera nya strategier baserat på högpresterande
        num_to_generate = max(1, iteration_results['removed_strategies'])
        
        if high_performing:
            # Välj en av de högpresterande strategierna att basera nya på
            base_strategy_id = random.choice(high_performing)
            base_strategy = self.strategy_pool.get_strategy(base_strategy_id)
            
            new_strategy_ids = self.discover_new_strategies(num_to_generate, base_strategy)
            iteration_results['new_strategies'] = len(new_strategy_ids)
        else:
            # Om inga högpresterande strategier finns, generera helt nya
            new_strategy_ids = self.discover_new_strategies(num_to_generate)
            iteration_results['new_strategies'] = len(new_strategy_ids)
        
        # 4. Notera den bästa strategin
        best_strategy = self.strategy_pool.get_best_strategy()
        if best_strategy:
            iteration_results['best_strategy'] = best_strategy['id']
        
        logging.info(f"🔄 Discovery iteration: Removed {iteration_results['removed_strategies']}, "
                    f"Added {iteration_results['new_strategies']} strategies")
        
        return iteration_results
    
    def save_state(self, filename=None):
        """Spara hela tillståndet för strategiupptäckt"""
        if filename is None:
            filename = f"data/strategies/discovery_state_{int(time.time())}.json"
        
        # Spara strategipoolen
        pool_file = self.strategy_pool.save()
        
        # Spara statistik
        state_data = {
            'discovery_stats': self.discovery_stats,
            'strategy_pool_file': pool_file,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        logging.info(f"💾 Discovery state sparad till {filename}")
        return filename
    
    def load_state(self, filename):
        """Ladda tillståndet för strategiupptäckt"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                state_data = json.load(f)
            
            # Ladda strategipoolen
            if 'strategy_pool_file' in state_data and os.path.exists(state_data['strategy_pool_file']):
                self.strategy_pool.load(state_data['strategy_pool_file'])
            
            # Ladda statistik
            if 'discovery_stats' in state_data:
                self.discovery_stats = state_data['discovery_stats']
            
            logging.info(f"📂 Discovery state laddad från {filename}")
            return True
        
        logging.warning(f"⚠️ Kunde inte ladda discovery state från {filename}")
        return False