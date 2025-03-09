"""
Database Backend - Databas för AI Desktop Controller

Detta modul implementerar en databas för persistens av strategier,
resultat och inlärningsdata med stöd för flera databaser (SQLite, PostgreSQL).
"""

import os
import time
import logging
import json
import sqlite3
import threading
import pickle
from datetime import datetime
import uuid

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("database.log"),
        logging.StreamHandler()
    ]
)

# Globala konstanter
DEFAULT_DB_PATH = "data/db/ai_system.db"

class DatabaseManager:
    """Hanterare för AI-systemets databas"""
    
    def __init__(self, db_type="sqlite", connection_params=None):
        """
        Initiera databashanteraren
        
        Args:
            db_type: Typ av databas ("sqlite", "postgresql", "mysql")
            connection_params: Anslutningsparametrar (dict eller sökväg för SQLite)
        """
        self.db_type = db_type.lower()
        self.connection_params = connection_params
        self.connection = None
        self.cursor = None
        
        # Trådsäkerhet
        self.connection_lock = threading.Lock()
        
        # Skapa databaskatalog om den inte finns
        if self.db_type == "sqlite":
            db_path = connection_params or DEFAULT_DB_PATH
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
        # Initiera databas
        self._initialize_database()
        
        logging.info(f"✅ DatabaseManager initierad med {db_type}")
    
    def _initialize_database(self):
        """Initiera databasanslutning och schema"""
        if self.db_type == "sqlite":
            self._initialize_sqlite()
        elif self.db_type == "postgresql":
            self._initialize_postgresql()
        else:
            raise ValueError(f"Databastyp {self.db_type} stöds inte")
    
    def _initialize_sqlite(self):
        """Initiera SQLite-databas"""
        try:
            db_path = self.connection_params or DEFAULT_DB_PATH
            
            # Skapa anslutning
            self.connection = sqlite3.connect(db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            self.cursor = self.connection.cursor()
            
            # Skapa tabeller om de inte finns
            self._create_tables()
            
            logging.info(f"✅ SQLite-databas initierad: {db_path}")
            
        except Exception as e:
            logging.error(f"Fel vid initiering av SQLite-databas: {e}")
            raise
    
    def _initialize_postgresql(self):
        """Initiera PostgreSQL-databas"""
        try:
            import psycopg2
            import psycopg2.extras
            
            # Kontrollera anslutningsparametrar
            if not self.connection_params:
                raise ValueError("Anslutningsparametrar krävs för PostgreSQL")
            
            # Skapa anslutning
            self.connection = psycopg2.connect(**self.connection_params)
            self.connection.autocommit = False
            self.cursor = self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Skapa tabeller om de inte finns
            self._create_tables()
            
            logging.info(f"✅ PostgreSQL-databas initierad: {self.connection_params.get('database')}")
            
        except ImportError:
            logging.error("psycopg2 är inte installerat. Kör: pip install psycopg2")
            raise
        except Exception as e:
            logging.error(f"Fel vid initiering av PostgreSQL-databas: {e}")
            raise
    
    def _create_tables(self):
        """Skapa databastabeller om de inte finns"""
        # Skapa strategitabell
        self.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                params TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                performance REAL,
                is_active INTEGER
            )
        """)
        
        # Skapa resultattabell
        self.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY,
                strategy_id TEXT,
                cluster_id TEXT,
                instance_id TEXT,
                success INTEGER,
                data TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (strategy_id) REFERENCES strategies (id)
            )
        """)
        
        # Skapa inlärningsdatatabell
        self.execute("""
            CREATE TABLE IF NOT EXISTS learning_data (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                state TEXT,
                action TEXT,
                reward REAL,
                next_state TEXT,
                done INTEGER,
                created_at TIMESTAMP
            )
        """)
        
        # Skapa index
        self.execute("CREATE INDEX IF NOT EXISTS idx_results_strategy ON results (strategy_id)")
        self.execute("CREATE INDEX IF NOT EXISTS idx_results_created ON results (created_at)")
        self.execute("CREATE INDEX IF NOT EXISTS idx_learning_type ON learning_data (type)")
        
        # Spara ändringar
        self.commit()
    
    def execute(self, query, params=None):
        """
        Utför en databasfråga
        
        Args:
            query: SQL-fråga
            params: Parametrar till frågan
            
        Returns:
            cursor: Databasmarkör
        """
        with self.connection_lock:
            try:
                if params:
                    self.cursor.execute(query, params)
                else:
                    self.cursor.execute(query)
                    
                return self.cursor
                
            except Exception as e:
                logging.error(f"Fel vid databasfråga: {e}")
                logging.error(f"Query: {query}")
                logging.error(f"Params: {params}")
                self.rollback()
                raise
    
    def commit(self):
        """Spara ändringar i databasen"""
        with self.connection_lock:
            try:
                self.connection.commit()
            except Exception as e:
                logging.error(f"Fel vid commit: {e}")
                self.rollback()
                raise
    
    def rollback(self):
        """Återställ ändringar i databasen"""
        with self.connection_lock:
            try:
                self.connection.rollback()
            except Exception as e:
                logging.error(f"Fel vid rollback: {e}")
    
    def close(self):
        """Stäng databasanslutning"""
        with self.connection_lock:
            try:
                if self.connection:
                    self.connection.close()
                    self.connection = None
                    self.cursor = None
                    logging.info("⏹️ Databasanslutning stängd")
            except Exception as e:
                logging.error(f"Fel vid stängning av databasanslutning: {e}")
    
    def save_strategy(self, strategy):
        """
        Spara en strategi i databasen
        
        Args:
            strategy: Strategiobjekt
            
        Returns:
            str: Strategi-ID
        """
        try:
            # Kontrollera om strategin redan har ett ID
            if not strategy.get('id'):
                strategy['id'] = f"strategy_{uuid.uuid4().hex[:8]}"
            
            # Konvertera parametrar till JSON
            if 'params' in strategy and not isinstance(strategy['params'], str):
                strategy['params'] = json.dumps(strategy['params'])
            
            # Aktuell tidsstämpel
            now = datetime.now().isoformat()
            if not strategy.get('created_at'):
                strategy['created_at'] = now
            strategy['updated_at'] = now
            
            # Standardvärden
            if 'performance' not in strategy:
                strategy['performance'] = 0.0
            if 'is_active' not in strategy:
                strategy['is_active'] = 1
            
            # Sätt in eller uppdatera strategi
            self.execute("""
                INSERT OR REPLACE INTO strategies 
                (id, name, description, params, created_at, updated_at, performance, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy['id'],
                strategy.get('name', 'Unnamed Strategy'),
                strategy.get('description', ''),
                strategy.get('params', '{}'),
                strategy['created_at'],
                strategy['updated_at'],
                strategy.get('performance', 0.0),
                strategy.get('is_active', 1)
            ))
            
            self.commit()
            logging.info(f"💾 Strategi sparad: {strategy['id']}")
            
            return strategy['id']
            
        except Exception as e:
            logging.error(f"Fel vid sparande av strategi: {e}")
            self.rollback()
            raise
    
    def get_strategy(self, strategy_id):
        """
        Hämta en strategi från databasen
        
        Args:
            strategy_id: Strategi-ID
            
        Returns:
            dict: Strategi eller None
        """
        try:
            cursor = self.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,))
            row = cursor.fetchone()
            
            if row:
                strategy = dict(row)
                
                # Konvertera JSON-parametrar
                if 'params' in strategy and strategy['params']:
                    try:
                        strategy['params'] = json.loads(strategy['params'])
                    except:
                        pass
                
                return strategy
            
            return None
            
        except Exception as e:
            logging.error(f"Fel vid hämtning av strategi: {e}")
            return None
    
    def get_all_strategies(self, active_only=False):
        """
        Hämta alla strategier från databasen
        
        Args:
            active_only: Om endast aktiva strategier ska returneras
            
        Returns:
            list: Lista med strategier
        """
        try:
            if active_only:
                cursor = self.execute("SELECT * FROM strategies WHERE is_active = 1 ORDER BY performance DESC")
            else:
                cursor = self.execute("SELECT * FROM strategies ORDER BY performance DESC")
            
            strategies = []
            for row in cursor.fetchall():
                strategy = dict(row)
                
                # Konvertera JSON-parametrar
                if 'params' in strategy and strategy['params']:
                    try:
                        strategy['params'] = json.loads(strategy['params'])
                    except:
                        pass
                
                strategies.append(strategy)
            
            return strategies
            
        except Exception as e:
            logging.error(f"Fel vid hämtning av strategier: {e}")
            return []
    
    def delete_strategy(self, strategy_id):
        """
        Ta bort en strategi från databasen
        
        Args:
            strategy_id: Strategi-ID
            
        Returns:
            bool: True om borttagning lyckades
        """
        try:
            self.execute("DELETE FROM strategies WHERE id = ?", (strategy_id,))
            self.commit()
            logging.info(f"🗑️ Strategi borttagen: {strategy_id}")
            return True
            
        except Exception as e:
            logging.error(f"Fel vid borttagning av strategi: {e}")
            self.rollback()
            return False
    
    def update_strategy_performance(self, strategy_id, performance):
        """
        Uppdatera prestanda för en strategi
        
        Args:
            strategy_id: Strategi-ID
            performance: Prestationspoäng
            
        Returns:
            bool: True om uppdatering lyckades
        """
        try:
            self.execute("""
                UPDATE strategies 
                SET performance = ?, updated_at = ?
                WHERE id = ?
            """, (performance, datetime.now().isoformat(), strategy_id))
            
            self.commit()
            return True
            
        except Exception as e:
            logging.error(f"Fel vid uppdatering av strategiprestanda: {e}")
            self.rollback()
            return False
    
    def save_result(self, result):
        """
        Spara ett resultat i databasen
        
        Args:
            result: Resultatobjekt
            
        Returns:
            str: Resultat-ID
        """
        try:
            # Kontrollera om resultatet redan har ett ID
            if not result.get('id'):
                result['id'] = f"result_{uuid.uuid4().hex[:8]}"
            
            # Konvertera data till JSON
            if 'data' in result and not isinstance(result['data'], str):
                result['data'] = json.dumps(result['data'])
            
            # Aktuell tidsstämpel
            if not result.get('created_at'):
                result['created_at'] = datetime.now().isoformat()
            
            # Standardvärden
            if 'success' not in result:
                result['success'] = 0
            
            # Sätt in resultat
            self.execute("""
                INSERT INTO results 
                (id, strategy_id, cluster_id, instance_id, success, data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result['id'],
                result.get('strategy_id'),
                result.get('cluster_id'),
                result.get('instance_id'),
                result.get('success', 0),
                result.get('data', '{}'),
                result['created_at']
            ))
            
            self.commit()
            return result['id']
            
        except Exception as e:
            logging.error(f"Fel vid sparande av resultat: {e}")
            self.rollback()
            raise
    
    def get_results(self, strategy_id=None, limit=100, offset=0):
        """
        Hämta resultat från databasen
        
        Args:
            strategy_id: Filtrera på strategi (eller None för alla)
            limit: Maxantal resultat
            offset: Startposition
            
        Returns:
            list: Lista med resultat
        """
        try:
            if strategy_id:
                cursor = self.execute("""
                    SELECT * FROM results 
                    WHERE strategy_id = ? 
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (strategy_id, limit, offset))
            else:
                cursor = self.execute("""
                    SELECT * FROM results 
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                
                # Konvertera JSON-data
                if 'data' in result and result['data']:
                    try:
                        result['data'] = json.loads(result['data'])
                    except:
                        pass
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logging.error(f"Fel vid hämtning av resultat: {e}")
            return []
    
    def get_strategy_performance_history(self, strategy_id, limit=100):
        """
        Hämta prestationshistorik för en strategi
        
        Args:
            strategy_id: Strategi-ID
            limit: Maxantal resultat
            
        Returns:
            list: Lista med prestationsdata
        """
        try:
            cursor = self.execute("""
                SELECT created_at, success, data
                FROM results 
                WHERE strategy_id = ? 
                ORDER BY created_at DESC
                LIMIT ?
            """, (strategy_id, limit))
            
            history = []
            for row in cursor.fetchall():
                data = dict(row)
                
                # Konvertera JSON-data
                if 'data' in data and data['data']:
                    try:
                        data_obj = json.loads(data['data'])
                        # Om data innehåller en performance-nyckel, använd den
                        if 'performance' in data_obj:
                            data['performance'] = data_obj['performance']
                        # Annars använd success-flaggan
                        else:
                            data['performance'] = 1.0 if data['success'] else 0.0
                    except:
                        data['performance'] = 1.0 if data['success'] else 0.0
                else:
                    data['performance'] = 1.0 if data['success'] else 0.0
                
                history.append({
                    'timestamp': data['created_at'],
                    'performance': data['performance']
                })
            
            return history
            
        except Exception as e:
            logging.error(f"Fel vid hämtning av strategiprestationshistorik: {e}")
            return []
    
    def save_learning_data(self, learning_data):
        """
        Spara inlärningsdata i databasen
        
        Args:
            learning_data: Inlärningsdataobjekt
            
        Returns:
            str: Inlärningsdata-ID
        """
        try:
            # Kontrollera om inlärningsdata redan har ett ID
            if not learning_data.get('id'):
                learning_data['id'] = f"learning_{uuid.uuid4().hex[:8]}"
            
            # Konvertera state/action/next_state till stringar
            for field in ['state', 'action', 'next_state']:
                if field in learning_data and not isinstance(learning_data[field], str):
                    if isinstance(learning_data[field], (list, dict)):
                        learning_data[field] = json.dumps(learning_data[field])
                    else:
                        learning_data[field] = str(learning_data[field])
            
            # Aktuell tidsstämpel
            if not learning_data.get('created_at'):
                learning_data['created_at'] = datetime.now().isoformat()
            
            # Sätt in inlärningsdata
            self.execute("""
                INSERT INTO learning_data 
                (id, type, state, action, reward, next_state, done, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                learning_data['id'],
                learning_data.get('type', 'default'),
                learning_data.get('state', ''),
                learning_data.get('action', ''),
                learning_data.get('reward', 0.0),
                learning_data.get('next_state', ''),
                learning_data.get('done', 0),
                learning_data['created_at']
            ))
            
            self.commit()
            return learning_data['id']
            
        except Exception as e:
            logging.error(f"Fel vid sparande av inlärningsdata: {e}")
            self.rollback()
            raise
    
    def get_learning_data_batch(self, data_type="default", batch_size=64):
        """
        Hämta en batch med inlärningsdata för träning
        
        Args:
            data_type: Typ av inlärningsdata
            batch_size: Storlek på batchen
            
        Returns:
            list: Lista med inlärningsdata
        """
        try:
            cursor = self.execute("""
                SELECT * FROM learning_data 
                WHERE type = ? 
                ORDER BY RANDOM() 
                LIMIT ?
            """, (data_type, batch_size))
            
            batch = []
            for row in cursor.fetchall():
                data = dict(row)
                
                # Konvertera JSON-data
                for field in ['state', 'action', 'next_state']:
                    if field in data and data[field]:
                        try:
                            data[field] = json.loads(data[field])
                        except:
                            pass
                
                batch.append(data)
            
            return batch
            
        except Exception as e:
            logging.error(f"Fel vid hämtning av inlärningsdata: {e}")
            return []
    
    def get_learning_data_count(self, data_type="default"):
        """
        Hämta antal inlärningsdata av en viss typ
        
        Args:
            data_type: Typ av inlärningsdata
            
        Returns:
            int: Antal inlärningsdata
        """
        try:
            cursor = self.execute("SELECT COUNT(*) as count FROM learning_data WHERE type = ?", (data_type,))
            result = cursor.fetchone()
            return result['count'] if result else 0
        except Exception as e:
            logging.error(f"Fel vid hämtning av inlärningsdataantal: {e}")
            return 0
    
    def clear_old_learning_data(self, data_type="default", max_age_days=30):
        """
        Rensa gamla inlärningsdata
        
        Args:
            data_type: Typ av inlärningsdata
            max_age_days: Maximal ålder i dagar
            
        Returns:
            int: Antal borttagna poster
        """
        try:
            # Beräkna tidsgräns
            cutoff_date = (datetime.now() - datetime.timedelta(days=max_age_days)).isoformat()
            
            cursor = self.execute("""
                DELETE FROM learning_data 
                WHERE type = ? AND created_at < ?
            """, (data_type, cutoff_date))
            
            deleted_count = cursor.rowcount
            self.commit()
            
            logging.info(f"🧹 {deleted_count} gamla inlärningsdata borttagna")
            return deleted_count
            
        except Exception as e:
            logging.error(f"Fel vid rensning av gamla inlärningsdata: {e}")
            self.rollback()
            return 0
    
    def execute_custom_query(self, query, params=None):
        """
        Utför en anpassad fråga
        
        Args:
            query: SQL-fråga
            params: Parameteruppsättning
            
        Returns:
            list: Resultat som lista med dictionaries
        """
        try:
            cursor = self.execute(query, params)
            
            # För SELECT-frågor, returnera resultatet
            if query.strip().lower().startswith("select"):
                results = []
                for row in cursor.fetchall():
                    results.append(dict(row))
                return results
            # För andra frågor, returnera antal påverkade rader
            else:
                self.commit()
                return [{"affected_rows": cursor.rowcount}]
                
        except Exception as e:
            logging.error(f"Fel vid anpassad fråga: {e}")
            self.rollback()
            raise
    
    def get_database_stats(self):
        """
        Hämta statistik om databasen
        
        Returns:
            dict: Databasstatistik
        """
        stats = {
            "db_type": self.db_type,
            "tables": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Strategistatistik
            cursor = self.execute("SELECT COUNT(*) as count, AVG(performance) as avg_performance FROM strategies")
            strategy_stats = dict(cursor.fetchone())
            stats["tables"]["strategies"] = {
                "count": strategy_stats["count"],
                "avg_performance": strategy_stats["avg_performance"]
            }
            
            # Resultatstatistik
            cursor = self.execute("""
                SELECT COUNT(*) as count, 
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count
                FROM results
            """)
            result_stats = dict(cursor.fetchone())
            stats["tables"]["results"] = {
                "count": result_stats["count"],
                "success_count": result_stats["success_count"],
                "success_rate": result_stats["success_count"] / result_stats["count"] if result_stats["count"] > 0 else 0
            }
            
            # Inlärningsdatastatistik
            cursor = self.execute("""
                SELECT type, COUNT(*) as count, AVG(reward) as avg_reward
                FROM learning_data
                GROUP BY type
            """)
            stats["tables"]["learning_data"] = {
                "total_count": 0,
                "by_type": {}
            }
            
            for row in cursor.fetchall():
                data = dict(row)
                stats["tables"]["learning_data"]["by_type"][data["type"]] = {
                    "count": data["count"],
                    "avg_reward": data["avg_reward"]
                }
                stats["tables"]["learning_data"]["total_count"] += data["count"]
            
            # Databasstorlek
            if self.db_type == "sqlite" and self.connection_params:
                try:
                    db_path = self.connection_params
                    stats["size_bytes"] = os.path.getsize(db_path)
                    stats["size_mb"] = stats["size_bytes"] / (1024 * 1024)
                except:
                    pass
            
            return stats
            
        except Exception as e:
            logging.error(f"Fel vid hämtning av databasstatistik: {e}")
            return stats
    
    def backup_database(self, backup_path=None):
        """
        Skapa en säkerhetskopia av databasen
        
        Args:
            backup_path: Sökväg för säkerhetskopia (eller None för auto)
            
        Returns:
            str: Sökväg till säkerhetskopian
        """
        if self.db_type == "sqlite":
            return self._backup_sqlite(backup_path)
        else:
            logging.warning(f"Säkerhetskopiering stöds inte för {self.db_type}")
            return None
    
    def _backup_sqlite(self, backup_path=None):
        """
        Skapa en säkerhetskopia av SQLite-databasen
        
        Args:
            backup_path: Sökväg för säkerhetskopia (eller None för auto)
            
        Returns:
            str: Sökväg till säkerhetskopian
        """
        try:
            # Generera sökväg om ingen anges
            if backup_path is None:
                backup_dir = "data/db/backups"
                os.makedirs(backup_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(backup_dir, f"ai_system_{timestamp}.db")
            
            # Skapa ny databasanslutning för backup
            source_path = self.connection_params or DEFAULT_DB_PATH
            
            # Skapa en temporär anslutning för att undvika låsproblem
            with self.connection_lock:
                source_conn = sqlite3.connect(source_path)
                backup_conn = sqlite3.connect(backup_path)
                
                source_conn.backup(backup_conn)
                
                backup_conn.close()
                source_conn.close()
            
            logging.info(f"💾 Databasbackup skapad: {backup_path}")
            return backup_path
            
        except Exception as e:
            logging.error(f"Fel vid säkerhetskopiering av databas: {e}")
            return None

# Singleton-instans
_db_manager = None

def get_database_manager(db_type="sqlite", connection_params=None):
    """Hämta den globala databashanteraren"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_type, connection_params)
    return _db_manager

def shutdown_database():
    """Stäng databaskopplingen"""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None

# Om vi kör direkt, utför ett enkelt test
if __name__ == "__main__":
    # Skapa testdata
    db = get_database_manager()
    
    # Lägg till en teststrategi
    strategy = {
        "name": "Test Strategy",
        "description": "A test strategy for database functionality",
        "params": {
            "focus_area": "desktop",
            "interaction_type": "clicks_first",
            "risk_level": "low"
        },
        "performance": 0.75
    }
    
    strategy_id = db.save_strategy(strategy)
    print(f"Sparad strategi med ID: {strategy_id}")
    
    # Lägg till några testresultat
    for i in range(5):
        success = i % 2 == 0  # Varannan lyckas
        result = {
            "strategy_id": strategy_id,
            "cluster_id": "test_cluster",
            "instance_id": f"instance_{i}",
            "success": 1 if success else 0,
            "data": {
                "action": "click",
                "x": 100 + i * 10,
                "y": 200 + i * 5,
                "performance": 0.5 + (i * 0.1)
            }
        }
        
        result_id = db.save_result(result)
        print(f"Sparade resultat {i+1}: {result_id}")
    
    # Hämta och visa strategin
    retrieved_strategy = db.get_strategy(strategy_id)
    print(f"\nHämtad strategi: {retrieved_strategy['name']}")
    print(f"Parametrar: {retrieved_strategy['params']}")
    
    # Visa resultat
    results = db.get_results(strategy_id)
    print(f"\nResultat för strategi {strategy_id}:")
    for result in results:
        success = "Lyckades" if result['success'] else "Misslyckades"
        print(f"  {result['id']}: {success} - {result['data']}")
    
    # Spara inlärningsdata
    for i in range(10):
        state = {"position": i, "screen_elements": 5 + i}
        action = {"type": "click", "x": 100 + i, "y": 200 + i}
        reward = 0.5 if i % 2 == 0 else -0.2
        
        learning_data = {
            "type": "qlearning",
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": {"position": i + 1, "screen_elements": 5 + i + 1},
            "done": 1 if i == 9 else 0
        }
        
        db.save_learning_data(learning_data)
    
    # Hämta inlärningsdata
    batch = db.get_learning_data_batch("qlearning", 5)
    print(f"\nInlärningsdata (5 slumpmässiga):")
    for data in batch:
        print(f"  Reward: {data['reward']}, Action: {data['action']}")
    
    # Visa statistik
    stats = db.get_database_stats()
    print(f"\nDatabasstatistik:")
    print(f"  Strategier: {stats['tables']['strategies']['count']}")
    print(f"  Resultat: {stats['tables']['results']['count']} (framgångsfrekvens: {stats['tables']['results']['success_rate']:.2f})")
    print(f"  Inlärningsdata: {stats['tables']['learning_data']['total_count']}")
    
    # Skapa backup
    backup_path = db.backup_database()
    print(f"\nSäkerhetskopia skapad: {backup_path}")
    
    # Stäng databasen
    shutdown_database()