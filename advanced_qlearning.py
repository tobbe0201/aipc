"""
Advanced Q-learning implementering för AI Desktop Controller

Detta modul tillhandahåller avancerade reinforcement learning-funktioner 
med Deep Q-Networks (DQN), experience replay och target networks.
"""

import os
import numpy as np
import random
import time
import json
import logging
import pickle
from collections import deque
from datetime import datetime

# Kontrollera om vi har GPU-stöd
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
    print(f"🧠 Neural networks kommer att köras på: {DEVICE}")
except ImportError:
    print("⚠️ PyTorch är inte installerat. Faller tillbaka på numpy-baserad implementation.")
    USE_CUDA = False
    

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("qlearning.log"),
        logging.StreamHandler()
    ]
)

class ReplayBuffer:
    """Experience replay buffer för att lagra och sampla tidigare erfarenheter"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        """Lägg till en erfarenhet i buffern"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Sampla en slumpmässig batch av erfarenheter"""
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
        
    def __len__(self):
        return len(self.buffer)
    
    def save(self, filename):
        """Spara buffern till fil"""
        with open(filename, 'wb') as f:
            pickle.dump(self.buffer, f)
        
    def load(self, filename):
        """Ladda buffern från fil"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.buffer = pickle.load(f)


class DQNModel:
    """Deep Q-Network modell"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        if USE_CUDA:
            self.model = self._build_torch_model(state_dim, action_dim, hidden_dim)
            self.target_model = self._build_torch_model(state_dim, action_dim, hidden_dim)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            # Initiera target model med samma vikter
            self.update_target_model()
        else:
            # Numpy-baserad fallback implementation
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
            # Enkla matriser för vikter i ett två-lagers nätverk
            self.weights1 = np.random.randn(state_dim, hidden_dim) / np.sqrt(state_dim)
            self.weights2 = np.random.randn(hidden_dim, action_dim) / np.sqrt(hidden_dim)
            self.target_weights1 = self.weights1.copy()
            self.target_weights2 = self.weights2.copy()
            self.learning_rate = 0.001
    
    def _build_torch_model(self, state_dim, action_dim, hidden_dim):
        """Bygg PyTorch DQN-modell"""
        model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(DEVICE)
        return model
    
    def predict(self, state, use_target=False):
        """Förutsäg Q-värden för ett givet tillstånd"""
        if USE_CUDA:
            state_tensor = torch.FloatTensor(state).to(DEVICE)
            with torch.no_grad():
                if use_target:
                    return self.target_model(state_tensor).cpu().numpy()
                else:
                    return self.model(state_tensor).cpu().numpy()
        else:
            # Numpy forward pass
            if use_target:
                hidden = np.dot(state, self.target_weights1)
                hidden = np.maximum(0, hidden)  # ReLU
                return np.dot(hidden, self.target_weights2)
            else:
                hidden = np.dot(state, self.weights1)
                hidden = np.maximum(0, hidden)  # ReLU
                return np.dot(hidden, self.weights2)
    
    def train(self, states, actions, rewards, next_states, dones, gamma=0.99):
        """Träna modellen på en batch av erfarenheter"""
        if USE_CUDA:
            # Konvertera till tensorer
            states_tensor = torch.FloatTensor(states).to(DEVICE)
            actions_tensor = torch.LongTensor(actions).to(DEVICE)
            rewards_tensor = torch.FloatTensor(rewards).to(DEVICE)
            next_states_tensor = torch.FloatTensor(next_states).to(DEVICE)
            dones_tensor = torch.FloatTensor(dones).to(DEVICE)
            
            # Beräkna förväntade Q-värden
            q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            
            # Beräkna nästa tillstånds Q-värden med target_model
            with torch.no_grad():
                next_q_values = self.target_model(next_states_tensor).max(1)[0]
                expected_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)
            
            # Beräkna loss och uppdatera modellen
            loss = F.mse_loss(q_values, expected_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
        else:
            # Numpy training implementation
            batch_size = len(states)
            losses = []
            
            for i in range(batch_size):
                state = states[i]
                action = actions[i]
                reward = rewards[i]
                next_state = next_states[i]
                done = dones[i]
                
                # Förutsäg Q-värden
                q_values = self.predict(state)
                
                # Beräkna target Q-värde
                target_q = q_values.copy()
                if done:
                    target_q[action] = reward
                else:
                    next_q = self.predict(next_state, use_target=True)
                    target_q[action] = reward + gamma * np.max(next_q)
                
                # Beräkna gradient och uppdatera vikter (förenklad)
                # Första lagret
                hidden = np.dot(state, self.weights1)
                hidden_activated = np.maximum(0, hidden)  # ReLU
                
                # Forward pass
                output = np.dot(hidden_activated, self.weights2)
                
                # Error i output layer
                error = output - target_q
                loss = np.sum(error**2)
                losses.append(loss)
                
                # Backpropagation (förenklad)
                d_output = error
                d_weights2 = np.outer(hidden_activated, d_output)
                
                # Uppdatera andra lagrets vikter
                self.weights2 -= self.learning_rate * d_weights2
                
                # Bakåtproparera till första lagret
                d_hidden = np.dot(d_output, self.weights2.T)
                d_hidden[hidden <= 0] = 0  # ReLU-derivatan
                d_weights1 = np.outer(state, d_hidden)
                
                # Uppdatera första lagrets vikter
                self.weights1 -= self.learning_rate * d_weights1
            
            return np.mean(losses)
    
    def update_target_model(self):
        """Uppdatera target model med vikter från huvudmodellen"""
        if USE_CUDA:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            self.target_weights1 = self.weights1.copy()
            self.target_weights2 = self.weights2.copy()
    
    def save(self, filename):
        """Spara modellen till fil"""
        if USE_CUDA:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'target_model_state_dict': self.target_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, filename)
        else:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'weights1': self.weights1,
                    'weights2': self.weights2,
                    'target_weights1': self.target_weights1,
                    'target_weights2': self.target_weights2
                }, f)
    
    def load(self, filename):
        """Ladda modellen från fil"""
        if os.path.exists(filename):
            if USE_CUDA:
                checkpoint = torch.load(filename)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✅ DQN-modell laddad från {filename}")
            else:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    self.weights1 = data['weights1']
                    self.weights2 = data['weights2']
                    self.target_weights1 = data['target_weights1']
                    self.target_weights2 = data['target_weights2']
                print(f"✅ Numpy-modell laddad från {filename}")
            return True
        return False


class AdvancedQLearningAgent:
    """Advanced Q-learning agent med DQN, experience replay och target networks"""
    
    def __init__(self, state_features, action_space, 
                 learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update_freq=10):
        """
        Initiera AdvancedQLearningAgent
        
        Args:
            state_features: Lista med funktioner/egenskaper i tillståndet
            action_space: Lista med möjliga handlingar
            learning_rate: Inlärningshastighet
            gamma: Diskonteringsfaktor för framtida belöningar
            epsilon_start: Startvärde för utforskningssannolikhet
            epsilon_min: Minimivärde för utforskningssannolikhet
            epsilon_decay: Minskningstakt för utforskningssannolikhet
            buffer_size: Storlek på experience replay buffer
            batch_size: Antal erfarenheter att träna på i varje batch
            target_update_freq: Hur ofta (antal träningssteg) target network uppdateras
        """
        # Möjligheten att representera tillstånd och handlingar
        self.state_features = state_features
        self.state_dim = len(state_features)
        self.action_space = action_space
        self.action_dim = len(action_space)
        
        # Hyperparametrar
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Skapa DQN-modell
        self.model = DQNModel(self.state_dim, self.action_dim)
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        # Träningsstatistik
        self.training_stats = {
            'episodes': 0,
            'steps': 0,
            'rewards': [],
            'losses': [],
            'epsilons': []
        }
        
        # Skapa mappar för att spara modeller och statistik
        os.makedirs("data/models", exist_ok=True)
        os.makedirs("data/stats", exist_ok=True)
        
        logging.info(f"✅ Avancerad Q-learning agent initialiserad med {self.state_dim} state features och {self.action_dim} handlingar")
    
    def select_action(self, state, explore=True):
        """Välj handling baserat på epsilon-greedy policy"""
        if explore and random.random() < self.epsilon:
            # Utforska: välj en slumpmässig handling
            return random.randint(0, self.action_dim - 1)
        else:
            # Utnyttja: välj handlingen med högst Q-värde
            q_values = self.model.predict(state)
            return np.argmax(q_values)
    
    def get_action_from_index(self, action_idx):
        """Konvertera handling-index till faktisk handling"""
        if 0 <= action_idx < len(self.action_space):
            return self.action_space[action_idx]
        return None
    
    def get_action_index(self, action):
        """Konvertera handling till index"""
        try:
            return self.action_space.index(action)
        except ValueError:
            return -1
    
    def remember(self, state, action, reward, next_state, done):
        """Lagra erfarenhet i replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self):
        """Träna modellen på en batch av erfarenheter från replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        # Sampla en batch från replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Förbered data för träning
        states = np.array([exp[0] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3] for exp in experiences])
        dones = np.array([exp[4] for exp in experiences], dtype=np.float32)
        
        # Träna modellen
        loss = self.model.train(states, actions, rewards, next_states, dones, self.gamma)
        
        # Uppdatera träningsstatistik
        self.training_stats['steps'] += 1
        self.training_stats['losses'].append(loss)
        
        # Uppdatera target model periodiskt
        if self.training_stats['steps'] % self.target_update_freq == 0:
            self.model.update_target_model()
            logging.info("🔄 Target network uppdaterad")
        
        return loss
    
    def update_epsilon(self):
        """Uppdatera epsilon (utforskningssannolikhet)"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.training_stats['epsilons'].append(self.epsilon)
    
    def end_episode(self, total_reward):
        """Slutför en episode och uppdatera statistik"""
        self.training_stats['episodes'] += 1
        self.training_stats['rewards'].append(total_reward)
        self.update_epsilon()
        
        # Logga statistik var 10:e episode
        if self.training_stats['episodes'] % 10 == 0:
            recent_rewards = self.training_stats['rewards'][-10:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            logging.info(f"📊 Episode {self.training_stats['episodes']}: "
                        f"Avg belöning={avg_reward:.2f}, "
                        f"Epsilon={self.epsilon:.4f}")
        
        # Spara modell och statistik var 100:e episode
        if self.training_stats['episodes'] % 100 == 0:
            self.save_model()
            self.save_stats()
    
    def save_model(self, filename=None):
        """Spara modell till fil"""
        if filename is None:
            filename = f"data/models/dqn_model_{int(time.time())}.pkl"
        
        self.model.save(filename)
        logging.info(f"💾 Modell sparad till {filename}")
    
    def load_model(self, filename):
        """Ladda modell från fil"""
        if self.model.load(filename):
            logging.info(f"📂 Modell laddad från {filename}")
            return True
        logging.warning(f"⚠️ Kunde inte ladda modell från {filename}")
        return False
    
    def save_stats(self, filename=None):
        """Spara träningsstatistik till fil"""
        if filename is None:
            filename = f"data/stats/training_stats_{int(time.time())}.json"
        
        # Konvertera numpy arrays till listor för JSON
        stats_json = {
            'episodes': self.training_stats['episodes'],
            'steps': self.training_stats['steps'],
            'rewards': [float(r) for r in self.training_stats['rewards']],
            'losses': [float(l) for l in self.training_stats['losses']],
            'epsilons': [float(e) for e in self.training_stats['epsilons']],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(stats_json, f, indent=2)
        
        logging.info(f"📊 Träningsstatistik sparad till {filename}")
    
    def load_stats(self, filename):
        """Ladda träningsstatistik från fil"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                stats = json.load(f)
            
            self.training_stats['episodes'] = stats['episodes']
            self.training_stats['steps'] = stats['steps']
            self.training_stats['rewards'] = stats['rewards']
            self.training_stats['losses'] = stats['losses']
            self.training_stats['epsilons'] = stats['epsilons']
            
            logging.info(f"📊 Träningsstatistik laddad från {filename}")
            return True
        
        logging.warning(f"⚠️ Kunde inte ladda statistik från {filename}")
        return False
    
    def save_buffer(self, filename=None):
        """Spara replay buffer till fil"""
        if filename is None:
            filename = f"data/models/replay_buffer_{int(time.time())}.pkl"
        
        self.replay_buffer.save(filename)
        logging.info(f"💾 Replay buffer sparad till {filename}")
    
    def load_buffer(self, filename):
        """Ladda replay buffer från fil"""
        self.replay_buffer.load(filename)
        logging.info(f"📂 Replay buffer laddad från {filename}")
    
    def get_q_values(self, state):
        """Hämta Q-värden för alla handlingar i ett givet tillstånd"""
        return self.model.predict(state)
    
    def get_state_features(self, screen_info, active_window, ocr_text):
        """
        Extrahera state features från skärminformation
        
        Detta är en förenklad implementation. I en faktisk implementation
        skulle denna metod extrahera meningsfulla funktioner från skärmdata.
        """
        # Exempel på feature extraction
        features = np.zeros(self.state_dim)
        
        # Detta är bara platshållare. En riktig implementation skulle
        # extrahera meningsfulla funktioner från skärmdata, aktivt fönster och OCR-text.
        
        return features