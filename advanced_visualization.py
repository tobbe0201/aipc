"""
Advanced Visualization - Detaljerade visualiseringsverktyg för AI Desktop Controller

Detta modul erbjuder avancerade diagramverktyg för att visualisera 
strategiprestanda och systemstatistik.
"""

import os
import time
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk

# Konfigurera loggning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("advanced_visualization.log"),
        logging.StreamHandler()
    ]
)

# Sätt Seaborn-tema för snyggare grafer
sns.set_theme(style="darkgrid")

class PerformanceData:
    """Klass för att hantera prestationsdata för visualisering"""
    
    def __init__(self):
        """Initiera prestationsdatahanterare"""
        self.strategy_performance = {}  # strategy_id -> [data_points]
        self.cluster_performance = {}   # cluster_id -> [data_points]
        self.qlearning_metrics = {
            'rewards': [],
            'losses': [],
            'epsilons': [],
            'action_distribution': {}
        }
        self.system_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'detection_counts': [],
            'success_rates': []
        }
        
        # Datamappar
        os.makedirs("data/visualization", exist_ok=True)
        
        logging.info("✅ PerformanceData initierad")
    
    def add_strategy_performance(self, strategy_id, performance, timestamp=None):
        """
        Lägg till prestationsdata för en strategi
        
        Args:
            strategy_id: ID för strategin
            performance: Prestationsvärde (0-1)
            timestamp: Tidsstämpel eller None för aktuell tid
        """
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = []
        
        self.strategy_performance[strategy_id].append({
            'value': performance,
            'timestamp': timestamp or time.time()
        })
    
    def add_cluster_performance(self, cluster_id, performance, strategy_id=None, timestamp=None):
        """
        Lägg till prestationsdata för ett kluster
        
        Args:
            cluster_id: ID för klustret
            performance: Prestationsvärde (0-1)
            strategy_id: ID för strategin klustret använder (eller None)
            timestamp: Tidsstämpel eller None för aktuell tid
        """
        if cluster_id not in self.cluster_performance:
            self.cluster_performance[cluster_id] = []
        
        self.cluster_performance[cluster_id].append({
            'value': performance,
            'strategy_id': strategy_id,
            'timestamp': timestamp or time.time()
        })
    
    def add_qlearning_metrics(self, reward=None, loss=None, epsilon=None, actions=None):
        """
        Lägg till Q-learning-mätvärden
        
        Args:
            reward: Belöningsvärde eller None
            loss: Förlustvärde eller None
            epsilon: Epsilon-värde eller None
            actions: Dict med {action: count} eller None
        """
        timestamp = time.time()
        
        if reward is not None:
            self.qlearning_metrics['rewards'].append({
                'value': reward,
                'timestamp': timestamp
            })
        
        if loss is not None:
            self.qlearning_metrics['losses'].append({
                'value': loss,
                'timestamp': timestamp
            })
        
        if epsilon is not None:
            self.qlearning_metrics['epsilons'].append({
                'value': epsilon,
                'timestamp': timestamp
            })
        
        if actions is not None:
            for action, count in actions.items():
                if action not in self.qlearning_metrics['action_distribution']:
                    self.qlearning_metrics['action_distribution'][action] = []
                
                self.qlearning_metrics['action_distribution'][action].append({
                    'value': count,
                    'timestamp': timestamp
                })
    
    def add_system_metrics(self, cpu_usage=None, memory_usage=None, 
                          detection_count=None, success_rate=None):
        """
        Lägg till systemmätvärden
        
        Args:
            cpu_usage: CPU-användning (0-100) eller None
            memory_usage: Minnesanvändning i MB eller None
            detection_count: Antal detektioner eller None
            success_rate: Framgångsfrekvens (0-1) eller None
        """
        timestamp = time.time()
        
        if cpu_usage is not None:
            self.system_metrics['cpu_usage'].append({
                'value': cpu_usage,
                'timestamp': timestamp
            })
        
        if memory_usage is not None:
            self.system_metrics['memory_usage'].append({
                'value': memory_usage,
                'timestamp': timestamp
            })
        
        if detection_count is not None:
            self.system_metrics['detection_counts'].append({
                'value': detection_count,
                'timestamp': timestamp
            })
        
        if success_rate is not None:
            self.system_metrics['success_rates'].append({
                'value': success_rate,
                'timestamp': timestamp
            })
    
    def get_strategy_performance(self, strategy_id=None, time_range=None):
        """
        Hämta prestationsdata för strategier
        
        Args:
            strategy_id: ID för specifik strategi eller None för alla
            time_range: Tidintervall i sekunder eller None för all data
            
        Returns:
            dict eller list: Prestationsdata
        """
        if strategy_id is not None:
            if strategy_id not in self.strategy_performance:
                return []
            
            data = self.strategy_performance[strategy_id]
        else:
            data = {sid: points for sid, points in self.strategy_performance.items()}
        
        # Filtrera baserat på tidsintervall om angivet
        if time_range is not None:
            now = time.time()
            min_time = now - time_range
            
            if strategy_id is not None:
                data = [point for point in data if point['timestamp'] >= min_time]
            else:
                for sid in data:
                    data[sid] = [point for point in data[sid] if point['timestamp'] >= min_time]
        
        return data
    
    def get_cluster_performance(self, cluster_id=None, time_range=None):
        """
        Hämta prestationsdata för kluster
        
        Args:
            cluster_id: ID för specifikt kluster eller None för alla
            time_range: Tidintervall i sekunder eller None för all data
            
        Returns:
            dict eller list: Prestationsdata
        """
        if cluster_id is not None:
            if cluster_id not in self.cluster_performance:
                return []
            
            data = self.cluster_performance[cluster_id]
        else:
            data = {cid: points for cid, points in self.cluster_performance.items()}
        
        # Filtrera baserat på tidsintervall om angivet
        if time_range is not None:
            now = time.time()
            min_time = now - time_range
            
            if cluster_id is not None:
                data = [point for point in data if point['timestamp'] >= min_time]
            else:
                for cid in data:
                    data[cid] = [point for point in data[cid] if point['timestamp'] >= min_time]
        
        return data
    
    def get_qlearning_metrics(self, metric_type, time_range=None):
        """
        Hämta Q-learning-mätvärden
        
        Args:
            metric_type: Typ av mätvärde ('rewards', 'losses', 'epsilons', 'action_distribution')
            time_range: Tidintervall i sekunder eller None för all data
            
        Returns:
            list eller dict: Mätvärden
        """
        if metric_type not in self.qlearning_metrics:
            return []
        
        data = self.qlearning_metrics[metric_type]
        
        # För action_distribution, returnera hela dictionaries
        if metric_type == 'action_distribution':
            if time_range is not None:
                now = time.time()
                min_time = now - time_range
                
                result = {}
                for action, points in data.items():
                    result[action] = [point for point in points if point['timestamp'] >= min_time]
                
                return result
            
            return data
        
        # För andra mätvärden, filtrera baserat på tidsintervall om angivet
        if time_range is not None:
            now = time.time()
            min_time = now - time_range
            
            data = [point for point in data if point['timestamp'] >= min_time]
        
        return data
    
    def get_system_metrics(self, metric_type, time_range=None):
        """
        Hämta systemmätvärden
        
        Args:
            metric_type: Typ av mätvärde ('cpu_usage', 'memory_usage', 'detection_counts', 'success_rates')
            time_range: Tidintervall i sekunder eller None för all data
            
        Returns:
            list: Mätvärden
        """
        if metric_type not in self.system_metrics:
            return []
        
        data = self.system_metrics[metric_type]
        
        # Filtrera baserat på tidsintervall om angivet
        if time_range is not None:
            now = time.time()
            min_time = now - time_range
            
            data = [point for point in data if point['timestamp'] >= min_time]
        
        return data
    
    def save_to_file(self, filename=None):
        """
        Spara all data till fil
        
        Args:
            filename: Filnamn eller None för automatgenererat
            
        Returns:
            str: Filnamn
        """
        if filename is None:
            filename = f"data/visualization/perf_data_{int(time.time())}.json"
        
        data = {
            'strategy_performance': self.strategy_performance,
            'cluster_performance': self.cluster_performance,
            'qlearning_metrics': self.qlearning_metrics,
            'system_metrics': self.system_metrics,
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logging.info(f"💾 Prestationsdata sparad till {filename}")
        return filename
    
    def load_from_file(self, filename):
        """
        Ladda data från fil
        
        Args:
            filename: Filnamn att ladda från
            
        Returns:
            bool: True om laddning lyckades
        """
        if not os.path.exists(filename):
            logging.warning(f"⚠️ Filen {filename} hittades inte")
            return False
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            if 'strategy_performance' in data:
                self.strategy_performance = data['strategy_performance']
            
            if 'cluster_performance' in data:
                self.cluster_performance = data['cluster_performance']
            
            if 'qlearning_metrics' in data:
                self.qlearning_metrics = data['qlearning_metrics']
            
            if 'system_metrics' in data:
                self.system_metrics = data['system_metrics']
            
            logging.info(f"📂 Prestationsdata laddad från {filename}")
            return True
            
        except Exception as e:
            logging.error(f"Fel vid laddning av data från {filename}: {e}")
            return False

class AdvancedVisualization:
    """Huvudklass för avancerad visualisering"""
    
    def __init__(self, perf_data=None):
        """
        Initiera avancerad visualisering
        
        Args:
            perf_data: PerformanceData-instans eller None för att skapa en ny
        """
        self.perf_data = perf_data or PerformanceData()
        
        # Figurstilar
        self.style = {
            'figsize': (10, 6),
            'dpi': 100,
            'cmap': 'viridis',
            'fontsize': 12,
            'title_fontsize': 14,
            'dark_mode': True
        }
        
        # Initiera olika graftypsgeneratorer
        self.chart_generators = {
            'line': self._create_line_chart,
            'bar': self._create_bar_chart,
            'scatter': self._create_scatter_chart,
            'heatmap': self._create_heatmap,
            'pie': self._create_pie_chart,
            'box': self._create_box_chart,
            'histogram': self._create_histogram,
            'area': self._create_area_chart,
            'candlestick': self._create_candlestick_chart,
            'radar': self._create_radar_chart,
            'bubble': self._create_bubble_chart,
            'parallel_coordinates': self._create_parallel_coordinates,
            'violin': self._create_violin_chart,
            'ridgeline': self._create_ridgeline_chart,
            'correlation': self._create_correlation_chart
        }
        
        # Ställ in Seaborn-tema baserat på dark mode
        if self.style['dark_mode']:
            sns.set_theme(style="darkgrid", palette="muted")
            plt.rcParams.update({
                'figure.facecolor': 'black',
                'axes.facecolor': '#222222',
                'axes.edgecolor': 'white',
                'axes.labelcolor': 'white',
                'text.color': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white',
                'grid.color': '#444444'
            })
        else:
            sns.set_theme(style="whitegrid")
        
        logging.info("✅ AdvancedVisualization initierad")
    
    def create_chart(self, chart_type, data=None, title=None, xlabel=None, ylabel=None, 
                    figsize=None, output_file=None, **kwargs):
        """
        Skapa ett diagram
        
        Args:
            chart_type: Typ av diagram ('line', 'bar', 'scatter', etc.)
            data: Data att visualisera (eller None för att använda perf_data)
            title: Titel för diagrammet
            xlabel: Etikett för x-axeln
            ylabel: Etikett för y-axeln
            figsize: Storlek på figuren (tuple) eller None för standardvärdet
            output_file: Filnamn att spara till (eller None)
            **kwargs: Extra parametrar för specifika diagramtyper
            
        Returns:
            matplotlib.figure.Figure: Diagramfigur
        """
        # Kontrollera att diagramtypen stöds
        if chart_type not in self.chart_generators:
            logging.warning(f"⚠️ Diagramtypen {chart_type} stöds inte")
            return None
        
        # Skapa figur
        figsize = figsize or self.style['figsize']
        fig = plt.figure(figsize=figsize, dpi=self.style['dpi'])
        
        # Anropa specifik diagramgenerator
        generator = self.chart_generators[chart_type]
        fig = generator(fig, data, **kwargs)
        
        # Lägg till titel och etiketter
        if title:
            plt.title(title, fontsize=self.style['title_fontsize'])
        
        if xlabel:
            plt.xlabel(xlabel, fontsize=self.style['fontsize'])
        
        if ylabel:
            plt.ylabel(ylabel, fontsize=self.style['fontsize'])
        
        # Använd Seaborns tema
        sns.despine()
        
        # Justera layout
        plt.tight_layout()
        
        # Spara till fil om angivet
        if output_file:
            try:
                plt.savefig(output_file, dpi=self.style['dpi'])
                logging.info(f"💾 Diagram sparat till {output_file}")
            except Exception as e:
                logging.error(f"Fel vid sparande av diagram: {e}")
        
        return fig
    
    def create_strategy_performance_chart(self, strategy_ids=None, time_range=3600,
                                        chart_type='line', output_file=None):
        """
        Skapa diagram för strategiprestanda
        
        Args:
            strategy_ids: Lista med strategi-ID:n att visa (eller None för alla)
            time_range: Tidintervall i sekunder (eller None för all data)
            chart_type: Typ av diagram ('line', 'area', 'bar')
            output_file: Filnamn att spara till (eller None)
            
        Returns:
            matplotlib.figure.Figure: Diagramfigur
        """
        # Hämta data
        if strategy_ids is None:
            data = self.perf_data.get_strategy_performance(time_range=time_range)
            strategy_ids = list(data.keys())
        else:
            data = {sid: self.perf_data.get_strategy_performance(sid, time_range)
                  for sid in strategy_ids}
        
        # Skapa DataFrame för enklare plotting
        df_data = []
        
        for sid in strategy_ids:
            if sid in data and data[sid]:
                for point in data[sid]:
                    df_data.append({
                        'strategy_id': sid,
                        'timestamp': point['timestamp'],
                        'value': point['value'],
                        'datetime': datetime.fromtimestamp(point['timestamp'])
                    })
        
        if not df_data:
            logging.warning("⚠️ Ingen data att visualisera")
            return None
        
        df = pd.DataFrame(df_data)
        
        # Skapa diagram
        if chart_type == 'line':
            return self.create_chart(
                'line',
                df,
                title="Strategiprestanda över tid",
                xlabel="Tid",
                ylabel="Prestanda",
                output_file=output_file,
                x_column='datetime',
                y_column='value',
                group_column='strategy_id'
            )
        elif chart_type == 'area':
            return self.create_chart(
                'area',
                df,
                title="Strategiprestanda över tid",
                xlabel="Tid",
                ylabel="Prestanda",
                output_file=output_file,
                x_column='datetime',
                y_column='value',
                group_column='strategy_id'
            )
        elif chart_type == 'bar':
            # För stapeldiagram, beräkna genomsnittlig prestanda per strategi
            avg_perf = df.groupby('strategy_id')['value'].mean().reset_index()
            
            return self.create_chart(
                'bar',
                avg_perf,
                title="Genomsnittlig strategiprestanda",
                xlabel="Strategi",
                ylabel="Genomsnittlig prestanda",
                output_file=output_file,
                x_column='strategy_id',
                y_column='value'
            )
        else:
            logging.warning(f"⚠️ Diagramtypen {chart_type} stöds inte för strategiprestanda")
            return None
    
    def create_cluster_comparison_chart(self, cluster_ids=None, time_range=3600,
                                      include_strategies=True, output_file=None):
        """
        Skapa jämförelsediagram för kluster
        
        Args:
            cluster_ids: Lista med kluster-ID:n att visa (eller None för alla)
            time_range: Tidintervall i sekunder (eller None för all data)
            include_strategies: Om strategiinformation ska inkluderas
            output_file: Filnamn att spara till (eller None)
            
        Returns:
            matplotlib.figure.Figure: Diagramfigur
        """
        # Hämta data
        if cluster_ids is None:
            data = self.perf_data.get_cluster_performance(time_range=time_range)
            cluster_ids = list(data.keys())
        else:
            data = {cid: self.perf_data.get_cluster_performance(cid, time_range)
                  for cid in cluster_ids}
        
        # Skapa DataFrame för enklare plotting
        df_data = []
        
        for cid in cluster_ids:
            if cid in data and data[cid]:
                for point in data[cid]:
                    df_data.append({
                        'cluster_id': cid,
                        'strategy_id': point.get('strategy_id', 'unknown'),
                        'timestamp': point['timestamp'],
                        'value': point['value'],
                        'datetime': datetime.fromtimestamp(point['timestamp'])
                    })
        
        if not df_data:
            logging.warning("⚠️ Ingen data att visualisera")
            return None
        
        df = pd.DataFrame(df_data)
        
        # För stapeldiagram, beräkna genomsnittlig prestanda per kluster
        avg_perf = df.groupby(['cluster_id', 'strategy_id'] if include_strategies else ['cluster_id'])['value'].mean().reset_index()
        
        # Skapa diagram
        if include_strategies:
            # Skapa grupperade staplar per strategi
            return self.create_chart(
                'bar',
                avg_perf,
                title="Klusterprestanda per strategi",
                xlabel="Kluster",
                ylabel="Genomsnittlig prestanda",
                output_file=output_file,
                x_column='cluster_id',
                y_column='value',
                group_column='strategy_id'
            )
        else:
            # Skapa vanliga staplar
            return self.create_chart(
                'bar',
                avg_perf,
                title="Klusterprestanda",
                xlabel="Kluster",
                ylabel="Genomsnittlig prestanda",
                output_file=output_file,
                x_column='cluster_id',
                y_column='value'
            )
    
    def create_qlearning_dashboard(self, time_range=3600, output_file=None):
        """
        Skapa en dashboard med Q-learning-mätvärden
        
        Args:
            time_range: Tidintervall i sekunder (eller None för all data)
            output_file: Filnamn att spara till (eller None)
            
        Returns:
            matplotlib.figure.Figure: Diagramfigur
        """
        # Hämta data
        rewards = self.perf_data.get_qlearning_metrics('rewards', time_range)
        losses = self.perf_data.get_qlearning_metrics('losses', time_range)
        epsilons = self.perf_data.get_qlearning_metrics('epsilons', time_range)
        actions = self.perf_data.get_qlearning_metrics('action_distribution', time_range)
        
        # Skapa DataFrames
        rewards_df = pd.DataFrame([
            {'timestamp': point['timestamp'], 'value': point['value'], 
             'datetime': datetime.fromtimestamp(point['timestamp'])}
            for point in rewards
        ]) if rewards else pd.DataFrame()
        
        losses_df = pd.DataFrame([
            {'timestamp': point['timestamp'], 'value': point['value'], 
             'datetime': datetime.fromtimestamp(point['timestamp'])}
            for point in losses
        ]) if losses else pd.DataFrame()
        
        epsilons_df = pd.DataFrame([
            {'timestamp': point['timestamp'], 'value': point['value'], 
             'datetime': datetime.fromtimestamp(point['timestamp'])}
            for point in epsilons
        ]) if epsilons else pd.DataFrame()
        
        # Förbereda handlingsdata
        actions_data = []
        for action, points in actions.items():
            for point in points:
                actions_data.append({
                    'action': action,
                    'timestamp': point['timestamp'],
                    'value': point['value'],
                    'datetime': datetime.fromtimestamp(point['timestamp'])
                })
        
        actions_df = pd.DataFrame(actions_data) if actions_data else pd.DataFrame()
        
        # Kontrollera om det finns data att visa
        if (rewards_df.empty and losses_df.empty and 
            epsilons_df.empty and actions_df.empty):
            logging.warning("⚠️ Ingen Q-learning-data att visualisera")
            return None
        
        # Skapa dashboard med flera diagram
        fig = plt.figure(figsize=(12, 10), dpi=self.style['dpi'])
        
        # Lägg till underdiagram
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])
        
        # Belöningar över tid
        if not rewards_df.empty:
            ax1 = fig.add_subplot(gs[0, 0])
            sns.lineplot(data=rewards_df, x='datetime', y='value', ax=ax1)
            ax1.set_title("Belöningar över tid")
            ax1.set_xlabel("")
            ax1.set_ylabel("Belöning")
        
        # Förluster över tid
        if not losses_df.empty:
            ax2 = fig.add_subplot(gs[0, 1])
            sns.lineplot(data=losses_df, x='datetime', y='value', ax=ax2)
            ax2.set_title("Träningsförluster över tid")
            ax2.set_xlabel("")
            ax2.set_ylabel("Förlust")
        
        # Epsilon över tid
        if not epsilons_df.empty:
            ax3 = fig.add_subplot(gs[1, 0])
            sns.lineplot(data=epsilons_df, x='datetime', y='value', ax=ax3)
            ax3.set_title("Epsilon-utveckling")
            ax3.set_xlabel("")
            ax3.set_ylabel("Epsilon")
            ax3.set_ylim(0, 1)
        
        # Handlingsfördelning
        if not actions_df.empty:
            # Beräkna totalt antal handlingar per typ
            action_counts = actions_df.groupby('action')['value'].sum().reset_index()
            
            ax4 = fig.add_subplot(gs[1, 1])
            sns.barplot(data=action_counts, x='action', y='value', ax=ax4)
            ax4.set_title("Handlingsfördelning")
            ax4.set_xlabel("Handling")
            ax4.set_ylabel("Antal")
            plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")
        
        # Om vi har tillräckligt med belöningsdata, visa rullande medelvärde
        if len(rewards_df) > 10:
            ax5 = fig.add_subplot(gs[2, 0])
            rewards_df['rolling_avg'] = rewards_df['value'].rolling(window=10).mean()
            sns.lineplot(data=rewards_df, x='datetime', y='value', alpha=0.3, ax=ax5)
            sns.lineplot(data=rewards_df, x='datetime', y='rolling_avg', color='red', ax=ax5)
            ax5.set_title("Belöning med rullande medelvärde (10 episoder)")
            ax5.set_xlabel("")
            ax5.set_ylabel("Belöning")
            ax5.legend(['Belöning', 'Rullande medelvärde'])
        
        # För varje belöningsintervall, visa fördelning över episoder
        if not rewards_df.empty:
            ax6 = fig.add_subplot(gs[2, 1])
            sns.histplot(data=rewards_df, x='value', kde=True, ax=ax6)
            ax6.set_title("Belöningsfördelning")
            ax6.set_xlabel("Belöning")
            ax6.set_ylabel("Antal episoder")
        
        # Statistik om framgång och misslyckande
        if not rewards_df.empty:
            # Visa stapeldiagram med olika belöningsnivåer
            ax7 = fig.add_subplot(gs[3, 0])
            
            # Skapa belöningskategorier
            rewards_df['category'] = pd.cut(
                rewards_df['value'],
                bins=[-float('inf'), 0, 50, 100, float('inf')],
                labels=['Negativ', 'Låg', 'Medel', 'Hög']
            )
            
            # Räkna antal episoder per kategori
            category_counts = rewards_df['category'].value_counts().reset_index()
            category_counts.columns = ['category', 'count']
            
            # Sortera kategorierna
            category_order = ['Negativ', 'Låg', 'Medel', 'Hög']
            category_counts['category'] = pd.Categorical(
                category_counts['category'],
                categories=category_order,
                ordered=True
            )
            category_counts = category_counts.sort_values('category')
            
            sns.barplot(data=category_counts, x='category', y='count', ax=ax7)
            ax7.set_title("Belöningskategorier")
            ax7.set_xlabel("Belöningsnivå")
            ax7.set_ylabel("Antal episoder")
        
        # Epsilon vs belöning (scatter plot)
        if not rewards_df.empty and not epsilons_df.empty:
            # Slå ihop dataframes baserat på tidsstämpel
            merged_df = pd.merge_asof(
                rewards_df.sort_values('timestamp'),
                epsilons_df.sort_values('timestamp'),
                on='timestamp',
                suffixes=('_reward', '_epsilon')
            )
            
            if not merged_df.empty:
                ax8 = fig.add_subplot(gs[3, 1])
                sns.scatterplot(data=merged_df, x='value_epsilon', y='value_reward', ax=ax8)
                ax8.set_title("Epsilon vs Belöning")
                ax8.set_xlabel("Epsilon")
                ax8.set_ylabel("Belöning")
        
        # Justera layout
        plt.tight_layout()
        
        # Spara till fil om angivet
        if output_file:
            try:
                plt.savefig(output_file, dpi=self.style['dpi'])
                logging.info(f"💾 Q-learning-dashboard sparad till {output_file}")
            except Exception as e:
                logging.error(f"Fel vid sparande av dashboard: {e}")
        
        return fig
    
    def create_system_dashboard(self, time_range=3600, output_file=None):
        """
        Skapa en dashboard med systemmätvärden
        
        Args:
            time_range: Tidintervall i sekunder (eller None för all data)
            output_file: Filnamn att spara till (eller None)
            
        Returns:
            matplotlib.figure.Figure: Diagramfigur
        """
        # Hämta data
        cpu_usage = self.perf_data.get_system_metrics('cpu_usage', time_range)
        memory_usage = self.perf_data.get_system_metrics('memory_usage', time_range)
        detection_counts = self.perf_data.get_system_metrics('detection_counts', time_range)
        success_rates = self.perf_data.get_system_metrics('success_rates', time_range)
        
        # Skapa DataFrames
        cpu_df = pd.DataFrame([
            {'timestamp': point['timestamp'], 'value': point['value'], 
             'datetime': datetime.fromtimestamp(point['timestamp'])}
            for point in cpu_usage
        ]) if cpu_usage else pd.DataFrame()
        
        memory_df = pd.DataFrame([
            {'timestamp': point['timestamp'], 'value': point['value'], 
             'datetime': datetime.fromtimestamp(point['timestamp'])}
            for point in memory_usage
        ]) if memory_usage else pd.DataFrame()
        
        detection_df = pd.DataFrame([
            {'timestamp': point['timestamp'], 'value': point['value'], 
             'datetime': datetime.fromtimestamp(point['timestamp'])}
            for point in detection_counts
        ]) if detection_counts else pd.DataFrame()
        
        success_df = pd.DataFrame([
            {'timestamp': point['timestamp'], 'value': point['value'], 
             'datetime': datetime.fromtimestamp(point['timestamp'])}
            for point in success_rates
        ]) if success_rates else pd.DataFrame()
        
        # Kontrollera om det finns data att visa
        if (cpu_df.empty and memory_df.empty and 
            detection_df.empty and success_df.empty):
            logging.warning("⚠️ Inga systemmätvärden att visualisera")
            return None
        
        # Skapa dashboard med flera diagram
        fig = plt.figure(figsize=(12, 10), dpi=self.style['dpi'])
        
        # Lägg till underdiagram
        gs = fig.add_gridspec(2, 2)
        
        # CPU-användning över tid
        if not cpu_df.empty:
            ax1 = fig.add_subplot(gs[0, 0])
            sns.lineplot(data=cpu_df, x='datetime', y='value', ax=ax1)
            ax1.set_title("CPU-användning över tid")
            ax1.set_xlabel("")
            ax1.set_ylabel("CPU-användning (%)")
            ax1.set_ylim(0, 100)
        
        # Minnesanvändning över tid
        if not memory_df.empty:
            ax2 = fig.add_subplot(gs[0, 1])
            sns.lineplot(data=memory_df, x='datetime', y='value', ax=ax2)
            ax2.set_title("Minnesanvändning över tid")
            ax2.set_xlabel("")
            ax2.set_ylabel("Minnesanvändning (MB)")
        
        # Antal detektioner över tid
        if not detection_df.empty:
            ax3 = fig.add_subplot(gs[1, 0])
            sns.lineplot(data=detection_df, x='datetime', y='value', ax=ax3)
            ax3.set_title("Antal detektioner över tid")
            ax3.set_xlabel("")
            ax3.set_ylabel("Detektioner")
        
        # Framgångsfrekvens över tid
        if not success_df.empty:
            ax4 = fig.add_subplot(gs[1, 1])
            sns.lineplot(data=success_df, x='datetime', y='value', ax=ax4)
            ax4.set_title("Framgångsfrekvens över tid")
            ax4.set_xlabel("")
            ax4.set_ylabel("Framgångsfrekvens")
            ax4.set_ylim(0, 1)
        
        # Justera layout
        plt.tight_layout()
        
        # Spara till fil om angivet
        if output_file:
            try:
                plt.savefig(output_file, dpi=self.style['dpi'])
                logging.info(f"💾 Systemdashboard sparad till {output_file}")
            except Exception as e:
                logging.error(f"Fel vid sparande av dashboard: {e}")
        
        return fig
    
    def create_correlation_matrix(self, metrics=None, time_range=3600, output_file=None):
        """
        Skapa korrelationsmatris för systemets mätvärden
        
        Args:
            metrics: Lista med mätvärden att inkludera (eller None för alla)
            time_range: Tidintervall i sekunder (eller None för all data)
            output_file: Filnamn att spara till (eller None)
            
        Returns:
            matplotlib.figure.Figure: Diagramfigur
        """
        # Förbered data
        all_metrics = {}
        
        # Hämta Q-learning-mätvärden
        all_metrics['reward'] = [{'timestamp': p['timestamp'], 'value': p['value']}
                               for p in self.perf_data.get_qlearning_metrics('rewards', time_range)]
        
        all_metrics['loss'] = [{'timestamp': p['timestamp'], 'value': p['value']}
                             for p in self.perf_data.get_qlearning_metrics('losses', time_range)]
        
        all_metrics['epsilon'] = [{'timestamp': p['timestamp'], 'value': p['value']}
                                for p in self.perf_data.get_qlearning_metrics('epsilons', time_range)]
        
        # Hämta systemmätvärden
        all_metrics['cpu_usage'] = [{'timestamp': p['timestamp'], 'value': p['value']}
                                  for p in self.perf_data.get_system_metrics('cpu_usage', time_range)]
        
        all_metrics['memory_usage'] = [{'timestamp': p['timestamp'], 'value': p['value']}
                                     for p in self.perf_data.get_system_metrics('memory_usage', time_range)]
        
        all_metrics['detection_count'] = [{'timestamp': p['timestamp'], 'value': p['value']}
                                        for p in self.perf_data.get_system_metrics('detection_counts', time_range)]
        
        all_metrics['success_rate'] = [{'timestamp': p['timestamp'], 'value': p['value']}
                                     for p in self.perf_data.get_system_metrics('success_rates', time_range)]
        
        # Filtrera mätvärden om angivet
        if metrics is not None:
            all_metrics = {k: v for k, v in all_metrics.items() if k in metrics}
        
        # Skapa en DataFrame för varje mätvärde
        dfs = {}
        for metric_name, metric_data in all_metrics.items():
            if metric_data:
                dfs[metric_name] = pd.DataFrame(metric_data)
                dfs[metric_name]['datetime'] = dfs[metric_name]['timestamp'].apply(
                    lambda x: datetime.fromtimestamp(x)
                )
                dfs[metric_name].set_index('datetime', inplace=True)
                dfs[metric_name] = dfs[metric_name]['value']
        
        # Slå ihop alla DataFrames till en
        if dfs:
            df = pd.DataFrame(dfs)
            
            # Ta bort rader med NaN-värden
            df = df.dropna()
            
            if not df.empty:
                # Beräkna korrelationsmatris
                corr = df.corr()
                
                # Skapa heatmap
                fig = plt.figure(figsize=(10, 8), dpi=self.style['dpi'])
                ax = fig.add_subplot(111)
                
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", 
                          linewidths=0.5, ax=ax)
                
                plt.title("Korrelationsmatris för systemmätvärden", fontsize=self.style['title_fontsize'])
                
                # Justera layout
                plt.tight_layout()
                
                # Spara till fil om angivet
                if output_file:
                    try:
                        plt.savefig(output_file, dpi=self.style['dpi'])
                        logging.info(f"💾 Korrelationsmatris sparad till {output_file}")
                    except Exception as e:
                        logging.error(f"Fel vid sparande av korrelationsmatris: {e}")
                
                return fig
        
        logging.warning("⚠️ Otillräcklig data för korrelationsmatris")
        return None
    
    def _create_line_chart(self, fig, data, x_column=None, y_column=None, 
                         group_column=None, **kwargs):
        """
        Skapa ett linjediagram
        
        Args:
            fig: Matplotlib Figure
            data: Pandas DataFrame
            x_column: Kolumnnamn för x-axeln
            y_column: Kolumnnamn för y-axeln
            group_column: Kolumnnamn för gruppering (eller None)
            **kwargs: Extra parametrar
            
        Returns:
            matplotlib.figure.Figure: Figur med diagrammet
        """
        ax = fig.add_subplot(111)
        
        if group_column is not None:
            # Gruppera efter kolumn
            for name, group in data.groupby(group_column):
                ax.plot(group[x_column], group[y_column], label=name)
            
            ax.legend()
        else:
            # Enkelt linjediagram
            ax.plot(data[x_column], data[y_column])
        
        return fig
    
    def _create_bar_chart(self, fig, data, x_column=None, y_column=None, 
                        group_column=None, **kwargs):
        """
        Skapa ett stapeldiagram
        
        Args:
            fig: Matplotlib Figure
            data: Pandas DataFrame
            x_column: Kolumnnamn för x-axeln
            y_column: Kolumnnamn för y-axeln
            group_column: Kolumnnamn för gruppering (eller None)
            **kwargs: Extra parametrar
            
        Returns:
            matplotlib.figure.Figure: Figur med diagrammet
        """
        ax = fig.add_subplot(111)
        
        if group_column is not None:
            # Grupperade staplar
            sns.barplot(data=data, x=x_column, y=y_column, hue=group_column, ax=ax)
        else:
            # Enkla staplar
            sns.barplot(data=data, x=x_column, y=y_column, ax=ax)
        
        # Rotera x-etiketter om det är många värden
        if len(data[x_column].unique()) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        return fig
    
    def _create_scatter_chart(self, fig, data, x_column=None, y_column=None, 
                            group_column=None, **kwargs):
        """
        Skapa ett spridningsdiagram
        
        Args:
            fig: Matplotlib Figure
            data: Pandas DataFrame
            x_column: Kolumnnamn för x-axeln
            y_column: Kolumnnamn för y-axeln
            group_column: Kolumnnamn för gruppering (eller None)
            **kwargs: Extra parametrar
            
        Returns:
            matplotlib.figure.Figure: Figur med diagrammet
        """
        ax = fig.add_subplot(111)
        
        if group_column is not None:
            # Grupperat spridningsdiagram
            sns.scatterplot(data=data, x=x_column, y=y_column, hue=group_column, ax=ax)
        else:
            # Enkelt spridningsdiagram
            sns.scatterplot(data=data, x=x_column, y=y_column, ax=ax)
        
        return fig
    
    def _create_heatmap(self, fig, data, **kwargs):
        """
        Skapa en heatmap
        
        Args:
            fig: Matplotlib Figure
            data: Pandas DataFrame eller numpy-array
            **kwargs: Extra parametrar
            
        Returns:
            matplotlib.figure.Figure: Figur med diagrammet
        """
        ax = fig.add_subplot(111)
        
        # Om data är en DataFrame, använd den direkt
        # annars, skapa en korrelationsmatris
        if isinstance(data, pd.DataFrame):
            if len(data.columns) <= 1:
                logging.warning("⚠️ Otillräcklig data för heatmap")
                return fig
            
            if 'correlation' in kwargs and kwargs['correlation']:
                data = data.corr()
        
        sns.heatmap(data, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5, ax=ax)
        
        return fig
    
    def _create_pie_chart(self, fig, data, value_column=None, label_column=None, **kwargs):
        """
        Skapa ett cirkeldiagram
        
        Args:
            fig: Matplotlib Figure
            data: Pandas DataFrame
            value_column: Kolumnnamn för värden
            label_column: Kolumnnamn för etiketter
            **kwargs: Extra parametrar
            
        Returns:
            matplotlib.figure.Figure: Figur med diagrammet
        """
        ax = fig.add_subplot(111)
        
        values = data[value_column].values
        labels = data[label_column].values if label_column else None
        
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # För att få en cirkulär form
        
        return fig
    
    def _create_box_chart(self, fig, data, x_column=None, y_column=None, **kwargs):
        """
        Skapa ett lådagram
        
        Args:
            fig: Matplotlib Figure
            data: Pandas DataFrame
            x_column: Kolumnnamn för x-axeln
            y_column: Kolumnnamn för y-axeln
            **kwargs: Extra parametrar
            
        Returns:
            matplotlib.figure.Figure: Figur med diagrammet
        """
        ax = fig.add_subplot(111)
        
        sns.boxplot(data=data, x=x_column, y=y_column, ax=ax)
        
        return fig
    
    def _create_histogram(self, fig, data, column=None, bins=None, **kwargs):
        """
        Skapa ett histogram
        
        Args:
            fig: Matplotlib Figure
            data: Pandas DataFrame
            column: Kolumnnamn för värden
            bins: Antal bins (eller None för auto)
            **kwargs: Extra parametrar
            
        Returns:
            matplotlib.figure.Figure: Figur med diagrammet
        """
        ax = fig.add_subplot(111)
        
        sns.histplot(data=data, x=column, bins=bins, kde=True, ax=ax)
        
        return fig
    
    def _create_area_chart(self, fig, data, x_column=None, y_column=None, 
                         group_column=None, **kwargs):
        """
        Skapa ett areadiagram
        
        Args:
            fig: Matplotlib Figure
            data: Pandas DataFrame
            x_column: Kolumnnamn för x-axeln
            y_column: Kolumnnamn för y-axeln
            group_column: Kolumnnamn för gruppering (eller None)
            **kwargs: Extra parametrar
            
        Returns:
            matplotlib.figure.Figure: Figur med diagrammet
        """
        ax = fig.add_subplot(111)
        
        if group_column is not None:
            # Grupperad area
            pivot_data = data.pivot_table(index=x_column, columns=group_column, values=y_column)
            pivot_data.plot.area(ax=ax, stacked=kwargs.get('stacked', False))
        else:
            # Enkel area
            ax.fill_between(data[x_column], data[y_column])
        
        return fig
    
    def _create_candlestick_chart(self, fig, data, **kwargs):
        """
        Skapa ett candlestick-diagram
        
        Args:
            fig: Matplotlib Figure
            data: Pandas DataFrame med kolumnerna 'date', 'open', 'high', 'low', 'close'
            **kwargs: Extra parametrar
            
        Returns:
            matplotlib.figure.Figure: Figur med diagrammet
        """
        from matplotlib.finance import candlestick_ohlc
        import matplotlib.dates as mdates
        
        ax = fig.add_subplot(111)
        
        # Konvertera datum till matplotlib-format
        data['date'] = mdates.date2num(data['date'])
        
        # Skapa ett OHLC-diagram
        candlestick_ohlc(ax, data[['date', 'open', 'high', 'low', 'close']].values, 
                        width=0.6, colorup='green', colordown='red')
        
        # Formatera x-axeln
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        return fig
    
    def _create_radar_chart(self, fig, data, categories=None, values=None, 
                          group_column=None, **kwargs):
        """
        Skapa ett radardiagram
        
        Args:
            fig: Matplotlib Figure
            data: Pandas DataFrame
            categories: Kolumnnamn för kategorier
            values: Kolumnnamn för värden
            group_column: Kolumnnamn för gruppering (eller None)
            **kwargs: Extra parametrar
            
        Returns:
            matplotlib.figure.Figure: Figur med diagrammet
        """
        from math import pi
        
        # Antal kategorier
        N = len(categories)
        
        # Vinklar för varje axel
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Stäng cirkeln
        
        # Skapa subplot med polära koordinater
        ax = fig.add_subplot(111, polar=True)
        
        # Om vi har ett gruppkolumn, dela upp data
        if group_column is not None:
            groups = data[group_column].unique()
            
            for i, group in enumerate(groups):
                group_data = data[data[group_column] == group]
                
                # Hämta värdena
                values = group_data[categories].iloc[0].values.flatten().tolist()
                values += values[:1]  # Stäng cirkeln
                
                # Plotta värdena
                ax.plot(angles, values, linewidth=1, linestyle='solid', label=group)
                ax.fill(angles, values, alpha=0.1)
        else:
            # Hämta värdena
            values = data[categories].iloc[0].values.flatten().tolist()
            values += values[:1]  # Stäng cirkeln
            
            # Plotta värdena
            ax.plot(angles, values, linewidth=1, linestyle='solid')
            ax.fill(angles, values, alpha=0.1)
        
        # Sätt etiketter för varje axel
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Visa legend om vi har gruppering
        if group_column is not None:
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        return fig
    
    def _create_bubble_chart(self, fig, data, x_column=None, y_column=None, 
                           size_column=None, group_column=None, **kwargs):
        """
        Skapa ett bubbeldiagram
        
        Args:
            fig: Matplotlib Figure
            data: Pandas DataFrame
            x_column: Kolumnnamn för x-axeln
            y_column: Kolumnnamn för y-axeln
            size_column: Kolumnnamn för bubbelstorlek
            group_column: Kolumnnamn för gruppering (eller None)
            **kwargs: Extra parametrar
            
        Returns:
            matplotlib.figure.Figure: Figur med diagrammet
        """
        ax = fig.add_subplot(111)
        
        if group_column is not None:
            # Grupperat bubbeldiagram
            for name, group in data.groupby(group_column):
                ax.scatter(group[x_column], group[y_column], 
                        s=group[size_column], label=name, alpha=0.6)
            
            ax.legend()
        else:
            # Enkelt bubbeldiagram
            ax.scatter(data[x_column], data[y_column], s=data[size_column], alpha=0.6)
        
        return fig
    
    def _create_parallel_coordinates(self, fig, data, columns=None, group_column=None, **kwargs):
        """
        Skapa ett parallellkoordinatdiagram
        
        Args:
            fig: Matplotlib Figure
            data: Pandas DataFrame
            columns: Lista med kolumnnamn att inkludera
            group_column: Kolumnnamn för gruppering (eller None)
            **kwargs: Extra parametrar
            
        Returns:
            matplotlib.figure.Figure: Figur med diagrammet
        """
        from pandas.plotting import parallel_coordinates
        
        ax = fig.add_subplot(111)
        
        # Om inga kolumner anges, använd alla
        if columns is None:
            columns = data.columns.tolist()
            if group_column in columns:
                columns.remove(group_column)
        
        # Om vi har ett gruppkolumn
        if group_column is not None:
            parallel_coordinates(data, group_column, cols=columns, ax=ax)
        else:
            # Skapa en dummy-kolumn för gruppering
            data['_group'] = 0
            parallel_coordinates(data, '_group', cols=columns, ax=ax)
        
        return fig
    
    def _create_violin_chart(self, fig, data, x_column=None, y_column=None, **kwargs):
        """
        Skapa ett violindiagram
        
        Args:
            fig: Matplotlib Figure
            data: Pandas DataFrame
            x_column: Kolumnnamn för x-axeln
            y_column: Kolumnnamn för y-axeln
            **kwargs: Extra parametrar
            
        Returns:
            matplotlib.figure.Figure: Figur med diagrammet
        """
        ax = fig.add_subplot(111)
        
        sns.violinplot(data=data, x=x_column, y=y_column, ax=ax)
        
        return fig
    
    def _create_ridgeline_chart(self, fig, data, x_column=None, y_column=None, **kwargs):
        """
        Skapa ett ridgeline-diagram (joyplot)
        
        Args:
            fig: Matplotlib Figure
            data: Pandas DataFrame
            x_column: Kolumnnamn för x-axeln
            y_column: Kolumnnamn för y-axeln (gruppering)
            **kwargs: Extra parametrar
            
        Returns:
            matplotlib.figure.Figure: Figur med diagrammet
        """
        from matplotlib.collections import PolyCollection
        
        # Hämta unika grupper
        groups = data[y_column].unique()
        
        # Skapa subplot
        ax = fig.add_subplot(111)
        
        # Skapa ett KDE-plot för varje grupp
        for i, group in enumerate(groups):
            group_data = data[data[y_column] == group][x_column]
            
            # Beräkna KDE
            density = sns.kdeplot(group_data, bw_adjust=.5, cut=0,
                                fill=True, alpha=1.0, linewidth=1.5,
                                y=i, clip_on=False)
        
        # Ställ in y-ticks
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups)
        
        # Ta bort y-axel
        ax.set_ylabel('')
        
        return fig
    
    def _create_correlation_chart(self, fig, data, **kwargs):
        """
        Skapa ett korrelationsdiagram
        
        Args:
            fig: Matplotlib Figure
            data: Pandas DataFrame
            **kwargs: Extra parametrar
            
        Returns:
            matplotlib.figure.Figure: Figur med diagrammet
        """
        # Beräkna korrelationsmatris
        corr = data.corr()
        
        # Skapa heatmap
        ax = fig.add_subplot(111)
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        
        return fig

class AdvancedVisualizationDashboard(ctk.CTk):
    """Dashboard för avancerad visualisering"""
    
    def __init__(self, perf_data=None):
        """
        Initiera dashboarden
        
        Args:
            perf_data: PerformanceData-instans eller None för att skapa en ny
        """
        super().__init__()
        
        # Konfigurera fönster
        self.title("Advanced Visualization Dashboard")
        self.geometry("1200x800")
        
        # PerformanceData och visualiseringshanterare
        self.perf_data = perf_data or PerformanceData()
        self.visualizer = AdvancedVisualization(self.perf_data)
        
        # Skapa UI
        self._create_ui()
        
        logging.info("✅ AdvancedVisualizationDashboard initierad")
    
    def _create_ui(self):
        """Skapa användargränssnittet"""
        # Konfigurera grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)
        
        # Skapa vänster kontrollpanel
        self.control_panel = ctk.CTkFrame(self)
        self.control_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self._create_control_panel()
        
        # Skapa höger visningspanel
        self.display_panel = ctk.CTkFrame(self)
        self.display_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self._create_display_panel()
    
    def _create_control_panel(self):
        """Skapa kontrollpanelen"""
        # Konfigurera grid
        self.control_panel.grid_columnconfigure(0, weight=1)
        
        # Titel
        title_label = ctk.CTkLabel(self.control_panel, text="Visualiseringskontroller",
                                 font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        # Diagramtypsväljare
        chart_frame = ctk.CTkFrame(self.control_panel)
        chart_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        chart_label = ctk.CTkLabel(chart_frame, text="Diagramtyp:")
        chart_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.chart_type_var = tk.StringVar(value="line")
        chart_types = ["line", "bar", "scatter", "heatmap", "area", "pie", 
                      "box", "histogram", "radar", "bubble", "correlation"]
        
        chart_dropdown = ctk.CTkOptionMenu(chart_frame, values=chart_types, 
                                         variable=self.chart_type_var,
                                         command=self._on_chart_type_changed)
        chart_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Tidsintervall
        time_frame = ctk.CTkFrame(self.control_panel)
        time_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        time_label = ctk.CTkLabel(time_frame, text="Tidsintervall:")
        time_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.time_range_var = tk.StringVar(value="3600")
        time_ranges = {
            "1 timme": "3600",
            "3 timmar": "10800",
            "12 timmar": "43200",
            "1 dag": "86400",
            "1 vecka": "604800",
            "Allt": "0"
        }
        
        time_dropdown = ctk.CTkOptionMenu(time_frame, 
                                        values=list(time_ranges.keys()), 
                                        command=lambda x: self.time_range_var.set(time_ranges[x]))
        time_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Dashboards
        dashboard_frame = ctk.CTkFrame(self.control_panel)
        dashboard_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        
        dashboard_label = ctk.CTkLabel(dashboard_frame, text="Dashboards:", 
                                      font=("Arial", 14, "bold"))
        dashboard_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        qlearning_btn = ctk.CTkButton(dashboard_frame, text="Q-Learning Dashboard", 
                                     command=self._show_qlearning_dashboard)
        qlearning_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        system_btn = ctk.CTkButton(dashboard_frame, text="System Dashboard", 
                                  command=self._show_system_dashboard)
        system_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        correlation_btn = ctk.CTkButton(dashboard_frame, text="Correlation Matrix", 
                                       command=self._show_correlation_matrix)
        correlation_btn.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        
        # Strategi- och klusterjämförelser
        comparison_frame = ctk.CTkFrame(self.control_panel)
        comparison_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        
        comparison_label = ctk.CTkLabel(comparison_frame, text="Jämförelser:", 
                                      font=("Arial", 14, "bold"))
        comparison_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        strategy_btn = ctk.CTkButton(comparison_frame, text="Strategiprestanda", 
                                    command=self._show_strategy_performance)
        strategy_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        cluster_btn = ctk.CTkButton(comparison_frame, text="Klusterjämförelse", 
                                   command=self._show_cluster_comparison)
        cluster_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        # Data-hantering
        data_frame = ctk.CTkFrame(self.control_panel)
        data_frame.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
        
        data_label = ctk.CTkLabel(data_frame, text="Datahantering:", 
                                 font=("Arial", 14, "bold"))
        data_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        save_btn = ctk.CTkButton(data_frame, text="Spara data", 
                               command=self._save_data)
        save_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        load_btn = ctk.CTkButton(data_frame, text="Ladda data", 
                               command=self._load_data)
        load_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        # Demo-data
        demo_btn = ctk.CTkButton(data_frame, text="Generera demo-data", 
                              command=self._generate_demo_data)
        demo_btn.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        
        # Export-knapp
        export_btn = ctk.CTkButton(self.control_panel, text="Exportera aktuell bild", 
                               command=self._export_current_chart)
        export_btn.grid(row=6, column=0, padx=10, pady=20, sticky="ew")
    
    def _create_display_panel(self):
        """Skapa visningspanelen"""
        # Konfigurera grid
        self.display_panel.grid_columnconfigure(0, weight=1)
        self.display_panel.grid_rowconfigure(0, weight=0)  # Titel
        self.display_panel.grid_rowconfigure(1, weight=1)  # Diagram
        
        # Titel
        self.chart_title = ctk.CTkLabel(self.display_panel, text="Visualisering", 
                                     font=("Arial", 16, "bold"))
        self.chart_title.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        # Diagram
        self.chart_frame = ctk.CTkFrame(self.display_panel)
        self.chart_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # Standarddiagram
        self._show_empty_chart()
    
    def _show_empty_chart(self):
        """Visa ett tomt diagram"""
        # Rensa befintligt diagram
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # Skapa ett tomt diagram
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlabel("X-axel")
        ax.set_ylabel("Y-axel")
        ax.set_title("Ingen data att visa")
        
        # Visa diagrammet
        self._show_figure(fig)
    
    def _show_figure(self, fig):
        """
        Visa en matplotlib-figur i chart_frame
        
        Args:
            fig: matplotlib.figure.Figure
        """
        # Rensa befintligt diagram
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # Ställ in bakgrundsfärg för figuren
        if hasattr(fig, 'patch'):
            fig.patch.set_facecolor('#2b2b2b')
        
        # Skapa canvas
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        
        # Lägg till canvas i frame
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Spara aktuell figur
        self.current_figure = fig
    
    def _on_chart_type_changed(self, chart_type):
        """Hantera byte av diagramtyp"""
        # Här kan vi uppdatera diagrammet baserat på vald typ
        pass
    
    def _show_qlearning_dashboard(self):
        """Visa Q-Learning dashboard"""
        self.chart_title.configure(text="Q-Learning Dashboard")
        
        # Hämta tidsintervall
        time_range = int(self.time_range_var.get())
        if time_range == 0:
            time_range = None
        
        # Skapa dashboard
        fig = self.visualizer.create_qlearning_dashboard(time_range)
        
        if fig:
            self._show_figure(fig)
        else:
            self._show_empty_chart()
    
    def _show_system_dashboard(self):
        """Visa systemdashboard"""
        self.chart_title.configure(text="System Dashboard")
        
        # Hämta tidsintervall
        time_range = int(self.time_range_var.get())
        if time_range == 0:
            time_range = None
        
        # Skapa dashboard
        fig = self.visualizer.create_system_dashboard(time_range)
        
        if fig:
            self._show_figure(fig)
        else:
            self._show_empty_chart()
    
    def _show_correlation_matrix(self):
        """Visa korrelationsmatris"""
        self.chart_title.configure(text="Korrelationsmatris")
        
        # Hämta tidsintervall
        time_range = int(self.time_range_var.get())
        if time_range == 0:
            time_range = None
        
        # Skapa korrelationsmatris
        fig = self.visualizer.create_correlation_matrix(time_range=time_range)
        
        if fig:
            self._show_figure(fig)
        else:
            self._show_empty_chart()
    
    def _show_strategy_performance(self):
        """Visa strategiprestanda"""
        self.chart_title.configure(text="Strategiprestanda")
        
        # Hämta tidsintervall
        time_range = int(self.time_range_var.get())
        if time_range == 0:
            time_range = None
        
        # Skapa diagram
        chart_type = self.chart_type_var.get()
        if chart_type not in ['line', 'area', 'bar']:
            chart_type = 'line'
        
        fig = self.visualizer.create_strategy_performance_chart(
            time_range=time_range,
            chart_type=chart_type
        )
        
        if fig:
            self._show_figure(fig)
        else:
            self._show_empty_chart()
    
    def _show_cluster_comparison(self):
        """Visa klusterjämförelse"""
        self.chart_title.configure(text="Klusterjämförelse")
        
        # Hämta tidsintervall
        time_range = int(self.time_range_var.get())
        if time_range == 0:
            time_range = None
        
        # Skapa diagram
        fig = self.visualizer.create_cluster_comparison_chart(
            time_range=time_range,
            include_strategies=True
        )
        
        if fig:
            self._show_figure(fig)
        else:
            self._show_empty_chart()
    
    def _save_data(self):
        """Spara data till fil"""
        import tkinter.filedialog as filedialog
        
        filename = filedialog.asksaveasfilename(
            initialdir="./data/visualization",
            title="Spara prestationsdata",
            filetypes=(("JSON-filer", "*.json"), ("Alla filer", "*.*")),
            defaultextension=".json"
        )
        
        if filename:
            self.perf_data.save_to_file(filename)
    
    def _load_data(self):
        """Ladda data från fil"""
        import tkinter.filedialog as filedialog
        
        filename = filedialog.askopenfilename(
            initialdir="./data/visualization",
            title="Ladda prestationsdata",
            filetypes=(("JSON-filer", "*.json"), ("Alla filer", "*.*"))
        )
        
        if filename:
            self.perf_data.load_from_file(filename)
    
    def _generate_demo_data(self):
        """Generera demodata för visualisering"""
        # Rensa befintlig data
        self.perf_data = PerformanceData()
        self.visualizer.perf_data = self.perf_data
        
        # Generera strategidata
        strategy_ids = ["strategy_1", "strategy_2", "strategy_3"]
        
        # 100 datapunkter över senaste timmen
        now = time.time()
        for i in range(100):
            timestamp = now - (100 - i) * 36  # En timme tillbaka
            
            # Olika trender för olika strategier
            self.perf_data.add_strategy_performance(
                "strategy_1",
                0.5 + 0.4 * (i / 100) + 0.1 * np.sin(i / 5),  # Ökande trend med brus
                timestamp
            )
            
            self.perf_data.add_strategy_performance(
                "strategy_2",
                0.7 - 0.3 * (i / 100) + 0.1 * np.sin(i / 10),  # Minskande trend med brus
                timestamp
            )
            
            self.perf_data.add_strategy_performance(
                "strategy_3",
                0.6 + 0.2 * np.sin(i / 15),  # Cyklisk trend
                timestamp
            )
        
        # Generera klusterdata
        cluster_ids = ["cluster_1", "cluster_2", "cluster_3", "cluster_4"]
        
        for i in range(100):
            timestamp = now - (100 - i) * 36  # En timme tillbaka
            
            # Klusters använder olika strategier
            self.perf_data.add_cluster_performance(
                "cluster_1",
                0.5 + 0.4 * (i / 100) + 0.05 * np.sin(i / 5),
                "strategy_1",
                timestamp
            )
            
            self.perf_data.add_cluster_performance(
                "cluster_2",
                0.45 + 0.4 * (i / 100) + 0.05 * np.cos(i / 5),
                "strategy_1",
                timestamp
            )
            
            self.perf_data.add_cluster_performance(
                "cluster_3",
                0.7 - 0.3 * (i / 100) + 0.05 * np.sin(i / 10),
                "strategy_2",
                timestamp
            )
            
            self.perf_data.add_cluster_performance(
                "cluster_4",
                0.6 + 0.2 * np.sin(i / 15),
                "strategy_3",
                timestamp
            )
        
        # Generera Q-learning-mätvärden
        for i in range(100):
            timestamp = now - (100 - i) * 36  # En timme tillbaka
            
            # Belöning med ökande trend och brus
            reward = -50 + 150 * (i / 100) + 20 * np.sin(i / 10)
            
            # Förlust med minskande trend
            loss = 100 * np.exp(-i / 30) + 5 * np.random.randn()
            
            # Epsilon med exponentiell minskning
            epsilon = 1.0 * np.exp(-i / 40)
            
            # Handlingar
            actions = {
                "action_1": int(20 * (i / 100) + 5 * np.random.random()),
                "action_2": int(15 - 10 * (i / 100) + 3 * np.random.random()),
                "action_3": int(10 + 5 * np.sin(i / 20) + 2 * np.random.random())
            }
            
            self.perf_data.add_qlearning_metrics(reward, loss, epsilon, actions)
        
        # Generera systemmätvärden
        for i in range(100):
            timestamp = now - (100 - i) * 36  # En timme tillbaka
            
            # CPU-användning
            cpu = 20 + 10 * np.sin(i / 10) + 5 * np.random.random()
            
            # Minnesanvändning
            memory = 500 + 200 * (i / 100) + 20 * np.random.random()
            
            # Detektioner
            detections = 50 + 20 * np.sin(i / 15) + 10 * np.random.random()
            
            # Framgångsfrekvens
            success = 0.7 + 0.2 * (i / 100) + 0.05 * np.sin(i / 20)
            
            self.perf_data.add_system_metrics(cpu, memory, detections, success)
        
        # Visa en bekräftelsedialog
        import tkinter.messagebox as messagebox
        messagebox.showinfo("Demo-data", "Demo-data har genererats!")
        
        # Visa Q-learning-dashboard som standard
        self._show_qlearning_dashboard()
    
    def _export_current_chart(self):
        """Exportera aktuellt diagram till fil"""
        if not hasattr(self, 'current_figure'):
            return
        
        import tkinter.filedialog as filedialog
        
        filename = filedialog.asksaveasfilename(
            initialdir="./data/visualization",
            title="Exportera diagram",
            filetypes=(
                ("PNG-fil", "*.png"),
                ("PDF-fil", "*.pdf"),
                ("SVG-fil", "*.svg"),
                ("Alla filer", "*.*")
            ),
            defaultextension=".png"
        )
        
        if filename:
            self.current_figure.savefig(filename, dpi=300, bbox_inches='tight')
            
            # Visa bekräftelse
            self.chart_title.configure(text=f"Diagram exporterat till {filename}")

# Convenience-funktioner

def run_visualization_dashboard(perf_data=None):
    """Starta visualiseringsdashboard"""
    dashboard = AdvancedVisualizationDashboard(perf_data)
    dashboard.mainloop()

def create_strategy_performance_chart(perf_data, strategy_ids=None, 
                                    time_range=3600, chart_type='line',
                                    output_file=None):
    """Skapa ett strategiprestanda-diagram"""
    visualizer = AdvancedVisualization(perf_data)
    return visualizer.create_strategy_performance_chart(
        strategy_ids, time_range, chart_type, output_file
    )

def create_cluster_comparison_chart(perf_data, cluster_ids=None,
                                  time_range=3600, include_strategies=True,
                                  output_file=None):
    """Skapa ett klusterjämförelsediagram"""
    visualizer = AdvancedVisualization(perf_data)
    return visualizer.create_cluster_comparison_chart(
        cluster_ids, time_range, include_strategies, output_file
    )

def create_qlearning_dashboard(perf_data, time_range=3600, output_file=None):
    """Skapa en Q-learning-dashboard"""
    visualizer = AdvancedVisualization(perf_data)
    return visualizer.create_qlearning_dashboard(time_range, output_file)

def create_system_dashboard(perf_data, time_range=3600, output_file=None):
    """Skapa en systemdashboard"""
    visualizer = AdvancedVisualization(perf_data)
    return visualizer.create_system_dashboard(time_range, output_file)