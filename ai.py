import numpy as np
from collections import deque

class SuperTrendAI:
    def __init__(self):
        # Input parameters (as per original)
        self.length = 10  # ATR Length
        self.minMult = 1  # Minimum factor
        self.maxMult = 5  # Maximum factor
        self.step = 0.5   # Step between factors
        self.perfAlpha = 10  # Performance memory
        self.fromCluster = 'Best'  # Cluster selection ('Best', 'Average', 'Worst')
        self.maxIter = 1000  # Maximum iteration steps for k-means
        self.maxData = 10000  # Historical bars calculation
        
        # Initialize factors array
        self.factors = np.arange(self.minMult, self.maxMult + self.step, self.step)
        if len(self.factors) == 0 or self.factors[-1] > self.maxMult:
            self.factors = self.factors[:-1]
        
        # Initialize data structures
        self.supertrends = [{
            'upper': 0,
            'lower': 0,
            'output': 0,
            'perf': 0,
            'factor': factor,
            'trend': 0
        } for factor in self.factors]
        
        # State variables
        self.target_factor = None
        self.perf_idx = None
        self.perf_ama = None
        self.upper = 0
        self.lower = 0
        self.os = 0
        self.ts = 0
        
        # History buffers
        self.close_history = deque(maxlen=self.length + 1)
        self.high_history = deque(maxlen=self.length + 1)
        self.low_history = deque(maxlen=self.length + 1)
        
        # K-means clusters
        self.factors_clusters = [[], [], []]
        self.perfclusters = [[], [], []]
        self.centroids = [0, 0, 0]
        
    def calculate_atr(self):
        if len(self.close_history) < self.length + 1:
            return 0
            
        tr_sum = 0
        for i in range(1, self.length + 1):
            high = self.high_history[-i]
            low = self.low_history[-i]
            prev_close = self.close_history[-i-1] if i < len(self.close_history) else self.close_history[-i]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
            
        return tr_sum / self.length
        
    def update_supertrends(self, atr):
        hl2 = (self.high_history[-1] + self.low_history[-1]) / 2
        close = self.close_history[-1]
        prev_close = self.close_history[-2] if len(self.close_history) > 1 else close
        
        for st in self.supertrends:
            factor = st['factor']
            up = hl2 + atr * factor
            dn = hl2 - atr * factor
            
            # Update trend direction
            if close > st['upper']:
                st['trend'] = 1
            elif close < st['lower']:
                st['trend'] = 0
            else:
                st['trend'] = st['trend']  # keep previous trend
                
            # Update upper/lower bands
            if prev_close < st['upper']:
                st['upper'] = min(up, st['upper'])
            else:
                st['upper'] = up
                
            if prev_close > st['lower']:
                st['lower'] = max(dn, st['lower'])
            else:
                st['lower'] = dn
                
            # Calculate performance
            prev_output = st['output'] if 'output' in st else hl2
            diff = np.sign(prev_close - prev_output) if prev_close != prev_output else 0
            perf_increment = (close - prev_close) * diff if prev_close != close else 0
            st['perf'] += (2 / (self.perfAlpha + 1)) * (perf_increment - st['perf'])
            
            # Set current output
            st['output'] = st['lower'] if st['trend'] == 1 else st['upper']
            
    def kmeans_clustering(self):
        # Prepare data
        data = np.array([st['perf'] for st in self.supertrends])
        factor_array = np.array([st['factor'] for st in self.supertrends])
        
        if len(data) == 0:
            return
            
        # Initialize centroids using quartiles
        self.centroids = [
            np.percentile(data, 25),
            np.percentile(data, 50),
            np.percentile(data, 75)
        ]
        
        # K-means clustering
        for _ in range(self.maxIter):
            # Reset clusters
            self.factors_clusters = [[], [], []]
            self.perfclusters = [[], [], []]
            
            # Assign to clusters
            for i, value in enumerate(data):
                distances = [abs(value - c) for c in self.centroids]
                cluster_idx = np.argmin(distances)
                self.perfclusters[cluster_idx].append(value)
                self.factors_clusters[cluster_idx].append(factor_array[i])
                
            # Calculate new centroids
            new_centroids = [
                np.mean(cluster) if cluster else self.centroids[i]
                for i, cluster in enumerate(self.perfclusters)
            ]
            
            # Check for convergence
            if all(np.isclose(new_centroids[i], self.centroids[i]) for i in range(3)):
                break
                
            self.centroids = new_centroids
            
    def update_signals(self, atr):
        if not self.perfclusters or not any(self.perfclusters):
            return
            
        # Determine which cluster to use
        cluster_map = {'Best': 2, 'Average': 1, 'Worst': 0}
        from_cluster = cluster_map.get(self.fromCluster, 2)
        
        # Get target factor from selected cluster
        if self.factors_clusters[from_cluster]:
            self.target_factor = np.mean(self.factors_clusters[from_cluster])
        
        # Calculate performance index
        den = self.ema(np.abs(self.close_history[-1] - self.close_history[-2]), self.perfAlpha) if len(self.close_history) > 1 else 1
        if self.perfclusters[from_cluster]:
            cluster_avg = np.mean(self.perfclusters[from_cluster])
            self.perf_idx = max(cluster_avg, 0) / den if den != 0 else 0
        
        # Calculate supertrend
        hl2 = (self.high_history[-1] + self.low_history[-1]) / 2
        close = self.close_history[-1]
        prev_close = self.close_history[-2] if len(self.close_history) > 1 else close
        
        if self.target_factor is not None:
            up = hl2 + atr * self.target_factor
            dn = hl2 - atr * self.target_factor
            
            # Update upper/lower
            if prev_close < self.upper:
                self.upper = min(up, self.upper)
            else:
                self.upper = up
                
            if prev_close > self.lower:
                self.lower = max(dn, self.lower)
            else:
                self.lower = dn
                
            # Update trend direction
            if close > self.upper:
                self.os = 1
            elif close < self.lower:
                self.os = 0
                
            # Current trailing stop
            self.ts = self.lower if self.os == 1 else self.upper
            
            # Update adaptive MA
            if self.perf_ama is None:
                self.perf_ama = self.ts
            else:
                self.perf_ama += self.perf_idx * (self.ts - self.perf_ama)
                
    def ema(self, value, length):
        """Simple EMA calculation for performance denominator"""
        alpha = 2 / (length + 1)
        if not hasattr(self, '_ema_buffer'):
            self._ema_buffer = value
        else:
            self._ema_buffer += alpha * (value - self._ema_buffer)
        return self._ema_buffer
        
    def update(self, new_bar):
        """
        Update the indicator with a new bar of data.
        new_bar should be a dictionary with 'open', 'high', 'low', 'close' values.
        """
        # Add new data to history
        self.close_history.append(new_bar['close'])
        self.high_history.append(new_bar['high'])
        self.low_history.append(new_bar['low'])
        
        # Wait until we have enough data
        if len(self.close_history) < self.length + 1:
            return None
            
        # Calculate ATR
        atr = self.calculate_atr()
        
        # Update all supertrends
        self.update_supertrends(atr)
        
        # Perform clustering when we have enough data
        if len(self.close_history) >= self.length + 1:
            self.kmeans_clustering()
            
        # Update signals
        self.update_signals(atr)
        
        # Return current signal
        return {
            'ts': self.ts,
            'perf_ama': self.perf_ama,
            'os': self.os,
            'perf_idx': self.perf_idx,
            'target_factor': self.target_factor
        }
