import numpy as np
from collections import deque

class TemporalSmoother:
    def __init__(self, window_size=5):
        """
        Manages rolling average smoothing for scores linked to tracking IDs.
        """
        self.window_size = window_size
        self.history = {}  

    def smooth(self, track_id, metric_name, value):
        """
        Appends value and returns the smoothed average.
        """
        if track_id not in self.history:
            self.history[track_id] = {}
        
        if metric_name not in self.history[track_id]:
            self.history[track_id][metric_name] = deque(maxlen=self.window_size)

        self.history[track_id][metric_name].append(value)
        return float(np.mean(self.history[track_id][metric_name]))
