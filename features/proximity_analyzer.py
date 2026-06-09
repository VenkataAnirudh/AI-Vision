import numpy as np

class ProximityAnalyzer:
    def __init__(self, config=None):
        self.threshold_px = 120.0  # Pixel distance threshold

    def analyze(self, persons):
        """
        Analyzes the list of persons detected in a frame.
        persons: List[Dict] with keys 'track_id' and 'bbox' (x1, y1, x2, y2).
        Returns a list of proximity events.
        """
        alerts = []
        if len(persons) < 2:
            return alerts

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1 = persons[i]
                p2 = persons[j]

                # Bounding box centers
                c1_x = (p1['bbox'][0] + p1['bbox'][2]) / 2.0
                c1_y = (p1['bbox'][1] + p1['bbox'][3]) / 2.0
                c2_x = (p2['bbox'][0] + p2['bbox'][2]) / 2.0
                c2_y = (p2['bbox'][1] + p2['bbox'][3]) / 2.0

                dist = np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)

                if dist < self.threshold_px:
                    severity = "HIGH" if dist < (self.threshold_px / 2.0) else "MEDIUM"
                    alerts.append({
                        'track_ids': [p1['track_id'], p2['track_id']],
                        'distance_px': float(dist),
                        'severity': severity
                    })

        return alerts
