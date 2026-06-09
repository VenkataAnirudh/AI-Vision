import cv2

class Annotator:
    def __init__(self):
        self.colors = {
            'person': (255, 0, 0),    # Blue
            'face': (0, 255, 0),      # Green
            'event': (0, 0, 255),     # Red
            'text': (255, 255, 255)   # White
        }

    def draw_bboxes(self, frame, detections, label_key='identity', color_key='person'):
        """Draws bounding boxes and labels on a frame."""
        annotated = frame.copy()
        color = self.colors.get(color_key, (0, 255, 255))
        
        for det in detections:
            x1, y1, x2, y2 = det.get('bbox', [0, 0, 0, 0])
            label = str(det.get(label_key, 'Unknown'))
            conf = det.get('confidence', 0.0)
            
            text = f"{label} {conf:.2f}" if conf else label
            
            # Draw Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw solid background for text readability
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            
            # Overlay Text
            cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            
        return annotated