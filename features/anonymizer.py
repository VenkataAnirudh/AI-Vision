import cv2

class FaceAnonymizer:
    def __init__(self, config=None):
        self.blur_kernel = (51, 51)  # High values ensure full anonymization

    def anonymize(self, frame, faces):
        """
        Applies Gaussian blur to all face bounding boxes in the frame.
        faces: List[Dict] with key 'bbox' (x1, y1, x2, y2).
        """
        anonymized = frame.copy()
        h, w, _ = frame.shape

        for face in faces:
            bbox = face.get('bbox')
            if not bbox:
                continue

            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Clip boundary coordinates to image dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                roi = anonymized[y1:y2, x1:x2]
                # Apply strong Gaussian blur
                blurred_roi = cv2.GaussianBlur(roi, self.blur_kernel, 30)
                anonymized[y1:y2, x1:x2] = blurred_roi

        return anonymized
