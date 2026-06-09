import os
import cv2
import numpy as np
import faiss
import insightface

class FacePipeline:
    def __init__(self, config):
        self.config = config['models']['face']
        self.threshold = self.config['cosine_threshold']
        
        # Initialize InsightFace
        self.app = insightface.app.FaceAnalysis(
            name=self.config['model'],
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=tuple(self.config['det_size']))
        
        self.index = None
        self.identity_map = []
        self._build_library()

    def _build_library(self):
        """Builds FAISS index from the face_library directory."""
        lib_dir = self.config['library_path']
        embeddings = []
        
        if not os.path.exists(lib_dir):
            return

        for person_name in os.listdir(lib_dir):
            person_dir = os.path.join(lib_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                faces = self.app.get(img)
                if faces:
                    # L2 normalize the embedding for Cosine Similarity (Inner Product)
                    emb = faces[0].normed_embedding
                    embeddings.append(emb)
                    self.identity_map.append(person_name)
        
        if embeddings:
            dim = embeddings[0].shape[0]
            self.index = faiss.IndexFlatIP(dim)
            emb_matrix = np.array(embeddings).astype('float32')
            self.index.add(emb_matrix)
            print(f"[FacePipeline] Library built: {len(self.identity_map)} faces indexed.")

    def process(self, frame, person_bboxes):
        """Finds faces within detected person bounding boxes and identifies them."""
        faces_detected = self.app.get(frame)
        results = []
        
        for face in faces_detected:
            identity = "Unknown"
            match_conf = 0.0
            
            if self.index is not None:
                query_emb = np.array([face.normed_embedding]).astype('float32')
                distances, indices = self.index.search(query_emb, k=1)
                
                if distances[0][0] >= self.threshold:
                    identity = self.identity_map[indices[0][0]]
                    match_conf = float(distances[0][0])
            
            results.append({
                'bbox': [int(x) for x in face.bbox],
                'identity': identity,
                'confidence': match_conf,
                'age': int(face.age),
                'gender': 'M' if face.gender == 1 else 'F'
            })
            
        return results