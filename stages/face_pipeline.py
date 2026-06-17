import os
import cv2
import numpy as np
import faiss
import insightface

class FacePipeline:
    def __init__(self, config):
        self.config = config['models']['face']
        self.threshold = self.config['cosine_threshold']
        
        
        self.app = insightface.app.FaceAnalysis(
            name=self.config['model'],
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        # Lower det_thresh (default 0.5) to recover small/low-confidence CCTV faces.
        self.app.prepare(ctx_id=0, det_size=tuple(self.config['det_size']),
                         det_thresh=float(self.config.get('det_thresh', 0.5)))
        
        self.index = None
        self.identity_map = []
        self.dynamic_person_count = 0  
        self.pending_buffer = {}  
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
                    
                    emb = faces[0].normed_embedding
                    embeddings.append(emb)
                    self.identity_map.append(person_name)
        
        if embeddings:
            dim = embeddings[0].shape[0]
            self.index = faiss.IndexFlatIP(dim)
            emb_matrix = np.array(embeddings).astype('float32')
            self.index.add(emb_matrix)
            print(f"[FacePipeline] Library built: {len(self.identity_map)} faces indexed.")

    def process(self, frame, person_bboxes, timestamp=0.0):
        """Finds faces within detected person bounding boxes and identifies them."""
        def compute_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0

        raw_faces = self.app.get(frame)
        faces_detected = [f for f in raw_faces if any(compute_iou(f.bbox, pbbox) > 0.15 for pbbox in person_bboxes)]
        results = []
        
        
        if self.index is None:
            self.index = faiss.IndexFlatIP(512) 
            
        for face in faces_detected:
            identity = "Unknown"
            match_conf = 0.0
            
            query_emb = np.array([face.normed_embedding]).astype('float32')
            
            
            if self.index.ntotal > 0:
                k_search = min(3, self.index.ntotal)
                distances, indices = self.index.search(query_emb, k=k_search)
                
                votes = [self.identity_map[i] for d, i in zip(distances[0], indices[0]) if d >= self.threshold]
                if len(votes) > 0:
                    from collections import Counter
                    identity = Counter(votes).most_common(1)[0][0]
                    
                    match_conf = max([float(d) for d, i in zip(distances[0], indices[0]) if self.identity_map[i] == identity and d >= self.threshold])
            
            
            if identity == "Unknown":
                def _nearest_cluster(emb, buffer):
                    for k, cluster in buffer.items():
                        if np.dot(cluster[0][0], emb.T)[0][0] > 0.85: 
                            return k
                    return len(buffer)

                pending_key = _nearest_cluster(query_emb, self.pending_buffer)
                if pending_key not in self.pending_buffer:
                    self.pending_buffer[pending_key] = []
                self.pending_buffer[pending_key].append((query_emb, timestamp))
                
                
                cluster = self.pending_buffer[pending_key]
                if len(cluster) >= 3 and (cluster[-1][1] - cluster[0][1]) <= 5.0:
                    self.dynamic_person_count += 1
                    identity = f"Person {self.dynamic_person_count}"
                    match_conf = 1.0 
                    
                    
                    self.index.add(query_emb)
                    self.identity_map.append(identity)
                    print(f"[FacePipeline] Dynamically registered new face: {identity}")
                    
                    del self.pending_buffer[pending_key]
            
            results.append({
                'bbox': [int(x) for x in face.bbox],
                'identity': identity,
                'confidence': match_conf,
                'age': int(face.age),
                'gender': 'M' if face.gender == 1 else 'F',
                'face_obj': face
            })
            
        return results