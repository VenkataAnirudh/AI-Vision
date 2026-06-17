import numpy as np
import cv2
from collections import defaultdict


class TrajectoryVisualizer:
    """Draws per-person movement trajectories from tracking data.

    Each unique ``track_id`` gets a distinct colour.  Paths are rendered
    as polylines on a dark background with green start-markers, red
    end-markers, and track-ID labels.

    CPU-only — uses numpy and cv2 exclusively.
    """

    def __init__(self, config=None):
        """Initialise the trajectory visualizer.

        Args:
            config: Optional dict.  Recognised keys (under 'trajectories'):
                - line_thickness (int): Polyline thickness. Default 2.
                - marker_radius (int): Start/end circle radius. Default 6.
                - bg_intensity (int): Background grey level 0-255.
                  Default 30.
        """
        config = config or {}
        traj_cfg = config.get('trajectories', {})
        self.line_thickness = int(traj_cfg.get('line_thickness', 2))
        self.marker_radius = int(traj_cfg.get('marker_radius', 6))
        self.bg_intensity = int(traj_cfg.get('bg_intensity', 30))

        # 12 distinct BGR colours for different track IDs
        self.colors = [
            (255, 0, 0),       # blue
            (0, 255, 0),       # green
            (0, 0, 255),       # red
            (255, 255, 0),     # cyan
            (255, 0, 255),     # magenta
            (0, 255, 255),     # yellow
            (128, 0, 255),     # pink-ish
            (255, 128, 0),     # sky blue
            (0, 128, 255),     # orange
            (128, 255, 0),     # spring green
            (255, 255, 128),   # light cyan
            (128, 128, 255),   # light salmon
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, person_results_per_frame, frame_shape, output_path):
        """Create, save and return a trajectory image.

        Args:
            person_results_per_frame: List of per-frame dicts, each with a
                ``'persons'`` list whose entries contain ``'track_id'``
                and ``'bbox'`` ``[x1, y1, x2, y2]``.
            frame_shape: Tuple ``(H, W, 3)`` giving video resolution.
            output_path: Filesystem path where the image is saved.

        Returns:
            numpy.ndarray: BGR trajectory image.
        """
        height, width = frame_shape[0], frame_shape[1]

        # -- Collect centroids per track_id (order-preserved) -------------
        tracks = defaultdict(list)
        for frame_entry in (person_results_per_frame or []):
            persons = frame_entry.get('persons', [])
            for person in persons:
                tid = person.get('track_id')
                bbox = person.get('bbox')
                if tid is None or bbox is None or len(bbox) < 4:
                    continue
                cx = int((bbox[0] + bbox[2]) / 2.0)
                cy = int((bbox[1] + bbox[3]) / 2.0)
                tracks[tid].append((cx, cy))

        # -- Draw on dark background -------------------------------------
        canvas = np.full((height, width, 3), self.bg_intensity,
                         dtype=np.uint8)

        for idx, (tid, points) in enumerate(tracks.items()):
            color = self.colors[idx % len(self.colors)]
            pts = np.array(points, dtype=np.int32)

            # Polyline
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], isClosed=False, color=color,
                              thickness=self.line_thickness,
                              lineType=cv2.LINE_AA)

            # Start marker (green) and end marker (red)
            start_pt = tuple(pts[0])
            end_pt = tuple(pts[-1])
            cv2.circle(canvas, start_pt, self.marker_radius,
                       (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, end_pt, self.marker_radius,
                       (0, 0, 255), -1, cv2.LINE_AA)

            # Track-ID label at start position
            label_pos = (start_pt[0] + self.marker_radius + 2,
                         start_pt[1] - self.marker_radius - 2)
            cv2.putText(canvas, f"ID {tid}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                        cv2.LINE_AA)

        # -- Save and return ----------------------------------------------
        cv2.imwrite(output_path, canvas)
        return canvas
