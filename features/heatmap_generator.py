import numpy as np
import cv2


class HeatmapGenerator:
    """Generates a spatial activity heatmap from person detection centroids.

    Accumulates bounding-box centre points across all frames into a 2D
    histogram, applies Gaussian smoothing and the JET colour-map, then
    alpha-blends the result onto a black background.

    CPU-only — uses numpy and cv2 exclusively.
    """

    def __init__(self, config=None):
        """Initialise the heatmap generator.

        Args:
            config: Optional dict.  Recognised keys (all under 'heatmap'):
                - blur_ksize (int): Gaussian kernel size, must be odd.
                  Default 51.
                - alpha (float): Blend factor for the coloured overlay.
                  Default 0.6.
        """
        config = config or {}
        heatmap_cfg = config.get('heatmap', {})
        self.blur_ksize = int(heatmap_cfg.get('blur_ksize', 51))
        # Ensure the kernel size is odd
        if self.blur_ksize % 2 == 0:
            self.blur_ksize += 1
        self.alpha = float(heatmap_cfg.get('alpha', 0.6))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, person_results_per_frame, frame_shape, output_path):
        """Create, save and return a spatial-activity heatmap image.

        Args:
            person_results_per_frame: List of dicts, each containing a
                ``'persons'`` key whose value is a list of person dicts
                with ``'bbox'`` ``[x1, y1, x2, y2]``.
            frame_shape: Tuple ``(H, W, 3)`` giving the video resolution.
            output_path: Filesystem path where the heatmap PNG is saved.

        Returns:
            numpy.ndarray: BGR heatmap image (same size as *frame_shape*).
        """
        height, width = frame_shape[0], frame_shape[1]
        accumulator = np.zeros((height, width), dtype=np.float64)

        # -- Accumulate centroids -----------------------------------------
        for frame_entry in (person_results_per_frame or []):
            persons = frame_entry.get('persons', [])
            for person in persons:
                bbox = person.get('bbox')
                if bbox is None or len(bbox) < 4:
                    continue
                cx = int((bbox[0] + bbox[2]) / 2.0)
                cy = int((bbox[1] + bbox[3]) / 2.0)
                # Clamp to image bounds
                cx = max(0, min(cx, width - 1))
                cy = max(0, min(cy, height - 1))
                accumulator[cy, cx] += 1.0

        # -- Early exit for empty data ------------------------------------
        if accumulator.max() == 0:
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.imwrite(output_path, blank)
            return blank

        # -- Gaussian smoothing -------------------------------------------
        accumulator = cv2.GaussianBlur(
            accumulator,
            (self.blur_ksize, self.blur_ksize),
            0,
        )

        # -- Normalise to 0-255 ------------------------------------------
        norm = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)

        # -- Apply JET colour-map ----------------------------------------
        heatmap_colour = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        # -- Alpha-blend onto black background ----------------------------
        background = np.zeros((height, width, 3), dtype=np.uint8)
        blended = cv2.addWeighted(heatmap_colour, self.alpha, background,
                                  1.0 - self.alpha, 0)

        # -- Save and return ----------------------------------------------
        cv2.imwrite(output_path, blended)
        return blended
