import cv2
import numpy as np


# COCO-17 skeleton edges (same graph as the ST-GCN pose pipeline). Defined locally
# so this drawing utility stays free of the heavy torch import in stages/stgcn_model.py.
COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9),
              (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def ui_scale(h):
    """Resolution-adaptive overlay metrics derived from frame height.

    Returns ``(font_scale, thickness, pad)`` so HUD/boxes/labels stay legible and
    proportionate from ~360p up to 4K instead of being hardcoded in pixels.
    """
    font = float(np.clip(h / 900.0, 0.45, 1.6))
    thickness = max(1, int(round(h / 540.0)))
    pad = max(6, int(round(h / 90.0)))
    return font, thickness, pad


class Annotator:
    def __init__(self):
        self.colors = {
            'person': (255, 0, 0),
            'face': (0, 255, 0),
            'event': (0, 0, 255),
            'text': (255, 255, 255)
        }

    def draw_bboxes(self, frame, detections, label_key='identity', color_key='person', frame_idx=0):
        """Draws bounding boxes and labels on a frame with Dynamic State Color-Shifting."""
        annotated = frame.copy()
        h = frame.shape[0]
        font_scale, thickness, _ = ui_scale(h)
        box_t = max(1, thickness)
        label_font = max(0.4, font_scale * 0.55)
        label_t = max(1, box_t - 1)

        for det in detections:
            x1, y1, x2, y2 = det.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = str(det.get(label_key, 'Unknown'))
            conf = det.get('confidence', 0.0)

            state = det.get('state', 'normal')

            if state == 'critical':
                color = (0, 0, 255) if (frame_idx % 10) < 5 else (0, 0, 100)
            elif state == 'caution':
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            if 'color' in det:
                color = det['color']

            text = f"{label} {conf:.2f}" if conf else label

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_t)

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, label_font, label_t)
            cv2.rectangle(annotated, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)

            cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        label_font, self.colors['text'], label_t, cv2.LINE_AA)

        return annotated

    def draw_skeletons(self, frame, keypoints_list, conf_thresh=0.35):
        """Draw COCO-17 pose skeletons for every person in the frame.

        ``keypoints_list`` is an ndarray ``(N, 17, 3)`` (or list of ``(17, 3)``) in frame
        pixel coordinates. A limb is drawn only when *both* endpoints exceed ``conf_thresh``,
        which kills the (0,0)-anchored crisscross lines that low-confidence joints produce.
        Drawn in place on the passed frame (caller already holds a copy).
        """
        if keypoints_list is None:
            return frame
        h = frame.shape[0]
        _, thickness, _ = ui_scale(h)
        limb_t = max(1, thickness)
        joint_r = max(2, thickness + 1)
        limb_color = (0, 255, 180)    # teal-green — distinct from boxes (state colors) and events (red)
        joint_color = (0, 220, 255)   # amber

        for person in keypoints_list:
            if person is None or len(person) < 17:
                continue
            for a, b in COCO_EDGES:
                if person[a][2] > conf_thresh and person[b][2] > conf_thresh:
                    pa = (int(person[a][0]), int(person[a][1]))
                    pb = (int(person[b][0]), int(person[b][1]))
                    cv2.line(frame, pa, pb, limb_color, limb_t, cv2.LINE_AA)
            for j in range(17):
                if person[j][2] > conf_thresh:
                    cv2.circle(frame, (int(person[j][0]), int(person[j][1])),
                               joint_r, joint_color, -1, cv2.LINE_AA)
        return frame

    def draw_hud(self, frame, metrics):
        """Builds a resolution-adaptive telemetry panel anchored top-left.

        Panel size, fonts, and line spacing all derive from frame height/width so the HUD
        never shrinks to dots on 4K nor overflows on low-res. ``metrics`` is a dict of
        ``{label: value}`` strings.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        font_scale, thickness, pad = ui_scale(h)
        title_font = font_scale * 0.75
        line_font = font_scale * 0.6
        title_t = max(1, thickness)
        line_t = max(1, thickness - 1)

        title = "LIVE TELEMETRY HUD"
        lines = [f"{k}: {v}" for k, v in metrics.items()]

        (title_w, title_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_font, title_t)
        max_line_w = title_w
        line_h = title_h
        for ln in lines:
            (lw, lh), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, line_font, line_t)
            max_line_w = max(max_line_w, lw)
            line_h = max(line_h, lh)
        line_gap = int(line_h + pad)

        panel_w = min(int(0.40 * w), max_line_w + 2 * pad)
        panel_h = pad + title_h + pad + len(lines) * line_gap + pad
        x0, y0 = pad, pad

        overlay = annotated.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)
        cv2.rectangle(annotated, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 255, 255), line_t)

        y = y0 + pad + title_h
        cv2.putText(annotated, title, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    title_font, (0, 255, 255), title_t, cv2.LINE_AA)
        y += pad
        for ln in lines:
            y += line_gap
            cv2.putText(annotated, ln, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                        line_font, (255, 255, 255), line_t, cv2.LINE_AA)

        return annotated

    def draw_event_ticker(self, frame, messages):
        """Resolution-adaptive rolling event log strip at the bottom of the frame."""
        if not messages:
            return frame

        annotated = frame.copy()
        h, w = frame.shape[:2]
        font_scale, thickness, pad = ui_scale(h)
        line_font = font_scale * 0.6
        line_t = max(1, thickness - 1)

        (_, lh), _ = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, line_font, line_t)
        line_gap = int(lh + pad * 0.6)
        n = max(1, min(3, len(messages)))
        strip_h = pad + n * line_gap + pad // 2
        y0 = h - strip_h

        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

        y = y0 + pad + lh
        for msg in messages[-n:]:
            cv2.putText(annotated, msg, (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                        line_font, (0, 200, 255), line_t, cv2.LINE_AA)
            y += line_gap

        return annotated
