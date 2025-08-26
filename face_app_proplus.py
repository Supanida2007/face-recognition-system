import sys, os, time, csv, math, json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import cv2 as cv
import numpy as np
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QSlider, QComboBox, QGroupBox, QFormLayout,
    QLineEdit, QDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QKeySequence, QShortcut

import face_recognition

# -------------------------------
# PATHS & CONSTANTS
# -------------------------------
ROOT = Path(__file__).resolve().parent
KNOWN_FACES_DIR = ROOT / "known-faces"
UNKNOWN_DIR = ROOT / "unknown_snapshots"
REPORTS_DIR = ROOT / "reports"
SETTINGS_JSON = ROOT / "settings.json"
LOG_CSV = ROOT / "visitor_log.csv"
for p in [KNOWN_FACES_DIR, UNKNOWN_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DEFAULT_SIZE = (640, 480)
MAX_CAMS = 4  # multi-cam grid up to 4

# Defaults (can be changed in UI)
DOWNSCALE = 0.25         # pre-scale for detection/encoding
DETECT_EVERY_N = 10      # detect/encode every N frames
MATCH_THR = 0.6          # tolerance for compare_faces (lower=stricter)
NAME_SMOOTH_WIN = 7      # majority vote window
MODEL_MODE = "hog"       # "hog" (CPU) or "cnn" (needs dlib CUDA)
DEDUP_SECONDS = 10       # avoid duplicate logs within X seconds
AUTO_ENROLL_FRAMES = 120 # frames of Unknown before prompt (per camera)
BLINK_EAR_THR = 0.21     # eye-aspect-ratio threshold
BLINK_MIN_FRAMES = 2     # consecutive frames below thr counts as a blink
LIVE_TIMEOUT = 8.0       # seconds after last blink to keep "LIVE" state

# -------------------------------
# SETTINGS PERSISTENCE
# -------------------------------

def load_settings():
    if SETTINGS_JSON.exists():
        try:
            return json.loads(SETTINGS_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_settings(d: dict):
    try:
        SETTINGS_JSON.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass

# -------------------------------
# UTILITIES
# -------------------------------

def load_known_faces() -> Tuple[List[np.ndarray], List[str]]:
    encodings, names = [], []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    for f in KNOWN_FACES_DIR.iterdir():
        if f.suffix.lower() not in valid_ext:
            continue
        img = cv.imread(str(f))
        if img is None:
            continue
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if encs:
            encodings.append(encs[0])
            names.append(f.stem)
    return encodings, names


def to_qimage_bgr(frame_bgr: np.ndarray) -> QImage:
    rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)


def iou(a, b):
    # a,b = (x1,y1,x2,y2)
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    denom = areaA + areaB - inter + 1e-6
    return inter/denom


def make_tracker():
    # MOSSE fastest; fallback to KCF
    try:
        return cv.legacy.TrackerMOSSE_create()
    except Exception:
        try:
            return cv.legacy.TrackerKCF_create()
        except Exception:
            return cv.TrackerKCF_create()


def ensure_csv_header(csv_path: Path):
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "name", "confidence", "camera_id", "location", "session"])


def play_beep(freq=1800, dur=120):
    try:
        import winsound
        winsound.Beep(freq, dur)
    except Exception:
        pass


# -------------------------------
# ATTENDANCE MANAGER
# -------------------------------
@dataclass
class PersonState:
    first_seen: float = 0.0
    last_seen: float = 0.0
    total_sec: float = 0.0
    present: bool = False

@dataclass
class AttendanceSession:
    active: bool = False
    location: str = ""
    name: str = ""
    started_at: float = 0.0
    people: Dict[str, PersonState] = field(default_factory=dict)

class AttendanceManager:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.session = AttendanceSession()
        self.absent_timeout = 8.0  # seconds w/o seeing person = consider left

    def start(self, location: str, session_name: str):
        self.session = AttendanceSession(
            active=True,
            location=location.strip(),
            name=session_name.strip() or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            started_at=time.time(),
        )

    def stop(self):
        # finalize durations for present people
        now = time.time()
        for ps in self.session.people.values():
            if ps.present:
                ps.total_sec += now - ps.first_seen
                ps.present = False
        self.session.active = False

    def update_seen(self, person: str, now_ts: float):
        if not self.session.active:
            return
        ps = self.session.people.get(person)
        if ps is None:
            ps = PersonState(first_seen=now_ts, last_seen=now_ts, present=True)
            self.session.people[person] = ps
        else:
            if ps.present:
                ps.last_seen = now_ts
            else:
                # person re-enters
                ps.first_seen = now_ts
                ps.last_seen = now_ts
                ps.present = True

    def tick(self, now_ts: float):
        if not self.session.active:
            return
        for ps in self.session.people.values():
            if ps.present and (now_ts - ps.last_seen) > self.absent_timeout:
                # left area
                ps.total_sec += ps.last_seen - ps.first_seen
                ps.present = False

    def export_report(self, out_path: Optional[Path] = None) -> Path:
        if out_path is None:
            out_path = REPORTS_DIR / f"attendance_{self.session.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["session", "location", "person", "total_minutes"])
            for name, ps in sorted(self.session.people.items()):
                total = ps.total_sec
                if ps.present:
                    total += time.time() - ps.first_seen
                w.writerow([self.session.name, self.session.location, name, f"{total/60.0:.2f}"])
        return out_path


# -------------------------------
# DIALOGS (Stats & Unknown Reviewer)
# -------------------------------
class StatsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📊 Real-time Stats")
        self.resize(420, 360)
        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Name", "Count"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        self.btn_clear = QPushButton("Clear")
        self.btn_close = QPushButton("Close")
        self.btn_clear.clicked.connect(self.clear)
        self.btn_close.clicked.connect(self.close)

        lay = QVBoxLayout(self)
        lay.addWidget(self.table)
        btns = QHBoxLayout()
        btns.addWidget(self.btn_clear); btns.addWidget(self.btn_close)
        lay.addLayout(btns)

    def update_counts(self, counter: dict):
        self.table.setRowCount(0)
        for name, cnt in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(name))
            self.table.setItem(r, 1, QTableWidgetItem(str(cnt)))

    def clear(self):
        self.table.setRowCount(0)


class UnknownReviewer(QDialog):
    def __init__(self, parent=None, unknown_dir: Path = None, known_dir: Path = None):
        super().__init__(parent)
        self.setWindowTitle("❓ Review Unknown Snapshots")
        self.resize(720, 520)
        self.unknown_dir = unknown_dir
        self.known_dir = known_dir
        self.files = sorted([p for p in self.unknown_dir.glob("*.jpg")])
        self.idx = 0

        self.img_lbl = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.name_edit = QLineEdit(); self.name_edit.setPlaceholderText("Enter name…")

        self.btn_prev = QPushButton("⟵ Prev")
        self.btn_next = QPushButton("Next ⟶")
        self.btn_save = QPushButton("Save as Known")
        self.btn_delete = QPushButton("Delete")

        self.btn_prev.clicked.connect(self.prev_img)
        self.btn_next.clicked.connect(self.next_img)
        self.btn_save.clicked.connect(self.save_img)
        self.btn_delete.clicked.connect(self.delete_img)

        lay = QVBoxLayout(self)
        lay.addWidget(self.img_lbl)
        form = QHBoxLayout(); form.addWidget(QLabel("Name:")); form.addWidget(self.name_edit)
        lay.addLayout(form)
        btns = QHBoxLayout();
        for b in [self.btn_prev, self.btn_next, self.btn_save, self.btn_delete]:
            btns.addWidget(b)
        lay.addLayout(btns)

        self.refresh()

    def refresh(self):
        if not self.files:
            self.img_lbl.setText("No unknown snapshots.")
            return
        self.idx = max(0, min(self.idx, len(self.files)-1))
        img = cv.imread(str(self.files[self.idx]))
        if img is None:
            self.img_lbl.setText("Failed to read image.")
            return
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(680, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.img_lbl.setPixmap(pix)

    def prev_img(self):
        if self.files:
            self.idx = (self.idx - 1) % len(self.files)
            self.refresh()

    def next_img(self):
        if self.files:
            self.idx = (self.idx + 1) % len(self.files)
            self.refresh()

    def save_img(self):
        if not self.files:
            return
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.information(self, "Info", "Please enter a name.")
            return
        src = self.files[self.idx]
        dst = self.known_dir / f"{name}.jpg"
        img = cv.imread(str(src))
        cv.imwrite(str(dst), img)
        src.unlink(missing_ok=True)
        QMessageBox.information(self, "Saved", f"Added {name} to known.")
        self.files = sorted([p for p in self.unknown_dir.glob("*.jpg")])
        self.refresh()

    def delete_img(self):
        if not self.files:
            return
        self.files[self.idx].unlink(missing_ok=True)
        self.files = sorted([p for p in self.unknown_dir.glob("*.jpg")])
        self.refresh()


# -------------------------------
# OVERLAY DRAWING (rounded label + shadow)
# -------------------------------

def draw_fancy_box(img: np.ndarray, box: Tuple[int,int,int,int], label: str, live: bool=False, color=(0, 220, 255)):
    (l, t, r, b) = box
    h, w = img.shape[:2]
    l = max(0, l); t = max(0, t); r = min(w-1, r); b = min(h-1, b)

    # Shadow
    overlay = img.copy()
    shadow_offset = 4
    cv.rectangle(overlay, (l+shadow_offset, t+shadow_offset), (r+shadow_offset, b+shadow_offset), (0,0,0), -1)
    img[:] = cv.addWeighted(overlay, 0.25, img, 0.75, 0)

    # Main rounded rectangle (approx via thick rectangle + filled bar)
    cv.rectangle(img, (l, t), (r, b), color, 2)

    # Label bar
    label_h = 26
    overlay2 = img.copy()
    cv.rectangle(overlay2, (l, max(0, t-label_h)), (r, t), color, -1)
    img[:] = cv.addWeighted(overlay2, 0.6, img, 0.4, 0)

    text = ("LIVE · " if live else "") + label
    cv.putText(img, text, (l+8, t-6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, lineType=cv.LINE_AA)
    cv.putText(img, text, (l+8, t-6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, lineType=cv.LINE_AA)


# -------------------------------
# MULTI-CAM HANDLER
# -------------------------------
class CamState:
    def __init__(self, cam_id: int, label: QLabel):
        self.cam_id = cam_id
        self.label = label
        self.cap: Optional[cv.VideoCapture] = None
        self.frame_idx = 0
        self.trackers: List = []
        self.track_names: List[str] = []
        self.last_seen_map: Dict[str, float] = {}
        self.name_history: List[str] = []
        self.unknown_frames = 0
        # anti-spoof
        self.ear_low_consec = 0
        self.last_blink_t = 0.0

    def open(self):
        self.cap = cv.VideoCapture(self.cam_id)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, DEFAULT_SIZE[0])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, DEFAULT_SIZE[1])

    def release(self):       
        if self.cap:
            self.cap.release()
            self.cap = None
        self.trackers.clear(); self.track_names.clear()
        self.name_history.clear(); self.frame_idx = 0
        self.unknown_frames = 0
        self.ear_low_consec = 0


# -------------------------------
# MAIN APP (with Attendance + Auto-Enroll + Anti-Spoof + Multi-Cam)
# -------------------------------
class FaceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("💼 Face Attendance Pro (PyQt6, ultra-smooth, no onnxruntime)")
        self.resize(1300, 900)

        # --- TOP CONTROLS ---
        self.btn_start = QPushButton("▶ Start")
        self.btn_stop  = QPushButton("⏹ Stop")
        self.btn_switch = QPushButton("🔁 Cycle Cam IDs")
        self.btn_add   = QPushButton("➕ Add Known (File)")
        self.btn_cap   = QPushButton("📸 Capture & Register")
        self.btn_reload = QPushButton("🔄 Reload DB")
        self.btn_stats = QPushButton("📊 Stats")
        self.btn_review = QPushButton("🗂 Review Unknowns")
        self.btn_rec = QPushButton("⏺ Start Rec"); self.btn_rec.setCheckable(True)
        self.btn_privacy = QPushButton("🫥 Privacy Blur"); self.btn_privacy.setCheckable(True)

        self.btn_start.clicked.connect(self.start_all)
        self.btn_stop.clicked.connect(self.stop_all)
        self.btn_switch.clicked.connect(self.cycle_cam_ids)
        self.btn_add.clicked.connect(self.add_known_face_from_file)
        self.btn_cap.clicked.connect(self.capture_and_register)
        self.btn_reload.clicked.connect(self.reload_db)
        self.btn_stats.clicked.connect(self.show_stats)
        self.btn_review.clicked.connect(self.review_unknowns)
        self.btn_rec.toggled.connect(self.toggle_record)

        # --- SETTINGS ---
        self.model_combo = QComboBox(); self.model_combo.addItems(["hog", "cnn"]); self.model_combo.setCurrentText(MODEL_MODE)
        self.model_combo.currentTextChanged.connect(self.on_model_change)

        self.slider_detect_n = QSlider(Qt.Orientation.Horizontal); self.slider_detect_n.setRange(1, 30); self.slider_detect_n.setValue(DETECT_EVERY_N)
        self.slider_scale = QSlider(Qt.Orientation.Horizontal); self.slider_scale.setRange(10, 50); self.slider_scale.setValue(int(DOWNSCALE*100))
        self.slider_thr = QSlider(Qt.Orientation.Horizontal); self.slider_thr.setRange(30, 80); self.slider_thr.setValue(int(MATCH_THR*100))
        self.slider_detect_n.valueChanged.connect(self.on_settings_change)
        self.slider_scale.valueChanged.connect(self.on_settings_change)
        self.slider_thr.valueChanged.connect(self.on_settings_change)

        self.watch_input = QLineEdit(); self.watch_input.setPlaceholderText("Comma-separated e.g., Alice,Bob")

        # --- Attendance controls ---
        self.att_loc = QLineEdit(); self.att_loc.setPlaceholderText("Location e.g., Lobby A")
        self.att_name = QLineEdit(); self.att_name.setPlaceholderText("Session name e.g., Morning Shift")
        self.btn_att_start = QPushButton("📋 Start Attendance")
        self.btn_att_stop = QPushButton("🧾 Stop & Export")
        self.btn_att_start.clicked.connect(self.att_start)
        self.btn_att_stop.clicked.connect(self.att_stop_export)

        # shortcuts
        QShortcut(QKeySequence("Space"), self, activated=self.capture_and_register)
        QShortcut(QKeySequence("R"), self, activated=self.reload_db)
        QShortcut(QKeySequence("S"), self, activated=lambda: self.btn_start.click() if not self.timer.isActive() else self.btn_stop.click())

        # --- CAM GRID (1..4) ---
        self.cams_combo = QComboBox(); self.cams_combo.addItems(["1","2","3","4"])
        self.cams_combo.setCurrentText("1")

        self.video_labels: List[QLabel] = []
        grid = QHBoxLayout()
        self.grid_container = QHBoxLayout()

        self.status_lbl = QLabel("Idle")
        self.fps_lbl = QLabel("FPS: -")

        # arrange controls
        top_btns = QHBoxLayout()
        for b in [self.btn_start, self.btn_stop, self.btn_switch, self.btn_add, self.btn_cap, self.btn_reload, self.btn_stats, self.btn_review, self.btn_rec, self.btn_privacy]:
            top_btns.addWidget(b)

        form = QFormLayout()
        form.addRow("Cameras:", self.cams_combo)
        form.addRow("Detector:", self.model_combo)
        form.addRow("Detect every N:", self.slider_detect_n)
        form.addRow("Downscale (%):", self.slider_scale)
        form.addRow("Match thr (%):", self.slider_thr)
        form.addRow("Watchlist:", self.watch_input)

        att_form = QFormLayout()
        att_form.addRow("Location:", self.att_loc)
        att_form.addRow("Session:", self.att_name)
        att_btns = QHBoxLayout(); att_btns.addWidget(self.btn_att_start); att_btns.addWidget(self.btn_att_stop)

        left = QVBoxLayout()
        left.addLayout(top_btns)
        left.addLayout(form)
        left.addLayout(att_form)
        left.addLayout(att_btns)
        left.addWidget(self.status_lbl)
        left.addWidget(self.fps_lbl)

        # create video placeholders
        self.video_panel = QVBoxLayout()
        self._rebuild_video_grid(int(self.cams_combo.currentText()))
        self.cams_combo.currentTextChanged.connect(lambda _: self._rebuild_video_grid(int(self.cams_combo.currentText())))

        root = QHBoxLayout()
        root.addLayout(left, 0)
        root.addLayout(self.video_panel, 1)
        self.setLayout(root)

        # state
        self.timer = QTimer(self); self.timer.timeout.connect(self.on_timer)
        self.cam_states: List[CamState] = []
        self.known_enc, self.known_names = load_known_faces()
        self.counter: Dict[str,int] = {}
        self.stats_dlg = None
        self.rec_writers: Dict[int, cv.VideoWriter] = {}
        self.att = AttendanceManager(LOG_CSV)

        # settings load
        cfg = load_settings()
        try:
            if "detect_n" in cfg: self.slider_detect_n.setValue(int(cfg["detect_n"]))
            if "downscale" in cfg: self.slider_scale.setValue(int(float(cfg["downscale"]) * 100))
            if "match_thr" in cfg: self.slider_thr.setValue(int(float(cfg["match_thr"]) * 100))
            if "detector" in cfg:  self.model_combo.setCurrentText(cfg["detector"]) 
            if "watchlist" in cfg: self.watch_input.setText(cfg["watchlist"]) 
        except Exception:
            pass

    # ------------- UI helpers -------------
    def _rebuild_video_grid(self, n: int):
        # เคลียร์ของเก่า
        while self.video_panel.count():
            item = self.video_panel.takeAt(0)
            w = item.widget()
            if w: w.deleteLater()
        self.video_labels.clear()

        if n <= 2:
            for _ in range(n):
                lbl = QLabel("Camera off"); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setMinimumSize(640, 360)
                self.video_panel.addWidget(lbl)
                self.video_labels.append(lbl)
        else:
            row1 = QHBoxLayout(); row2 = QHBoxLayout()
            for i in range(2):
                lbl = QLabel("Camera off"); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setMinimumSize(480, 270)
                row1.addWidget(lbl); self.video_labels.append(lbl)
            for i in range(n-2):
                lbl = QLabel("Camera off"); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setMinimumSize(480, 270)
                row2.addWidget(lbl); self.video_labels.append(lbl)
            self.video_panel.addLayout(row1)
            self.video_panel.addLayout(row2)

        # >>> เพิ่มเงื่อนไขกันยังไม่มี self.timer
        if hasattr(self, "timer") and isinstance(self.timer, QTimer):
            if self.timer.isActive():
                self.stop_all()
                self.start_all()


    # ------------- Settings -------------
    def on_model_change(self, text):
        global MODEL_MODE
        MODEL_MODE = text

    def on_settings_change(self):
        global DETECT_EVERY_N, DOWNSCALE, MATCH_THR
        DETECT_EVERY_N = int(self.slider_detect_n.value())
        DOWNSCALE = max(0.1, self.slider_scale.value()/100.0)
        MATCH_THR = self.slider_thr.value()/100.0
        save_settings({
            "detect_n": DETECT_EVERY_N,
            "downscale": DOWNSCALE,
            "match_thr": MATCH_THR,
            "detector": MODEL_MODE,
            "watchlist": self.watch_input.text(),
        })

    # ------------- Attendance -------------
    def att_start(self):
        loc = self.att_loc.text().strip() or "Unknown location"
        sess = self.att_name.text().strip() or f"session_{datetime.now().strftime('%H%M%S')}"
        self.att.start(loc, sess)
        QMessageBox.information(self, "Attendance", f"Started session '{sess}' @ {loc}")

    def att_stop_export(self):
        self.att.stop()
        out = self.att.export_report()
        QMessageBox.information(self, "Attendance", f"Exported report:\n{out}")

    # ------------- Camera Flow -------------
    def start_all(self):
        if self.timer.isActive():
            return
        self.cam_states = []
        for i, lbl in enumerate(self.video_labels):
            cs = CamState(cam_id=i, label=lbl)
            cs.open()
            self.cam_states.append(cs)
        self.timer.start(1)
        self.status_lbl.setText("Running…")

    def stop_all(self):
        self.timer.stop()
        for cs in self.cam_states:
            if cs.cap:
                cs.release()
            cs.label.clear(); cs.label.setText("Camera off")
        for w in self.rec_writers.values():
            w.release()
        self.rec_writers.clear()
        self.status_lbl.setText("Stopped.")
        self.fps_lbl.setText("FPS: -")

    def cycle_cam_ids(self):
        # try next IDs (very simple: shift base index)
        if self.timer.isActive():
            self.stop_all()
        # shift starting id by 1 (user can click multiple times)
        first = getattr(self, "_first_cam_offset", 0)
        self._first_cam_offset = (first + 1) % 5
        # rebind labels to new ids
        self.cam_states = []
        for idx, lbl in enumerate(self.video_labels):
            cs = CamState(cam_id=self._first_cam_offset + idx, label=lbl)
            cs.open(); self.cam_states.append(cs)
        self.timer.start(1)

    # ------------- DB / Add Known -------------
    def reload_db(self):
        self.known_enc, self.known_names = load_known_faces()
        QMessageBox.information(self, "Database", f"Loaded {len(self.known_names)} identities.")

    def add_known_face_from_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select face image", "", "Images (*.jpg *.jpeg *.png *.bmp *.tiff)")
        if not path:
            return
        img = cv.imread(path)
        if img is None:
            QMessageBox.warning(self, "Error", "Cannot read image.")
            return
        name = Path(path).stem
        cv.imwrite(str(KNOWN_FACES_DIR / f"{name}.jpg"), img)
        self.reload_db()

    def capture_and_register(self):
        # use first camera
        if not self.cam_states:
            QMessageBox.information(self, "Info", "No camera running.")
            return
        cs = self.cam_states[0]
        if not cs.cap:
            return
        ok, frame = cs.cap.read()
        if not ok:
            return
        small = cv.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb_small = cv.cvtColor(small, cv.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb_small, model="hog")
        if not locs:
            QMessageBox.information(self, "Info", "Face not found.")
            return
        t, r, b, l = locs[0]
        box = (int(l/0.5), int(t/0.5), int(r/0.5), int(b/0.5))
        crop = self.auto_crop_face(frame, box, margin=0.25)
        name = f"person_{int(time.time())}"
        cv.imwrite(str(KNOWN_FACES_DIR / f"{name}.jpg"), crop)
        self.reload_db()
        QMessageBox.information(self, "Saved", f"Registered {name} from camera.")

    # ------------- Recording -------------
    def toggle_record(self, checked):
        if checked:
            for cs in self.cam_states:
                fourcc = cv.VideoWriter_fourcc(*"mp4v")
                out_path = ROOT / f"rec_cam{cs.cam_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                self.rec_writers[cs.cam_id] = cv.VideoWriter(str(out_path), fourcc, 25.0, DEFAULT_SIZE)
            self.btn_rec.setText("⏹ Stop Rec")
            self.status_lbl.setText("Recording…")
        else:
            for w in self.rec_writers.values():
                w.release()
            self.rec_writers.clear()
            self.btn_rec.setText("⏺ Start Rec")
            self.status_lbl.setText("Recording stopped.")

    # ------------- Stats / Unknown Review -------------
    def show_stats(self):
        if self.stats_dlg is None:
            self.stats_dlg = StatsDialog(self)
        self.stats_dlg.show()

    def review_unknowns(self):
        dlg = UnknownReviewer(self, unknown_dir=UNKNOWN_DIR, known_dir=KNOWN_FACES_DIR)
        dlg.exec()
        self.reload_db()

    # ------------- Helpers -------------
    def auto_crop_face(self, frame, box, margin=0.2):
        l, t, r, b = box
        h, w = frame.shape[:2]
        bw, bh = r-l, b-t
        dl = int(bw * margin); dt = int(bh * margin)
        l2 = max(0, l - dl); r2 = min(w, r + dl)
        t2 = max(0, t - dt); b2 = min(h, b + dt)
        return frame[t2:b2, l2:r2]

    def log_visit(self, name, confidence, cam_id):
        ensure_csv_header(LOG_CSV)
        with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            sess = self.att.session.name if self.att.session.active else "-"
            loc = self.att.session.location if self.att.session.active else "-"
            w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, f"{confidence:.3f}", cam_id, loc, sess])

    # ------------- Anti-spoof: EAR / blink -------------
    def compute_ear(self, eye_pts: List[Tuple[int,int]]):
        # eye points: 6 points around the eye
        def dist(a,b):
            return math.dist(a,b)
        # p1..p6 = around the eye; here we map from face_recognition order
        p2p6 = dist(eye_pts[1], eye_pts[5])
        p3p5 = dist(eye_pts[2], eye_pts[4])
        p1p4 = dist(eye_pts[0], eye_pts[3])
        if p1p4 == 0: return 1.0
        return (p2p6 + p3p5) / (2.0 * p1p4)

    # ------------- Main Timer -------------
    def on_timer(self):
        if not self.cam_states:
            return
        t0 = time.time()
        watch = [x.strip() for x in self.watch_input.text().split(",") if x.strip()]

        for cs in self.cam_states:
            if not cs.cap:
                continue
            ok, frame = cs.cap.read()
            if not ok:
                continue
            cs.frame_idx += 1
            updated_boxes: List[Tuple[int,int,int,int]] = []
            names_to_draw: List[str] = []
            lives_to_draw: List[bool] = []

            # 1) Update trackers (fast)
            for trk in list(cs.trackers):
                ok2, box = trk.update(frame)
                if ok2:
                    x, y, w, h = box
                    updated_boxes.append((int(x), int(y), int(x+w), int(y+h)))
                else:
                    idx = cs.trackers.index(trk)
                    cs.trackers.pop(idx)
                    cs.track_names.pop(idx)

            # 2) Detect+Encode every N frames or if none tracking
            if (cs.frame_idx % DETECT_EVERY_N == 0) or (len(cs.trackers) == 0):
                small = cv.resize(frame, (0,0), fx=DOWNSCALE, fy=DOWNSCALE)
                rgb_small = cv.cvtColor(small, cv.COLOR_BGR2RGB)
                locs_small = face_recognition.face_locations(rgb_small, model=MODEL_MODE)
                locs = [(int(t/DOWNSCALE), int(r/DOWNSCALE), int(b/DOWNSCALE), int(l/DOWNSCALE)) for (t,r,b,l) in locs_small]

                # encodings
                encs = []
                for (t, r, b, l) in locs:
                    face_img = frame[t:b, l:r]
                    if face_img.size == 0:
                        encs.append(None); continue
                    rgb_face = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
                    e = face_recognition.face_encodings(rgb_face)
                    encs.append(e[0] if len(e) else None)

                new_trackers, new_names = [], []
                for (t, r, b, l), enc in zip(locs, encs):
                    name, conf = "Unknown", 0.0
                    if enc is not None and len(self.known_enc) > 0:
                        matches = face_recognition.compare_faces(self.known_enc, enc, tolerance=MATCH_THR)
                        if any(matches):
                            dists = face_recognition.face_distance(self.known_enc, enc)
                            idx = int(np.argmin(dists)); name = self.known_names[idx]
                            conf = 1.0 - float(dists[idx])

                    # associate with previous box
                    best_iou, best_idx = 0.0, -1
                    cur_box = (l, t, r, b)
                    for i, ub in enumerate(updated_boxes):
                        val = iou(cur_box, ub)
                        if val > best_iou:
                            best_iou, best_idx = val, i
                    if best_iou > 0.3 and 0 <= best_idx < len(cs.track_names):
                        if cs.track_names[best_idx] != "Unknown" and name == "Unknown":
                            name = cs.track_names[best_idx]

                    # create tracker
                    trk = make_tracker(); trk.init(frame, (l, t, r-l, b-t))
                    new_trackers.append(trk); new_names.append(name)

                    # counters, logging, attendance, watchlist
                    self.counter[name] = self.counter.get(name, 0) + 1
                    now_ts = time.time()
                    if name != "Unknown":
                        # event log (dedup handled externally via last_seen_map if needed)
                        if (name not in cs.last_seen_map) or (now_ts - cs.last_seen_map[name] > DEDUP_SECONDS):
                            self.log_visit(name, conf, cs.cam_id)
                            cs.last_seen_map[name] = now_ts
                        # attendance update
                        self.att.update_seen(name, now_ts)
                        if name in watch:
                            play_beep();
                    else:
                        cs.unknown_frames += 1
                        # auto-enroll proposal
                        if cs.unknown_frames >= AUTO_ENROLL_FRAMES:
                            crop = self.auto_crop_face(frame, cur_box, margin=0.25)
                            tmp_path = UNKNOWN_DIR / f"auto_unknown_{int(time.time())}.jpg"
                            cv.imwrite(str(tmp_path), crop)
                            cs.unknown_frames = 0
                            QMessageBox.information(self, "Auto-Enrollment", f"Saved snapshot for enrollment:\n{tmp_path}")

                cs.trackers, cs.track_names = new_trackers, new_names
                updated_boxes = [(l, t, r, b) for (t, r, b, l) in locs]

                # --- Anti-spoof (blink) on small frame using landmarks ---
                try:
                    lms = face_recognition.face_landmarks(rgb_small, face_locations=locs_small)
                except Exception:
                    lms = []
                live_flags = []
                for lm in lms:
                    # use one eye (right_eye/left_eye) if exists
                    live = False
                    for eye_key in ["left_eye", "right_eye"]:
                        if eye_key in lm and len(lm[eye_key]) >= 6:
                            # scale back to full-res points
                            eye = [(int(x/DOWNSCALE), int(y/DOWNSCALE)) for (x,y) in lm[eye_key]]
                            ear = self.compute_ear(eye[:6])
                            if ear < BLINK_EAR_THR:
                                cs.ear_low_consec += 1
                            else:
                                if cs.ear_low_consec >= BLINK_MIN_FRAMES:
                                    cs.last_blink_t = time.time()
                                cs.ear_low_consec = 0
                            if time.time() - cs.last_blink_t < LIVE_TIMEOUT:
                                live = True
                    live_flags.append(live)
            else:
                live_flags = [False]*len(updated_boxes)

            # 3) Draw overlays
            # (sync live flags length)
            if len(live_flags) < len(updated_boxes):
                live_flags += [False]*(len(updated_boxes)-len(live_flags))

            for (box, name, live) in zip(updated_boxes, cs.track_names, live_flags):
                draw_fancy_box(frame, box, name, live=live)

            # 4) Attendance tick (finalize people who left)
            self.att.tick(time.time())

            # 5) Privacy blur
            if self.btn_privacy.isChecked():
                blur = cv.GaussianBlur(frame, (35,35), 0)
                for box in updated_boxes:
                    l,t,r,b = box
                    l=max(0,l); t=max(0,t); r=min(frame.shape[1],r); b=min(frame.shape[0],b)
                    blur[t:b, l:r] = frame[t:b, l:r]
                frame = blur

            # 6) Write recording per camera
            if self.rec_writers.get(cs.cam_id):
                self.rec_writers[cs.cam_id].write(frame)

            cs.label.setPixmap(QPixmap.fromImage(to_qimage_bgr(frame)))

        # FPS overall (rough)
        dt = time.time() - t0
        if dt > 0:
            fps = len(self.cam_states) / dt
            self.fps_lbl.setText(f"FPS ~ {fps:.1f} (cams={len(self.cam_states)})")

        # update stats dialog if visible
        if self.stats_dlg and self.stats_dlg.isVisible():
            self.stats_dlg.update_counts(self.counter)


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    ensure_csv_header(LOG_CSV)
    app = QApplication(sys.argv)
    w = FaceApp()
    w.show()
    sys.exit(app.exec())
