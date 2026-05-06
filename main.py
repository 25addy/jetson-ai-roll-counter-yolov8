import sys
import os
import cv2
import time
import math
import csv
from datetime import datetime
from zoneinfo import ZoneInfo

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QMessageBox, QFrame, QGridLayout,
    QDialog, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView
)

from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_DIR = os.getenv("MODEL_DIR", "models/best.engine")

CAMERA_INDEX = 0
USE_V4L2 = True
USE_MJPG = True

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_FPS = 30

CONFIDENCE = 0.22
IOU = 0.45
IMG_SIZE = 512

# =========================
# 3-ZONE + DIRECTION LOCK
# =========================
TOP_ZONE_Y1_RATIO = 0.08
TOP_ZONE_Y2_RATIO = 0.33

COUNT_ZONE_Y1_RATIO = 0.34
COUNT_ZONE_Y2_RATIO = 0.63

EXIT_ZONE_Y1_RATIO = 0.64
EXIT_ZONE_Y2_RATIO = 0.94

COUNT_LINE_Y_RATIO = 0.49

TRACK_MATCH_DIST = 140
TRACK_TIMEOUT_SEC = 1.0
MIN_MOVE_PX = 2
MIN_TRACK_HITS = 2
COUNT_COOLDOWN_SEC = 0.10

PROCESS_EVERY_N_FRAMES = 1

APP_TITLE = os.getenv("APP_TITLE", "AI Roll Counter")
CSV_PATH = "roller_batches.csv"
CONVEYOR_DIRECTION = "down"


# =========================
# MYSQL - SEARCH PALLET / PRODUCT
# =========================
ENABLE_PALLET_MYSQL = os.getenv("ENABLE_PALLET_MYSQL", "false").lower() == "true"
PALLET_MYSQL_CONFIG = {
    "host": os.getenv("PALLET_DB_HOST", "YOUR_DB_HOST"),
    "user": os.getenv("PALLET_DB_USER", "YOUR_DB_USER"),
    "password": os.getenv("PALLET_DB_PASSWORD", "YOUR_DB_PASSWORD"),
    "database": os.getenv("PALLET_DB_NAME", "YOUR_DATABASE"),
    "ssl_disabled": True
}

PALLET_DB_NAME = os.getenv("PALLET_DB_NAME", "YOUR_DATABASE")
PALLET_TABLE_NAME = "pallet_dt"
PALLET_COL_REFNO = "ref_no"
PALLET_COL_PRODUCT = "product_code"
PALLET_COL_GROSS_WEIGHT = "gross_weight"

PRODUCT_TABLE_NAME = "product"
PRODUCT_COL_CODE = "code"
PRODUCT_COL_ROLL_CTN = "roll_ctn"

# =========================
# MYSQL - SAVE LOG
# =========================
ENABLE_LOG_MYSQL = os.getenv("ENABLE_LOG_MYSQL", "false").lower() == "true"
LOG_MYSQL_CONFIG = {
    "host": os.getenv("LOG_DB_HOST", "YOUR_LOG_DB_HOST"),
    "user": os.getenv("LOG_DB_USER", "YOUR_DB_USER"),
    "password": os.getenv("LOG_DB_PASSWORD", "YOUR_DB_PASSWORD"),
    "database": os.getenv("LOG_DB_NAME", "YOUR_LOG_DATABASE"),
    "ssl_disabled": True
}

LOG_TABLE_NAME = "roll_counter_log"

LOG_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {LOG_TABLE_NAME} (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pallet_no VARCHAR(100) DEFAULT '',
    product_code VARCHAR(100) DEFAULT '',
    target_qty INT NOT NULL DEFAULT 0,
    actual_qty INT NOT NULL DEFAULT 0,
    event_type VARCHAR(20) NOT NULL,
    event_time DATETIME NOT NULL,
    duration_sec DECIMAL(10,2) DEFAULT 0.00,
    remark VARCHAR(255) DEFAULT ''
)
"""

def now_my():
    return datetime.now(ZoneInfo("Asia/Kuala_Lumpur"))

def get_mysql_server_month():
    try:
        import mysql.connector
        conn = mysql.connector.connect(**LOG_MYSQL_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT MONTH(NOW())")
        row = cur.fetchone()
        cur.close()
        conn.close()
        return int(row[0]) if row and row[0] else now_my().month
    except Exception:
        return now_my().month


def get_month_prefix():
    month_letters = {
        1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F",
        7: "G", 8: "H", 9: "I", 10: "J", 11: "K", 12: "L",
    }
    return month_letters.get(get_mysql_server_month(), "")

def normalize_pallet_input(text: str):
    text = text.strip().upper()
    if not text:
        return ""

    prefix = get_month_prefix()

    if text[0].isalpha():
        letter = text[0]
        rest = text[1:].strip()
        return f"{letter} {rest}".strip()

    return f"{prefix} {text}".strip()


def fetch_pallet_data(pallet_no: str):
    pallet_no = pallet_no.strip()
    if not pallet_no:
        return False, "Please enter Pallet No.", None

    if not ENABLE_PALLET_MYSQL:
        return False, "Pallet MySQL is disabled. Set ENABLE_PALLET_MYSQL=true and configure DB environment variables.", None

    try:
        try:
            import mysql.connector
        except ModuleNotFoundError:
            return False, "MySQL module not installed. Run: pip3 install mysql-connector-python", None

        cfg = dict(PALLET_MYSQL_CONFIG)
        cfg["database"] = PALLET_DB_NAME

        conn = mysql.connector.connect(**cfg)
        cur = conn.cursor(dictionary=True)

        sql_pallet = f"""
            SELECT {PALLET_COL_REFNO}, {PALLET_COL_PRODUCT}, {PALLET_COL_GROSS_WEIGHT}
            FROM {PALLET_TABLE_NAME}
            WHERE {PALLET_COL_REFNO} = %s
              AND COALESCE({PALLET_COL_GROSS_WEIGHT}, 0) = 0
        """
        cur.execute(sql_pallet, (pallet_no,))
        rows = cur.fetchall()

        if not rows:
            cur.close()
            conn.close()
            return False, f"No rows found for Pallet No {pallet_no} with gross_weight = 0.", None

        product_code = str(rows[0].get(PALLET_COL_PRODUCT, "") or "").strip()
        pallet_row_count = len(rows)

        sql_product = f"""
            SELECT {PRODUCT_COL_ROLL_CTN}
            FROM {PRODUCT_TABLE_NAME}
            WHERE {PRODUCT_COL_CODE} = %s
            LIMIT 1
        """
        cur.execute(sql_product, (product_code,))
        product_row = cur.fetchone()

        cur.close()
        conn.close()

        if not product_row:
            return False, f"Product code {product_code} not found in table {PRODUCT_TABLE_NAME}.", None

        roll_ctn = int(product_row.get(PRODUCT_COL_ROLL_CTN) or 0)
        target_qty = pallet_row_count * roll_ctn

        data = {
            "pallet_no": pallet_no,
            "product_code": product_code,
            "pallet_row_count": pallet_row_count,
            "roll_ctn": roll_ctn,
            "target_qty": target_qty,
            "rows": rows,
        }

        return True, (
            f"Loaded Pallet No {pallet_no}: {pallet_row_count} row(s), "
            f"roll_ctn={roll_ctn}, target_qty={target_qty}"
        ), data

    except Exception as e:
        return False, f"MySQL search error: {e}", None


class PalletSearchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_data = None
        self.setWindowTitle("Search Pallet No")
        self.setModal(True)
        self.resize(780, 460)
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet("""
            QDialog {
                background: #0f172a;
                color: white;
                font-family: Arial;
            }
            QLabel {
                color: white;
            }
            QLineEdit {
                background: white;
                color: black;
                border-radius: 10px;
                padding: 10px;
                font-size: 18px;
            }
            QPushButton {
                border: none;
                border-radius: 10px;
                padding: 10px;
                font-weight: bold;
                color: white;
                min-height: 44px;
            }
            QTableWidget {
                background: white;
                color: black;
                border-radius: 10px;
                gridline-color: #cbd5e1;
            }
            QHeaderView::section {
                background: #1e293b;
                color: white;
                padding: 6px;
                border: none;
            }
        """)

        title = QLabel("Search Pallet No")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")

        self.pallet_edit = QLineEdit()
        self.pallet_edit.setPlaceholderText("Enter Pallet No")
        self.pallet_edit.setText(f"{get_month_prefix()} ")
        self.pallet_edit.setCursorPosition(len(self.pallet_edit.text()))
        self.pallet_edit.returnPressed.connect(self.search_pallet)

        self.search_btn = QPushButton("SEARCH")
        self.search_btn.setStyleSheet("background: #0ea5e9; font-size: 18px;")
        self.search_btn.clicked.connect(self.search_pallet)

        self.use_btn = QPushButton("USE THIS PALLET")
        self.use_btn.setStyleSheet("background: #16a34a; font-size: 18px;")
        self.use_btn.clicked.connect(self.use_selected)
        self.use_btn.setEnabled(False)

        self.close_btn = QPushButton("CLOSE")
        self.close_btn.setStyleSheet("background: #475569; font-size: 18px;")
        self.close_btn.clicked.connect(self.reject)

        self.result_info = QLabel("Enter Pallet No and click SEARCH")
        self.result_info.setWordWrap(True)
        self.result_info.setStyleSheet("font-size: 16px; color: #cbd5e1;")

        self.summary_label = QLabel("Pallet No: -\nProduct: -\nPallet Rows: -\nRoll/Ctn: -\nTarget Qty: -")
        self.summary_label.setStyleSheet(
            "background: #111827; border-radius: 12px; padding: 12px; font-size: 16px;"
        )

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Pallet No", "Product", "Gross Weight"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)

        top_row = QHBoxLayout()
        top_row.addWidget(self.pallet_edit, 1)
        top_row.addWidget(self.search_btn)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.use_btn)
        bottom_row.addWidget(self.close_btn)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(top_row)
        layout.addWidget(self.result_info)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.table, 1)
        layout.addLayout(bottom_row)
        self.setLayout(layout)

    def search_pallet(self):
        pallet_no = normalize_pallet_input(self.pallet_edit.text())
        self.pallet_edit.setText(pallet_no)
        self.pallet_edit.setCursorPosition(len(self.pallet_edit.text()))

        ok, msg, data = fetch_pallet_data(pallet_no)
        self.result_info.setText(msg)

        if not ok:
            self.selected_data = None
            self.use_btn.setEnabled(False)
            self.summary_label.setText("Pallet No: -\nProduct: -\nPallet Rows: -\nRoll/Ctn: -\nTarget Qty: -")
            self.table.setRowCount(0)
            return

        self.selected_data = data
        self.use_btn.setEnabled(True)
        self.summary_label.setText(
            f"Pallet No: {data['pallet_no']}\n"
            f"Product: {data['product_code'] or '-'}\n"
            f"Pallet Rows: {data['pallet_row_count']}\n"
            f"Roll/Ctn: {data['roll_ctn']}\n"
            f"Target Qty: {data['target_qty']}"
        )

        rows = data["rows"]
        self.table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(row.get(PALLET_COL_REFNO, ""))))
            self.table.setItem(row_idx, 1, QTableWidgetItem(str(row.get(PALLET_COL_PRODUCT, ""))))
            self.table.setItem(row_idx, 2, QTableWidgetItem(str(row.get(PALLET_COL_GROSS_WEIGHT, ""))))

    def use_selected(self):
        if self.selected_data is None:
            QMessageBox.warning(self, "No Data", "Please search Pallet No first.")
            return
        self.accept()

class AddRollerDialog(QDialog):
    def __init__(self, current_count=0, current_target=0, parent=None):
        super().__init__(parent)
        self.add_qty = 0
        self.operator_name = ""
        self.current_count = int(current_count)
        self.current_target = int(current_target)
        self.setWindowTitle("Add Roll")
        self.setModal(True)
        self.resize(430, 260)
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet("""
            QDialog {
                background: #0f172a;
                color: white;
                font-family: Arial;
            }
            QLabel {
                color: white;
                font-size: 16px;
            }
            QLineEdit {
                background: white;
                color: black;
                border-radius: 10px;
                padding: 10px;
                font-size: 18px;
            }
            QPushButton {
                border: none;
                border-radius: 10px;
                padding: 10px;
                font-weight: bold;
                color: white;
                min-height: 44px;
                font-size: 18px;
            }
        """)

        title = QLabel("ADD ROLL")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")

        self.info_label = QLabel(
            f"Current Count: {self.current_count}\n"
            f"Current Target: {self.current_target}"
        )
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #cbd5e1;")

        qty_label = QLabel("Add Roll Qty")
        self.qty_edit = QLineEdit()
        self.qty_edit.setPlaceholderText("Example: 50")

        user_label = QLabel("User")
        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("Operator name")

        self.ok_btn = QPushButton("CONFIRM")
        self.ok_btn.setStyleSheet("background: #16a34a;")
        self.ok_btn.clicked.connect(self.submit)

        self.cancel_btn = QPushButton("CANCEL")
        self.cancel_btn.setStyleSheet("background: #475569;")
        self.cancel_btn.clicked.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.info_label)
        layout.addWidget(qty_label)
        layout.addWidget(self.qty_edit)
        layout.addWidget(user_label)
        layout.addWidget(self.user_edit)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.ok_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    def submit(self):
        qty_text = self.qty_edit.text().strip()
        operator_name = self.user_edit.text().strip()

        if not qty_text:
            QMessageBox.warning(self, "Missing Qty", "Please enter add roll quantity.")
            return

        try:
            qty = int(qty_text)
        except ValueError:
            QMessageBox.warning(self, "Invalid Qty", "Add roll quantity must be integer.")
            return

        if qty <= 0:
            QMessageBox.warning(self, "Invalid Qty", "Add roll quantity must be more than 0.")
            return

        if not operator_name:
            QMessageBox.warning(self, "Missing User", "Please enter operator name.")
            self.user_edit.setFocus()
            return

        self.add_qty = qty
        self.operator_name = operator_name
        self.accept()


class CounterWorker(QThread):
    frame_ready = Signal(QImage)
    count_changed = Signal(int)
    status_changed = Signal(str)
    finished_target = Signal()
    batch_saved = Signal(str)
    info_changed = Signal(str)

    def __init__(self, target_count: int, pallet_no: str = "", product_code: str = "", parent=None):
        super().__init__(parent)
        self.target_count = target_count
        self.pallet_no = pallet_no
        self.product_code = product_code
        self.running = False
        self.paused = False
        self.total_count = 0

        self.next_track_id = 1
        self.tracks = {}
        self.batch_start_time = None
        self.last_count_time = 0.0
        self.camera_source_text = ""
        self.extra_remark = ""

    def stop(self):
        self.running = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def center_of_box(self, x1, y1, x2, y2):
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def euclidean(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def in_zone(self, cy, y1, y2):
        return y1 <= cy <= y2

    def cleanup_tracks(self, now_ts):
        dead_ids = []
        for tid, tr in self.tracks.items():
            if now_ts - tr["last_seen"] > TRACK_TIMEOUT_SEC:
                dead_ids.append(tid)
        for tid in dead_ids:
            del self.tracks[tid]

    def match_detection_to_track(self, det_center, used_track_ids):
        best_id = None
        best_dist = 1e9
        for tid, tr in self.tracks.items():
            if tid in used_track_ids:
                continue
            d = self.euclidean(det_center, tr["center"])
            if d < TRACK_MATCH_DIST and d < best_dist:
                best_dist = d
                best_id = tid
        return best_id

    def save_to_csv(self, target_qty, actual_qty, status, duration_sec, remark=""):
        file_exists = os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "batch_time", "pallet_no", "product_code", "target_qty", "actual_qty",
                    "status", "duration_sec", "remark"
                ])
            writer.writerow([
                now_my().strftime("%Y-%m-%d %H:%M:%S"),
                self.pallet_no,
                self.product_code,
                target_qty,
                actual_qty,
                status,
                f"{duration_sec:.2f}",
                remark
            ])

    def save_to_mysql(self, target_qty, actual_qty, status, duration_sec, remark=""):
        if not ENABLE_LOG_MYSQL:
            return

        try:
            import mysql.connector
            conn = mysql.connector.connect(**LOG_MYSQL_CONFIG)
            cur = conn.cursor()

            cur.execute(LOG_TABLE_SQL)

            cur.execute(
                f"""
                INSERT INTO {LOG_TABLE_NAME}
                (pallet_no, product_code, target_qty, actual_qty, event_type, event_time, duration_sec, remark)
                VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s)
                """,
                (
                    self.pallet_no,
                    self.product_code,
                    int(target_qty),
                    int(actual_qty),
                    str(status),
                    float(duration_sec),
                    str(remark),
                )
            )

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            self.batch_saved.emit(f"MySQL save error: {e}")

    def save_batch_result(self, status, remark="", save_mysql=True, save_csv=True):
        duration_sec = 0.0 if self.batch_start_time is None else (time.time() - self.batch_start_time)

        if save_csv:
            self.save_to_csv(
                target_qty=self.target_count,
                actual_qty=self.total_count,
                status=status,
                duration_sec=duration_sec,
                remark=remark
            )

        if save_mysql:
            self.save_to_mysql(
                target_qty=self.target_count,
                actual_qty=self.total_count,
                status=status,
                duration_sec=duration_sec,
                remark=remark
            )

        where_saved = []
        if save_csv:
            where_saved.append("CSV")
        if save_mysql:
            where_saved.append("MySQL")

        save_text = " + ".join(where_saved) if where_saved else "nowhere"

        self.batch_saved.emit(
            f"Saved batch ({save_text}): pallet={self.pallet_no}, product={self.product_code}, "
            f"target={self.target_count}, actual={self.total_count}, status={status}"
        )

    def open_camera(self):
        backend_candidates = []
        if USE_V4L2:
            backend_candidates.append(("V4L2", cv2.CAP_V4L2))
        backend_candidates.append(("AUTO", None))

        mjpg_candidates = [True, False] if USE_MJPG else [False]
        index_candidates = [CAMERA_INDEX] + [i for i in range(4) if i != CAMERA_INDEX]

        last_cap = None
        for cam_index in index_candidates:
            for backend_name, backend in backend_candidates:
                for use_mjpg in mjpg_candidates:
                    if last_cap is not None:
                        try:
                            last_cap.release()
                        except Exception:
                            pass
                        last_cap = None

                    cap = cv2.VideoCapture(cam_index, backend) if backend is not None else cv2.VideoCapture(cam_index)
                    last_cap = cap
                    if not cap.isOpened():
                        continue

                    if use_mjpg:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

                    ok, frame = cap.read()
                    if ok and frame is not None:
                        self.camera_source_text = f"/dev/video{cam_index} [{backend_name}{' MJPG' if use_mjpg else ''}]"
                        self.info_changed.emit(f"Camera connected: {self.camera_source_text}")
                        return cap

        if last_cap is None:
            last_cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2 if USE_V4L2 else 0)
        self.camera_source_text = ""
        return last_cap

    def draw_zone_box(self, frame, y1, y2, color, text):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, y1), (w, y2), color, 2)
        cv2.putText(frame, text, (10, max(30, y1 + 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def run(self):
        self.running = True
        self.total_count = 0
        self.next_track_id = 1
        self.tracks.clear()
        self.batch_start_time = time.time()
        self.last_count_time = 0.0

        self.status_changed.emit("LOADING MODEL")
        model = YOLO(MODEL_DIR, task="detect")

        cap = self.open_camera()
        if not cap.isOpened():
            self.status_changed.emit("CAMERA FAIL")
            self.info_changed.emit("Cannot open USB camera. Check /dev/videoX or camera busy.")
            return

        ok, test_frame = cap.read()
        if not ok or test_frame is None:
            cap.release()
            self.status_changed.emit("FRAME FAIL")
            self.info_changed.emit("Camera opened but no frame received")
            return

        prev_time = time.time()
        fps = 0.0
        frame_index = 0
        last_detections = []

        self.status_changed.emit("RUNNING")

        while self.running:
            if self.paused:
                self.msleep(100)
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                self.status_changed.emit("FRAME FAIL")
                self.info_changed.emit("Camera frame lost")
                break

            frame_index += 1
            now_ts = time.time()
            self.cleanup_tracks(now_ts)

            h, w = frame.shape[:2]
            top_y1 = int(h * TOP_ZONE_Y1_RATIO)
            top_y2 = int(h * TOP_ZONE_Y2_RATIO)
            count_y1 = int(h * COUNT_ZONE_Y1_RATIO)
            count_y2 = int(h * COUNT_ZONE_Y2_RATIO)
            exit_y1 = int(h * EXIT_ZONE_Y1_RATIO)
            exit_y2 = int(h * EXIT_ZONE_Y2_RATIO)
            count_line_y = int(h * COUNT_LINE_Y_RATIO)

            run_ai = (frame_index % PROCESS_EVERY_N_FRAMES == 0) or (not last_detections)
            detections = last_detections

            if run_ai:
                results = model.predict(
                    frame,
                    conf=CONFIDENCE,
                    iou=IOU,
                    imgsz=IMG_SIZE,
                    verbose=False,
                    device=0
                )[0]

                detections = []
                if results.boxes is not None and len(results.boxes) > 0:
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        conf = float(box.conf[0])
                        cx, cy = self.center_of_box(x1, y1, x2, y2)
                        detections.append({
                            "box": (x1, y1, x2, y2),
                            "center": (cx, cy),
                            "conf": conf
                        })

                last_detections = detections

            used_track_ids = set()

            for det in detections:
                x1, y1, x2, y2 = det["box"]
                cx, cy = det["center"]
                conf = det["conf"]

                matched_id = self.match_detection_to_track((cx, cy), used_track_ids)

                if matched_id is None:
                    matched_id = self.next_track_id
                    self.next_track_id += 1

                    born_in_top = self.in_zone(cy, top_y1, top_y2)
                    born_in_count = self.in_zone(cy, count_y1, count_y2)
                    born_in_exit = self.in_zone(cy, exit_y1, exit_y2)

                    self.tracks[matched_id] = {
                        "center": (cx, cy),
                        "prev_center": (cx, cy),
                        "last_seen": now_ts,
                        "counted": False,
                        "done": False,
                        "hits": 1,
                        "first_seen_y": cy,
                        "seen_top": born_in_top,
                        "seen_count": born_in_count,
                        "seen_exit": born_in_exit,
                        "move_down_hits": 0,
                        "move_up_hits": 0,
                        "direction_locked": None,
                    }
                else:
                    tr = self.tracks[matched_id]
                    tr["prev_center"] = tr["center"]
                    tr["center"] = (cx, cy)
                    tr["last_seen"] = now_ts
                    tr["hits"] += 1

                used_track_ids.add(matched_id)
                tr = self.tracks[matched_id]

                prev_cx, prev_cy = tr["prev_center"]
                dy = cy - prev_cy

                if dy > MIN_MOVE_PX:
                    tr["move_down_hits"] += 1
                elif dy < -MIN_MOVE_PX:
                    tr["move_up_hits"] += 1

                if tr["move_down_hits"] >= 2 and tr["direction_locked"] is None:
                    tr["direction_locked"] = "down"
                if tr["move_up_hits"] >= 2 and tr["direction_locked"] is None:
                    tr["direction_locked"] = "up"

                in_top = self.in_zone(cy, top_y1, top_y2)
                in_count = self.in_zone(cy, count_y1, count_y2)
                in_exit = self.in_zone(cy, exit_y1, exit_y2)

                if in_top:
                    tr["seen_top"] = True
                if in_count:
                    tr["seen_count"] = True
                if in_exit:
                    tr["seen_exit"] = True

                count_ready = False

                if tr["hits"] >= MIN_TRACK_HITS and not tr["counted"] and not tr["done"]:
                    if CONVEYOR_DIRECTION == "down":
                        correct_dir = (tr["direction_locked"] == "down")
                        crossed_line = (prev_cy < count_line_y <= cy)
                        valid_progress = tr["seen_top"] or tr["first_seen_y"] <= (count_y1 + 12)
                        if correct_dir and crossed_line and valid_progress:
                            count_ready = True
                    else:
                        correct_dir = (tr["direction_locked"] == "up")
                        crossed_line = (prev_cy > count_line_y >= cy)
                        valid_progress = tr["seen_exit"] or tr["first_seen_y"] >= (count_y2 - 12)
                        if correct_dir and crossed_line and valid_progress:
                            count_ready = True

                if count_ready and (now_ts - self.last_count_time) > COUNT_COOLDOWN_SEC:
                    self.total_count += 1
                    tr["counted"] = True
                    self.last_count_time = now_ts
                    self.count_changed.emit(self.total_count)

                    if self.total_count >= self.target_count:
                        self.status_changed.emit("COMPLETE")
                        self.save_batch_result("COMPLETE", self.extra_remark)
                        self.finished_target.emit()
                        self.running = False
                        break

                if tr["counted"]:
                    if CONVEYOR_DIRECTION == "down" and in_exit:
                        tr["done"] = True
                    elif CONVEYOR_DIRECTION == "up" and in_top:
                        tr["done"] = True

                box_color = (255, 0, 0) if not tr["counted"] else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

                state_txt = (
                    f"ID:{matched_id} "
                    f"T{int(tr['seen_top'])} C{int(tr['seen_count'])} E{int(tr['seen_exit'])} "
                    f"D:{tr['direction_locked']}"
                )
                if tr["counted"]:
                    state_txt += " COUNTED"

                cv2.putText(frame, f"Roll {conf:.2f}", (x1, max(25, y1 - 28)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(frame, state_txt, (x1, max(45, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            self.draw_zone_box(frame, top_y1, top_y2, (255, 255, 0), "TOP ZONE")
            self.draw_zone_box(frame, count_y1, count_y2, (0, 255, 255), "COUNT ZONE")
            self.draw_zone_box(frame, exit_y1, exit_y2, (0, 128, 255), "EXIT ZONE")

            cv2.line(frame, (0, count_line_y), (w, count_line_y), (0, 0, 255), 2)
            cv2.putText(frame, "COUNT LINE", (10, max(20, count_line_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, f"Pallet: {self.pallet_no}", (15, 94),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, f"Product: {self.product_code}", (15, 126),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            current_time = time.time()
            dt = current_time - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = current_time

            cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Count: {self.total_count}/{self.target_count}", (15, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.strides[0],
                QImage.Format_RGB888
            ).copy()

            self.frame_ready.emit(qimg)

        cap.release()

        if self.total_count < self.target_count and self.batch_start_time is not None:
            stop_remark = "Stopped before target"
            if self.extra_remark:
                stop_remark = f"{self.extra_remark} | {stop_remark}"

        self.status_changed.emit("STOPPED")


def save_add_roller_log(pallet_no, product_code, target_qty, actual_qty, remark, duration_sec=0.0):
    if not ENABLE_LOG_MYSQL:
        return True, "MySQL log disabled"

    try:
        import mysql.connector
        conn = mysql.connector.connect(**LOG_MYSQL_CONFIG)
        cur = conn.cursor()

        cur.execute(LOG_TABLE_SQL)
        cur.execute(
            f"""
            INSERT INTO {LOG_TABLE_NAME}
            (pallet_no, product_code, target_qty, actual_qty, event_type, event_time, duration_sec, remark)
            VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s)
            """,
            (
                str(pallet_no or ""),
                str(product_code or ""),
                int(target_qty),
                int(actual_qty),
                "ADD_ROLL",
                float(duration_sec),
                str(remark),
            )
        )

        conn.commit()
        cur.close()
        conn.close()
        return True, "Add roller log saved"

    except Exception as e:
        return False, f"MySQL save error: {e}"


class RollerCounterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.target_count = 0
        self.current_count = 0
        self.pallet_no = ""
        self.product_code = ""
        self.batch_remark = ""
        self.is_paused = False
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle(APP_TITLE)
        self.showFullScreen()

        screen = QApplication.primaryScreen().availableGeometry()
        sw = screen.width()
        sh = screen.height()

        left_panel_w = 290 if sw <= 1024 else 420
        title_fs = 18 if sw <= 1024 else 24
        count_fs = 42 if sw <= 1024 else 64
        status_fs = 24 if sw <= 1024 else 32
        button_fs = 14 if sw <= 1024 else 22
        info_fs = 14 if sw <= 1024 else 18
        label_fs = 14 if sw <= 1024 else 18
        camera_w = 680 if sw <= 1024 else 960
        camera_h = 500 if sh <= 768 else 540

        self.setStyleSheet("""
            QWidget {
                background: #0f172a;
                color: white;
                font-family: Arial;
            }
            QPushButton {
                border: none;
                border-radius: 10px;
                padding: 10px;
                font-weight: bold;
                color: white;
            }
        """)

        self.title_label = QLabel(APP_TITLE)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet(
            f"font-size: {title_fs}px; font-weight: bold; padding: 4px;"
        )

        self.pallet_btn = QPushButton("PALLET NO")
        self.pallet_btn.setMinimumHeight(52 if sw <= 1024 else 64)
        self.pallet_btn.setStyleSheet(f"background: #0ea5e9; font-size: {button_fs}px; font-weight: bold;")
        self.pallet_btn.clicked.connect(self.open_pallet_dialog)

        self.pause_btn = QPushButton("PAUSE")
        self.pause_btn.setMinimumHeight(52 if sw <= 1024 else 64)
        self.pause_btn.setStyleSheet(f"background: #8b5cf6; font-size: {button_fs}px; font-weight: bold;")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)

        self.pallet_label = QLabel("Pallet No: -")
        self.pallet_label.setAlignment(Qt.AlignCenter)
        self.pallet_label.setWordWrap(True)
        self.pallet_label.setStyleSheet(f"font-size: {label_fs}px; color: #e2e8f0;")

        self.product_label = QLabel("Product: -")
        self.product_label.setAlignment(Qt.AlignCenter)
        self.product_label.setWordWrap(True)
        self.product_label.setStyleSheet(f"font-size: {label_fs}px; color: #e2e8f0;")

        self.target_label = QLabel("Target Qty: 0")
        self.target_label.setAlignment(Qt.AlignCenter)
        self.target_label.setWordWrap(True)
        self.target_label.setStyleSheet(f"font-size: {label_fs}px; color: #e2e8f0;")

        self.count_label = QLabel("0 / 0")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet(
            f"font-size: {count_fs}px; font-weight: bold; color: #38bdf8;"
        )

        self.status_box = QLabel("IDLE")
        self.status_box.setAlignment(Qt.AlignCenter)
        self.status_box.setFixedHeight(60 if sw <= 1024 else 80)
        self.status_box.setStyleSheet(f"""
            background: #334155;
            color: white;
            font-size: {status_fs}px;
            font-weight: bold;
            border-radius: 16px;
        """)

        self.camera_label = QLabel("Camera View")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(camera_w, camera_h)
        self.camera_label.setStyleSheet("""
            background: black;
            border: 3px solid #475569;
            border-radius: 16px;
        """)

        self.info_label = QLabel("Ready")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(f"font-size: {info_fs}px; color: #cbd5e1;")

        self.pause_btn = QPushButton("PAUSE")
        self.start_btn = QPushButton("START")
        self.stop_btn = QPushButton("STOP")
        self.reset_btn = QPushButton("RESET")
        self.add_roller_btn = QPushButton("ADD ROLL")
        self.exit_btn = QPushButton("EXIT")

        self.pause_btn.setMinimumHeight(52 if sw <= 1024 else 64)
        self.start_btn.setMinimumHeight(52 if sw <= 1024 else 64)
        self.stop_btn.setMinimumHeight(52 if sw <= 1024 else 64)
        self.reset_btn.setMinimumHeight(52 if sw <= 1024 else 64)
        self.add_roller_btn.setMinimumHeight(52 if sw <= 1024 else 64)
        self.exit_btn.setMinimumHeight(52 if sw <= 1024 else 64)

        self.pause_btn.setStyleSheet(f"background: #8b5cf6; font-size: {button_fs}px; font-weight: bold;")
        self.start_btn.setStyleSheet(f"background: #16a34a; font-size: {button_fs}px; font-weight: bold;")
        self.stop_btn.setStyleSheet(f"background: #dc2626; font-size: {button_fs}px; font-weight: bold;")
        self.reset_btn.setStyleSheet(f"background: #2563eb; font-size: {button_fs}px; font-weight: bold;")
        self.add_roller_btn.setStyleSheet(f"background: #f59e0b; font-size: {button_fs}px; font-weight: bold;")
        self.exit_btn.setStyleSheet(f"background: #475569; font-size: {button_fs}px; font-weight: bold;")

        self.pause_btn.clicked.connect(self.toggle_pause)
        self.start_btn.clicked.connect(self.start_counting)
        self.stop_btn.clicked.connect(self.stop_counting)
        self.reset_btn.clicked.connect(self.reset_counting)
        self.add_roller_btn.clicked.connect(self.add_roller)
        self.exit_btn.clicked.connect(self.close)

        button_grid = QGridLayout()
        button_grid.setSpacing(8)
        button_grid.addWidget(self.pallet_btn, 0, 0, 1, 2)
        button_grid.addWidget(self.pause_btn, 1, 0, 1, 2)
        button_grid.addWidget(self.start_btn, 2, 0)
        button_grid.addWidget(self.stop_btn, 2, 1)
        button_grid.addWidget(self.reset_btn, 3, 0)
        button_grid.addWidget(self.add_roller_btn, 3, 1)
        button_grid.addWidget(self.exit_btn, 4, 0, 1, 2)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        left_box = QVBoxLayout()
        left_box.setSpacing(12)
        left_box.addWidget(self.title_label)
        left_box.addWidget(self.pallet_label)
        left_box.addWidget(self.product_label)
        left_box.addWidget(self.target_label)
        left_box.addWidget(self.count_label)
        left_box.addWidget(self.status_box)
        left_box.addWidget(self.info_label)
        left_box.addStretch()
        left_box.addLayout(button_grid)

        left_frame = QFrame()
        left_frame.setLayout(left_box)
        left_frame.setStyleSheet("QFrame { background: #111827; border-radius: 18px; }")
        left_frame.setFixedWidth(left_panel_w)

        top_row.addWidget(left_frame)
        top_row.addWidget(self.camera_label, 1)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(top_row)
        self.setLayout(layout)

    def beep_complete(self):
        QApplication.beep()
        QApplication.beep()
        QApplication.beep()

    def open_pallet_dialog(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Running", "Cannot change pallet while counter is running.")
            return

        dialog = PalletSearchDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        data = dialog.selected_data
        if not data:
            return

        self.pallet_no = str(data.get("pallet_no", "") or "")
        self.product_code = str(data.get("product_code", "") or "")
        self.target_count = int(data.get("target_qty", 0) or 0)
        self.current_count = 0
        self.batch_remark = ""

        self.update_info_panel()
        self.update_count_label()
        self.info_label.setText(
            f"Loaded {self.pallet_no} | Product: {self.product_code or '-'} | Target: {self.target_count}"
        )

        # auto start after click USE THIS PALLET
        QTimer.singleShot(200, self.start_counting)

    def update_frame(self, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        self.camera_label.setPixmap(
            pix.scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def on_count_changed(self, count):
        self.current_count = int(count)
        self.update_count_label()

    def on_status_changed(self, text):
        self.set_status(text)

    def on_finished_target(self):
        self.beep_complete()
        self.info_label.setText("Roll complete")

        QMessageBox.information(
            self,
            "Roll Complete",
            f"Roll is complete\n\nCounter: {self.current_count}/{self.target_count}"
        )

        self.reset_counting()

    def on_batch_saved(self, msg):
        self.info_label.setText(msg)

    def on_info_changed(self, msg):
        self.info_label.setText(msg)

    def on_worker_finished(self):
        pass

    def update_info_panel(self):
        self.pallet_label.setText(f"Pallet No: {self.pallet_no or '-'}")
        self.product_label.setText(f"Product: {self.product_code or '-'}")
        self.target_label.setText(f"Target Qty: {self.target_count}")

    def start_counting(self):
        if not self.pallet_no:
            QMessageBox.warning(self, "No Pallet", "Please click PALLET NO and search first.")
            return

        if self.target_count <= 0:
            QMessageBox.warning(self, "Invalid Target", "This pallet has no row with gross_weight = 0.")
            return

        if self.worker is not None and self.worker.isRunning():
            return

        self.current_count = 0
        self.batch_remark = ""
        self.is_paused = False
        self.update_count_label()
        self.info_label.setText("Starting...")
        self.pause_btn.setText("PAUSE")
        self.pause_btn.setEnabled(True)

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

        self.worker = CounterWorker(
            self.target_count,
            pallet_no=self.pallet_no,
            product_code=self.product_code
        )
        self.worker.extra_remark = self.batch_remark
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.count_changed.connect(self.on_count_changed)
        self.worker.status_changed.connect(self.on_status_changed)
        self.worker.finished_target.connect(self.on_finished_target)
        self.worker.batch_saved.connect(self.on_batch_saved)
        self.worker.info_changed.connect(self.on_info_changed)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def stop_counting(self):
        if self.worker is not None and self.worker.isRunning():
            self.info_label.setText("Stopping...")
            self.worker.stop()
            self.worker.wait(2000)

        self.is_paused = False
        self.pause_btn.setText("PAUSE")
        self.pause_btn.setEnabled(False)

    def reset_counting(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)

        self.is_paused = False
        self.pause_btn.setText("PAUSE")
        self.pause_btn.setEnabled(False)

        self.current_count = 0
        self.target_count = 0
        self.pallet_no = ""
        self.product_code = ""
        self.batch_remark = ""

        self.update_info_panel()
        self.count_label.setText("0 / 0")
        self.info_label.setText("Ready")
        self.camera_label.clear()
        self.camera_label.setText("Camera View")
        self.set_status("IDLE")

    def add_roller(self):
        if not self.pallet_no:
            QMessageBox.warning(self, "No Pallet", "Please load Pallet No first.")
            return

        if self.target_count <= 0:
            QMessageBox.warning(self, "Invalid Target", "No active target to add roller.")
            return

        dialog = AddRollerDialog(
            current_count=self.current_count,
            current_target=self.target_count,
            parent=self
        )

        if dialog.exec() != QDialog.Accepted:
            return

        add_qty = int(dialog.add_qty)
        operator_name = dialog.operator_name.strip()

        old_target = int(self.target_count)
        new_target = old_target + add_qty

        self.target_count = new_target

        if self.worker is not None:
            self.worker.target_count = new_target

        self.update_info_panel()
        self.update_count_label()

        remark = f"{operator_name} add roll {add_qty} (target {old_target} -> {new_target})"

        if self.batch_remark:
            self.batch_remark = f"{self.batch_remark} | {remark}"
        else:
            self.batch_remark = remark

        if self.worker is not None:
            self.worker.extra_remark = self.batch_remark


        self.info_label.setText(
            f"Added roll {add_qty}: {self.current_count} / {new_target}"
        )

        QMessageBox.information(
            self,
            "Add Roll",
            f"Add roll success\n\n"
            f"User: {operator_name}\n"
            f"Current count: {self.current_count}\n"
            f"Old target: {old_target}\n"
            f"Added: {add_qty}\n"
            f"New target: {new_target}"
        )
        
    def update_count_label(self):
        self.count_label.setText(f"{self.current_count} / {self.target_count}")

    def toggle_pause(self):
        if self.worker is None or not self.worker.isRunning():
            QMessageBox.warning(self, "Not Running", "Counter is not running.")
            return

        if not self.is_paused:
            self.worker.pause()
            self.is_paused = True
            self.pause_btn.setText("PLAY")
            self.info_label.setText("Paused")
            self.set_status("PAUSED")
        else:
            self.worker.resume()
            self.is_paused = False
            self.pause_btn.setText("PAUSE")
            self.info_label.setText("Running...")
            self.set_status("RUNNING")

    def set_status(self, text):
        self.status_box.setText(text)

        if text == "RUNNING":
            self.status_box.setStyleSheet("""
                background: #16a34a;
                color: white;
                font-size: 24px;
                font-weight: bold;
                border-radius: 16px;
            """)
        elif text == "COMPLETE":
            self.status_box.setStyleSheet("""
                background: #22c55e;
                color: white;
                font-size: 24px;
                font-weight: bold;
                border-radius: 16px;
            """)
        elif text == "PAUSED":
            self.status_box.setStyleSheet("""
                background: #8b5cf6;
                color: white;
                font-size: 24px;
                font-weight: bold;
                border-radius: 16px;
            """)
        elif text in ("STOPPED", "IDLE"):
            self.status_box.setStyleSheet("""
                background: #475569;
                color: white;
                font-size: 24px;
                font-weight: bold;
                border-radius: 16px;
            """)
        elif text == "LOADING MODEL":
            self.status_box.setStyleSheet("""
                background: #f59e0b;
                color: white;
                font-size: 24px;
                font-weight: bold;
                border-radius: 16px;
            """)
        elif text in ("CAMERA FAIL", "FRAME FAIL"):
            self.status_box.setStyleSheet("""
                background: #dc2626;
                color: white;
                font-size: 24px;
                font-weight: bold;
                border-radius: 16px;
            """)
        else:
            self.status_box.setStyleSheet("""
                background: #334155;
                color: white;
                font-size: 24px;
                font-weight: bold;
                border-radius: 16px;
            """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RollerCounterApp()
    window.show()
    sys.exit(app.exec())