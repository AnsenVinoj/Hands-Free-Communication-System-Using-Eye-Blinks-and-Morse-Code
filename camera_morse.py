import cv2
import time
import os
import threading
import glob
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()  # read .env (CAMERA_SOURCE, etc.)

class MorseCam:
    def __init__(self, predictor):
        self.model = YOLO("best.pt")
        self.predictor = predictor


        # ── Camera source from environment ────────────────────────────
        # CAMERA_SOURCE in .env controls what camera to open:
        #
        #   CAMERA_SOURCE=0               → USB / built-in, index 0
        #   CAMERA_SOURCE=2               → USB / built-in, index 2
        #   CAMERA_SOURCE=http://192.168.1.100:8080/video  → DroidCam / IP cam
        #   CAMERA_SOURCE=rtsp://192.168.1.100:554/stream  → RTSP (ESP32-CAM)
        #   (leave blank or unset)        → auto-detect first working USB cam
        #
        cam_src = os.environ.get("CAMERA_SOURCE", "").strip()

        self.cap = None
        self._camera_idx = None   # None when using a URL stream

        if cam_src.startswith("http") or cam_src.startswith("rtsp"):
            # ── WiFi / IP camera (DroidCam, ESP32-CAM, RTSP, etc.) ────────
            print(f"[MorseCam] Connecting to WiFi/IP camera: {cam_src}")
            cap = cv2.VideoCapture(cam_src)
            # Give RTSP/HTTP streams time to buffer
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    self._stream_url = cam_src
                    print(f"[MorseCam] ✅ WiFi camera connected: {cam_src}")
                else:
                    cap.release()
                    print(f"[MorseCam] ⚠️  No frame from {cam_src}")
            else:
                print(f"[MorseCam] ⚠️  Could not open {cam_src}")

        elif cam_src.isdigit():
            # ── Explicit USB index ───────────────────────────────────────
            idx = int(cam_src)
            print(f"[MorseCam] Using explicit camera index {idx}")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    self._camera_idx = idx
                    print(f"[MorseCam] ✅ Camera index {idx} opened")
                else:
                    cap.release()

        if self.cap is None:
            # ── Auto-scan all /dev/video* devices ─────────────────────────
            print("[MorseCam] Auto-scanning cameras...")
            candidates = []
            try:
                dev_nodes = sorted(glob.glob("/dev/video*"))
                candidates = [int(d.replace("/dev/video", "")) for d in dev_nodes]
            except Exception:
                pass
            for i in range(5):
                if i not in candidates:
                    candidates.append(i)

            for idx in candidates:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        self.cap = cap
                        self._camera_idx = idx
                        print(f"[MorseCam] ✅ Using camera index {idx} (/dev/video{idx})")
                        break
                    cap.release()

        if self.cap is None:
            raise RuntimeError(
                "Camera not accessible.\n"
                "  • For USB camera: set CAMERA_SOURCE=0 (or 1, 2...) in .env\n"
                "  • For WiFi/DroidCam: set CAMERA_SOURCE=http://IP:PORT/video in .env\n"
                "  • For RTSP/ESP32-CAM: set CAMERA_SOURCE=rtsp://IP:PORT/stream in .env"
            )

        # ── Thread-safety lock ──────────────────────────────────────────────
        # generate_frames() runs in the MJPEG streaming thread.
        # /api/sync_text reads state from the Flask/Werkzeug thread.
        # Without a lock, decoded_text can be read mid-update → empty string.
        self._lock = threading.Lock()

        # State (always mutate/read under self._lock)
        self.blink_start = None
        self.last_event = time.time()

        self.current_symbol = ""
        self.morse_code = ""
        self.decoded_text = ""
        self.eye_detected = False

        self.suggestions = []
        self.alert_message = None
        self.action_queue = []

        # Timing (IMPORTANT – tuned for webcam)
        self.DOT_TIME = 0.5
        self.LETTER_GAP = 2.5
        self.WORD_GAP = 3

        self.MORSE_DICT = {
            '.-':'A','-...':'B','-.-.':'C','-..':'D','.':'E',
            '..-.':'F','--.':'G','....':'H','..':'I','.---':'J',
            '-.-':'K','.-..':'L','--':'M','-.':'N','---':'O',
            '.--.':'P','--.-':'Q','.-.':'R','...':'S','-':'T',
            '..-':'U','...-':'V','.--':'W','-..-':'X','-.--':'Y',
            '--..':'Z'
        }

    # ==================================================
    def get_state_snapshot(self):
        """Return a thread-safe snapshot of all state needed by /api/sync_text."""
        with self._lock:
            action = self.action_queue.pop(0) if self.action_queue else None
            return {
                "text":         self.decoded_text,
                "morse":        self.morse_code + self.current_symbol,
                "eye_detected": self.eye_detected,
                "action":       action,
            }

    def manual_control(self, action_type):
        """Handle manual UI button presses for space, backspace, etc."""
        with self._lock:
            if action_type == "space":
                if not self.decoded_text.endswith(" ") and self.decoded_text != "":
                    self.decoded_text += " "
                self.last_event = time.time()
            elif action_type == "backspace":
                if self.decoded_text:
                    self.decoded_text = self.decoded_text[:-1]
                self.last_event = time.time()
            elif action_type == "clear":
                self.decoded_text = ""
                self.morse_code = ""
                self.action_queue.append({"type": "clear"})
                self.last_event = time.time()

    def generate_frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                # Try to re-open the camera before giving up
                # Works for both USB (integer index) and WiFi URL streams
                self.cap.release()
                src = getattr(self, "_stream_url", None) or self._camera_idx
                if src is None:
                    break
                self.cap = cv2.VideoCapture(src)
                if src and str(src).startswith(("http", "rtsp")):
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, frame = self.cap.read()
                if not ok:
                    break

            now = time.time()
            labels = []

            results = self.model(frame, conf=0.25, verbose=False)

            for r in results:
                for box in r.boxes:
                    label = self.model.names[int(box.cls)].lower()
                    labels.append(label)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0) if label == "open" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            eye_detected = len(labels) > 0
            eye_closed   = "close" in labels

            # ── All state mutations happen under the lock ────────────
            with self._lock:
                self.eye_detected = eye_detected

                # -------- BLINK START --------
                if eye_closed and self.blink_start is None:
                    self.blink_start = now

                # -------- BLINK END --------
                if not eye_closed and self.blink_start:
                    duration = now - self.blink_start
                    self.current_symbol += "." if duration < self.DOT_TIME else "-"
                    self.blink_start = None
                    self.last_event = now

                # -------- LETTER COMMIT --------
                if self.current_symbol and (now - self.last_event) > self.LETTER_GAP:
                    self.morse_code += self.current_symbol + " "

                    if self.current_symbol == ".....":
                        if not self.decoded_text.endswith(" "):
                            self.decoded_text += " "
                            
                    elif self.current_symbol in ["....-", "-....", "--..."]:
                        idx = ["....-", "-....", "--..."].index(self.current_symbol)
                        text_str  = self.decoded_text.strip()
                        space_end = self.decoded_text.endswith(" ") or self.decoded_text == ""

                        if space_end:
                            preds = self.predictor.predict_next_word(text_str)
                        else:
                            parts  = text_str.split()
                            last   = parts[-1] if parts else ""
                            prefix = " ".join(parts[:-1]) if len(parts) > 1 else ""
                            preds  = self.predictor.complete_word(last, prefix)

                        if idx < len(preds):
                            pred_word = preds[idx].word
                            if not space_end:
                                if " " in self.decoded_text.rstrip():
                                    prefix_str = self.decoded_text.rstrip().rsplit(" ", 1)[0]
                                    self.decoded_text = prefix_str + " " + pred_word
                                else:
                                    self.decoded_text = pred_word
                            else:
                                if not self.decoded_text.endswith(" ") and self.decoded_text != "":
                                    self.decoded_text += " "
                                self.decoded_text += pred_word

                    elif self.current_symbol == "......":  # 6 dots = Backspace
                        if self.decoded_text:
                            self.decoded_text = self.decoded_text[:-1]

                    elif self.current_symbol in ("----", "..--"):  # Clear All
                        self.decoded_text = ""
                        self.morse_code   = ""
                        self.action_queue.append({"type": "clear"})

                    elif self.current_symbol == ".-.-.": # Speak
                        if self.decoded_text.strip():
                            self.action_queue.append({"type": "speak",
                                                      "text": self.decoded_text.strip()})
                            self.decoded_text = ""

                    elif self.current_symbol == "...---...": # SOS Alert
                        self.action_queue.append({"type": "alert",
                                                  "text": "EMERGENCY! HELP NEEDED!"})
                        self.decoded_text = "EMERGENCY! "

                    else:
                        letter = self.MORSE_DICT.get(self.current_symbol, "")
                        self.decoded_text += letter

                    self.current_symbol = ""
                    self.last_event = now

                # -------- WORD SPACE --------
                # Avoid unnecessary spaces: we just process alerts here
                if self.decoded_text and (now - self.last_event) > self.WORD_GAP:
                    alerts = {
                        "INH": "HELP! ",
                        "DR":  "NEED DOCTOR! ",
                        "EM":  "SOS! ",
                        "WP":  "WATER ",
                        "FD":  "FOOD ",
                        "Y":   "YES ",
                        "N":   "NO "
                    }
                    parts = self.decoded_text.split()
                    if parts:
                        last_word = parts[-1]
                        if last_word in alerts:
                            v = alerts[last_word]
                            prefix = " ".join(parts[:-1])
                            self.decoded_text = (prefix + " " + v) if prefix else v
                            # Trigger speech/alert for all predefined quick commands
                            action_type = "alert" if "!" in v else "speak"
                            self.action_queue.append({"type": action_type,
                                                       "text": v.strip()})
                    self.last_event = now

                # Take an overlay snapshot while we hold the lock
                overlay_morse = self.morse_code + self.current_symbol
                overlay_text  = self.decoded_text
            # ── Lock released ───────────────────────────────────────

            # -------- UI overlay (uses snapshot, no lock needed) ----
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Morse: {overlay_morse}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Text: {overlay_text}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")

    def release(self):
        self.cap.release()