from ultralytics import YOLO
import cv2
import torch
import time
import threading
import json
import os
import random
from datetime import datetime
from collections import deque, Counter

# ============================================================
# CẤU HÌNH - Chỉnh tại đây
# ============================================================
MODEL_PATH    = "best.pt"
CONF_THRESH   = 0.55
IOU_THRESH    = 0.5
IMGSZ         = 640
FRAME_SKIP    = 1
SMOOTH_WINDOW = 5
MIN_VOTE      = 3
CAM_W, CAM_H  = 640, 480
HISTORY_FILE  = "word_history.json"

# Quiz settings
QUIZ_QUESTIONS   = 10     # Số câu mỗi ván
QUIZ_HOLD_FRAMES = 18     # Giữ đúng bao nhiêu frame liên tiếp → tính điểm
QUIZ_TIME = {             # Giây giới hạn mỗi câu
    "EASY"  : 15,
    "NORMAL": 10,
    "HARD"  : 6,
}
QUIZ_SKIP_PENALTY = 5
QUIZ_NEXT_DELAY   = 2.0   # Giây chờ sau khi trả lời → sang câu tiếp
# ============================================================


# =====================
# 1. THIẾT BỊ & MODEL
# =====================
device   = "cuda" if torch.cuda.is_available() else "cpu"
use_half = device == "cuda"
print(f"🚀 Thiết bị: {device} | FP16: {use_half}")

try:
    model = YOLO(MODEL_PATH).to(device)
    if use_half:
        model.model.half()
    dummy = torch.zeros(1, 3, IMGSZ, IMGSZ, device=device)
    if use_half:
        dummy = dummy.half()
    model.predict(source=dummy, verbose=False, imgsz=IMGSZ)
    print("✅ Model sẵn sàng (warmup xong)")
except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    exit()

ALL_CLASSES = list(model.names.values())

# =====================
# 2. CAMERA THREAD
# =====================
class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.lock    = threading.Lock()
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def release(self):
        self.running = False
        self.cap.release()

cam = CameraStream(0, CAM_W, CAM_H)
if not cam.ret:
    print("❌ Không mở được camera")
    exit()
print("✅ Camera sẵn sàng — Nhấn 'Q' để thoát")


# =====================
# 3. TEMPORAL SMOOTHER
# =====================
class TemporalSmoother:
    def __init__(self, window=5, min_vote=3):
        self.window   = window
        self.min_vote = min_vote
        self.history  = deque(maxlen=window)

    def update(self, labels: list):
        self.history.append(labels)

    def get_stable_labels(self) -> list:
        if not self.history:
            return []
        all_labels = [lbl for frame_labels in self.history for lbl in frame_labels]
        counts = Counter(all_labels)
        return [lbl for lbl, cnt in counts.items() if cnt >= self.min_vote]

    def top(self):
        stable = self.get_stable_labels()
        return stable[0] if stable else None

smoother = TemporalSmoother(window=SMOOTH_WINDOW, min_vote=MIN_VOTE)


# =====================
# 3.5 WORD BUILDER
# =====================
class WordBuilder:
    def __init__(self, hold_frames=10, reset_frames=30):
        self.hold_frames       = hold_frames
        self.reset_frames      = reset_frames
        self.current_letter    = None
        self.letter_counter    = 0
        self.idle_counter      = 0
        self.current_word      = ""
        self.completed_words   = []
        self.saved_words       = []
        self.max_words_history = 5
        self.show_history_mode = False
        self.show_session_panel= True
        self.building_enabled  = True
        self.load_history()

    def load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.saved_words = data.get('words', [])
                print(f"📚 Đã load {len(self.saved_words)} từ từ lịch sử")
            except Exception as e:
                print(f"⚠️ Không thể load lịch sử: {e}")
                self.saved_words = []

    def save_history(self):
        try:
            if len(self.saved_words) > 100:
                self.saved_words = self.saved_words[-100:]
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump({'words': self.saved_words}, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"⚠️ Không thể lưu lịch sử: {e}")
            return False

    def update(self, stable_labels):
        if not self.building_enabled:
            if stable_labels:
                self.current_letter = stable_labels[0]
                self.letter_counter = 1
            else:
                self.current_letter = None
                self.letter_counter = 0
            return

        if stable_labels:
            current_detection = stable_labels[0]
            if self.current_letter != current_detection:
                if (self.current_letter is not None and
                        self.letter_counter >= self.hold_frames):
                    self.current_word += self.current_letter
                self.current_letter = current_detection
                self.letter_counter = 1
            else:
                self.letter_counter += 1
            self.idle_counter = 0
        else:
            if self.current_letter is not None:
                if self.letter_counter >= self.hold_frames:
                    self.current_word += self.current_letter
                self.current_letter = None
                self.letter_counter = 0
            self.idle_counter += 1
            if self.idle_counter >= self.reset_frames and self.current_word:
                self.complete_current_word()

    def complete_current_word(self):
        if self.current_word:
            self.completed_words.append(self.current_word)
            if len(self.completed_words) > self.max_words_history:
                self.completed_words.pop(0)
            self.current_word = ""
        if self.current_letter:
            self.current_letter = None
            self.letter_counter = 0

    def force_complete_word(self):
        self.complete_current_word()

    def save_current_word(self):
        if self.current_word:
            word_to_save = self.current_word
            self.completed_words.append(word_to_save)
            self.saved_words.append({
                'word': word_to_save,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            self.save_history()
            self.current_word   = ""
            self.current_letter = None
            self.letter_counter = 0
            return True
        elif self.completed_words:
            last_word = self.completed_words[-1]
            self.saved_words.append({
                'word': last_word,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            self.save_history()
            return True
        return False

    def delete_last_letter(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]

    def clear_current_word(self):
        self.current_word   = ""
        self.current_letter = None
        self.letter_counter = 0

    def clear_session(self):
        self.completed_words    = []
        self.show_session_panel = False

    def toggle_session_panel(self):
        self.show_session_panel = not self.show_session_panel

    def toggle_history_mode(self):
        self.show_history_mode = not self.show_history_mode

    def toggle_building_mode(self):
        self.building_enabled = not self.building_enabled
        if not self.building_enabled:
            self.current_word    = ""
            self.completed_words = []
        return self.building_enabled

    def clear_history(self):
        self.saved_words = []
        self.save_history()

    def get_display_text(self):
        if self.show_history_mode:
            history_lines = []
            for i, item in enumerate(self.saved_words[-10:]):
                if isinstance(item, dict):
                    history_lines.append(f"{i+1}. {item['word']} ({item['time']})")
                else:
                    history_lines.append(f"{i+1}. {item}")
            return {
                'mode'            : 'history',
                'lines'           : history_lines,
                'current_word'    : self.current_word,
                'holding'         : f" → {self.current_letter}" if self.current_letter else "",
                'building_enabled': self.building_enabled,
            }
        else:
            holding_text = f" → {self.current_letter}" if self.current_letter else ""
            if not self.building_enabled:
                return {
                    'mode'            : 'view_only',
                    'current_label'   : self.current_letter,
                    'saved_count'     : len(self.saved_words),
                    'building_enabled': False,
                }
            return {
                'mode'           : 'normal',
                'current_word'   : self.current_word,
                'holding'        : holding_text,
                'session_history': ' | '.join(self.completed_words[-3:]) if self.show_session_panel else "",
                'saved_count'    : len(self.saved_words),
                'show_session'   : self.show_session_panel,
                'building_enabled': True,
            }

word_builder = WordBuilder(hold_frames=10, reset_frames=30)


# ══════════════════════════════════════════════
# 3.6  QUIZ ENGINE
# ══════════════════════════════════════════════
class QuizEngine:
    def __init__(self, difficulty="NORMAL"):
        self.difficulty     = difficulty
        self.time_limit     = QUIZ_TIME[difficulty]
        self.questions      = random.sample(ALL_CLASSES, min(QUIZ_QUESTIONS, len(ALL_CLASSES)))
        self.q_index        = 0
        self.score          = 0
        self.streak         = 0
        self.max_streak     = 0
        self.correct_count  = 0
        self.skipped_count  = 0
        self.q_start_time   = time.time()
        self.hold_counter   = 0
        self.last_detection = None
        self.answered       = False
        self.answer_result  = None   # "correct" | "skip" | "timeout"
        self.answer_time    = 0.0
        self.finished       = False
        self.history        = []     # [(letter, result, time_taken, score_gained)]

    # ── properties ──
    @property
    def current_q(self):
        return self.questions[self.q_index] if self.q_index < len(self.questions) else None

    @property
    def remaining(self):
        return max(0.0, self.time_limit - (time.time() - self.q_start_time))

    @property
    def time_pct(self):
        return self.remaining / self.time_limit

    @property
    def elapsed(self):
        return time.time() - self.q_start_time

    @property
    def accuracy(self):
        total = len(self.history)
        return self.correct_count / max(total, 1) * 100

    # ── logic ──
    def calc_score(self):
        base         = 100
        time_bonus   = int(self.remaining / self.time_limit * 50)
        streak_bonus = min(self.streak * 10, 50)
        return base + time_bonus + streak_bonus

    def update(self, detected_label):
        if self.answered or self.finished:
            return
        if self.remaining <= 0:
            self._record("timeout", 0)
            return
        if detected_label == self.current_q:
            self.hold_counter = self.hold_counter + 1 if detected_label == self.last_detection else 1
            self.last_detection = detected_label
            if self.hold_counter >= QUIZ_HOLD_FRAMES:
                self._record("correct", self.calc_score())
        else:
            self.hold_counter   = 0
            self.last_detection = detected_label

    def skip(self):
        if not self.answered and not self.finished:
            self._record("skip", -QUIZ_SKIP_PENALTY)

    def _record(self, result, gained):
        self.history.append((self.current_q, result, self.elapsed, gained))
        self.answer_result = result
        self.answer_time   = time.time()
        self.answered      = True
        if result == "correct":
            self.score        += gained
            self.streak       += 1
            self.max_streak    = max(self.max_streak, self.streak)
            self.correct_count += 1
        elif result == "skip":
            self.score  = max(0, self.score - QUIZ_SKIP_PENALTY)
            self.streak = 0
            self.skipped_count += 1
        else:
            self.streak = 0

    def next_question(self):
        self.q_index       += 1
        self.answered       = False
        self.answer_result  = None
        self.hold_counter   = 0
        self.last_detection = None
        self.q_start_time   = time.time()
        if self.q_index >= len(self.questions):
            self.finished = True

    def grade(self):
        acc = self.accuracy
        if acc >= 90: return "S", (30,  215, 255)
        if acc >= 75: return "A", (60,  220, 120)
        if acc >= 60: return "B", (255, 200,  80)
        if acc >= 45: return "C", (80,  180, 255)
        return               "D", (80,   80, 230)


# ══════════════════════════════════════════════
# 3.7  QUIZ RENDERER  (vẽ lên annotated_frame gốc 640×480)
# ══════════════════════════════════════════════
def _rect(img, x1, y1, x2, y2, color, alpha=0.72):
    """Vẽ hình chữ nhật mờ"""
    sub = img[y1:y2, x1:x2]
    if sub.size == 0:
        return
    overlay = sub.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

def _txt(img, text, x, y, scale=0.5, color=(255,255,255), thick=1):
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y  ), cv2.FONT_HERSHEY_SIMPLEX, scale, color,   thick,   cv2.LINE_AA)

def _txt_c(img, text, cx, cy, scale=0.5, color=(255,255,255), thick=1):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    _txt(img, text, cx - w//2, cy + h//2, scale, color, thick)

def _bar(img, x, y, w, h, pct, bg=(40,40,40), fg=(80,200,255)):
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    fw = max(0, int(w * pct))
    if fw > 0:
        cv2.rectangle(img, (x, y), (x+fw, y+h), fg, -1)


# ── Màn hình chọn độ khó ──
def draw_quiz_menu(frame):
    H, W = frame.shape[:2]
    overlay = frame.copy()
    overlay[:] = (15, 15, 25)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    _txt_c(frame, "SIGN LANGUAGE QUIZ", W//2, 70,  scale=1.1, color=(80,200,255), thick=2)
    _txt_c(frame, "Chon do kho:",       W//2, 115, scale=0.6, color=(200,200,200))

    options = [
        ("1", "EASY",   "15s / cau", (60,200,80)),
        ("2", "NORMAL", "10s / cau", (40,200,255)),
        ("3", "HARD",   " 6s / cau", (80,80,230)),
    ]
    bw, bh = 150, 70
    gap    = 20
    total  = bw * 3 + gap * 2
    sx     = (W - total) // 2

    for i, (key, name, desc, col) in enumerate(options):
        bx = sx + i * (bw + gap)
        by = 145
        _rect(frame, bx, by, bx+bw, by+bh, (25,28,45), alpha=0.9)
        cv2.rectangle(frame, (bx, by), (bx+bw, by+3), col, -1)
        _txt_c(frame, f"[{key}] {name}", bx+bw//2, by+33, scale=0.6, color=col, thick=2)
        _txt_c(frame, desc,              bx+bw//2, by+58, scale=0.4, color=(160,160,160))

    _txt_c(frame, f"{QUIZ_QUESTIONS} cau  |  {len(ALL_CLASSES)} ky hieu",
           W//2, 245, scale=0.45, color=(120,120,140))
    _txt_c(frame, "G: thoat Quiz mode  |  Q: thoat chuong trinh",
           W//2, H-15, scale=0.35, color=(100,100,120))


# ── Màn hình chơi ──
def draw_quiz_playing(frame, quiz: QuizEngine, detected):
    H, W = frame.shape[:2]

    letter = quiz.current_q or "?"

    # ── Nền panel trái (thông tin) ──
    _rect(frame, 0, 0, 210, H, (10, 12, 22), alpha=0.80)
    cv2.line(frame, (210, 0), (210, H), (50, 55, 80), 1)

    # Score & streak
    _txt(frame, f"SCORE",                  10, 25,  scale=0.38, color=(120,120,140))
    _txt(frame, f"{quiz.score:,}",         10, 50,  scale=0.8,  color=(80,200,255), thick=2)
    _txt(frame, f"STREAK",                 115, 25, scale=0.38, color=(120,120,140))
    scol = (40,200,255) if quiz.streak >= 3 else (200,200,200)
    _txt(frame, f"x{quiz.streak}",         115, 50, scale=0.8,  color=scol, thick=2)

    cv2.line(frame, (8, 60), (202, 60), (40,42,60), 1)

    # Progress câu
    done  = quiz.q_index
    total = len(quiz.questions)
    _txt(frame, f"Q {done+1} / {total}", 10, 80, scale=0.42, color=(160,160,180))
    _bar(frame, 8, 88, 194, 7, done/total, bg=(30,32,48), fg=(80,200,255))

    # Chữ cần giơ (BIG)
    _rect(frame, 8, 105, 202, 240, (20,22,38), alpha=0.88)
    (lw, lh), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_DUPLEX, 3.8, 6)
    lx = 8 + (194 - lw) // 2
    # Glow
    cv2.putText(frame, letter, (lx+3, 210+3), cv2.FONT_HERSHEY_DUPLEX, 3.8, (20,60,100), 8, cv2.LINE_AA)
    cv2.putText(frame, letter, (lx,   210   ), cv2.FONT_HERSHEY_DUPLEX, 3.8, (255,255,255), 5, cv2.LINE_AA)
    _txt_c(frame, "SHOW THIS SIGN", 105, 232, scale=0.38, color=(100,100,120))

    # Timer
    pct  = quiz.time_pct
    tcol = (60,200,80) if pct > 0.5 else ((40,200,255) if pct > 0.25 else (80,80,230))
    _txt(frame, f"{quiz.remaining:.1f}s", 10, 262, scale=0.5, color=tcol, thick=1)
    _bar(frame, 8, 270, 194, 10, pct, bg=(30,32,48), fg=tcol)

    # Detection box
    _rect(frame, 8, 290, 202, 360, (20,22,38), alpha=0.85)
    _txt_c(frame, "DETECTED:", 105, 305, scale=0.38, color=(100,100,120))
    if detected:
        dcol = (60,220,80) if detected == quiz.current_q else (80,80,230)
        _txt_c(frame, detected, 105, 345, scale=1.1, color=dcol, thick=2)
        # Hold progress bar (chỉ khi đang đúng)
        if detected == quiz.current_q and not quiz.answered:
            hold_pct = quiz.hold_counter / QUIZ_HOLD_FRAMES
            _bar(frame, 8, 355, 194, 6, hold_pct, bg=(30,32,48), fg=(60,220,80))
    else:
        _txt_c(frame, "---", 105, 340, scale=0.7, color=(50,50,70))

    # Answer flash
    if quiz.answered and quiz.answer_result:
        age = time.time() - quiz.answer_time
        if age < QUIZ_NEXT_DELAY:
            result = quiz.answer_result
            if result == "correct":
                msg, bcol = f"+{quiz.history[-1][3]} CORRECT!", (20,100,30)
            elif result == "skip":
                msg, bcol = f"-{QUIZ_SKIP_PENALTY} SKIPPED",   (80,70,10)
            else:
                msg, bcol = "TIME'S UP!",                       (80,20,20)
            alpha = min(1.0, (QUIZ_NEXT_DELAY - age) / 0.4)
            _rect(frame, 8, 368, 202, 400, bcol, alpha=alpha * 0.9)
            _txt_c(frame, msg, 105, 388, scale=0.5, color=(255,255,255), thick=1)

    # History mini (cuối panel)
    _txt(frame, "RECENT:", 10, 420, scale=0.35, color=(80,80,100))
    for i, (ltr, res, t, pts) in enumerate(reversed(quiz.history[-4:])):
        ry  = 438 + i * 18
        col = (60,200,80) if res == "correct" else ((200,180,60) if res == "skip" else (80,80,200))
        ico = "+" if res == "correct" else (">" if res == "skip" else "x")
        _txt(frame, f"{ico} {ltr}  {pts:+}pt", 10, ry, scale=0.35, color=col)

    # Help
    _txt_c(frame, "SPACE:skip  G:exit  Q:quit",
           105, H-12, scale=0.32, color=(80,80,100))


# ── Màn hình kết quả ──
def draw_quiz_result(frame, quiz: QuizEngine):
    H, W = frame.shape[:2]
    overlay = frame.copy()
    overlay[:] = (12, 14, 22)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    grade, gcol = quiz.grade()

    # Grade circle
    cv2.circle(frame, (W//2, 95), 55, (25,28,45), -1)
    cv2.circle(frame, (W//2, 95), 55, gcol, 2)
    _txt_c(frame, grade, W//2, 108, scale=2.5, color=gcol, thick=5)

    _txt_c(frame, "RESULT", W//2, 175, scale=0.9, color=(220,220,240), thick=2)

    # Stats
    stats = [
        (f"Score: {quiz.score:,}",               (80,200,255)),
        (f"Correct: {quiz.correct_count}/{len(quiz.questions)}", (60,200,80)),
        (f"Accuracy: {quiz.accuracy:.0f}%",       (40,200,255)),
        (f"Max Streak: x{quiz.max_streak}",       (200,160,40)),
    ]
    for i, (txt, col) in enumerate(stats):
        _txt_c(frame, txt, W//2, 210 + i*28, scale=0.6, color=col, thick=1)

    # History table
    cv2.line(frame, (40, 325), (W-40, 325), (40,42,60), 1)
    _txt_c(frame, "-- Question History --", W//2, 342, scale=0.38, color=(80,80,100))
    for i, (ltr, res, t, pts) in enumerate(quiz.history):
        ry  = 360 + i * 18
        if ry > H - 45:
            break
        col = (60,200,80) if res == "correct" else ((200,180,60) if res == "skip" else (80,80,200))
        ico = "✓" if res == "correct" else ("→" if res == "skip" else "✗")
        _txt_c(frame, f"{ico} {ltr}  {pts:+}pt  {t:.1f}s", W//2, ry, scale=0.38, color=col)

    _txt_c(frame, "R: choi lai  |  G: thoat quiz  |  Q: thoat",
           W//2, H-15, scale=0.38, color=(100,100,120))


# ══════════════════════════════════════════════
# QUIZ STATE MANAGER  (gắn vào main loop)
# ══════════════════════════════════════════════
class QuizManager:
    """Quản lý toàn bộ luồng quiz: menu → playing → result"""
    def __init__(self):
        self.active     = False     # True khi đang ở quiz mode
        self.sub_state  = "menu"    # "menu" | "playing" | "result"
        self.quiz       = None

    def enter(self):
        self.active    = True
        self.sub_state = "menu"
        self.quiz      = None

    def exit(self):
        self.active    = False
        self.sub_state = "menu"
        self.quiz      = None

    def start_game(self, difficulty):
        self.quiz      = QuizEngine(difficulty)
        self.sub_state = "playing"

    def handle_key(self, key):
        """Trả về True nếu key đã được xử lý bởi quiz"""
        if not self.active:
            return False

        if self.sub_state == "menu":
            if key == ord('1'):
                self.start_game("EASY");   return True
            if key == ord('2'):
                self.start_game("NORMAL"); return True
            if key == ord('3'):
                self.start_game("HARD");   return True
            if key in (ord('g'), ord('G')):
                self.exit();               return True

        elif self.sub_state == "playing":
            if key == ord(' '):
                self.quiz.skip();          return True
            if key in (ord('g'), ord('G')):
                self.exit();               return True

        elif self.sub_state == "result":
            if key in (ord('r'), ord('R')):
                self.sub_state = "menu";   return True
            if key in (ord('g'), ord('G')):
                self.exit();               return True

        return False

    def update_and_draw(self, frame, detected):
        """Gọi mỗi frame khi quiz đang active. Vẽ lên frame trực tiếp."""
        if not self.active:
            return

        if self.sub_state == "menu":
            draw_quiz_menu(frame)

        elif self.sub_state == "playing":
            q = self.quiz
            # Auto-advance
            if q.answered:
                if time.time() - q.answer_time >= QUIZ_NEXT_DELAY:
                    q.next_question()
                    if q.finished:
                        self.sub_state = "result"
            else:
                q.update(detected)
            draw_quiz_playing(frame, q, detected)

        elif self.sub_state == "result":
            draw_quiz_result(frame, self.quiz)


quiz_manager = QuizManager()


# =====================
# 4. VÒNG LẶP CHÍNH
# =====================
fps_buffer  = deque(maxlen=30)
prev_time   = time.perf_counter()
frame_count = 0
last_result = None

while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        print("❌ Không đọc được frame")
        break

    frame_count += 1
    run_inference = (frame_count % (FRAME_SKIP + 1) == 0)

    # ---- INFERENCE ----
    if run_inference:
        results = model.predict(
            source      = frame,
            conf        = CONF_THRESH,
            iou         = IOU_THRESH,
            imgsz       = IMGSZ,
            device      = device,
            half        = use_half,
            verbose     = False,
            stream      = False,
            augment     = False,
            agnostic_nms= True,
        )
        last_result = results[0]

        detected_labels = []
        if last_result.boxes is not None:
            for box in last_result.boxes:
                cls_id = int(box.cls[0])
                label  = model.names[cls_id]
                detected_labels.append(label)
        smoother.update(detected_labels)

    # ---- VẼ KẾT QUẢ (YOLO annotations) ----
    if last_result is not None:
        annotated_frame = last_result.plot(
            labels=True, conf=True, line_width=2, font_size=0.5,
        )
    else:
        annotated_frame = frame.copy()

    stable   = smoother.get_stable_labels()
    detected = smoother.top()

    # ════════════════════════════════════════
    # QUIZ MODE — ghi đè toàn bộ UI
    # ════════════════════════════════════════
    if quiz_manager.active:
        quiz_manager.update_and_draw(annotated_frame, detected)

        # FPS nhỏ góc phải dưới
        now = time.perf_counter()
        fps_buffer.append(1.0 / (now - prev_time + 1e-9))
        prev_time = now
        avg_fps   = sum(fps_buffer) / len(fps_buffer)
        cv2.putText(annotated_frame, f"FPS:{avg_fps:.0f}",
                    (CAM_W - 75, CAM_H - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60,60,80), 1, cv2.LINE_AA)

        cv2.imshow("YOLOv11 - Sign Language", annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        quiz_manager.handle_key(key)
        continue   # ← bỏ qua phần UI Word Builder bên dưới

    # ════════════════════════════════════════
    # NORMAL MODE — Word Builder (giữ nguyên)
    # ════════════════════════════════════════
    word_builder.update(stable)
    display = word_builder.get_display_text()

    # ---- HIỂN THỊ MODE (góc phải trên) ----
    mode_text  = "BUILD" if display['building_enabled'] else "VIEW"
    mode_color = (0, 255, 0) if display['building_enabled'] else (255, 255, 0)
    (mode_w, mode_h), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    mode_x = annotated_frame.shape[1] - mode_w - 30
    cv2.rectangle(annotated_frame, (mode_x-2, 18), (mode_x+mode_w+2, 38), (0,0,0), -1)
    cv2.putText(annotated_frame, mode_text, (mode_x, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, mode_color, 1, cv2.LINE_AA)
    cv2.putText(annotated_frame, "[Q]", (annotated_frame.shape[1]-25, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)

    if display['mode'] == 'history':
        cv2.rectangle(annotated_frame, (5,45), (annotated_frame.shape[1]-5, 350), (0,0,0), -1)
        cv2.putText(annotated_frame, "HISTORY (H to exit)",
                    (15,65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
        y_pos = 90
        for line in display['lines']:
            cv2.putText(annotated_frame, line, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,255), 1, cv2.LINE_AA)
            y_pos += 20
        if display['current_word'] or display['holding']:
            cv2.putText(annotated_frame,
                        f"Current: {display['current_word']}{display['holding']}",
                        (20, annotated_frame.shape[0]-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)

    elif display['mode'] == 'view_only':
        if display['current_label']:
            label_text = display['current_label']
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
            tx = (annotated_frame.shape[1] - tw) // 2
            ty = (annotated_frame.shape[0] + th) // 2 - 30
            cv2.rectangle(annotated_frame, (tx-10, ty-th-10), (tx+tw+10, ty+10), (0,0,0), -1)
            cv2.putText(annotated_frame, label_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,255), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Saved: {display['saved_count']}", (10,95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,255,100), 1, cv2.LINE_AA)

    else:
        if display['current_word'] or display['holding']:
            word_text = f"Word: {display['current_word']}{display['holding']}"
            (tw, th), _ = cv2.getTextSize(word_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tx = annotated_frame.shape[1] - tw - 15
            ty = 70
            cv2.rectangle(annotated_frame, (tx-3, ty-th-3), (tx+tw+3, ty+3), (0,0,0), -1)
            cv2.putText(annotated_frame, word_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        if display['show_session'] and display['session_history']:
            history_text = f"Session: {display['session_history']}"
            (th2, _), _  = cv2.getTextSize(history_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tx = annotated_frame.shape[1] - len(history_text)*10 - 15
            ty = annotated_frame.shape[0] - 50
            cv2.rectangle(annotated_frame, (tx-3, ty-th2-3),
                          (tx+len(history_text)*10+3, ty+3), (0,0,0), -1)
            cv2.putText(annotated_frame, history_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,255), 1, cv2.LINE_AA)
        if display['show_session'] and display['building_enabled']:
            close_btn_text = "[X] Close"
            (btn_w, btn_h), _ = cv2.getTextSize(close_btn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            btn_x = annotated_frame.shape[1] - btn_w - 15
            btn_y = annotated_frame.shape[0] - 25
            cv2.rectangle(annotated_frame, (btn_x-3, btn_y-btn_h-3),
                          (btn_x+btn_w+3, btn_y+3), (0,0,255), -1)
            cv2.putText(annotated_frame, close_btn_text, (btn_x, btn_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Saved: {display['saved_count']}", (10,95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,255,100), 1, cv2.LINE_AA)

    # ---- FPS ----
    now = time.perf_counter()
    fps_buffer.append(1.0 / (now - prev_time + 1e-9))
    prev_time = now
    avg_fps   = sum(fps_buffer) / len(fps_buffer)
    cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10,35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    # ---- DEBUG ----
    cv2.putText(annotated_frame, f"{device.upper()}|S{FRAME_SKIP}", (10,55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1, cv2.LINE_AA)

    # ---- HELP BAR ----
    cv2.putText(annotated_frame,
                "SPACE:cmp S:sv H:his M:mod DEL:del C:clr G:QUIZ Q:quit",
                (10, annotated_frame.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150,150,150), 1, cv2.LINE_AA)

    cv2.imshow("YOLOv11 - Sign Language Word Builder", annotated_frame)

    # ---- KEY HANDLING ----
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key in (ord("g"), ord("G")):        # ← PHÍM MỚI: vào Quiz mode
        quiz_manager.enter()
        print("🎮 Đã vào QUIZ mode — nhấn G để thoát")
    elif key == ord(" "):
        if word_builder.building_enabled:
            word_builder.force_complete_word()
    elif key in (ord("s"), ord("S")):
        if word_builder.save_current_word():
            print("💾 Đã lưu từ vào lịch sử")
    elif key in (ord("h"), ord("H")):
        word_builder.toggle_history_mode()
    elif key in (ord("m"), ord("M")):
        new_mode  = word_builder.toggle_building_mode()
        mode_name = "BUILD" if new_mode else "VIEW"
        print(f"🔄 Đã chuyển sang chế độ: {mode_name}")
    elif key in (ord("x"), ord("X")):
        if word_builder.building_enabled:
            word_builder.clear_session()
            print("🗑️ Đã đóng session")
    elif key in (ord("\x7f"), 8):
        if word_builder.building_enabled:
            word_builder.delete_last_letter()
    elif key in (ord("c"), ord("C")):
        if word_builder.building_enabled:
            word_builder.clear_current_word()
    elif key in (ord("d"), ord("D")):
        word_builder.clear_history()
        print("🗑️ Đã xóa toàn bộ lịch sử")


# =====================
# 5. DỌN DẸP
# =====================
word_builder.save_history()
cam.release()
cv2.destroyAllWindows()
print("👋 Chương trình kết thúc.")
print(f"📚 Đã lưu {len(word_builder.saved_words)} từ vào lịch sử")
