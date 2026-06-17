"""
collect_words.py — collect word-level ISL signs via webcam
Run: python collect_words.py
Hold each sign steady when prompted. Press SPACE to start collecting, Q to quit.
"""
import os
import cv2
import mediapipe as mp
import pickle
import numpy as np
import time

# ── Signs to collect — add or remove words here ──────────────────────────────
WORD_SIGNS = [
    "namaste",
    "hello",
    "help",
    "water",
    "eat",
    "call",
    "please",
    "thankyou",
    "yes",
    "no",
    "stop",
    "good",
    "bad",
    "home",
    "mother",
    "father",
]

SAMPLES_PER_SIGN = 150  # how many frames to capture per sign
COUNTDOWN        = 3    # seconds to get ready before capture starts

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
hands       = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_features(hand_landmarks):
    """Extract normalised 42-feature vector from one hand."""
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]
    data = []
    for lm in hand_landmarks.landmark:
        data.append(lm.x - min(x_))
        data.append(lm.y - min(y_))
    return data  # 42 values

def draw_text(frame, text, pos, color=(0, 255, 0), size=1.0, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

# ── Load existing data so we can append ───────────────────────────────────────
DATA_FILE = "data_words.pickle"
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        saved = pickle.load(f)
    all_data   = saved["data"]
    all_labels = saved["labels"]
    print(f"Loaded existing data: {len(all_data)} samples")
    existing = set(all_labels)
    print(f"Existing signs: {sorted(existing)}")
else:
    all_data   = []
    all_labels = []
    existing   = set()

# ── Main collection loop ───────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("\n=== ISL Word Sign Collector ===")
print("Press SPACE to start collecting each sign")
print("Press S to skip a sign")
print("Press Q to quit and save\n")

for sign in WORD_SIGNS:
    if sign in existing:
        print(f"⏭  Skipping '{sign}' — already collected")
        continue

    sign_data = []
    print(f"\n📌 Next sign: '{sign.upper()}'")
    print(f"   Get ready to sign '{sign}' and press SPACE...")

    # ── Wait for SPACE ────────────────────────────────────────────────────────
    waiting = True
    while waiting:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # show hand landmarks while waiting
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(display, hl, mp_hands.HAND_CONNECTIONS)

        draw_text(display, f"Sign: {sign.upper()}", (20, 50), color=(0, 255, 100), size=1.4)
        draw_text(display, "SPACE = start  |  S = skip  |  Q = quit", (20, 90), color=(200, 200, 200), size=0.6)
        draw_text(display, f"Collected so far: {len(all_data)} total samples", (20, 130), color=(150, 150, 150), size=0.6)

        cv2.imshow("ISL Collector", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            waiting = False
        elif key == ord('s') or key == ord('S'):
            print(f"   Skipped '{sign}'")
            waiting = False
            sign = None
            break
        elif key == ord('q') or key == ord('Q'):
            print("\nQuitting early...")
            cap.release()
            cv2.destroyAllWindows()
            # save whatever we have
            with open(DATA_FILE, "wb") as f:
                pickle.dump({"data": all_data, "labels": all_labels}, f)
            print(f"Saved {len(all_data)} samples to {DATA_FILE}")
            exit()

    if sign is None:
        continue

    # ── Countdown ─────────────────────────────────────────────────────────────
    for i in range(COUNTDOWN, 0, -1):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        draw_text(frame, f"Starting in {i}...", (20, 50), color=(0, 200, 255), size=1.6)
        draw_text(frame, f"HOLD the '{sign}' sign!", (20, 110), color=(255, 255, 255), size=1.0)
        cv2.imshow("ISL Collector", frame)
        cv2.waitKey(1000)

    # ── Capture frames ────────────────────────────────────────────────────────
    collected   = 0
    no_hand_cnt = 0

    while collected < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            no_hand_cnt = 0
            hl = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(display, hl, mp_hands.HAND_CONNECTIONS)
            features = extract_features(hl)
            sign_data.append(features)
            collected += 1
        else:
            no_hand_cnt += 1
            if no_hand_cnt > 30:
                draw_text(display, "NO HAND DETECTED", (20, 200), color=(0, 0, 255), size=1.2)

        # progress bar
        progress = int((collected / SAMPLES_PER_SIGN) * 400)
        cv2.rectangle(display, (20, 460), (420, 480), (50, 50, 50), -1)
        cv2.rectangle(display, (20, 460), (20 + progress, 480), (0, 255, 100), -1)

        draw_text(display, f"Sign: {sign.upper()}", (20, 50), color=(0, 255, 100), size=1.4)
        draw_text(display, f"Capturing: {collected}/{SAMPLES_PER_SIGN}", (20, 90), color=(255, 255, 255), size=0.8)
        draw_text(display, "Keep holding the sign!", (20, 440), color=(200, 200, 200), size=0.6)

        cv2.imshow("ISL Collector", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if len(sign_data) >= 50:
        all_data.extend(sign_data)
        all_labels.extend([sign] * len(sign_data))
        print(f"✅ Collected {len(sign_data)} samples for '{sign}'")
    else:
        print(f"⚠️  Only got {len(sign_data)} samples for '{sign}' — skipped (need 50+)")

    # auto-save after each sign
    with open(DATA_FILE, "wb") as f:
        pickle.dump({"data": all_data, "labels": all_labels}, f)
    print(f"   💾 Saved — total: {len(all_data)} samples")

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Done! Total samples: {len(all_data)}")
print(f"Signs collected: {sorted(set(all_labels))}")
print(f"Saved to: {DATA_FILE}")