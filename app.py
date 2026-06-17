import asyncio
import base64
import json
import pickle
import time
from contextlib import asynccontextmanager
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Load model ───────────────────────────────────────────────────────────────
MODEL_PATH = "./model_combined.p"
model = None
model_loaded = False

label_encoder = None

def load_model():
    global model, model_loaded, label_encoder
    try:
        model_dict  = pickle.load(open(MODEL_PATH, "rb"))
        model       = model_dict["model"]
        label_encoder = model_dict.get("encoder", None)
        model_loaded = True
        print("Model loaded")
    except FileNotFoundError:
        print(f"Model not found at {MODEL_PATH}")


# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands        = mp.solutions.hands
hands_detector  = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)


LABELS = {
    0:"A",1:"B",2:"C",3:"D",4:"E",5:"F",6:"G",7:"H",
    8:"I",9:"J",10:"K",11:"L",12:"M",13:"N",14:"O",
    15:"P",16:"Q",17:"R",18:"S",19:"T",20:"U",21:"V",
    22:"W",23:"X",24:"Y",25:"Z"
}

# ── TTS (runs in a thread to avoid blocking) ─────────────────────────────────
def speak_word(word: str, rate: float = 1.0):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", int(150 * rate))
        engine.say(word)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS error: {e}")

# ── Inference ────────────────────────────────────────────────────────────────
def run_inference(frame_bytes: bytes) -> dict[str, Any]:
    """Decode a JPEG frame, run MediaPipe + classifier, return prediction dict."""
    t0 = time.perf_counter()

    # decode
    arr   = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "bad frame"}

    H, W, _ = frame.shape
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results  = hands_detector.process(rgb)

    out: dict[str, Any] = {
        "letter":      None,
        "confidence":  0.0,
        "hands":       {"left": False, "right": False},
        "landmarks":   [],
        "latency":     0,
        "model_loaded": model_loaded,
    }

    if not results.multi_hand_landmarks:
        out["latency"] = int((time.perf_counter() - t0) * 1000)
        return out

    # hand side detection
    if results.multi_handedness:
        for h in results.multi_handedness:
            side = h.classification[0].label.lower()
            out["hands"][side] = True

    # collect landmarks for frontend drawing
    for hand_lm in results.multi_hand_landmarks:
        pts = [{"x": lm.x, "y": lm.y} for lm in hand_lm.landmark]
        out["landmarks"].append(pts)

    # build feature vector — first hand only, 42 features
    first_hand = results.multi_hand_landmarks[0]
    data_aux = []
    x_ = [lm.x for lm in first_hand.landmark]
    y_ = [lm.y for lm in first_hand.landmark]

    for lm in first_hand.landmark:
        data_aux.append(lm.x - min(x_))
        data_aux.append(lm.y - min(y_))

    if model_loaded and model is not None:
        try:
            arr_input  = np.asarray(data_aux).reshape(1, -1)
            proba      = model.predict_proba(arr_input)[0]
            pred_idx   = int(np.argmax(proba))
            confidence = float(proba[pred_idx])
            label_name = label_encoder.inverse_transform([pred_idx])[0] if label_encoder else str(pred_idx)
            print(f"Pred: {label_name} conf: {confidence:.2f}")
            if confidence > 0.45:
                if label_encoder:
                    out["letter"] = label_encoder.inverse_transform([pred_idx])[0].upper()
                else:
                    out["letter"] = LABELS.get(pred_idx, str(pred_idx))
                out["confidence"] = confidence
        except Exception as e:
            print(f"Inference error: {e}")

    out["latency"] = int((time.perf_counter() - t0) * 1000)
    return out


# ── App ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="ISL-to-Speech API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the React/HTML frontend
try:
    app.mount("/app", StaticFiles(directory="../frontend", html=True), name="frontend")
except RuntimeError:
    pass  # frontend directory doesn't exist yet


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model_loaded}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("🟢 Client connected")
    loop = asyncio.get_event_loop()

    try:
        while True:
            # Receive frame as base64-encoded JPEG
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "frame":
                # strip data URI prefix if present
                b64 = msg["data"]
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                frame_bytes = base64.b64decode(b64)

                # run inference in thread so we don't block the event loop
                result = await loop.run_in_executor(None, run_inference, frame_bytes)
                await ws.send_text(json.dumps(result))

            elif msg.get("type") == "speak":
                word  = msg.get("word", "")
                rate  = float(msg.get("rate", 1.0))
                await loop.run_in_executor(None, speak_word, word, rate)

    except WebSocketDisconnect:
        print("🔴 Client disconnected")
    except Exception as e:
        print(f"WS error: {e}")