# ISL-to-Speech — Phase 1: Web Dashboard

Replaces the OpenCV popup window with a full browser-based dashboard.

## Project layout

```
isl-to-speech/
├── backend/
│   ├── app.py              ← FastAPI + WebSocket server  (NEW)
│   └── requirements.txt
├── frontend/
│   └── index.html          ← Single-file dashboard       (NEW)
│
│   # your original files — unchanged:
├── collect_imgs.py
├── create_database.py
├── train_classifier.py
├── data.pickle
└── model.p
```

## Setup

### 1. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Copy your model

Make sure `model.p` is in the project root (one level above `backend/`).
The server looks for it at `../model.p`.

### 3. Start the backend

```bash
cd backend
uvicorn app:app --reload --port 8000
```

You should see:
```
✅ Model loaded
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Open the dashboard

Open `frontend/index.html` directly in Chrome/Firefox.

Or serve it via the FastAPI static mount:
```
http://localhost:8000/app
```

### 5. Connect

1. Click **Start camera** — allow camera access
2. Click the **settings** tab → **Connect WebSocket**
   (URL defaults to `ws://localhost:8000/ws`)
3. Start signing — letters appear in the word builder

---

## How the WebSocket protocol works

**Frontend → Backend** (every ~40ms):
```json
{ "type": "frame", "data": "<base64 JPEG>" }
```

**Backend → Frontend** (per frame):
```json
{
  "letter": "A",
  "confidence": 0.91,
  "hands": { "left": false, "right": true },
  "landmarks": [ [{"x":0.4,"y":0.6}, ...] ],
  "latency": 18,
  "model_loaded": true
}
```

**Frontend → Backend** (speak a word):
```json
{ "type": "speak", "word": "NAMASTE", "rate": 1.0 }
```

---

## Dashboard features

| Feature | Description |
|---|---|
| Live camera feed | Webcam with flipped mirror view |
| Hand landmarks | Green skeleton drawn on the feed |
| Big letter display | Large letter flashes when sign is stable |
| Confidence bar | Real-time certainty for current sign |
| Word builder | Letters accumulate; auto-spoken after 2.5s pause |
| Session history | Every spoken word logged with timestamp + replay |
| Stats panel | Letter, confidence, latency, stable-frame counter |
| Settings | Auto-speak toggle, landmark toggle, stability threshold, speech rate |
| Demo mode | Works without a backend to test the UI |

---

## Running without a model (demo mode)

Click **Settings → Run demo mode** — the frontend simulates detection
with a built-in word list so you can test the full UI without any backend.

---

## Next: Phase 2 — Claude AI integration

Once Phase 1 is running, Phase 2 adds:
- `backend/ai_handler.py` — Claude API sentence correction
- Conversation panel in the right sidebar
- Next-word prediction chips (already stubbed in the UI)