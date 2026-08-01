"""
api.py - FastAPI backend for Speech Emotion Recognition.
Serves predictions to the React frontend.
"""

import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from predict import EmotionPredictor

app = FastAPI(
    title="Listenly Speech Emotion API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Optional[EmotionPredictor] = None


def get_predictor() -> EmotionPredictor:
    global predictor
    if predictor is None:
        model_path = os.path.join(config.SAVED_MODELS_DIR, "best_model_cnn_lstm.h5")
        if not os.path.exists(model_path):
            model_path = os.path.join(config.SAVED_MODELS_DIR, "final_model_cnn_lstm.h5")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=503, detail="Model not found. Train a model first.")
        predictor = EmotionPredictor(model_path, model_type="cnn_lstm")
    return predictor


@app.on_event("startup")
def warmup():
    try:
        get_predictor()
    except Exception:
        pass


@app.get("/api/health")
def health():
    model_ready = False
    try:
        get_predictor()
        model_ready = True
    except Exception:
        model_ready = False
    return {"status": "ok", "model_ready": model_ready}


@app.get("/api/emotions")
def emotions():
    return {
        "emotions": config.EMOTION_LABELS,
        "colors": config.EMOTION_COLORS,
        "emoji": config.EMOTION_EMOJI,
    }


@app.get("/api/metrics")
def metrics():
    import json

    path = os.path.join(config.SAVED_MODELS_DIR, "metrics_cnn_lstm.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Metrics not found")
    with open(path, "r") as f:
        return json.load(f)


def _parse_classification_report(text: str):
    """Parse sklearn-style classification report into structured rows."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        name = parts[0]
        if name in {"precision", "accuracy", "macro", "weighted", "Model:", "Test"}:
            continue
        if name in config.EMOTION_LABELS:
            try:
                rows.append(
                    {
                        "emotion": name,
                        "precision": float(parts[1]),
                        "recall": float(parts[2]),
                        "f1": float(parts[3]),
                        "support": int(parts[4]),
                        "color": config.EMOTION_COLORS.get(name, "#888"),
                    }
                )
            except ValueError:
                continue
    return rows


@app.get("/api/insights")
def insights():
    import json

    metrics_path = os.path.join(config.SAVED_MODELS_DIR, "metrics_cnn_lstm.json")
    report_path = os.path.join(config.SAVED_MODELS_DIR, "classification_report_cnn_lstm.txt")

    metrics_data = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

    report_text = ""
    classes = []
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report_text = f.read()
        classes = _parse_classification_report(report_text)

    charts = []
    chart_defs = [
        ("training_history_cnn_lstm.png", "Training history", "Accuracy and loss over epochs"),
        ("confusion_matrix_cnn_lstm.png", "Confusion matrix", "True vs predicted emotion classes"),
        ("emotion_distribution.png", "Emotion distribution", "Class balance across the dataset"),
    ]
    for filename, title, caption in chart_defs:
        path = os.path.join(config.SAVED_MODELS_DIR, filename)
        if os.path.exists(path):
            charts.append(
                {
                    "id": filename.replace(".png", ""),
                    "title": title,
                    "caption": caption,
                    "url": f"/api/artifacts/{filename}",
                }
            )

    return {
        "metrics": metrics_data,
        "classes": classes,
        "report": report_text,
        "charts": charts,
        "emotions": config.EMOTION_LABELS,
        "colors": config.EMOTION_COLORS,
    }


@app.get("/api/artifacts/{filename}")
def artifact(filename: str):
    safe = os.path.basename(filename)
    if not safe.lower().endswith((".png", ".jpg", ".jpeg", ".txt", ".json")):
        raise HTTPException(status_code=400, detail="Unsupported artifact type")
    path = os.path.join(config.SAVED_MODELS_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    allowed = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Use: {', '.join(sorted(allowed))}",
        )

    contents = await file.read()
    if len(contents) > config.MAX_UPLOAD_SIZE * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    suffix = ext if ext else ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = get_predictor().predict(tmp_path)
        result["filename"] = file.filename
        result["emoji"] = config.EMOTION_EMOJI.get(result["emotion"], "")
        result["color"] = config.EMOTION_COLORS.get(result["emotion"], "#666")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Serve built React app if present
FRONTEND_DIST = os.path.join(config.BASE_DIR, "frontend", "dist")
if os.path.isdir(FRONTEND_DIST):
    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        file_path = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
