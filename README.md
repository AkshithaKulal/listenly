# Listenly

Speech emotion recognition with a React UI and FastAPI + TensorFlow backend.

## Local run

```bash
# Backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python train.py --model cnn_lstm   # needs dataset/RAVDESS
python -m uvicorn api:app --host 127.0.0.1 --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173

## Deploy

### Frontend (Vercel)
1. Import this repo in Vercel
2. Set **Root Directory** to `frontend`
3. Add env var `VITE_API_URL` = your backend URL (e.g. `https://listenly-api.onrender.com`)
4. Deploy

### Backend (not Vercel)
TensorFlow models need a long-running server. Good options:
- [Render](https://render.com) Web Service
- [Railway](https://railway.app)
- [Fly.io](https://fly.io)

Upload `saved_models/best_model_cnn_lstm.h5` on the host (it is gitignored because it is large).

## Repo notes
- `dataset/` and `*.h5` model weights are gitignored
- Charts/metrics under `saved_models/` (png/json/txt) are included
