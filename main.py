from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os
import sys

# 1. ساخت اپلیکیشن
app = FastAPI()

# 2. حل مشکل دسترسی (CORS)
# این بخش اجازه می‌دهد مرورگر یا برنامه‌های دیگر به API وصل شوند
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # اجازه به همه منابع
    allow_credentials=True,
    allow_methods=["*"],  # اجازه به همه متدها (POST, GET, OPTIONS, ...)
    allow_headers=["*"],
)

# 3. پیدا کردن مسیر دقیق فایل (حل مشکل پیدا نشدن فایل)
# این کد مسیر پوشه‌ای که main.py در آن است را پیدا می‌کند
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'co2_model.pkl')

print(f"Searching for model at: {model_path}")

# 4. بارگذاری مدل
model = None
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ مدل با موفقیت لود شد.")
    except Exception as e:
        print(f"❌ خطا در خواندن فایل مدل: {e}")
else:
    print("⚠️ فایل مدل پیدا نشد! لطفاً ابتدا train.py را اجرا کنید.")

# تعریف ورودی
class InputData(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"status": "Server is running", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found on server")
    
    if len(data.features) != 5:
        raise HTTPException(status_code=400, detail="Please send exactly 5 numbers")

    try:
        input_array = np.array([data.features])
        prediction = model.predict(input_array)
        return {"prediction": float(prediction[0][0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))