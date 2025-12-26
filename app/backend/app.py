from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
import logging
from inference import predict

# 1. إعداد اللوجز (عشان نشوف الأخطاء بوضوح)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 2. إعداد CORS (للسماح للفرونت إند بالتواصل مع الباك إند)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. إعداد مسار الفيديوهات (التعديل المهم للدوكر) 🛠️
# هنا بنستخدم المسار المباشر اللي حددناه في docker-compose volumes
ANIMATION_DIR = "/app/animation"

# التأكد من وجود الفولدر (لتجنب الأخطاء عند التشغيل)
if not os.path.exists(ANIMATION_DIR):
    # بننشئه احتياطياً، لكن المفروض الدوكر هو اللي يربطه بالفولدر اللي عندك
    os.makedirs(ANIMATION_DIR)
    logger.warning(f"Created animation directory at {ANIMATION_DIR}. Make sure Docker volume is mounted!")

# ربط رابط /animations بالفولدر الفعلي لعرض الفيديوهات
app.mount("/animations", StaticFiles(directory=ANIMATION_DIR), name="animations")

@app.post("/predict")
async def predict_sign(file: UploadFile = File(...)):
    path = ""
    try:
        # إنشاء اسم فريد للملف المؤقت
        filename = f"{uuid.uuid4()}.webm"
        path = f"/tmp/{filename}"
        
        logger.info(f"Receiving file request. Saving to {path}")
        
        # حفظ الفيديو القادم من الفرونت
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # التأكد من أن الملف اتحفظ وله حجم
        if not os.path.exists(path):
             logger.error("File path does not exist after write attempt!")
             raise HTTPException(status_code=500, detail="File save failed")

        file_size = os.path.getsize(path)
        logger.info(f"File saved successfully. Size: {file_size} bytes")

        if file_size == 0:
            logger.error("Received empty file")
            raise HTTPException(status_code=400, detail="Empty video file received")

        # تشغيل الموديل
        logger.info("Starting prediction...")
        result = predict(path)
        logger.info(f"Prediction success. Result: {result}")
        
        # تنظيف الملف المؤقت
        os.remove(path)
        
        return {"result": result}

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        
        # تنظيف الملف لو حصل خطأ وهو لسه موجود
        if path and os.path.exists(path):
            os.remove(path)
            
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")