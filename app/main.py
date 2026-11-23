
from fastapi import FastAPI

# Membuat aplikasi FastAPI
app = FastAPI(
    title="SkinSense API",
    description="Backend API untuk aplikasi SkinSense",
    version="1.0.0"
)

# Endpoint halaman utama
@app.get("/")
def home():
    return {
        "message": "Welcome to SkinSense API",
        "status": "success"
    }
