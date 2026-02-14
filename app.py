from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from PIL import Image
import io

from inference import load_model
import download_model

# download model first
download_model

MODEL_PATH = "swin_best.pth"

model = load_model(MODEL_PATH)

CLASS_NAMES = [
     "infectious",
    "organic",
    "recyclable",
    "sharps"
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Swin API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        pred = torch.argmax(outputs, dim=1).item()

    return {
        "prediction": CLASS_NAMES[pred]
    }
