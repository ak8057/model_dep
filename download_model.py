import gdown
import os

MODEL_PATH = "swin_best.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    
    url = "https://drive.google.com/uc?id=1SSI8fGRgg5xzvAOGGue3v0yYYAV8sFqh"
    
    gdown.download(url, MODEL_PATH, quiet=False)

else:
    print("Model already exists.")
