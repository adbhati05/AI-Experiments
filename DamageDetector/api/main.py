import io
import torch
import threading
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from pi_heif import register_heif_opener
from pathlib import Path
from ultralytics import YOLO
from pydantic import BaseModel

# Note: the model weights were copied from the fifth training session's (negatives) run folder to the weights folder in the directory that this file is located in (api).
WEIGHTS = Path(__file__).resolve().parent / "weights" / "best.pt" # Resolving the path from this file's location to ensure that the model weights can be loaded properly.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" # Ensuring that the model is loaded on this computer's GPU (Apple Silicon) since CPU inference is significantly slower.
CONF_FLOOR = 0.10 # Setting a confidence floor of 0.10 
MAX_BYTES = 10 * 1024 * 1024 # Capping the maximum size of the uploaded image to 10 MB to ensure that the API can handle image processing without running into memory issues or timeouts.

# Registering the HEIF opener to ensure that the API can read HEIC/HEIF images, which are commonly used on Apple devices. Even though Ultralytics' replacement for Image.open can read such images, I've explicitly registered the opener here to clarify what formats are supported by the API.
register_heif_opener()

# Initializing the YOLO model with the specified weights.
model = YOLO(WEIGHTS)

# Initializing a thread lock so that the prediction step in /predict below can be wrapped in the lock to ensure that the model is not accessed by multiple threads at the same time (when multiple users are making requests to the predict endpoint simultaneously).
# Such an event could lead to a race condition and cause the model to crash or return incorrect results.
model_lock = threading.Lock()

app = FastAPI(
    title="Damage Detector API",
    version="1.0.0",
    description="This API allows you to upload an image and receive an annotated image with bounding boxes around detected damages.",
)

# Seting up CORS middleware to allow requests from the frontend which is a React Vite app.
# Vite's dev server runs on port 5173, which is different from Fast API's default port 8000, so we need to allow cross-origin requests from the frontend to the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], # Ensuring that the frontend can make requests to the backend by allowing the two origins that Vite's dev server uses.
    allow_credentials=False, # Set to False here since my app won't have any auth set up that requires credentials to be sent with requests.
    allow_methods=["GET", "POST"], # Allowing only GET and POST methods since the API will only have two endpoints, one for health check and one for prediction.
    allow_headers=["*"],
)

# Defining the Pydantic models for the response schema of the /predict endpoint. This ensures that the API returns a consistent and well-defined structure for the predictions, which can be easily consumed by the frontend or other clients.
# The box model represents the bounding box coordinates, the detection model represents a single detection with its class name, confidence score, and bounding box coords (Box object), and the PredictResponse model represents the overall response structure containing the image dimensions and a list of detections.
class Box(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class Detection(BaseModel):
    class_name: str
    confidence: float
    box: Box

class PredictResponse(BaseModel):
    width: int
    height: int
    detections: list[Detection]

# Setting up a health check endpoint to verify that the API is running and responsive.
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Setting up the prediction endpoint to accept STRICTLY image files and then doing the necessary processing to return the image with bounding boxes around detected damage.
@app.post("/predict")
def predict(file: UploadFile) -> PredictResponse:
    contents = file.file.read()

    # Exiting with an exception if the uploaded image is larger than 10 MB.
    if len(contents) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB. Please upload a smaller image.")

    # Retrieving the image's bytes and converting it to a PIL Image object. If the image is not readable, an exception is thrown to ensure the user uploads a valid image file.
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError): # OSError is included here to catch non-file-format related cases like a truncated JEPG file, which is the correct format but is not readable due to being corrupted or incomplete.
        raise HTTPException(status_code=400, detail="That file is not a readable image. Please upload a file with a valid image format (e.g., JPEG, PNG, etc).")

    # Using exif transpose to ensure that the image is oriented correctly based on its EXIF metadata and then converting it to RGB format so the model can process it.
    image = ImageOps.exif_transpose(image).convert("RGB")

    # Running the loaded model's prediction method on the uploaded image with the specified confidence floor and device.
    # Here, augment is set to True to enable test-time augmentation, which can improve the model's performance by making predictions on multiple augmented versions of the image and averaging the results (in place for now but might be removed later if the API can't handle the extra processing).
    with model_lock:
        results = model.predict(image, imgsz=640, conf=CONF_FLOOR, augment=True, device=DEVICE, verbose=False)[0]

    # Lastly, extracting the bounding box coordinates, class names, and confidence scores from the results object above and returning them as a list of dictionaries (each dictionary belonging to a single detection) along with the original image's width and height.
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_name": results.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "box": {"x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2)},
        })

    return {"width": image.width, "height": image.height, "detections": detections}