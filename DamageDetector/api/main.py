import os
import io
import hmac
import torch
import hashlib
import secrets
import threading
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from pi_heif import register_heif_opener
from pathlib import Path
from ultralytics import YOLO
from typing import Literal
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from google.cloud import firestore
from google.api_core import exceptions as gcloud_exceptions

# Declaring Literals for the class names and prediction outcomes to restrict the values that can be used in the feedback and stats endpoints. This ensures that only valid class names and outcomes are accepted, preventing the code from breaking and maintaining data integrity.
ClassName = Literal["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
Outcome = Literal["correct", "missed_or_wrong", "false_positive"]

# Note: the model weights were copied from the fifth training session's (negatives) run folder to the weights folder in the directory that this file is located in (api).
WEIGHTS = Path(__file__).resolve().parent / "weights" / "best.pt" # Resolving the path from this file's location to ensure that the model weights can be loaded properly.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" # Ensuring that the model is loaded on this computer's GPU (Apple Silicon) since CPU inference is significantly slower.
CONF_FLOOR = 0.10 # Setting a confidence floor of 0.10 which is the lowest confidence the API will return. 
MAX_BYTES = 10 * 1024 * 1024 # Capping the maximum size of the uploaded image to 10 MB to ensure that the API can handle image processing without running into memory issues or timeouts.

# ALLOWED_ORIGINS is an environment variable that will hold the URL that the frontend will be deployed on in Vercel. This allows the frontend to make requests to the backend without running into CORS issues. If the environment variable is not set, it defaults to allowing requests from localhost (which matches the Vite dev server's default port).
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")] # Stripping each origin so a space after a comma does not stop it from matching.

# PREDICTION_SECRET is an environment variable that will hold a private key that will be used to sign the prediction token that is returned in the /predict endpoint.
PREDICTION_SECRET = os.environ["PREDICTION_SECRET"].encode() # Using environ here to ensure there is no default value (no fall back).

# Initializing the Firestore client to interact with the Firestore database (specifically from the project I set up for this app).
db = firestore.Client(project="adbhati-damage-detector")
stats_ref = db.collection("stats").document("totals") # Choosing to initialize this up here so that it can be used in both the /feedback and /stats endpoints to read and update the stats document in Firestore.

# Registering the HEIF opener to ensure that the API can read HEIC/HEIF images, which are commonly used on Apple devices. Even though Ultralytics' replacement for Image.open can read such images, I've explicitly registered the opener here to clarify what formats are supported by the API.
register_heif_opener()

# Initializing the YOLO model with the specified weights.
model = YOLO(WEIGHTS)

# Initializing a thread lock so that the prediction step in /predict below can be wrapped in the lock to ensure that the model is not accessed by multiple threads at the same time (when multiple users are making requests to the predict endpoint simultaneously).
# Such an event could lead to a race condition and cause the model to crash or return incorrect results.
model_lock = threading.Lock()

app = FastAPI(
    title="Damage Detector API",
    version="1.1.0",
    description="Upload a photo of a car and receive the detected damage as bounding boxes with class names and confidence scores.",
)

# Seting up CORS middleware to allow requests from the frontend which is a React Vite app.
# Vite's dev server runs on port 5173, which is different from Fast API's default port 8000, so it's necessary to allow cross-origin requests from the frontend to the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, # Ensuring that the frontend can make requests to the backend by allowing only the origins listed in ALLOWED_ORIGINS above.
    allow_credentials=False, # Set to False here since my app won't have any auth set up that requires credentials to be sent with requests.
    allow_methods=["GET", "POST"], # Allowing only GET and POST methods since the API will only have two endpoints, one for health check and one for prediction.
    allow_headers=["*"],
)

# Defining the Pydantic models for the response schema of the /predict (POST) endpoint. This ensures that the API returns a consistent and well-defined structure for the predictions, which can be easily consumed by the frontend or other clients.
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
    prediction_token: str # A signed ID per prediction, this is to ensure a unique identifier can be generated for each prediction so that when users submit feedback on the prediction, it can be associated with the correct prediction in the database.
    width: int
    height: int
    detections: list[Detection]

# Defining the pydantic models for the response schema of the /feedback (POST) and /stats (GET) endpoints. 
class FeedbackRequest(BaseModel):
    prediction_token: str
    has_damage: bool                                                    # Question 1: "Did your image show a car with damage?" (Yes/No response)
    model_correct: bool                                                 # Question 2: "Did the model detect the damage accurately?" (Yes/No response | only if user answered "Yes" to Q1) or "Did the model detect any damage?" (Yes/No response | only if user answered "No" to Q1)
    missed_classes: list[ClassName] = Field(default=[], max_length=6)   # Question 3: "What damage was on the car?" (user selects classes | only rendered if on the user entered "Yes" then "No" to the previous question)
    threshold: float = Field(ge=0.10, le=1.0)                           # Where the threshold slider was set when the user made the prediction and submitted feedback.
    shown_classes: list[ClassName] = Field(max_length=6)                # Classes visible on screen at specified threshold (also this info is required as opposed to the missed classes).

# These models specifically follow the same structure as the stats and feedback documents in the Firestore database.
class FeedbackRecord(BaseModel):
    has_damage: bool
    model_correct: bool
    outcome: Outcome
    missed_classes: list[ClassName]
    threshold: float
    shown_classes: list[ClassName]
    created_at: datetime

class Stats(BaseModel):
    total: int = 0
    correct: int = 0
    missed_or_wrong: int = 0
    false_positive: int = 0


# This function generates a unique prediction token using the key stored in PREDICTION_SECRET to sign the token.
def make_token() -> str:
    random_id = secrets.token_hex(16)
    signature = hmac.new(PREDICTION_SECRET, random_id.encode(), hashlib.sha256).hexdigest()
    return f"{random_id}.{signature}"


# This function verifies a prediction token by checking its signature against the expected signature generated using PREDICTION_SECRET. If the token is valid, it returns the ID part of the token; none otherwise.
def verify_token(token: str) -> str | None:
    # If the split is unsuccessful, it means the token is malformed. Hence why None is returned to indicate that the token is invalid. 
    try:
        random_id, signature = token.split(".")
        expected_signature = hmac.new(PREDICTION_SECRET, random_id.encode(), hashlib.sha256).hexdigest()

        if hmac.compare_digest(expected_signature, signature):
            return random_id
        else:
            return None
    except (ValueError, TypeError):
        return None
    

# Setting up a health check endpoint to verify that the API is running and responsive.
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Setting up the prediction endpoint to accept STRICTLY image files and then doing the necessary processing to return the bounding boxes, class names and confidence scores of the detected damage.
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

    return {"width": image.width, "height": image.height, "detections": detections, "prediction_token": make_token()}


# Setting up the feedback endpoint to accept user feedback on the prediction made by the model. The feedback is then processed as a record and stored in the Firestore database.
@app.post("/feedback", status_code=201)
def feedback(request: FeedbackRequest) -> dict[str, str]:
    # Extracting the prediction ID by verifying the prediction token, returning an error if the token is invalid.
    prediction_id = verify_token(request.prediction_token)
    if not prediction_id:
        raise HTTPException(status_code=400, detail="Invalid prediction token.")

    # Determining the outcome of the feedback based on the user's responses to the questions in the questionnaire.
    # Only keeping missed classes if the user indicated damage was present and the model was incorrect.
    if request.model_correct:
        outcome = "correct"
        missed_classes = []
    elif request.has_damage:
        outcome = "missed_or_wrong"
        missed_classes = request.missed_classes
    else:
        outcome = "false_positive"
        missed_classes = []

    # Creating a feedback record to store in the Firestore database.
    feedback_record = FeedbackRecord(
        has_damage=request.has_damage,
        model_correct=request.model_correct,
        outcome=outcome,
        missed_classes=missed_classes,
        threshold=request.threshold,
        shown_classes=request.shown_classes,
        created_at=datetime.now(timezone.utc),
    )

    # Setting up a batch write to Firestore to store the feedback record and update the stats document atomically (since the totals depend on the outcome of the feedback).
    feedback_ref = db.collection("feedback").document(prediction_id)

    batch = db.batch()
    batch.create(feedback_ref, feedback_record.model_dump())
    batch.set(stats_ref, {"total": firestore.Increment(1), outcome: firestore.Increment(1)}, merge=True)

    # Wrapping the batch commit in a try-except block to handle potential errors when writing to Firestore. A success message is returned when none of the exceptions are raised.
    try:
        batch.commit()
        return {"status": "Feedback stored successfully."}
    except gcloud_exceptions.AlreadyExists:
        raise HTTPException(status_code=409, detail="Answer for this prediction token already exists. Please do not submit feedback multiple times for the same prediction.")
    except gcloud_exceptions.GoogleAPICallError as e:
        print(f"Firestore write failed: {e}")
        raise HTTPException(status_code=503, detail="Feedback is unavailable right now. Please try again later.")


# Setting up the stats endpoint to retrieve the current statistics of the feedback across all predictions. This will be leveraged by the frontend to display the overall performance of the model post-deployment.
@app.get("/stats")
def stats() -> Stats:
    # Wrapping the Firestore read operation in a try-except block to handle potential errors when reading from Firestore.
    try:
        snapshot = stats_ref.get()
    except gcloud_exceptions.GoogleAPICallError as e:
        print(f"Firestore read failed: {e}")
        raise HTTPException(status_code=503, detail="Stats are unavailable right now. Please try again later.")

    # Returning a default Stats object with all counts set to zero in case the stats document doesn't exist. Otherwise, returning the stats document's data as a Stats object via to_dict().
    if not snapshot.exists:
        return Stats()

    return Stats(**snapshot.to_dict())
