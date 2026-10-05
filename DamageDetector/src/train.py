from pathlib import Path
from ultralytics import YOLO

# Initializing the path to the dataset config, resolved from this file's location rather than the current working directory, so the script can run properly from anywhere.
DATA_CONFIG = Path(__file__).resolve().parent.parent / "data" / "data.yaml"

BATCH_SIZE = 16 # Capped at 16 since this computer has 16GB of RAM and a larger batch size would cause memory overflow errors during training.
EPOCHS = 100
PATIENCE = 25 # Setting patience to 25 since the model is expected to converge quickly. This will prevent overfitting and save time by stopping training early in case no improvement is seen in the validation loss for 25 epochs.
RESOLUTION = 640
WORKERS = 4 # Including 4 workers for data loading since this will speed up training by loading data in parallel.
DEVICE = "mps" # Ensuring that the model is trained on this computer's GPU (Apple Silicon) since CPU training is significantly slower.
RUN_NAME = "negatives" # Naming this run "negatives" since this is the fifth training session and the dataset has been augmented with negative samples to improve the model's ability to distinguish between damaged and undamaged images.
SEED = 0 # Setting the seed to 0 to ensure reproducibility of results across different runs of the model.

# Choosing to use the YOLOv8 small model for this fifth training session since it performed the best out of the 4 configurations tested in previous sessions.
model = YOLO("yolov8s.pt")

# Training the model using the specified hyperparameter values above and the yaml config file.
results = model.train(
    data=str(DATA_CONFIG),
    epochs=EPOCHS,
    patience=PATIENCE,
    workers=WORKERS,
    batch=BATCH_SIZE,
    imgsz=RESOLUTION,
    device=DEVICE,
    name=RUN_NAME,
    seed=SEED,
    plots=True, # This saves the per-class precision-recall curves, confusion matrix and training curves to the run folder, which are needed for the error analysis and the model page of the web app.
)