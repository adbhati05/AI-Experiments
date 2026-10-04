from pathlib import Path
from ultralytics import YOLO

# Initializing the path to the dataset config, resolved from this file's location rather than the current working directory, so the script can run properly from anywhere.
DATA_CONFIG = Path(__file__).resolve().parent.parent / "data" / "data.yaml"

BATCH_SIZE = 16 # Capped at 16 since this computer has 16GB of RAM and training with a batch size larger than 16 will result in an out-of-memory error.
EPOCHS = 100
PATIENCE = 25 # Setting patience to 25 since the model is expected to converge quickly. This will prevent overfitting and save time by stopping training early in case no improvement is seen in the validation loss for 25 epochs.
RESOLUTION = 640
WORKERS = 4 # Including 4 workers for data loading since this will speed up training by loading data in parallel.
DEVICE = "mps" # Ensuring that the model is trained on this computer's GPU (Apple Silicon) since CPU training is significantly slower.
RUN_NAME = "yolov8s" # Ensuring that the run name is set to "yolov8s" since this is the third run of the model in which the larger YOLOv8 small model is being used.
SEED = 0 # Setting the seed to 0 to ensure reproducibility of results across different runs of the model.

# Choosing to use the YOLOv8 small model for the third training session point in hopes of improving model performance.
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