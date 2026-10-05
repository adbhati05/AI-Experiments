from pathlib import Path
from ultralytics import YOLO

# Initializing the path to the dataset config, resolved from this file's location rather than the current working directory, so the script can run properly from anywhere.
DATA_CONFIG = Path(__file__).resolve().parent.parent / "data" / "data.yaml"

BATCH_SIZE = 8 # Back to 8 since for this 4th training run, the resolution is set to 960.
EPOCHS = 100
PATIENCE = 25 # Setting patience to 25 since the model is expected to converge quickly. This will prevent overfitting and save time by stopping training early in case no improvement is seen in the validation loss for 25 epochs.
RESOLUTION = 960
WORKERS = 4 # Including 4 workers for data loading since this will speed up training by loading data in parallel.
DEVICE = "mps" # Ensuring that the model is trained on this computer's GPU (Apple Silicon) since CPU training is significantly slower.
RUN_NAME = "yolov8sres960" # Naming this run "yolov8sres960" since it uses the YOLOv8 small model with a resolution of 960.
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