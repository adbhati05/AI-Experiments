import csv
import json
import statistics
from collections import Counter
from datetime import date
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# Resolving every path from this file's location so the script can be run from anywhere, the same way train.py does.
ROOT = Path(__file__).resolve().parent.parent
DATA_CONFIG = ROOT / "data" / "data.yaml"
RUNS = ROOT / "runs" / "detect"
OUTPUT = ROOT / "web" / "src" / "data" / "metrics.json" # The React app imports this file directly, so the Training and Performance pages never depend on the API being awake.
SCRATCH = "/tmp/damagedetector_eval" # Sending ultralytics' own val output here so it doesn't create extra folders inside runs/.

CLASSES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
DEVICE = "mps"

# Initializing the shipping configuration, which involves the fifth training sessiom (YOLOv8s trained on the modified dataset with 280 negative samples added).
# I've set the confidence threshold to 0.50 since that is pretty much the point where the precision and recall curves cross on the validation split, which ensures a good balance between false positives and false negatives.
SHIPPING_RUN = "negatives"
SHIPPING_CONF = 0.50
SHIPPING_IMGSZ = 640

# These are the five training sessions in order. I established a hypothesis for each one about what change would improve the model, and the verdict is whether or not that claim held.
SESSIONS = [
    {"id": 1, "run": "baseline", "label": "Baseline", "change": "YOLOv8n at 640", "verdict": "baseline"},
    {"id": 2, "run": "res960", "label": "Higher resolution", "change": "640 to 960", "verdict": "not confirmed"},
    {"id": 3, "run": "yolov8s", "label": "Larger model", "change": "YOLOv8n to YOLOv8s", "verdict": "partially confirmed"},
    {"id": 4, "run": "yolov8sres960", "label": "Larger model, higher resolution", "change": "YOLOv8s at 960", "verdict": "not confirmed"},
    {"id": 5, "run": "negatives", "label": "Negative examples", "change": "280 undamaged car images added", "verdict": "confirmed"},
]

# Initializing the share of each bounding box that is actual damage, measured once from CarDD's polygon masks on the val split. The masks are not part of this repo, so the values are kept here as constants.
BOX_FILL_PCT = {"dent": 67.4, "scratch": 56.0, "crack": 25.8, "glass shatter": 84.2, "lamp broken": 73.1, "tire flat": 81.0}


# This function is used to run ultralytics' validation on a given run folder and return the headline metrics (mAP50, mAP50-95, precision, recall) as well as the per-class metrics. 
def evaluate(run, split, imgsz, tta=False, conf=None):
    model = YOLO(RUNS / run / "weights" / "best.pt")
    extra = {} if conf is None else {"conf": conf}
    r = model.val(data=str(DATA_CONFIG), split=split, imgsz=imgsz, device=DEVICE, augment=tta,
                  verbose=False, plots=False, project=SCRATCH, name=f"{run}_{split}", exist_ok=True, **extra)
    per_class = {}
    for i, c in enumerate(r.ap_class_index):
        per_class[r.names[int(c)]] = {
            "ap50": round(float(r.box.ap50[i]), 4), "ap50_95": round(float(r.box.ap[i]), 4),
            "precision": round(float(r.box.p[i]), 4), "recall": round(float(r.box.r[i]), 4),
        }
    return {
        "map50": round(float(r.box.map50), 4), "map50_95": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
        "per_class": per_class,
    }


# This function summarizes the dataset by counting the number of images, bounding boxes, negative samples, and per-class instances in each split (train, val, test) by reading the label files directly.
def dataset_summary():
    splits, instances = {}, {}
    for split in ["train", "val", "test"]:
        counts, negatives = Counter(), 0
        labels = sorted((ROOT / "data" / "labels" / split).glob("*.txt"))
        for f in labels:
            lines = [l for l in f.read_text().splitlines() if l.strip()]
            negatives += not lines
            counts.update(CLASSES[int(l.split()[0])] for l in lines)
        splits[split] = {"images": len(labels), "boxes": sum(counts.values()), "negatives": negatives}
        instances[split] = {c: counts[c] for c in CLASSES}
    return splits, instances


# This function calculates the median bounding box area for each class in the validation split, expressed as a percentage of the image size.
def class_geometry():
    # Computing the median bounding box area by extracting the width/height from the label files in the val split and multiplying them together, then converting to a percentage of the image size (640x640).
    areas = {c: [] for c in CLASSES}
    for f in (ROOT / "data" / "labels" / "val").glob("*.txt"):
        for l in f.read_text().splitlines():
            p = l.split()
            if len(p) == 5:
                areas[CLASSES[int(p[0])]].append(float(p[3]) * float(p[4]) * 100)
    return {c: {"median_box_area_pct": round(statistics.median(areas[c]), 2), "box_fill_pct": BOX_FILL_PCT[c]} for c in CLASSES}


# This function constructs a record of a training session by reading the run folder, extracting the hyperparameters, epoch curves, and validation metrics. 
def session_record(s):
    run = RUNS / s["run"]
    args = yaml.safe_load((run / "args.yaml").read_text())
    rows = list(csv.DictReader((run / "results.csv").open()))
    rows = [{k.strip(): v for k, v in row.items()} for row in rows]
    curve50 = [round(float(r["metrics/mAP50(B)"]), 4) for r in rows]
    curve5095 = [round(float(r["metrics/mAP50-95(B)"]), 4) for r in rows]
    return {
        **{k: s[k] for k in ["id", "run", "label", "change", "verdict"]},
        "model": Path(args["model"]).stem, "imgsz": args["imgsz"], "batch": args["batch"],
        "negatives": 280 if s["run"] == "negatives" else 0,
        "epochs_run": len(rows), "best_epoch": curve5095.index(max(curve5095)) + 1,
        "wall_time_hours": round(float(rows[-1]["time"]) / 3600, 1),
        "val": evaluate(s["run"], "val", args["imgsz"]),
        "curve": {"map50": curve50, "map50_95": curve5095},
    }


# This function takes the inputted run folder and runs the model over the frozen clean car eval set (200 images of undamaged cars).
# Every detection here is a false positive since none of these cars are damaged, so this function will be used to draw a comparison between the model before and after adding negative samples to the training set.
def clean_car_false_positives(run):
    model = YOLO(RUNS / run / "weights" / "best.pt")
    images = sorted((ROOT / "data" / "clean" / "eval").glob("*.jpg"))
    detections = [] # One entry per image: a list of tuples of the form (class name, confidence).
    for i in range(0, len(images), 8): # Predicting in chunks of 8 since passing every image at once runs MPS out of memory.
        for r in model.predict([str(p) for p in images[i:i + 8]], imgsz=SHIPPING_IMGSZ, conf=0.25, augment=True, device=DEVICE, verbose=False):
            detections.append([(CLASSES[int(b.cls)], float(b.conf)) for b in r.boxes])
        torch.mps.empty_cache()

    # This function summarizes the false positives at a given confidence threshold by counting the number of images with detections, the total number of detections, the average number of detections per image, and the number of detections per class.
    # The goal here is to return two sets of metrics for each confidence threshold: 0.25 and SHIPPING_CONF (0.50).
    def summarize(indices, conf):
        kept = [[d for d in detections[i] if d[1] >= conf] for i in indices]
        total = sum(len(k) for k in kept)
        by_class = Counter(d[0] for k in kept for d in k)
        return {
            "images": len(indices), "false_positives": total,
            "per_image": round(total / len(indices), 3),
            "images_flagged_pct": round(100 * sum(1 for k in kept if k) / len(indices), 1),
            "by_class": {c: by_class[c] for c in CLASSES},
        }

    close = [i for i, p in enumerate(images) if "fullcar" not in p.name]
    full = [i for i, p in enumerate(images) if "fullcar" in p.name]
    everything = list(range(len(images)))
    return {
        f"conf_{conf:.2f}": {"all": summarize(everything, conf), "close_ups": summarize(close, conf), "full_car": summarize(full, conf)}
        for conf in [0.25, SHIPPING_CONF]
    }

# This function reads the specific precision, recall, and F1 score at fixed confidence thresholds off the full curves ultralytics computes during validation.
# The goal here is to ensure that legitimate values are being retrieved before drawing conclusions about the best confidence threshold for the shipping model.
def threshold_curve():
    # The precision and recall that val() reports are always taken at the confidence with the best F1 score, no matter what conf value is passed in, so they cannot be used to compare thresholds. 
    # So instead, I'm using the full curves that val() computes to retrieve these values.
    model = YOLO(RUNS / SHIPPING_RUN / "weights" / "best.pt")
    r = model.val(data=str(DATA_CONFIG), split="val", imgsz=SHIPPING_IMGSZ, device=DEVICE, augment=True,
                  verbose=False, plots=False, project=SCRATCH, name="threshold_curve", exist_ok=True)
    conf_axis = r.box.px
    precision, recall, f1 = r.box.p_curve.mean(0), r.box.r_curve.mean(0), r.box.f1_curve.mean(0) # Averaged over the six classes.

    def at(conf):
        i = abs(conf_axis - conf).argmin()
        return {"conf": round(float(conf), 3), "precision": round(float(precision[i]), 4), "recall": round(float(recall[i]), 4), "f1": round(float(f1[i]), 4)}

    return {
        "best_f1": at(conf_axis[f1.argmax()]),
        "points": [at(c) for c in [0.10, 0.25, 0.40, 0.50, 0.60, 0.70]],
        "curve": [at(c / 100) for c in range(2, 99, 2)], # Sampling the full curves every 0.02 so the web app can draw precision and recall against confidence.
    }


# This function returns the confusion matrix of the shipping model on the test split at the shipping confidence threshold.
# Rows are what the model predicted and columns are what was actually there, with "background" as the last row and column (a missed box or a false alarm).
def confusion_matrix():
    model = YOLO(RUNS / SHIPPING_RUN / "weights" / "best.pt")
    # Ultralytics only fills in the confusion matrix when plots is True, so the plots it writes are sent to the scratch folder and ignored.
    r = model.val(data=str(DATA_CONFIG), split="test", imgsz=SHIPPING_IMGSZ, device=DEVICE, augment=True, conf=SHIPPING_CONF,
                  verbose=False, plots=True, project=SCRATCH, name="confusion_matrix", exist_ok=True)
    return {"conf": SHIPPING_CONF, "labels": CLASSES + ["background"], "counts": r.confusion_matrix.matrix.astype(int).tolist()}


# This function returns the precision-recall curve of each class for the shipping model on the test split.
def pr_curves():
    model = YOLO(RUNS / SHIPPING_RUN / "weights" / "best.pt")
    r = model.val(data=str(DATA_CONFIG), split="test", imgsz=SHIPPING_IMGSZ, device=DEVICE, augment=True,
                  verbose=False, plots=False, project=SCRATCH, name="pr_curves", exist_ok=True)
    steps = range(0, 1000, 20) # The curves have 1000 points each, so every 20th point is kept to keep the file small.
    curves = {r.names[int(c)]: [round(float(r.box.prec_values[i][k]), 4) for k in steps] for i, c in enumerate(r.ap_class_index)}
    return {"recall": [round(float(r.box.px[k]), 3) for k in steps], "precision": curves}


# By leveraging the functions above, main() constructs a dictionary of metrics that summarizes the dataset, the model, the training sessions, and the evaluation results.
# The goal here is to serialize this dictionary to a JSON file so that it can be used by the web app to display the model's performance and the dataset's characteristics.
def main():
    splits, instances = dataset_summary()
    args = yaml.safe_load((RUNS / SHIPPING_RUN / "args.yaml").read_text())

    metrics = {
        "generated": date.today().isoformat(),
        "classes": CLASSES,
        "model": {
            "architecture": Path(args["model"]).stem, "run": SHIPPING_RUN, "imgsz": SHIPPING_IMGSZ,
            "conf": SHIPPING_CONF, "tta": True, "pretrained_on": "COCO",
            "parameters_millions": round(sum(p.numel() for p in YOLO(RUNS / SHIPPING_RUN / "weights" / "best.pt").model.parameters()) / 1e6, 1),
        },
        "dataset": {"splits": splits, "class_instances": instances, "clean_eval_images": len(list((ROOT / "data" / "clean" / "eval").glob("*.jpg")))},
        "class_geometry": class_geometry(),
        # Ensuring that the test split numbers are the headline results for the model, since this is the split that the model has never seen before and is therefore the most representative of real-world performance.
        "test": {"tta": evaluate(SHIPPING_RUN, "test", SHIPPING_IMGSZ, tta=True), "no_tta": evaluate(SHIPPING_RUN, "test", SHIPPING_IMGSZ)},
        "sessions": [session_record(s) for s in SESSIONS],
        # As mentioned above, clean_car_false_positives is used here to draw a comparison between Session #3 (the best performing model before adding negatives) and Session #5 (the shipping model, which is the exact same config as with Session #3 just with negatives added).
        "clean_cars": {"before": clean_car_false_positives("yolov8s"), "after": clean_car_false_positives(SHIPPING_RUN)},
        # Retrieving the precision, recall, and F1 score at differing confidence thresholds (0.10, 0.25, 0.40, 0.50, 0.60, 0.70), along with the threshold that gives the best F1 score.
        # The goal here is to validate that the chosen confidence threshold of 0.50 is indeed the best balance between precision and recall.
        "thresholds": threshold_curve(),
        "confusion_matrix": confusion_matrix(),
        "pr_curves": pr_curves(),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()