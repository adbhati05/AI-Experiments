# DamageDetector

A web app that finds damage in a photo of a car. Upload a photo and it marks dents, scratches, cracks, shattered glass, broken lamps and flat tires, each with a confidence score.

The model is YOLOv8s fine-tuned on the CarDD dataset. The backend is FastAPI, the frontend is React, and everything was trained on one laptop.

Live demo: coming soon.

## Results

Measured once on CarDD's 374 held-out test photos, after all tuning was finished.

| Metric | Score |
|---|---|
| mAP50 | 0.745 |
| mAP50-95 | 0.590 |
| Precision | 0.725 |
| Recall | 0.720 |

| Class | AP50 | AP50-95 |
|---|---|---|
| glass shatter | 0.990 | 0.923 |
| tire flat | 0.930 | 0.893 |
| lamp broken | 0.874 | 0.730 |
| scratch | 0.627 | 0.367 |
| dent | 0.617 | 0.379 |
| crack | 0.432 | 0.245 |

## What I found

**Accuracy follows the size of the damage, not how often it appears in training.** Scratch has the most training examples (2,560) and scores near the bottom. Tire flat has the fewest (225) and scores near the top. Ranking the six classes by box size, by how much of the box is actual damage, and by accuracy gives nearly the same order each time.

**The benchmark could not see the model's worst behavior.** CarDD contains no undamaged cars, so none of its metrics show false alarms. I built a separate set of 200 clean cars from CompCars to measure it, and the model flagged damage on 66% of them.

**Adding undamaged cars to training fixed most of it.** With 280 clean cars added as negative examples, the share of clean cars flagged fell from 66% to 14%, and recall on real damage moved by 0.001. At the shipping threshold of 0.50 it is 8%.

**A bigger model and a higher resolution did not help the hard classes.** Scratch did not improve under either change, which points at the data. Where a scratch or dent ends is a judgment call, so the labels themselves are inconsistent.

## Training sessions

Each session changed one thing and started with a written prediction of what would happen.

| # | Change | Val mAP50-95 | Hours | Prediction |
|---|---|---|---|---|
| 1 | Baseline, YOLOv8n at 640 px | 0.555 | 4.0 | baseline |
| 2 | Resolution raised to 960 px | 0.552 | 7.1 | not confirmed |
| 3 | Larger model, YOLOv8s | 0.571 | 6.0 | partially confirmed |
| 4 | YOLOv8s at 960 px | 0.561 | 12.0 | not confirmed |
| 5 | 280 undamaged cars added | 0.560 | 5.6 | confirmed |

Session 5 is the shipped model. It scores slightly below Session 3 on CarDD and far better on clean cars.

## Project layout

```
api/     FastAPI backend: main.py, Dockerfile, the model weights
web/     React frontend: Detect, Training and Performance pages
src/     train.py and export_metrics.py
data/    dataset config and instructions (the data itself is not in this repo)
```

## Running it locally

You need Python 3.12 and Node 22.

**Backend**

```
pip install torch torchvision
pip install -r api/requirements.txt
cd api
uvicorn main:app --reload
```

The API runs at `http://127.0.0.1:8000`, with interactive docs at `/docs`. It loads its weights from `api/weights/best.pt`.

**Frontend**

```
cd web
npm install
echo "VITE_API_URL=http://127.0.0.1:8000" > .env.local
npm run dev
```

The site runs at `http://localhost:5173`.

**Training**

Training needs the datasets, which are not in this repo. See [data/README.md](data/README.md) for how to get them and rebuild the labels. Then:

```
python src/train.py
python src/export_metrics.py
```

The second command rewrites `web/src/data/metrics.json`, which the Training and Performance pages read.

## A note on training on Apple Silicon

The first training runs looked broken: losses rose, accuracy stayed near zero and epochs took 40 minutes. Running the same setup on the CPU learned normally, which pointed at PyTorch's Apple GPU backend. PyTorch 2.4.1 was computing wrong results on it without any error. Upgrading to 2.14.1 fixed the results and made inference about 29 times faster.

Training on a Mac works well. The takeaway is to keep PyTorch current and, when a run looks wrong, compare a short run on the CPU before changing anything else.

## Limitations

- The model is small. YOLOv8s was chosen because it trains on a laptop.
- At the shipping threshold it misses about 35% of dents, 35% of scratches and 50% of cracks.
- Each configuration was trained once, so small differences between sessions may be chance.
- It has only been tested on CarDD-style photos. Night, rain and unusual angles are untested.

## Data and citations

This project uses two research datasets. Both are available for non-commercial research only and neither may be redistributed, so no photos or annotations from them are in this repository.

- **CarDD.** Wang, X., Li, W. and Wu, Z. "CarDD: A New Dataset for Vision-Based Car Damage Detection." IEEE Transactions on Intelligent Transportation Systems, 24(7), 7202-7214, 2023. https://doi.org/10.1109/TITS.2023.3258480
- **CompCars.** Yang, L., Luo, P., Loy, C. C. and Tang, X. "A Large-Scale Car Dataset for Fine-Grained Categorization and Verification." IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html

## License

The code is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE). It builds on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), which is released under the same license.

The datasets keep their own terms, described above. This project is non-commercial.
