# DamageDetector

A web app that finds damage in a photo of a car. Upload a photo and it marks dents, scratches, cracks, shattered glass, broken lamps and flat tires, each with a confidence score.

The model is YOLOv8s fine-tuned on the CarDD dataset. The backend is FastAPI, the frontend is React, and everything was trained on one laptop. After each scan, visitors can optionally say whether the model got it right, and the running totals are shown in the site's footer.

Live demo: [damagedetector.adityabdev.com](https://damagedetector.adityabdev.com)

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

There were five sessions. Each one changed a single thing about the setup, and before each run I wrote down what I expected that change to do, so the result could prove the guess wrong.

| # | Change | Val mAP50-95 | Hours |
|---|---|---|---|
| 1 | Baseline, YOLOv8n at 640 px | 0.555 | 4.0 |
| 2 | Resolution raised to 960 px | 0.552 | 7.1 |
| 3 | Larger model, YOLOv8s | 0.571 | 6.0 |
| 4 | YOLOv8s at 960 px | 0.561 | 12.0 |
| 5 | 280 undamaged cars added | 0.560 | 5.6 |

**Session 1: baseline.**
- Prediction: tire flat would be the weakest class, since it has the fewest training examples (225).
- Result: wrong. Tire flat scored second best. The weak classes were dent, scratch and crack, and they were being missed outright, not confused with each other.

**Session 2: higher resolution.**
- Prediction: small damage loses its detail when a photo is shrunk to 640 px, so 960 px should lift crack, dent and scratch and leave the other three unchanged.
- Result: not what happened. Recall rose from 0.658 to 0.695 and precision fell from 0.758 to 0.734, across all classes, so the two cancelled out. Scratch got worse.

**Session 3: larger model.**
- Prediction: if the weak classes are limited by model size, a larger model should lift dent, scratch and crack.
- Result: partly. Dent improved and overall mAP50-95 reached its best value, but scratch did not move.

**Session 4: larger model at higher resolution.**
- Prediction: the two changes should stack for a further gain of 0.01 to 0.02 in mAP50-95 over Session 3.
- Result: wrong. mAP50-95 fell by 0.010. The model found more damage and drew looser boxes around it, the same pattern as Session 2.

**Session 5: undamaged cars added to training.**
- Prediction: adding clean cars as negative examples should cut false alarms on undamaged cars, especially on intact lamps, without costing recall on real damage.
- Result: right. Clean cars flagged fell from 66% to 14%, and recall on real damage moved by 0.001.

Session 5 is the shipped model. It scores slightly below Session 3 on CarDD and far better on clean cars.

## Project layout

```
api/     FastAPI backend: main.py and its Dockerfile
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
echo "PREDICTION_SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" > .env
gcloud auth application-default login
uvicorn main:app --reload --env-file .env
```

The API runs at `http://127.0.0.1:8000`, with interactive docs at `/docs`. It loads its model from `api/weights/best.pt`. The trained weights are not published, because they are derived from the two datasets below, so you would need to train your own with `src/train.py` and copy `best.pt` there.

`PREDICTION_SECRET` signs the token that each scan returns, and the API will not start without it. Any value works locally. `api/.env` is gitignored. The feedback endpoints use Firestore, and `gcloud auth application-default login` lets the API reach it with your own Google account. The project ID is set in `main.py`, so to run them against your own database, change it to a Google Cloud project with Firestore enabled. `/predict` works without Firestore.

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

## Feedback questionnaire

Below each scan's results there is an optional prompt asking how the model did. It asks at most three questions, each appearing once the previous one is answered:

1. Did your photo show a car with damage?
2. If yes, did the model detect the damage accurately? If no, did the model detect any damage?
3. Only if the damage was not detected accurately: which damage was on the car, picked from the six classes.

Each answer is counted as correct, missed or wrong, or a false positive. The footer shows the running totals, labeled as user-reported, since only some visitors answer and the answers are not checked. The totals stay hidden until there are 10 answers, and they are also hidden if the database does not respond.

**What is stored.** Each answer is one Firestore document holding the answers, the outcome, any classes picked, the confidence threshold the slider was at, the classes visible at that threshold, and a timestamp. The photo is never stored, and nothing identifies the visitor.

**How it works.**

- The browser never talks to Firestore. It sends answers to the API, which writes them using the Cloud Run service account. The Firestore security rules deny all direct access.
- `/predict` returns a `prediction_token` with every scan. The token is a random ID followed by an HMAC signature of it, made with `PREDICTION_SECRET`. `POST /feedback` only accepts tokens whose signature checks out, so answers cannot be invented without a real scan.
- The random ID becomes the document's name, and the document is created with `create()`, which fails if it already exists. Each scan can therefore be answered only once, and a repeat gets a 409.
- The answer and the totals document (`stats/totals`) are written in one batch, with the totals updated through Firestore's atomic increments, so the counts always match the answers. `GET /stats` returns the totals.

## Deployment

The site and the API are hosted separately.

| Part | Host | Details |
|---|---|---|
| Website | Vercel | built from `web/`, served at `damagedetector.adityabdev.com` |
| API | Google Cloud Run | built from `api/Dockerfile`, region `us-east1` |
| Feedback database | Firestore | Native mode, region `us-east1`, same Google Cloud project as the API |

**API on Cloud Run**

- Runs on CPU with 2 vCPUs and 2 GB of memory. The container uses the CPU-only build of PyTorch, since Cloud Run has no GPU.
- Scales to zero when nobody is using it and is capped at one instance, which keeps it inside Cloud Run's free tier and limits what a traffic spike can cost. The first request after a quiet period waits for the container to start, and startup CPU boost is on to shorten that.
- `ALLOWED_ORIGINS` is set to the site's address, so only the site can call the API from a browser.
- `PREDICTION_SECRET` is a random key set on the service and kept out of the repo. Production uses a different key from local development.
- The API reaches Firestore as the service's default account, which has access through its Editor role, so no key file is needed.
- Deployed from inside `api/` with:

```
gcloud run deploy damagedetector-api \
  --source . \
  --region us-east1 \
  --memory 2Gi --cpu 2 \
  --max-instances 1 \
  --cpu-boost \
  --allow-unauthenticated \
  --update-env-vars ALLOWED_ORIGINS=https://damagedetector.adityabdev.com,PREDICTION_SECRET=<random key>
```

`--update-env-vars` adds or changes only the variables it lists, and the service keeps them between deploys, so later deploys can leave the flag out. Avoid `--set-env-vars` here: it replaces every variable on the service, so setting one would delete the other.

`--source .` uploads the folder and Cloud Build builds the image from the Dockerfile. The model weights are not in git, so they are uploaded from the local `api/weights/` folder; `api/.gcloudignore` makes sure they are included, since without it `gcloud` would follow `.gitignore` and leave them out.

**Website on Vercel**

- Root directory: `DamageDetector/web`.
- `VITE_API_URL` is set to the Cloud Run service URL, with no trailing slash.
- `web/vercel.json` sends every path to `index.html`, so refreshing `/training` or `/performance` works.

**Updating**

- Website changes deploy automatically when they are pushed to GitHub.
- API changes need the deploy command above run again from `api/`.

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

That license covers the code only. The datasets keep their own terms, described above, which limit them to non-commercial research use. For the same reason, the trained model weights are not included in this repository.
