# Data

The datasets are not in this repository. Both are licensed for non-commercial research only and do not allow redistribution, so the photos and the label files stay out of git. This page explains where the data comes from and how to rebuild it.

## What goes where

```
data/
  data.yaml              class names and the paths to each split
  images/{train,val,test}/   photos (not in git)
  labels/{train,val,test}/   one .txt per photo, YOLO format (not in git)
  clean/eval/            200 undamaged cars, used only for testing (not in git)
  clean/negatives/       280 undamaged cars, added to training (not in git)
```

`data.yaml` has an absolute `path` on its first line. Change it to the location of this `data` folder on your machine.

## CarDD: the damage photos

CarDD is requested from its authors by sending them a signed licensing form. The form and instructions are at https://cardd-ustc.github.io/.

The release has two versions. Use `CarDD_COCO`. `CarDD_SOD` is for a different task and has no class labels.

| Split | Photos | Boxes |
|---|---|---|
| train | 2,816 | 6,211 |
| val | 810 | 1,744 |
| test | 374 | 785 |

There are six classes: dent, scratch, crack, glass shatter, lamp broken and tire flat.

**Rebuilding the labels.** CarDD ships its annotations as COCO JSON. YOLOv8 needs one text file per photo, so they are converted with Ultralytics:

```python
from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="CarDD_release/CarDD_COCO/annotations",
    save_dir="cardd_converted",
    use_segments=False,   # boxes only
    cls91to80=False,      # that remapping is for the original COCO classes
)
```

Then:

1. Copy the photos from `train2017`, `val2017` and `test2017` into `data/images/train`, `val` and `test`.
2. Copy the converted labels from `cardd_converted/labels/train2017`, `val2017` and `test2017` into `data/labels/train`, `val` and `test`.

Each photo should end up with a label file of the same name. Class ids run from 0 to 5.

## CompCars: the undamaged cars

CarDD has no undamaged cars, so it cannot show how often the model raises a false alarm. CompCars fills that gap. Its terms and download instructions are at http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html.

480 photos were taken from it:

- **200 for testing** (`data/clean/eval`): 150 exterior close-ups and 50 whole-car photos. These are never used in training.
- **280 for training** (`data/clean/negatives`): exterior close-ups, added to the training split as negative examples.

The close-ups come from CompCars' part categories 1 to 4: headlights, taillights, fog lights and front grilles. Categories 5 to 8 are interiors and were left out.

The split was made by car model, so the same vehicle never appears in both sets. At most two photos were taken per model and part, and photos narrower than 400 pixels were skipped.

**Adding the negatives to training.** Copy each photo from `data/clean/negatives` into `data/images/train` and create an empty `.txt` file of the same name in `data/labels/train`. An empty label file tells YOLOv8 the photo contains no damage.
