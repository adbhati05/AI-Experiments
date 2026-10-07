// The response shape of POST /predict. These mirror the Pydantic models in api/main.py.
export interface Box {
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface Detection {
  class_name: string
  confidence: number
  box: Box
}

export interface PredictResponse {
  width: number
  height: number
  detections: Detection[]
}

// The shape of metrics.json, which is written by src/export_metrics.py.
export interface ClassMetrics {
  ap50: number
  ap50_95: number
  precision: number
  recall: number
}

export interface EvalResult {
  map50: number
  map50_95: number
  precision: number
  recall: number
  per_class: Record<string, ClassMetrics>
}

export interface Session {
  id: number
  run: string
  label: string
  change: string
  verdict: string
  model: string
  imgsz: number
  batch: number
  negatives: number
  epochs_run: number
  best_epoch: number
  wall_time_hours: number
  val: EvalResult
  curve: { map50: number[]; map50_95: number[] }
}

export interface FalsePositiveSummary {
  images: number
  false_positives: number
  per_image: number
  images_flagged_pct: number
  by_class: Record<string, number>
}

export type FalsePositivesByConf = Record<
  string,
  { all: FalsePositiveSummary; close_ups: FalsePositiveSummary; full_car: FalsePositiveSummary }
>

export interface ThresholdPoint {
  conf: number
  precision: number
  recall: number
  f1: number
}

type Split = 'train' | 'val' | 'test'

export interface Metrics {
  generated: string
  classes: string[]
  model: {
    architecture: string
    run: string
    imgsz: number
    conf: number
    tta: boolean
    pretrained_on: string
    parameters_millions: number
  }
  dataset: {
    splits: Record<Split, { images: number; boxes: number; negatives: number }>
    class_instances: Record<Split, Record<string, number>>
    clean_eval_images: number
  }
  class_geometry: Record<string, { median_box_area_pct: number; box_fill_pct: number }>
  test: { tta: EvalResult; no_tta: EvalResult }
  sessions: Session[]
  clean_cars: { before: FalsePositivesByConf; after: FalsePositivesByConf }
  thresholds: { best_f1: ThresholdPoint; points: ThresholdPoint[]; curve: ThresholdPoint[] }
  confusion_matrix: { conf: number; labels: string[]; counts: number[][] }
  pr_curves: { recall: number[]; precision: Record<string, number[]> }
}
