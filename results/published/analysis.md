# Verified experimental findings

- yolov8n.pt: shared AP 0.3567 (COCO) to 0.1943 (KITTI), difference -16.24 percentage points. Sources: experiments/coco_yolov8n/shared/metrics.json and experiments/kitti_yolov8n/shared/metrics.json.
  Timing baseline: 10.59 ms, 94.45 FPS; repeat mean SD 2.01 ms; isolated forward 7.78 ms. Source: timing_yolov8n_baseline/timing.json.
- yolo11n.pt: shared AP 0.3632 (COCO) to 0.1969 (KITTI), difference -16.63 percentage points. Sources: experiments/coco_yolo11n/shared/metrics.json and experiments/kitti_yolo11n/shared/metrics.json.
  Timing baseline: 14.10 ms, 70.90 FPS; repeat mean SD 3.80 ms; isolated forward 10.98 ms. Source: timing_yolo11n_baseline/timing.json.
- rtdetr-l.pt: shared AP 0.5017 (COCO) to 0.2860 (KITTI), difference -21.57 percentage points. Sources: experiments/coco_rtdetr-l/shared/metrics.json and experiments/kitti_rtdetr-l/shared/metrics.json.
  Timing baseline: 27.27 ms, 36.67 FPS; repeat mean SD 1.79 ms; isolated forward 24.61 ms. Source: timing_rtdetr-l_baseline/timing.json.
- Relative to yolov8n.pt, yolo11n.pt adds +1.20 COCO AP percentage points and has 1.33x measured full-prediction latency. This compares checkpoint configurations, not architecture alone; repeat spreads above bound interpretation.
- Relative to yolov8n.pt, rtdetr-l.pt adds +15.65 COCO AP percentage points and has 2.58x measured full-prediction latency. This compares checkpoint configurations, not architecture alone; repeat spreads above bound interpretation.
- Qualitative scene 002142.png (ID 139, small_objects): YOLOv8n shared-class diagnostics at IoU .50/conf .25: 6 TP, 1 FP, 6 FN. Full panel and crop are in figures/. Counts reflect the mapped protocol; missing DontCare and excluded taxonomy can explain some unmatched predictions.
- Qualitative scene 000535.png (ID 34, crowded_scene): YOLOv8n shared-class diagnostics at IoU .50/conf .25: 3 TP, 10 FP, 13 FN. Full panel and crop are in figures/. Counts reflect the mapped protocol; missing DontCare and excluded taxonomy can explain some unmatched predictions.
- Qualitative scene 005226.png (ID 344, pedestrians): YOLOv8n shared-class diagnostics at IoU .50/conf .25: 5 TP, 4 FP, 10 FN. Full panel and crop are in figures/. Counts reflect the mapped protocol; missing DontCare and excluded taxonomy can explain some unmatched predictions.
- Qualitative scene 000423.png (ID 27, trucks): YOLOv8n shared-class diagnostics at IoU .50/conf .25: 6 TP, 6 FP, 8 FN. Full panel and crop are in figures/. Counts reflect the mapped protocol; missing DontCare and excluded taxonomy can explain some unmatched predictions.
- Qualitative scene 000346.png (ID 23, overlapping_gt): YOLOv8n shared-class diagnostics at IoU .50/conf .25: 6 TP, 7 FP, 2 FN. Full panel and crop are in figures/. Counts reflect the mapped protocol; missing DontCare and excluded taxonomy can explain some unmatched predictions.
- Qualitative scene 006322.png (ID 420, highest_shared_false_positives_yolov8n): YOLOv8n shared-class diagnostics at IoU .50/conf .25: 4 TP, 11 FP, 3 FN. Full panel and crop are in figures/. Counts reflect the mapped protocol; missing DontCare and excluded taxonomy can explain some unmatched predictions.

## Parameter effects

- ablation_yolov8n_384_fp32: 200-image shared AP 0.1593, -2.78 percentage points relative to the SAME 200-image 512 FP32 baseline (0.1871).
- ablation_yolov8n_640_fp32: 200-image shared AP 0.2103, +2.32 percentage points relative to the SAME 200-image 512 FP32 baseline (0.1871).
- ablation_yolov8n_512_fp16: 200-image shared AP 0.1865, -0.06 percentage points relative to the SAME 200-image 512 FP32 baseline (0.1871).
- timing_yolov8n_resolution_384: latency change +16.2% vs timing baseline; repeat SD 3.23 ms, mean output count 7.89. Output counts alone do not measure recall.
- timing_yolov8n_resolution_640: latency change +10.4% vs timing baseline; repeat SD 2.97 ms, mean output count 13.79. Output counts alone do not measure recall.
- timing_yolov8n_fp16: latency change +4.6% vs timing baseline; repeat SD 1.83 ms, mean output count 11.21. Output counts alone do not measure recall.
- timing_yolov8n_conf_0.1: latency change -0.4% vs timing baseline; repeat SD 2.02 ms, mean output count 23.63. Output counts alone do not measure recall.
- timing_yolov8n_conf_0.5: latency change +1.7% vs timing baseline; repeat SD 2.49 ms, mean output count 4.89. Output counts alone do not measure recall.
- timing_yolov8n_iou_0.5: latency change -1.9% vs timing baseline; repeat SD 2.05 ms, mean output count 10.68. Output counts alone do not measure recall.
- ablation_yolo11n_384_fp32: 200-image shared AP 0.1637, -2.77 percentage points relative to the SAME 200-image 512 FP32 baseline (0.1915).
- ablation_yolo11n_640_fp32: 200-image shared AP 0.2327, +4.12 percentage points relative to the SAME 200-image 512 FP32 baseline (0.1915).
- ablation_yolo11n_512_fp16: 200-image shared AP 0.1907, -0.07 percentage points relative to the SAME 200-image 512 FP32 baseline (0.1915).
- timing_yolo11n_resolution_384: latency change -8.4% vs timing baseline; repeat SD 1.92 ms, mean output count 7.26. Output counts alone do not measure recall.
- timing_yolo11n_resolution_640: latency change -16.0% vs timing baseline; repeat SD 0.48 ms, mean output count 13.00. Output counts alone do not measure recall.
- timing_yolo11n_fp16: latency change -5.9% vs timing baseline; repeat SD 1.07 ms, mean output count 10.63. Output counts alone do not measure recall.
- timing_yolo11n_conf_0.1: latency change -15.5% vs timing baseline; repeat SD 0.82 ms, mean output count 22.42. Output counts alone do not measure recall.
- timing_yolo11n_conf_0.5: latency change +5.4% vs timing baseline; repeat SD 2.96 ms, mean output count 4.63. Output counts alone do not measure recall.
- timing_yolo11n_iou_0.5: latency change +9.6% vs timing baseline; repeat SD 1.39 ms, mean output count 9.79. Output counts alone do not measure recall.
- ablation_rtdetr-l_384_fp32: 200-image shared AP 0.2636, -3.16 percentage points relative to the SAME 200-image 512 FP32 baseline (0.2952).
- ablation_rtdetr-l_640_fp32: 200-image shared AP 0.3018, +0.66 percentage points relative to the SAME 200-image 512 FP32 baseline (0.2952).
- ablation_rtdetr-l_512_fp16: 200-image shared AP 0.2933, -0.19 percentage points relative to the SAME 200-image 512 FP32 baseline (0.2952).
- timing_rtdetr-l_resolution_384: latency change -9.0% vs timing baseline; repeat SD 0.13 ms, mean output count 30.37. Output counts alone do not measure recall.
- timing_rtdetr-l_resolution_640: latency change +11.3% vs timing baseline; repeat SD 0.40 ms, mean output count 40.21. Output counts alone do not measure recall.
- timing_rtdetr-l_fp16: latency change -10.0% vs timing baseline; repeat SD 0.28 ms, mean output count 36.37. Output counts alone do not measure recall.
- timing_rtdetr-l_conf_0.1: latency change -1.7% vs timing baseline; repeat SD 2.08 ms, mean output count 148.63. Output counts alone do not measure recall.
- timing_rtdetr-l_conf_0.5: latency change -6.8% vs timing baseline; repeat SD 0.45 ms, mean output count 11.79. Output counts alone do not measure recall.

The shared-class comparison uses identical evaluator thresholds, not identical scenes or annotation taxonomies. Differences are descriptive and confounded by dataset composition, object scale, model capacity and resizing. KITTI vans and sitting persons are excluded; mapped predictions overlapping them can become false positives. DontCare/difficulty data are absent. COCO retains original area metadata whereas KITTI uses clipped box area, so size bins are not directly equivalent. Bicycle/cyclist is supplementary and approximate. Timing uncertainty is repeat variability on one laptop and one fixed image set, not a population confidence interval. No threshold was tuned on evaluation data. No retraining was performed.

All exact tables come from tables/report_data.json. Primary KITTI table values use only the three shared classes. The raw KITTI all/ evaluator directory also includes supplementary bicycle/cyclist; use shared/ and domain_comparison.csv for the primary transfer macro. Counts, mappings and input hashes are in data/audit.json. Qualitative scene selection rules and exact image IDs are in figures/scene_selection.json; matched TP/FP/FN diagnostics use official COCOeval at IoU .50 and confidence .25. A visible disagreement alone does not establish its cause.
