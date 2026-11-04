# Repository validation

The reorganized source was checked locally in the existing `dtect` environment.

| Check | Result |
|---|---|
| Repository/unit tests | 11 passed |
| Official evaluator/data cases within the suite | 15 passed |
| Invocation outside the repository working directory | Passed |
| Default repository-relative input/output paths | Passed |
| Local configuration merge preserves common settings | Passed |
| Planned experiment matrix | 38 unique configurations; no RT-DETR NMS sweep |
| Missing versus empty labels / image-folder handling | Passed |
| Arbitrary annotated verification image IDs | Preserved |
| Worker subprocess package path | Passed |
| GPU verification worker: YOLOv8n | Passed; 5 output boxes on the bus test image |
| GPU verification worker: YOLO11n | Passed; 5 output boxes on the bus test image |
| GPU verification worker: RT-DETR-L | Passed; 10 output boxes on the bus test image |

The GPU check ran one existing test image from an ignored integration-input
folder. Tensor checks verified CUDA, FP32, batch one, no gradients and evaluation
mode. These output counts are not accuracy claims. No example image was added
to the user-upload directory.

The complete original COCO/KITTI experiment was not repeated for this source
reorganization. Its published artifacts are labeled separately in
`results/published/`. The GitHub Actions workflow is provided but has not been
executed on GitHub. Local integration logs are in ignored `outputs/`.

