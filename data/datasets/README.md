# Large validation datasets (not committed)

Default COCO layout: `coco/annotations/instances_val2017.json` and
`coco/images/val2017/`. Default prepared KITTI layout: `kitti/kitti.yaml`,
`kitti/images/val/`, and `kitti/labels/val/`.

To reuse data elsewhere, copy `configs/local.example.json` to `configs/local.json`
and set the existing dataset paths. Local configuration is ignored by Git.
The pipeline does not download COCO training data.

