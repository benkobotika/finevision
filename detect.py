import os
import json
import numpy as np
from tqdm import tqdm
import argparse

from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.structures import DetDataSample
from pycocotools import mask as maskUtils
from PIL import Image


def binary_mask_to_rle(mask):
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def convert_to_coco_annotations(
    results: DetDataSample.pred_instances, image_id, category_map, ann_id_start
):
    coco_results = []
    bboxes = results.bboxes.cpu().numpy()
    scores = results.scores.cpu().numpy()
    labels = results.labels.cpu().numpy()
    masks = getattr(results, "masks", None)

    ann_id = ann_id_start
    for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
        x1, y1, x2, y2 = bbox.tolist()
        width = x2 - x1
        height = y2 - y1

        ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_map[label],
            "bbox": [x1, y1, width, height],
            "score": float(score),
        }

        if masks is not None:
            mask = masks[i].cpu().numpy()
            ann["segmentation"] = binary_mask_to_rle(mask)

        coco_results.append(ann)
        ann_id += 1

    return coco_results, ann_id


def main(config_file, checkpoint_file, img_folder, out_file, score_thr=0.3):
    model = init_detector(config_file, checkpoint_file, device="cuda:0")

    if hasattr(model.cfg, "metainfo") and "classes" in model.cfg.metainfo:
        classes = model.cfg.metainfo["classes"]
    else:
        raise ValueError("Cannot find dataset classes in model.cfg.metainfo['classes']")

    category_map = {i: i + 1 for i in range(len(classes))}

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(classes)]

    img_files = [
        os.path.join(img_folder, f)
        for f in os.listdir(img_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    coco_images = []
    coco_annotations = []
    ann_id = 1

    for image_id, img_path in enumerate(tqdm(img_files, desc="Processing images")):
        img = Image.open(img_path)
        width, height = img.size
        coco_images.append(
            {
                "id": image_id,
                "file_name": os.path.basename(img_path),
                "width": width,
                "height": height,
            }
        )

        result: DetDataSample = inference_detector(model, img_path)
        pred_instances = result.pred_instances

        keep = pred_instances.scores >= score_thr
        pred_instances = pred_instances[keep]

        anns, ann_id = convert_to_coco_annotations(
            pred_instances, image_id, category_map, ann_id
        )
        coco_annotations.extend(anns)

    coco_output = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    with open(out_file, "w") as f:
        json.dump(coco_output, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to MMDetection config file"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to trained checkpoint"
    )
    parser.add_argument("--images", required=True, help="Folder with images to process")
    parser.add_argument("--out", required=True, help="Output COCO JSON file path")
    parser.add_argument(
        "--score_thr", type=float, default=0.3, help="Score threshold for detections"
    )
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.images, args.out, args.score_thr)
