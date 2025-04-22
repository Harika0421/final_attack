import os
import json
import argparse
import yaml
import logging
import base64
import time
import csv
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import torch
import torch.nn as nn
import numpy as np
import requests
from torchvision import transforms
from tqdm import tqdm

# --- Global Constants & Transforms ---
PREPROCESS = transforms.Compose([transforms.ToTensor()])
TO_PIL = transforms.ToPILImage()

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_to_csv(filename: str, row: Dict):
    path = os.path.join("logs", filename)
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def log_to_jsonl(filename: str, data: Dict):
    path = os.path.join("logs", filename)
    with open(path, 'a') as f:
        f.write(json.dumps(data) + '\n')

# --- Config Loader ---
def load_config(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def validate_config(cfg: Dict) -> None:
    required_keys = ['api_url', 'api_key', 'dataset', 'attack_type']
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")

# --- API Safety Wrapper ---
def safe_api_call(url: str, headers: Dict[str,str], payload: Dict, retries: int=3, timeout: int=10) -> Dict:
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            resp_json = response.json()
            if 'boxes' not in resp_json or 'labels' not in resp_json:
                raise ValueError("Malformed API response: missing 'boxes' or 'labels'")
            return resp_json
        except Exception as e:
            logging.warning(f"API call failed (attempt {attempt+1}): {e}")
            time.sleep(2)
    raise RuntimeError("API call failed after retries")

# --- Model Wrappers ---
class VisionModel:
    def predict(self, image: Image.Image) -> Dict:
        raise NotImplementedError

class GeminiWrapper(VisionModel):
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    def predict(self, image: Image.Image) -> Dict:
    # Fake prediction (2 dummy boxes)
     return {
        "boxes": [[30, 30, 150, 150], [200, 200, 300, 300]],
        "labels": ["dog", "cat"]
    }


class TorchVisionWrapper(VisionModel):
    def __init__(self, model: nn.Module):
        self.model = model.eval()

    def predict(self, image: Image.Image) -> Dict:
        with torch.no_grad():
            img_tensor = PREPROCESS(image).unsqueeze(0)
            out = self.model(img_tensor)[0]
            boxes = out['boxes'].tolist()
            labels = [str(l.item()) for l in out['labels']]
            return {'boxes': boxes, 'labels': labels}

# --- Utility Functions ---
def compute_iou(boxA: Tuple, boxB: Tuple) -> float:
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

def compute_ap(pred_boxes: List[Tuple], gt_boxes: List[Tuple], iou_thresh: float=0.5) -> float:
    matched, tp = set(), 0
    for pred in pred_boxes:
        for i, gt in enumerate(gt_boxes):
            if i not in matched and compute_iou(pred, gt) >= iou_thresh:
                tp += 1
                matched.add(i)
                break
    precision = tp / len(pred_boxes) if pred_boxes else 0.0
    recall = tp / len(gt_boxes) if gt_boxes else 0.0
    return precision * recall

def draw_boxes(image: Image.Image, boxes: List[Tuple], color: str, label: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0], box[1]), label, fill=color)
    return image

# --- Attack Base ---
class Attack:
    def run(self, image: Image.Image, gt_boxes: List[Tuple], target_label: Optional[str]) -> Image.Image:
        raise NotImplementedError

class NESAttack(Attack):
    def __init__(self, model: VisionModel, epsilon: float=0.05, iterations: int=40, sigma: float=0.01, alpha: float=1e-2):
        self.model, self.epsilon, self.iterations, self.sigma, self.alpha = model, epsilon, iterations, sigma, alpha

    def run(self, image: Image.Image, gt_boxes: List[Tuple], target_label: Optional[str]) -> Image.Image:
        img = PREPROCESS(image).unsqueeze(0)
        adv = img.clone()
        for _ in range(self.iterations):
            noise = torch.randn_like(adv) * self.sigma
            scores = []
            for sign in [-1,1]:
                trial = adv + sign * noise
                trial_img = TO_PIL(trial.squeeze(0).clamp(0,1))
                preds = self.model.predict(trial_img)
                boxes = [b for b,l in zip(preds['boxes'], preds['labels']) if not target_label or l==target_label]
                scores.append(-compute_ap(boxes, gt_boxes))
            grad = (scores[1] - scores[0])/(2*self.sigma) * noise
            adv = (adv + self.alpha*grad).clamp(0,1)
        return TO_PIL(adv.squeeze(0))

# --- Evaluation & Transferability ---
def evaluate_attack(entry: Dict, model: VisionModel, attack: Attack, target_label: Optional[str]=None) -> Dict:
    image = Image.open(entry['image']).convert("RGB")
    gt_boxes = entry['gt_boxes']
    orig_tensor = PREPROCESS(image).unsqueeze(0)

    preds_before = model.predict(image)
    ap_before = compute_ap(preds_before['boxes'], gt_boxes)

    adv_image = attack.run(image, gt_boxes, target_label)
    adv_tensor = PREPROCESS(adv_image).unsqueeze(0)

    preds_after = model.predict(adv_image)
    ap_after = compute_ap(preds_after['boxes'], gt_boxes)
    delta_ap = ap_before - ap_after
    perturb = (adv_tensor - orig_tensor).abs().mean().item()

    vis_orig = draw_boxes(image.copy(), preds_before['boxes'], 'green', 'Before')
    vis_adv = draw_boxes(adv_image.copy(), preds_after['boxes'], 'red', 'After')
    vis_orig.save(f'logs/vis_{os.path.basename(entry["image"])}_before.png')
    vis_adv.save(f'logs/vis_{os.path.basename(entry["image"])}_after.png')

    result = {
        "image": os.path.basename(entry['image']),
        "ap_before": ap_before,
        "ap_after": ap_after,
        "delta_ap": delta_ap,
        "perturbation": perturb
    }
    log_to_csv("results.csv", result)
    log_to_jsonl("results.jsonl", result)
    return result

def transferability_check(entry: Dict, models: Dict[str, VisionModel], attack: Attack) -> Dict[str, float]:
    image = Image.open(entry['image']).convert("RGB")
    gt_boxes = entry['gt_boxes']
    adv_image = attack.run(image, gt_boxes, None)
    results = {}
    for name, mdl in models.items():
        ap = compute_ap(mdl.predict(adv_image)['boxes'], gt_boxes)
        results[name] = ap
    return results

# --- Main CLI Entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)

    if not os.path.exists(cfg['dataset']):
        raise FileNotFoundError(f"Dataset file not found: {cfg['dataset']}")

    # ✅ FIX: Load dataset from YAML
    with open(cfg['dataset'], 'r') as f:
        dataset = yaml.safe_load(f)

    model_primary = GeminiWrapper(cfg['api_url'], cfg['api_key'])
    extra_models = {
        name: GeminiWrapper(m['api_url'], m['api_key']) for name, m in cfg.get('transfer_models', {}).items()
    }

    atk_type = cfg.get('attack_type', 'nes')
    if atk_type == 'nes':
        attack = NESAttack(model_primary, **cfg.get('nes_params', {}))
    else:
        raise ValueError(f"Unsupported attack type: {atk_type}")

    for entry in tqdm(dataset):
        try:
            result = evaluate_attack(entry, model_primary, attack, entry.get('target_label'))
            if extra_models:
                trans = transferability_check(entry, extra_models, attack)
                logging.info(f"Transferability for {entry['image']}: {trans}")
        except Exception as e:
            logging.error(f"Failed on {entry['image']}: {e}")
