from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_LABEL_MAP = {
    "Normal": "a normal driving scene",
    "RoadAccidents": "a traffic accident on the road",
}


@dataclass
class DemoConfig:
    src_root: str
    weights_path: str
    val_list: Optional[str] = None
    fixed_threshold: Optional[float] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    embed_dim: int = 512
    visual_length: int = 256
    visual_width: int = 512
    visual_head: int = 1
    visual_layers: int = 2
    attn_window: int = 8
    prompt_prefix: int = 16
    prompt_postfix: int = 10
    classes_num: int = 2
    peft_blocks: int = 3
    init_temp: float = 0.07

    crop_type: int = 10
    downsample_if_fps_ge: float = 30.0
    target_fps: float = 25.0
    stride: int = 1

    label_map: Dict[str, str] = None

    def __post_init__(self):
        if self.label_map is None:
            self.label_map = dict(DEFAULT_LABEL_MAP)


class DemoSystem:
    def __init__(self, config: DemoConfig):
        self.cfg = config
        self.src_root = self._setup_project_imports(config.src_root)
        self.device = torch.device(config.device)

        from clip import clip as thesis_clip  # local package inside src
        from crop import image_crop
        from model_with_MLP import CLIPVAD
        from torch.utils.data import DataLoader
        from utils.dataset import Dataset
        from utils.tools import get_batch_mask, get_prompt_text, process_split
        from test_joint import evaluate_dataset

        self._clip_module = thesis_clip
        self._image_crop = image_crop
        self._CLIPVAD = CLIPVAD
        self._DataLoader = DataLoader
        self._Dataset = Dataset
        self._get_batch_mask = get_batch_mask
        self._get_prompt_text = get_prompt_text
        self._process_split = process_split
        self._evaluate_dataset = evaluate_dataset

        self.prompt_text = self._get_prompt_text(self.cfg.label_map)
        self.model = self._load_model()
        self.clip_model, self.clip_preprocess = self._clip_module.load("ViT-B/16", self.device)
        self.threshold, self.validation_info = self._resolve_threshold()

    @staticmethod
    def _setup_project_imports(src_root: str) -> Path:
        src = Path(src_root).resolve()
        if not (src / "model_with_MLP.py").exists():
            raise FileNotFoundError(
                f"src_root non valido: {src}. Deve puntare alla cartella che contiene model_with_MLP.py"
            )
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        return src

    def _load_model(self):
        cfg = self.cfg
        model = self._CLIPVAD(
            cfg.classes_num,
            cfg.embed_dim,
            cfg.visual_length,
            cfg.visual_width,
            cfg.visual_head,
            cfg.visual_layers,
            cfg.attn_window,
            cfg.prompt_prefix,
            cfg.prompt_postfix,
            self.device,
            peft_blocks=cfg.peft_blocks,
            init_temp=cfg.init_temp,
        )

        weights_path = Path(cfg.weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {weights_path}")

        state = torch.load(str(weights_path), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        missing, unexpected = model.load_state_dict(state, strict=False)
        model.to(self.device)
        model.eval()

        self.model_load_info = {
            "weights_path": str(weights_path),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        }
        return model

    def _resolve_threshold(self) -> Tuple[float, Dict[str, Any]]:
        if self.cfg.fixed_threshold is not None:
            return float(self.cfg.fixed_threshold), {"source": "fixed", "threshold": float(self.cfg.fixed_threshold)}

        if not self.cfg.val_list:
            return 0.5, {"source": "default", "threshold": 0.5}

        val_list = Path(self.cfg.val_list)
        if not val_list.exists():
            return 0.5, {"source": "default_missing_val", "threshold": 0.5}

        val_dataset = self._Dataset(self.cfg.visual_length, str(val_list), True, self.cfg.label_map)
        val_loader = self._DataLoader(val_dataset, batch_size=1, shuffle=False)

        results = self._evaluate_dataset(
            model=self.model,
            dataloader=val_loader,
            maxlen=self.cfg.visual_length,
            prompt_text=self.prompt_text,
            device=self.device,
            fixed_threshold=None,
            desc="VAL_DEMO",
        )
        return float(results["thr"]), {
            "source": "validation",
            "threshold": float(results["thr"]),
            "auc": float(results["auc"]),
            "ap": float(results["ap"]),
            "f1": float(results["F1"]),
        }

    def _video_sampling_step(self, fps: float) -> int:
        if fps and fps >= self.cfg.downsample_if_fps_ge and self.cfg.target_fps > 0:
            return max(1, int(round(fps / self.cfg.target_fps)))
        return 1

    def extract_features_from_video(self, video_path: str) -> Tuple[np.ndarray, List[float], Dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Impossibile aprire il video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

        sample_step = self._video_sampling_step(fps)
        stride = max(1, int(self.cfg.stride))

        features: List[np.ndarray] = []
        timestamps: List[float] = []
        kept_idx = 0
        raw_idx = 0

        with torch.no_grad():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if raw_idx % sample_step != 0:
                    raw_idx += 1
                    continue

                if kept_idx % stride == 0:
                    rgb_crop = self._image_crop(frame, self.cfg.crop_type)
                    pil_img = Image.fromarray(rgb_crop)
                    tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
                    feat = self.clip_model.encode_image(tensor).detach().cpu().numpy()[0].astype(np.float32)
                    features.append(feat)
                    timestamps.append(raw_idx / fps)

                kept_idx += 1
                raw_idx += 1

        cap.release()

        if len(features) == 0:
            raise RuntimeError("Nessuna feature estratta dal video. Controlla fps/stride o il file caricato.")

        meta = {
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration,
            "sampling_step_frames": sample_step,
            "feature_stride_frames": stride,
            "num_features": len(features),
        }
        return np.stack(features, axis=0), timestamps, meta

    def _build_split_lengths(self, total_len: int, num_splits: int) -> torch.Tensor:
        split_lengths = []
        remaining = int(total_len)
        for _ in range(int(num_splits)):
            cur = min(self.cfg.visual_length, max(remaining, 0))
            split_lengths.append(cur)
            remaining -= cur
        return torch.tensor(split_lengths, dtype=torch.long, device=self.device)

    @staticmethod
    def _best_window_indices(full_scores: torch.Tensor, chunk_size: int) -> Tuple[int, int]:
        L = int(full_scores.numel())
        if L <= chunk_size:
            return 0, max(0, L - 1)
        windows = full_scores.unfold(0, chunk_size, 1)
        means = windows.mean(dim=1)
        start = int(torch.argmax(means).item())
        end = start + chunk_size - 1
        return start, end

    @staticmethod
    def _estimate_feature_duration(timestamps: List[float], fallback: float = 0.64) -> float:
        if len(timestamps) >= 2:
            diffs = np.diff(np.asarray(timestamps, dtype=np.float32))
            diffs = diffs[diffs > 0]
            if diffs.size > 0:
                return float(np.median(diffs))
        return float(fallback)

    def predict_from_features(self, features: np.ndarray, timestamps: List[float]) -> Dict[str, Any]:
        split_feat, total_len = self._process_split(features.astype(np.float32), self.cfg.visual_length)

        # Se il video è più corto di visual_length, process_split restituisce (T, C)
        # ma il modello vuole sempre (B, T, C)
        if split_feat.ndim == 2:
            split_feat = np.expand_dims(split_feat, axis=0)

        visual = torch.tensor(split_feat, dtype=torch.float32, device=self.device)
        split_lengths = self._build_split_lengths(total_len=int(total_len), num_splits=visual.size(0))
        padding_mask = self._get_batch_mask(split_lengths, self.cfg.visual_length).to(self.device)

        with torch.no_grad():
            _, _, logits2 = self.model(visual, padding_mask, self.prompt_text, split_lengths)
            video_logits = self.model.forward_video_logits_full_video(logits2, split_lengths)
            video_prob = float(torch.sigmoid(video_logits).item())

            frame_scores = self.model.video_score_aggregator.extract_anomaly_scores(logits2)
            full_scores = self.model.video_score_aggregator.rebuild_full_video_scores(frame_scores, split_lengths)
            full_scores_np = full_scores.detach().cpu().numpy().astype(np.float32)

        start_idx, end_idx = self._best_window_indices(full_scores, self.cfg.visual_length)
        feat_dur = self._estimate_feature_duration(timestamps)

        start_sec = float(timestamps[min(start_idx, len(timestamps) - 1)]) if timestamps else 0.0
        end_anchor = float(timestamps[min(end_idx, len(timestamps) - 1)]) if timestamps else 0.0
        end_sec = end_anchor + feat_dur

        pred_label = int(video_prob >= self.threshold)
        pred_name = "RoadAccidents" if pred_label == 1 else "Normal"

        topk = min(5, len(full_scores_np))
        top_idx = np.argsort(-full_scores_np)[:topk]
        top_segments = [
            {
                "feature_index": int(i),
                "time_sec": float(timestamps[i]) if i < len(timestamps) else None,
                "score": float(full_scores_np[i]),
            }
            for i in top_idx
        ]

        timeline = [
            {
                "feature_index": int(i),
                "time_sec": float(timestamps[i]) if i < len(timestamps) else None,
                "score": float(full_scores_np[i]),
            }
            for i in range(len(full_scores_np))
        ]

        return {
            "probability": video_prob,
            "threshold": float(self.threshold),
            "prediction": pred_name,
            "prediction_binary": pred_label,
            "best_window": {
                "start_feature": int(start_idx),
                "end_feature": int(end_idx),
                "start_sec": start_sec,
                "end_sec": end_sec,
            },
            "top_segments": top_segments,
            "timeline": timeline,
            "num_valid_features": int(total_len),
        }

    def predict_video(self, video_path: str) -> Dict[str, Any]:
        features, timestamps, video_meta = self.extract_features_from_video(video_path)
        pred = self.predict_from_features(features, timestamps)
        return {
            "video_path": str(video_path),
            "video_meta": video_meta,
            "model_load_info": self.model_load_info,
            "threshold_info": self.validation_info,
            "prediction": pred,
        }


def load_config(config_path: str) -> DemoConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DemoConfig(**data)


def save_temp_video(file_bytes: bytes, suffix: str = ".mp4") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(file_bytes)
        return f.name


def to_pretty_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def config_to_dict(cfg: DemoConfig) -> Dict[str, Any]:
    return asdict(cfg)
