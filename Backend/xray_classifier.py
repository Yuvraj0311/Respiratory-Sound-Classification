"""
xray_classifier.py - Chest X-ray classification with a fine-tuned ResNet18 model.

The model architecture mirrors the torchvision ResNet18 used in the training
notebook, but is implemented with plain PyTorch modules so inference does not
require torchvision at runtime.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18XRay(nn.Module):
    """ResNet18 with the notebook's custom classifier head."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, blocks=2)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512 * BasicBlock.expansion, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers: List[nn.Module] = [
            BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)
        ]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class LungXRayClassifier:
    """Chest X-ray image classifier with notebook-matched preprocessing."""

    CLASS_NAMES = ["Lung Opacity", "Normal", "Viral Pneumonia"]
    KNOWLEDGE_DISEASE_MAP = {
        "Lung Opacity": "Pneumonia",
        "Normal": "Healthy",
        "Viral Pneumonia": "Pneumonia",
    }
    SEVERITY_THRESHOLDS = {
        "high": 0.90,
        "moderate": 0.75,
        "mild": 0.60,
    }

    def __init__(
        self,
        model_path: str = "models/resnet18_finetuned.pth",
        image_size: Tuple[int, int] = (224, 224),
        device: Optional[str] = None,
    ):
        self.model_path = self._resolve_model_path(model_path)
        self.image_size = image_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ResNet18XRay(num_classes=len(self.CLASS_NAMES)).to(self.device)
        self.load_model()

        logger.info("LungXRayClassifier initialized")
        logger.info("Model path: %s", self.model_path)
        logger.info("Device: %s", self.device)

    def _resolve_model_path(self, model_path: str) -> Path:
        path = Path(model_path)
        if path.exists():
            return path

        backend_relative = Path(__file__).resolve().parent / model_path
        if backend_relative.exists():
            return backend_relative

        raise FileNotFoundError(f"X-ray model file not found: {model_path}")

    def load_model(self) -> None:
        try:
            try:
                checkpoint = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except TypeError:
                checkpoint = torch.load(self.model_path, map_location=self.device)

            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            logger.info("X-ray model loaded successfully")
        except Exception as exc:
            logger.error("Error loading X-ray model: %s", exc)
            raise

    def preprocess_for_inference(self, image_path: str) -> torch.Tensor:
        try:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image)
                image = image.convert("L")
                try:
                    resample = Image.Resampling.BILINEAR
                except AttributeError:
                    resample = Image.BILINEAR
                image = image.resize(self.image_size, resample=resample)
                image_array = np.asarray(image, dtype=np.float32) / 255.0

            # Match notebook transform:
            # Resize -> Grayscale(num_output_channels=3) -> ToTensor -> ImageNet normalize.
            image_array = np.stack([image_array, image_array, image_array], axis=0)
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            image_array = (image_array - mean) / std

            tensor = torch.from_numpy(image_array).unsqueeze(0).to(self.device)
            return tensor
        except Exception as exc:
            logger.error("Error preprocessing X-ray image %s: %s", image_path, exc)
            raise ValueError("Invalid or unreadable chest X-ray image") from exc

    def predict(self, image_path: str) -> Dict[str, Any]:
        logger.info("Starting X-ray prediction for: %s", image_path)

        features = self.preprocess_for_inference(image_path)
        with torch.inference_mode():
            logits = self.model(features)
            probabilities_tensor = torch.softmax(logits, dim=1).squeeze(0).cpu()

        probabilities = {
            class_name: float(probabilities_tensor[idx])
            for idx, class_name in enumerate(self.CLASS_NAMES)
        }
        predicted_idx = int(torch.argmax(probabilities_tensor).item())
        finding = self.CLASS_NAMES[predicted_idx]
        confidence = probabilities[finding]

        top_predictions = sorted(
            [
                {"finding": class_name, "confidence": score}
                for class_name, score in probabilities.items()
            ],
            key=lambda item: item["confidence"],
            reverse=True,
        )

        result = {
            "tool": "lung_xray_classifier",
            "modality": "chest_xray",
            "finding": finding,
            "knowledge_disease": self.KNOWLEDGE_DISEASE_MAP.get(finding, finding),
            "confidence": confidence,
            "severity": self._calculate_severity(confidence, finding),
            "probabilities": probabilities,
            "top_predictions": top_predictions,
            "clinical_note": self._clinical_note(finding, confidence),
        }

        logger.info("X-ray prediction: %s (confidence %.4f)", finding, confidence)
        return result

    def _calculate_severity(self, confidence: float, finding: str) -> str:
        if finding == "Normal":
            return "none"
        if confidence >= self.SEVERITY_THRESHOLDS["high"]:
            return "high"
        if confidence >= self.SEVERITY_THRESHOLDS["moderate"]:
            return "moderate"
        if confidence >= self.SEVERITY_THRESHOLDS["mild"]:
            return "mild"
        return "uncertain"

    def _clinical_note(self, finding: str, confidence: float) -> str:
        confidence_text = "high" if confidence >= 0.90 else "moderate" if confidence >= 0.75 else "limited"
        if finding == "Normal":
            return f"The uploaded image was classified as normal with {confidence_text} model confidence."
        if finding == "Viral Pneumonia":
            return (
                "The uploaded image pattern is consistent with viral pneumonia screening output; "
                "clinical symptoms and radiologist review are still required."
            )
        return (
            "The uploaded image shows a lung opacity screening output; opacity is a radiographic "
            "finding that needs clinical correlation and radiologist review."
        )

    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(param.numel() for param in self.model.parameters())
        trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)

        return {
            "model_path": str(self.model_path),
            "architecture": "ResNet18 with custom 3-class classifier head",
            "device": str(self.device),
            "image_size": self.image_size,
            "classes": self.CLASS_NAMES,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "preprocessing": {
                "color_mode": "grayscale converted to 3 channels",
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
            },
        }
