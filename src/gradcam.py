from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


CLASS_NAMES = [
    "Actinic keratoses / intraepithelial carcinoma",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]

IMAGE_SIZE = 224
NUM_CLASSES = len(CLASS_NAMES)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "v4_efficientnet_b0_dermamnist_224.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class GradCAM:
    """
    Grad-CAM implementation for EfficientNet-B0.

    The final convolutional feature layer is used as the target layer.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> tuple[np.ndarray, int, float]:
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        probabilities = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(probabilities, dim=1).item())

        score = logits[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations.")

        # Global-average-pool gradients over spatial dimensions.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps.
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()

        # Normalize heatmap to [0, 1].
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        confidence = float(probabilities[0, target_class].item())

        return cam, target_class, confidence

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def load_model() -> torch.nn.Module:
    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def generate_gradcam(
    image: Image.Image,
    target_class: Optional[int] = None,
) -> tuple[np.ndarray, int, float]:
    """
    Generate a normalized Grad-CAM heatmap.

    Returns:
        heatmap: numpy array with values in [0, 1]
        predicted_class: class index
        confidence: softmax confidence
    """

    model = load_model()

    # EfficientNet-B0's final convolutional feature block.
    target_layer = model.features[-1]

    gradcam = GradCAM(model, target_layer)

    try:
        image_rgb = image.convert("RGB")
        input_tensor = _transform(image_rgb).unsqueeze(0).to(DEVICE)

        heatmap, predicted_class, confidence = gradcam.generate(
            input_tensor,
            target_class=target_class,
        )

        return heatmap, predicted_class, confidence

    finally:
        gradcam.close()


def create_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Create a visual overlay of the Grad-CAM heatmap on the original image.
    """

    original = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert heatmap to uint8.
    heatmap_uint8 = np.uint8(np.clip(heatmap, 0, 1) * 255)

    # Use a perceptual matplotlib colormap.
    import matplotlib

    cmap = matplotlib.colormaps.get_cmap("jet")
    colored = cmap(heatmap_uint8 / 255.0)
    colored_rgb = np.uint8(colored[:, :, :3] * 255)

    heatmap_image = Image.fromarray(colored_rgb).convert("RGB")

    overlay = Image.blend(
        original,
        heatmap_image,
        alpha=float(alpha),
    )

    return overlay
