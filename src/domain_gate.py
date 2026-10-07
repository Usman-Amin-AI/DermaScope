from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "domain_gate_efficientnet_b0.pt"

IMAGE_SIZE = 224
DEFAULT_DOMAIN_THRESHOLD = 0.30
CONFIDENT_DOMAIN_THRESHOLD = 0.60

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b0(weights=None)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        2,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    return model, device


_MODEL = None
_DEVICE = None


def predict_domain(
    image: Image.Image,
    threshold: float = DEFAULT_DOMAIN_THRESHOLD,
) -> dict:
    global _MODEL, _DEVICE

    if _MODEL is None:
        _MODEL, _DEVICE = _load_model()

    image = image.convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        logits = _MODEL(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    non_dermoscopic_probability = float(probabilities[0])
    dermoscopic_probability = float(probabilities[1])

    is_dermoscopic = dermoscopic_probability >= threshold

    return {
        "is_dermoscopic": is_dermoscopic,
        "label": (
            "dermoscopic"
            if is_dermoscopic
            else "non_dermoscopic"
        ),
        "dermoscopic_probability": dermoscopic_probability,
        "non_dermoscopic_probability": non_dermoscopic_probability,
        "threshold": threshold,
    }
