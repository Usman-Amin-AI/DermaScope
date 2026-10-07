from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights


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

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "v4_efficientnet_b0_dermamnist_224.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():
    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        len(CLASS_NAMES),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


MODEL = load_model()


TRANSFORM = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict(
    image: Image.Image,
    uncertainty_threshold: float = 0.60,
):
    image = image.convert("RGB")

    tensor = TRANSFORM(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    confidence, index = torch.max(
        probabilities,
        dim=0,
    )

    confidence = float(confidence.cpu())
    index = int(index.cpu())

    prediction = CLASS_NAMES[index]

    if confidence < uncertainty_threshold:
        status = "Uncertain"
    else:
        status = "Prediction available"

    probabilities_dict = {
        CLASS_NAMES[i]: float(probabilities[i].cpu())
        for i in range(len(CLASS_NAMES))
    }

    return {
        "prediction": prediction,
        "confidence": confidence,
        "status": status,
        "probabilities": probabilities_dict,
    }
