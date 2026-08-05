from pathlib import Path

import torch

from src.model import get_model

from src.train import device, get_config_path, load_config
from contextlib import asynccontextmanager
import io

import torchvision.transforms as transforms
from torchvision.transforms import v2

from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image

# Global dictionary to store the loaded model
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown lifecycle.
    Loads the PyTorch model into memory once.
    """
    # Get model
    model = get_model()

    # Load configuration
    config_path = get_config_path()

    config = load_config(str(config_path))
    save_path = (
        Path(config["output"]["checkpoint_dir"]) / config["output"]["model_name"]
    )

    # load only model weights
    checkpoint = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Cache resources
    ml_models["model"] = model
    ml_models["device"] = device
    yield
    # Clean up and free memory on shutdown
    ml_models.clear()


# Initialize FastAPI with the lifespan handler
app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    if not ml_models.get("model"):
        raise HTTPException(
            status_code=500, detail={"status": "error", "message": "Model not loaded"}
        )
    return {"status": "ok", "message": "Model is loaded and ready for inference"}


# Define Image Transformations
transform = transforms.Compose(
    [
        v2.ToImage(),
        v2.Resize((28, 28)),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Converts raw image bytes into a preprocessed PyTorch tensor."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(ml_models["device"])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format")


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()

    input_tensor = preprocess_image(image_bytes)

    with torch.no_grad():
        outputs = ml_models["model"](input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        return {
            "probabilities": probabilities.to("cpu").numpy().tolist(),
        }
