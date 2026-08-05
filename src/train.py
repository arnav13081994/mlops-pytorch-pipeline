import json
import os
from pathlib import Path
import torch
import torch.nn as nn
import yaml
from src.dataset import get_dataloaders
from src.model import get_model

device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    total_loss = 0.0
    correct = 0
    total = 0

    # Set model to training mode
    model.train()

    for inputs, targets in loader:

        inputs, targets = inputs.to(device), targets.to(device)

        # Clear out previous gradients
        optimizer.zero_grad()

        # Compute predictions
        outputs = model(inputs)
        # Compute loss
        loss = criterion(outputs, targets)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    total_loss = 0.0
    correct = 0
    total = 0

    # set model to evaluation mode
    model.eval()

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute predictions
        outputs = model(inputs)
        # Compute loss
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def get_config_path() -> Path:
    # Explicit override via env var takes precedence over the mounted/baked-in default
    env_path = os.environ.get("CONFIG_PATH")
    if env_path:
        return Path(env_path)

    config_path = Path("/app/configs/training_config.yaml")
    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")
    return config_path


def main():
    # Load configuration
    config_path = get_config_path()
    config = load_config(str(config_path))

    #  Define Hyperparameters
    patience = config["training"]["early_stopping_patience"]
    data_dir = config["data"]["data_dir"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]
    epochs = config["training"]["epochs"]
    checkpoint_dir = config["output"]["checkpoint_dir"]
    model_name = config["output"]["model_name"]

    if not device:
        raise RuntimeError(
            "No suitable device found. Please ensure you have a GPU or CUDA or CPU installed."
        )

    # Create Checkpoint Directory if it doesn't exist
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get model
    model = get_model().to(device)

    # Get Data
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
    )
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()

    # Training Loop with Early Stopping
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }

        print(json.dumps(log_entry), flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = checkpoint_dir / model_name

            # save model checkpoint
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                save_path,
            )
            print(
                json.dumps({"event": "checkpoint_saved", "path": str(save_path)}),
                flush=True,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    json.dumps({"event": "early_stopping", "epoch": epoch + 1}),
                    flush=True,
                )
                break

    print(
        json.dumps(
            {"event": "training_complete", "best_val_loss": round(best_val_loss, 4)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
