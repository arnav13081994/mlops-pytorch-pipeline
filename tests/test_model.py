import torch

from src.model import NeuralNetwork, get_model


def test_get_model_returns_neural_network():
    model = get_model()
    assert isinstance(model, NeuralNetwork)


def test_forward_output_shape():
    model = get_model()
    model.eval()
    batch = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (4, 10)
