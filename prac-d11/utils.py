import torch
import matplotlib.pyplot as plt


def save_model(model, path):
    """
    Save model weights.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device):
    """
    Load model weights.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")


def count_parameters(model):
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_loss(losses):

    plt.figure(figsize=(8,5))
    plt.plot(losses)

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(True)
    plt.show()


def epoch_time(start_time, end_time):

    elapsed = end_time - start_time

    mins = int(elapsed / 60)
    secs = int(elapsed - mins * 60)

    return mins, secs