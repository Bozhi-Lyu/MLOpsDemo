import click
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from model import BasicCNN
from data import corruptmnist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=256, help="batch size to use for training")
@click.option("--num_epochs", default=20, help="number of epochs to train for")
def train(lr, batch_size, num_epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    model = BasicCNN.to(device)

    num_epochs = 20
    # acc_threshold = 0.85 
    training_losses = []

    train_dataset, _ = corruptmnist()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()  
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(num_epochs):
        loss_in_epoch = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            with torch.set_grad_enabled(True):
                outputs = model(images)
                # print(outputs.shape)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            loss_in_epoch.append(loss.item())
        training_losses.append(sum(loss_in_epoch)/len(loss_in_epoch))
        print(f"Epoch {epoch} Loss {training_losses[epoch]}")

    # Plot the training curve
    steps = np.arange(len(training_losses))
    plt.plot(steps, training_losses)
    plt.xlabel('Training Step')
    plt.ylabel('Training Loss')
    plt.title('Training Curve')
    plt.savefig("Training Loss VS Step")
    print(f"Training curve saved.")

    torch.save(model, "model.pt")



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_dataset = corruptmnist()
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    model.eval()

    pred = []
    truelabels = []
    with torch.no_grad():
        for batch in test_dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            pred.append(outputs.argmax(dim=1).cpu())
            truelabels.append(labels.cpu())
    
    pred = torch.cat(pred, dim = 0)
    truelabels = torch.cat(truelabels, dim = 0)

    print(f"Test Accuracy:", (pred == truelabels).float().mean().item())


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()