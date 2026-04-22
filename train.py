import torch
import torch.nn.functional as F
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import functional
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, SEED, set_seed, NUM_CHANNELS
from dataset import get_dataloaders
from model import SpikFormer

def train(model, device, train_loader, optimizer, epoch, writer, num_classes, loss_fn=F.mse_loss):
    model.train()
    total_loss = 0.0
    train_acc = 0.0
    data_count = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        functional.reset_net(model)

        target_one_hot = F.one_hot(target, num_classes=num_classes).float()

        output = model(data)
        loss = loss_fn(output, target_one_hot)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.numel()
        train_acc += (output.argmax(dim=1) == target).sum().item()
        data_count += target.numel()
        
    avg_loss = total_loss / len(train_loader.dataset)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc / data_count, epoch)
    print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")

def test(model, device, test_loader, epoch, writer, num_classes, loss_fn=F.mse_loss):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            functional.reset_net(model)

            output = model(data)
            test_loss += loss_fn(output, F.one_hot(target, num_classes=num_classes).float()).item() * target.numel()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", correct / len(test_loader.dataset), epoch)
    print(
        f"====> Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.2f}%)"
    )

def main():
    set_seed(SEED)
    
    print(f"Using device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}, Number of epochs: {NUM_EPOCHS}")

    train_loader, test_loader, num_classes, class_names = get_dataloaders()
    print(f"Number of classes: {num_classes}")
    
    model = SpikFormer(num_classes=num_classes, num_channels=NUM_CHANNELS, use_cupy=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    model_size = num_params * 4 / (1024 * 1024) 
    print(f"Model size: {model_size:.2f} MB")

    writer = SummaryWriter(log_dir=f"runs/spikformer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs"):
        train(model, DEVICE, train_loader, optimizer, epoch, writer, num_classes)
        test(model, DEVICE, test_loader, epoch, writer, num_classes)

    writer.close()

if __name__ == "__main__":
    main()