import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet
from dataloader import ProstateDataset
from utils import ToTensor, plot_images, visualize_predictions, save_model, load_model
import os
from torch.utils.data import random_split
import matplotlib.pyplot as plt

# Set the environment variable to use GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def train_model(model, dataloader, valid_loader, criterion, optimizer, num_epochs=25):
    model.train()  # Set the model to training mode

    train_losses = []
    valid_losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for i, batch in enumerate(dataloader):
            # Get the inputs and labels from the data loader
            inputs = batch['image'].to(device)
            labels = batch['mask'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

        train_losses.append(running_loss / len(dataloader))
        print(f'Epoch {epoch + 1}, Training Loss: {train_losses[-1]:.4f}')

        # Validation phase
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                inputs = batch['image'].to(device)
                labels = batch['mask'].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()

        valid_losses.append(valid_loss / len(valid_loader))
        print(f'Epoch {epoch + 1}, Validation Loss: {valid_losses[-1]:.4f}')
        model.train()  # Set the model back to training mode

    return train_losses, valid_losses

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")

    # Setting up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize your dataset and dataloader
    dataset = ProstateDataset(img_dir='./data/img', mask_dir='./data/mask', transform=ToTensor())

    # Define the sizes for train, validation, and test sets
    train_size = int(0.7 * len(dataset))  # 70% for training
    valid_size = int(0.2 * len(dataset))  # 20% for validation
    test_size = len(dataset) - train_size - valid_size  # Remaining 10% for testing

    # Split the dataset
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # print(len(train_dataset), len(valid_dataset), len(test_dataset))

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # for i, batch in enumerate(dataloader):
    #     print(batch['image'].shape)
    #     print(batch['mask'].shape)
    #     break

    # Initialize model
    model = UNet(n_channels=1, n_classes=1).to(device)

    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_loss, valid_loss = train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=25)

    # # Visualize training and validation loss
    plt.figure(figsize=(10, 10))
    plt.plot(train_loss, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('training_loss_plot.png')

    plt.figure(figsize=(10, 10))
    plt.plot(valid_loss, label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('validation_loss_plot.png')

    # save model
    save_model(model, 'model_unet.pth')

    # load model
    # load_model(model, 'model_unet.pth', device)

    # # Visualize predictions
    # visualize_predictions(test_loader, model, device, num_vis=2)