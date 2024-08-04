import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet
from dataloader import ProstateDataset
from utils import ToTensor, plot_images, visualize_predictions, save_model, load_model

def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    model.train()  # Set the model to training mode
    
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

    print('Finished Training')

# Setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Initialize your dataset and dataloader
dataset = ProstateDataset(img_dir='./data/img', mask_dir='./data/mask', transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# Initialize model
model = UNet(n_channels=1, n_classes=1).to(device)

# Loss function
criterion = torch.nn.BCEWithLogitsLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, dataloader, criterion, optimizer, num_epochs=25)

# save model
save_model(model, 'model_unet.pth')

# load model
# load_model(model, 'model_unet.pth', device)

# Visualize predictions
visualize_predictions(dataloader, model, device, num_vis=2)