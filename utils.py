from torchvision import transforms
import matplotlib.pyplot as plt
import random
import torch

# Custom transform to convert PIL images to tensors
class ToTensor:
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        return {'image': image, 'mask': mask}

# Plot images
def plot_images(data_loader, num_images=4):
    fig, ax = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
    
    for i in range(num_images):
        # Randomly select a batch and an image within that batch
        batch = next(iter(data_loader))
        index = random.randint(0, batch['image'].size(0) - 1)
        image = batch['image'][index].squeeze()  # Remove channel dim because it's grayscale
        mask = batch['mask'][index].squeeze()  # Remove channel dim because it's grayscale

        ax[i, 0].imshow(image.numpy(), cmap='gray')
        ax[i, 0].set_title('Prostate Image')
        ax[i, 0].axis('off')

        ax[i, 1].imshow(mask.numpy(), cmap='gray')
        ax[i, 1].set_title('Mask Image')
        ax[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Visualize predictions
def visualize_predictions(dataloader, model, device, num_vis=2):
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():  # Inference without gradient computation
        for batch in dataloader:
            images = batch['image'].to(device)
            true_masks = batch['mask'].to(device)
            preds = torch.sigmoid(model(images))
            preds = (preds > 0.5).float()  # Threshold predictions
            
            images = images.cpu()
            true_masks = true_masks.cpu()
            preds = preds.cpu()
            
            fig, ax = plt.subplots(nrows=num_vis, ncols=3, figsize=(15, 5 * num_vis))
            for i in range(num_vis):
                ax[i, 0].imshow(images[i].squeeze(), cmap='gray')
                ax[i, 0].set_title('Original Image')
                ax[i, 0].axis('off')
                
                ax[i, 1].imshow(true_masks[i].squeeze(), cmap='gray')
                ax[i, 1].set_title('Ground Truth Mask')
                ax[i, 1].axis('off')
                
                ax[i, 2].imshow(preds[i].squeeze(), cmap='gray')
                ax[i, 2].set_title('Predicted Mask')
                ax[i, 2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            break

# Save and load model
def save_model(model, path):
    """
    Save the model's state dictionary to a file.
    
    Args:
        model (torch.nn.Module): The model to save.
        path (str): Path where the model will be saved.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    """
    Load a model's state dictionary from a file.
    
    Args:
        model (torch.nn.Module): The model for which the state dictionary will be loaded.
        path (str): Path to the model's saved state dictionary.
        device (torch.device): The device to load the model onto.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")