import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from utils.scheduler import exponential_decay
from utils.data_processor import create_flower_dataloaders


# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Basic settings
data_root = "./flowers"  # Path to the dataset root directory
model_save_path = "./model/Bestmodel_diffusion.pkl"  # Path to save the trained model
vis_root = "./vis"

# Hyperparameters (adjustable)
batch_size = 16  # Batch size for training and validation
num_epochs = 1000  # Number of training epochs
img_channel = 3
img_width, img_height = 24, 24

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
print(f"using device {device}")

# Diffusion process settings (adjustable)
num_steps = 300  # Number of steps in the diffusion process
beta_min = 1e-4  # Minimum value of the beta schedule
beta_max = 5e-2  # Maximum value of the beta schedule

# Initialize diffusion parameters
# These parameters define the noise schedule and are crucial for the forward/reverse process
betas = torch.linspace(-6, 6, num_steps, device=device) # Linearly spaced values for noise schedule
betas = torch.sigmoid(betas) * (beta_max - beta_min) + beta_min  # Scale sigmoid outputs
alphas = 1 - betas  # Compute alpha values from beta
alphas_sqrt = torch.sqrt(alphas)  # Square root of alphas
alphas_bar = torch.cumprod(alphas, 0)  # Cumulative product of alphas over steps
alphas_bar_sqrt = torch.sqrt(alphas_bar)  # Square root of cumulative alphas
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_bar)  # Square root of 1 - cumulative alphas


# Define the diffusion model
class Diffuser(nn.Module):
    def __init__(self, n_steps, img_channels=3):
        """
        Initialize a diffusion model. The model predicts the added noise at each time step.

        Args:
          - n_steps (int): Number of diffusion steps (t).
          - img_channels (int): Number of image channels (e.g., 3 for RGB images).
        """
        super(Diffuser, self).__init__()
        # TODO: Define an encoder-decoder architecture for epsilon 
        # prediction, and add time step embeddings to condition the 
        # predictions.
        raise NotImplementedError()

    def forward(self, x, t):
        """
        Forward pass of the diffusion model:
        Predict the noise added to x at a given time step t.

        Input:
          - x (torch.Tensor): Noisy input image, shape (batch_size, img_channels, height, width).
          - t (torch.Tensor): Time step indices, shape (batch_size,).
        Output:
          - Predicted noise, shape same as x.
        """
        # TODO: Implement the forward computation for a diffusion model
        raise NotImplementedError()

    def sample(self, shape, n_steps):
        """
        Generate images by iteratively sampling from pure noise.

        Args:
        - shape (tuple): Shape of the generated images (batch_size, img_channels, height, width).
        - n_steps (int): Number of diffusion steps.

        Returns:
        - x_seq (list): List of images at each step of the reverse process.
        """
        self.eval()

        x_t = torch.randn(shape, device=next(self.parameters()).device)
        x_seq = [x_t]
        for t in reversed(range(n_steps)):
            x_t = self.p_theta_sampling(x_t, t)
            x_seq.append(x_t)
        return x_seq

    def p_theta_sampling(self, x, t):
        """
        Estimate x[t-1] given x[t] using the reverse diffusion process.

        Steps:
        1. Predict the noise added to x[t] using the model.
        2. Compute the mean of the reverse process based on the formula.
        3. Optionally add Gaussian noise for stochasticity if t > 0.

        Args:
        - model (Diffuser): The diffusion model.
        - x (torch.Tensor): Image at time step t, shape (batch_size, img_channels, height, width).
        - t (int): Current time step.

        Returns:
        - x_t_minus_1 (torch.Tensor): Estimated image at time step t-1.
        """
        # TODO:
        # Predict noise using the model and compute the mean for the reverse process.
        # Add Gaussian noise if t > 0 to simulate the stochastic nature of the process.
        raise NotImplementedError()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

# Loss function for training
def diffusion_loss_fn(model, x_0):
    """
    Compute the training loss for the diffusion model:
    This function compares the model's predicted noise with the actual noise.

    Steps:
      1. Randomly sample a batch of time steps t.
      2. Add Gaussian noise to x_0 to simulate the forward process.
      3. Predict the noise using the model.
      4. Compute the mean squared error between predicted and actual noise.

    Args:
      - model (Diffuser): The diffusion model.
      - x_0 (torch.Tensor): Original clean image, shape (batch_size, img_channels, height, width).

    Returns:
      - loss (torch.Tensor): Scalar value representing the loss.
    """
    batch_size = x_0.size(0)
    t = torch.randint(0, num_steps, size=(batch_size,), device=device)
    # Randomly sample time steps
    # TODO:
    # Compute coefficients for x_0 and noise at step t.
    # Generate noisy images using the diffusion forward process.
    # Use the model to predict noise and compute MSE loss.
    raise NotImplementedError()

# Sampling function with visualization
def visualize_sampling(model, sample_shape, n_steps, save_path):
    """Visualize the sampling process during training."""
    model.eval()
    x_seq = model.sample(sample_shape, n_steps)
    
    num_shows = 10  # Number of steps to display
    step_interval = n_steps // num_shows
    fig, axes = plt.subplots(1, num_shows, figsize=(15, 5))
    for i, ax in enumerate(axes):
        step = i * step_interval
        image = x_seq[step].squeeze().permute(1, 2, 0).detach().cpu().numpy()
        
        # Normalize the image to [-1, 1]
        image_min, image_max = image.min(), image.max()
        normalized_image = 2 * (image - image_min) / (image_max - image_min) - 1
        
        # Clip to [-1, 1] for visualization (optional)
        normalized_image = normalized_image.clip(-1, 1)
        
        # Shift back to [0, 1] for imshow visualization
        imshow_image = (normalized_image + 1) / 2  # [0, 1]
        
        ax.imshow(imshow_image)
        ax.axis('off')
        ax.set_title(f"Step {step}")
    plt.suptitle(f"Sampling Visualization")
    plt.savefig(save_path)


if __name__ == "__main__":
    # Load data
    training_dataloader, validation_dataloader = create_flower_dataloaders(
        batch_size, data_root, img_width, img_height
    )

    # Initialize model, optimizer, and scheduler
    model = Diffuser(n_steps=num_steps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = exponential_decay(initial_learning_rate=1e-3, decay_rate=0.9, decay_epochs=5)

    sample_shape = (1, img_channel, img_width, img_height)  # Shape for visualization samples

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for images, _ in training_dataloader:
            images = images.to(device).float()
            # normalize images input to [-1, 1]
            images = 2 * images - 1
            loss = diffusion_loss_fn(model, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = sum(train_losses) / len(train_losses)

        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Visualize sampling process every 10 epochs
        if (epoch + 1) % 10 == 0:
            visualize_sampling(model, sample_shape, num_steps, os.path.join(vis_root, f"random_images_diffusion_epoch_{epoch + 1}"))

    model.save(model_save_path)
    visualize_sampling(model, sample_shape, num_steps, os.path.join(vis_root, "random_images_diffusion"))