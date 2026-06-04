"""
GradCAM visualization for vision-language models.

This module provides GradCAM (Gradient-weighted Class Activation Mapping)
visualization for understanding which regions of an image are important
for zero-shot predictions.
"""

from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


class GradCAM:
    """
    GradCAM implementation for vision-language models.

    Args:
        model: Vision-language model
        target_layer: Name of the layer to compute GradCAM on

    Example:
        >>> from src.models import create_model
        >>> model = create_model('conch', config_path='config/model_configs.yaml')
        >>> gradcam = GradCAM(model)
        >>> heatmap = gradcam.generate(image, text_prompt)
    """

    def __init__(self, model, target_layer: Optional[str] = None):
        """Initialize GradCAM."""
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self.hooks = []

    def _get_target_layer(self):
        """
        Get the target layer for GradCAM.

        For vision transformers, we typically use the last layer before pooling.
        """
        if self.target_layer:
            return self._get_layer_by_name(self.target_layer)

        # Auto-detect target layer based on model architecture
        if hasattr(self.model.model, 'visual'):
            # CLIP-style models
            visual = self.model.model.visual

            # Try different ViT architectures
            if hasattr(visual, 'transformer') and hasattr(visual.transformer, 'resblocks'):
                # Standard CLIP ViT
                return visual.transformer.resblocks[-1]
            elif hasattr(visual, 'blocks'):
                # timm-style ViT (CONCH uses this)
                return visual.blocks[-1]
            elif hasattr(visual, 'layer4'):
                # ResNet architecture
                return visual.layer4
            elif hasattr(visual, 'trunk') and hasattr(visual.trunk, 'blocks'):
                # Another timm variant
                return visual.trunk.blocks[-1]

        elif hasattr(self.model.model, 'vision_model'):
            # Transformers-style models (PLIP, KEEP)
            vision_model = self.model.model.vision_model
            if hasattr(vision_model, 'encoder') and hasattr(vision_model.encoder, 'layers'):
                return vision_model.encoder.layers[-1]

        # Fallback: try to find a suitable layer by searching all modules
        target = None
        for name, module in self.model.model.named_modules():
            # Look for transformer blocks or ResNet layers
            if any(keyword in name.lower() for keyword in ['block', 'layer', 'resblock']):
                # Skip Identity, LayerNorm, and other utility layers
                if not isinstance(module, (nn.Identity, nn.LayerNorm, nn.Dropout)):
                    target = module

        return target

    def _get_layer_by_name(self, layer_name: str):
        """Get layer by name from model."""
        parts = layer_name.split('.')
        layer = self.model.model
        for part in parts:
            layer = getattr(layer, part)
        return layer

    def _save_gradient(self, grad):
        """Hook to save gradients."""
        self.gradients = grad

    def _save_activation(self, module, input, output):
        """Hook to save activations."""
        # Handle tuple outputs (some transformers layers return tuples)
        if isinstance(output, tuple):
            self.activations = output[0]
        else:
            self.activations = output

    def _register_hooks(self, target_layer):
        """Register forward and backward hooks."""
        # Forward hook
        forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.hooks.append(forward_hook)

        # Backward hook
        if hasattr(target_layer, 'register_full_backward_hook'):
            backward_hook = target_layer.register_full_backward_hook(
                lambda module, grad_in, grad_out: self._save_gradient(grad_out[0])
            )
        else:
            backward_hook = target_layer.register_backward_hook(
                lambda module, grad_in, grad_out: self._save_gradient(grad_out[0])
            )
        self.hooks.append(backward_hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate(
        self,
        image: torch.Tensor,
        text_prompts: List[str],
        target_class: int = 0
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap.

        Args:
            image: Input image tensor [1, C, H, W]
            text_prompts: List of text prompts for zero-shot classification
            target_class: Index of target class to visualize

        Returns:
            GradCAM heatmap as numpy array [H, W]
        """
        # Put model in train mode temporarily to enable gradients
        was_training = self.model.training
        self.model.train()

        # Register hooks
        target_layer = self._get_target_layer()
        if target_layer is None:
            # Fallback to uniform heatmap
            self.model.eval() if not was_training else None
            return np.ones((image.shape[2], image.shape[3]))

        self._register_hooks(target_layer)

        # Ensure image requires grad
        image = image.clone().detach()
        image.requires_grad_(True)

        try:
            # Forward pass with gradients enabled
            # Move image to correct device
            if image.device != self.model.device:
                image = image.to(self.model.device)

            # Get text features (these don't need gradients)
            with torch.no_grad():
                text_features = self.model.encode_text(text_prompts)
                text_features = F.normalize(text_features, dim=-1)

            # Encode image WITH gradients - need to call visual encoder directly
            # We completely bypass the wrapper to avoid torch.no_grad()

            # Try different model architectures
            if hasattr(self.model.model, 'visual'):
                # CLIP-style models (CONCH, PathGen-CLIP, QuiltNet, KEEP)
                # Always use visual encoder directly to ensure gradients flow through hooked layers
                visual_encoder = self.model.model.visual
                image_features = visual_encoder(image)

                # Handle tuple output (some models return (features, intermediate))
                if isinstance(image_features, tuple):
                    image_features = image_features[0]

                # Check if we need projection (for compatibility with text embeddings)
                if hasattr(self.model.model, 'visual_projection'):
                    # Standard CLIP projection (matrix or module)
                    proj = self.model.model.visual_projection
                    if isinstance(proj, (torch.Tensor, nn.Parameter)):
                        image_features = image_features @ proj
                    elif isinstance(proj, nn.Module):
                        image_features = proj(image_features)
                elif hasattr(self.model.model, 'visual_head'):
                    # KEEP-style: has separate visual_head module (not visual.head)
                    image_features = self.model.model.visual_head(image_features)
                elif hasattr(self.model.model.visual, 'head') and not isinstance(self.model.model.visual.head, nn.Identity):
                    # Some ViTs have a head layer
                    image_features = self.model.model.visual.head(image_features)
                # Otherwise features are already in the right dimensional space

            elif hasattr(self.model.model, 'vision_model'):
                # Transformers-style models (PLIP, KEEP)
                # Get vision features
                vision_outputs = self.model.model.vision_model(pixel_values=image)
                image_features = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs.last_hidden_state[:, 0]

                # Apply projection if it exists
                if hasattr(self.model.model, 'visual_projection'):
                    proj = self.model.model.visual_projection
                    # Check if it's a matrix or a module
                    if isinstance(proj, (torch.Tensor, nn.Parameter)):
                        image_features = image_features @ proj
                    elif isinstance(proj, nn.Module):
                        image_features = proj(image_features)

            else:
                raise RuntimeError(f"Cannot identify visual encoder for model type: {type(self.model.model)}")

            # Normalize features
            image_features = F.normalize(image_features, dim=-1)

            # Compute similarity with temperature scaling
            temperature = getattr(self.model.model, 'logit_scale', torch.tensor(1.0)).exp()
            similarity = temperature * (image_features @ text_features.T)

            # Get score for target class
            target_score = similarity[0, target_class]

            # Backward pass
            self.model.zero_grad()
            if image.grad is not None:
                image.grad.zero_()

            target_score.backward(retain_graph=False)

        except Exception as e:
            # Clean up and return uniform heatmap
            self._remove_hooks()
            self.model.eval() if not was_training else None
            raise e

        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations

        # Remove hooks
        self._remove_hooks()

        # Restore model state
        if not was_training:
            self.model.eval()

        # Compute GradCAM
        if gradients is None or activations is None:
            # Fallback: return uniform heatmap
            return np.ones((image.shape[2], image.shape[3]))

        # Handle different gradient shapes
        if gradients.dim() == 2:
            # For ViT: [B, N] where N = num_patches
            # Reshape to spatial grid
            num_patches = gradients.shape[1]
            grid_size = int(np.sqrt(num_patches))
            gradients = gradients.reshape(1, grid_size, grid_size, 1).permute(0, 3, 1, 2)
            activations = activations.reshape(1, grid_size, grid_size, 1).permute(0, 3, 1, 2)
        elif gradients.dim() == 3:
            # For ViT: [B, N, D] where N may include CLS token
            B, N, D = gradients.shape
            grid_size = int(np.sqrt(N))

            # Check if we have CLS token (N != perfect square)
            if grid_size * grid_size != N:
                # N includes CLS token - in ViT, only CLS gradients are non-zero for classification
                # We need to broadcast CLS gradients to all patches
                # Extract CLS token gradients: [B, 1, D]
                cls_gradients = gradients[:, 0:1, :]  # [B, 1, D]

                # Remove CLS from activations to get patch activations
                patch_activations = activations[:, 1:, :]  # [B, N-1, D]
                N = patch_activations.shape[1]
                grid_size = int(np.sqrt(N))

                # Reshape patch activations to spatial grid
                patch_activations = patch_activations.reshape(B, grid_size, grid_size, D).permute(0, 3, 1, 2)  # [B, D, H, W]

                # Broadcast CLS gradients to all spatial locations
                # cls_gradients: [B, 1, D] -> [B, D, 1, 1] -> [B, D, H, W]
                cls_gradients = cls_gradients.transpose(1, 2)  # [B, D, 1]
                cls_gradients = cls_gradients.unsqueeze(-1)  # [B, D, 1, 1]
                gradients = cls_gradients.expand(-1, -1, grid_size, grid_size)  # [B, D, H, W]
                activations = patch_activations
            else:
                # N is already a perfect square (no CLS token)
                try:
                    gradients = gradients.reshape(B, grid_size, grid_size, D).permute(0, 3, 1, 2)
                    activations = activations.reshape(B, grid_size, grid_size, D).permute(0, 3, 1, 2)
                except RuntimeError:
                    # If reshape fails, use global pooling
                    gradients = gradients.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
                    activations = activations.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        elif gradients.dim() == 4:
            # Already in [B, C, H, W] format
            pass
        else:
            # Unexpected shape - use global pooling
            if gradients.dim() > 2:
                gradients = gradients.mean(dim=tuple(range(1, gradients.dim()-1)), keepdim=True).unsqueeze(-1).unsqueeze(-1)
                activations = activations.mean(dim=tuple(range(1, activations.dim()-1)), keepdim=True).unsqueeze(-1).unsqueeze(-1)

        # Compute GradCAM using gradient-weighted activations
        if gradients.dim() == 4 and gradients.shape[2] > 1:
            # For Vision Transformers, compute channel-wise importance
            # Use absolute value to avoid cancellation, then average across spatial dims
            weights = torch.mean(torch.abs(gradients), dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

            # Apply ReLU to activations first to keep only positive contributions
            # This prevents negative activations from canceling out the heatmap
            activations_positive = F.relu(activations)

            # Weight activations by gradient importance
            # cam[i,j] = sum_k |grad_k| * ReLU(activation_k[i,j])
            cam = torch.sum(weights * activations_positive, dim=1, keepdim=True)  # [B, 1, H, W]
        else:
            # Fallback for unexpected shapes or single pixel
            cam = torch.mean(activations, dim=1, keepdim=True) if activations.dim() > 2 else activations

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to image size
        cam = self._resize_cam(cam, (image.shape[2], image.shape[3]))

        return cam

    def _resize_cam(self, cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize CAM to target size."""
        from scipy.ndimage import zoom

        # Calculate zoom factors
        zoom_factors = (target_size[0] / cam.shape[0], target_size[1] / cam.shape[1])

        # Resize
        cam_resized = zoom(cam, zoom_factors, order=1)

        return cam_resized

    def visualize(
        self,
        image: torch.Tensor,
        cam: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Overlay GradCAM heatmap on image.

        Args:
            image: Original image tensor [1, C, H, W]
            cam: GradCAM heatmap [H, W]
            alpha: Transparency of heatmap overlay
            colormap: Matplotlib colormap name

        Returns:
            Visualization as numpy array [H, W, 3]
        """
        # Convert image to numpy
        img_np = image.squeeze().cpu().numpy()

        # Denormalize image (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        # Convert to HWC format
        img_np = np.transpose(img_np, (1, 2, 0))

        # Apply colormap to heatmap
        cmap = cm.get_cmap(colormap)
        cam_colored = cmap(cam)[:, :, :3]  # Remove alpha channel

        # Overlay
        visualization = (1 - alpha) * img_np + alpha * cam_colored
        visualization = np.clip(visualization, 0, 1)

        return visualization

    def save_visualization(
        self,
        visualization: np.ndarray,
        save_path: str,
        title: Optional[str] = None,
        original_image: Optional[np.ndarray] = None,
        heatmap: Optional[np.ndarray] = None
    ):
        """
        Save visualization to file.

        Args:
            visualization: Visualization array [H, W, 3] (overlay)
            save_path: Path to save image
            title: Optional title for the plot
            original_image: Optional original image [H, W, 3]
            heatmap: Optional heatmap [H, W]
        """
        if original_image is not None and heatmap is not None:
            # Create a 3-panel visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            axes[0].imshow(original_image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Heatmap
            im = axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Attention Heatmap')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046)

            # Overlay
            axes[2].imshow(visualization)
            axes[2].set_title('Overlay')
            axes[2].axis('off')

            if title:
                fig.suptitle(title, fontsize=12)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            # Single panel
            plt.figure(figsize=(8, 8))
            plt.imshow(visualization)
            plt.axis('off')
            if title:
                plt.title(title)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()


def generate_gradcam_batch(
    model,
    images: torch.Tensor,
    text_prompts: List[str],
    save_dir: str,
    predictions: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    sample_ids: Optional[List[int]] = None
) -> List[np.ndarray]:
    """
    Generate GradCAM visualizations for a batch of images.

    Args:
        model: Vision-language model
        images: Batch of images [B, C, H, W]
        text_prompts: List of text prompts
        save_dir: Directory to save visualizations
        predictions: Model predictions [B]
        labels: Ground truth labels [B]
        sample_ids: List of sample IDs for naming

    Returns:
        List of GradCAM heatmaps
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    gradcam = GradCAM(model)
    heatmaps = []

    for i in range(images.shape[0]):
        image = images[i:i+1]

        # Generate GradCAM for predicted class
        pred_class = predictions[i].item() if predictions is not None else 0
        cam = gradcam.generate(image, text_prompts, target_class=pred_class)
        heatmaps.append(cam)

        # Create visualization
        vis = gradcam.visualize(image, cam)

        # Create title
        title = f"Sample {sample_ids[i] if sample_ids else i}"
        if predictions is not None and labels is not None:
            pred_label = text_prompts[pred_class]
            true_label = text_prompts[labels[i].item()]
            correct = "✓" if pred_class == labels[i].item() else "✗"
            title += f"\nPred: {pred_label} | True: {true_label} {correct}"

        # Save
        save_path = save_dir / f"sample_{sample_ids[i] if sample_ids else i}.png"
        gradcam.save_visualization(vis, str(save_path), title=title)

    return heatmaps


def create_gradcam_grid(
    visualizations: List[np.ndarray],
    save_path: str,
    titles: Optional[List[str]] = None,
    grid_size: Optional[Tuple[int, int]] = None
):
    """
    Create a grid of GradCAM visualizations.

    Args:
        visualizations: List of visualization arrays
        save_path: Path to save grid
        titles: Optional titles for each visualization
        grid_size: Optional (rows, cols) for grid layout
    """
    n = len(visualizations)

    if grid_size is None:
        # Auto-determine grid size
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_size

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if n > 1 else [axes]

    for i, (ax, vis) in enumerate(zip(axes, visualizations)):
        ax.imshow(vis)
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=8)

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
