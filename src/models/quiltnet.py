"""
QuiltNet model wrapper.

QuiltNet is a vision-language model trained on the Quilt-1M dataset,
specifically designed for histopathology image analysis.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from .base_model import BaseVLModel


class QuiltNetModel(BaseVLModel):
    """
    QuiltNet model wrapper.

    QuiltNet uses ViT-B/32 architecture trained on Quilt-1M dataset
    (1 million histopathology image-text pairs).

    Args:
        model_id: Model identifier for OpenCLIP (default: "hf-hub:wisdomik/QuiltNet-B-32")
        device: Device to load model on
        embedding_dim: Embedding dimension (inferred from model)

    Example:
        >>> model = QuiltNetModel(device='cuda')
        >>> images = torch.randn(4, 3, 224, 224).cuda()
        >>> features = model.encode_image(images)
        >>> features.shape
        torch.Size([4, 512])
    """

    def __init__(
        self,
        model_id: str = "hf-hub:wisdomik/QuiltNet-B-32",
        device: str = "cuda",
        embedding_dim: Optional[int] = None,
        **kwargs  # Accept and ignore extra config keys
    ):
        super().__init__(device=device)

        self.model_id = model_id
        self.embedding_dim = embedding_dim  # Will be set after loading

        # Load the model
        self._load_model()

    def _load_model(self):
        """
        Load QuiltNet model using OpenCLIP.

        QuiltNet is available via HuggingFace Hub integration in OpenCLIP.
        """
        try:
            import open_clip

            print(f"Loading QuiltNet model from: {self.model_id}")

            # Load model and preprocessing
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_id,
                device=self.device
            )

            # Load tokenizer
            self.tokenizer = open_clip.get_tokenizer(self.model_id)

            # Set model to eval mode
            self.model.eval()

            # Infer embedding dimension if not provided
            if self.embedding_dim is None:
                # Try to get embedding dim from model
                # For CLIP-style models, we can test with a dummy input
                with torch.no_grad():
                    dummy_img = torch.randn(1, 3, 224, 224).to(self.device)
                    dummy_features = self.model.encode_image(dummy_img)
                    self.embedding_dim = dummy_features.shape[-1]

            print(f"QuiltNet model loaded successfully on {self.device}")
            print(f"Embedding dimension: {self.embedding_dim}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import required libraries for QuiltNet: {e}\n"
                "Please install: pip install open_clip_torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load QuiltNet model: {e}")

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images into feature embeddings.

        Args:
            images: Batch of images [B, C, H, W]

        Returns:
            Image embeddings [B, D]
        """
        if images.device != self.device:
            images = images.to(self.device)

        with torch.no_grad():
            # Use QuiltNet's encode_image method
            image_features = self.model.encode_image(images)

            # Normalize features for zero-shot classification
            image_features = torch.nn.functional.normalize(image_features, dim=-1)

        return image_features

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text prompts into feature embeddings.

        Args:
            texts: List of text strings

        Returns:
            Text embeddings [N, D]
        """
        with torch.no_grad():
            # Tokenize texts using QuiltNet's tokenizer
            text_tokens = self.tokenizer(texts).to(self.device)

            # Encode text using QuiltNet's encode_text method
            text_features = self.model.encode_text(text_tokens)

            # Normalize features for zero-shot classification
            text_features = torch.nn.functional.normalize(text_features, dim=-1)

        return text_features

    def get_preprocessor(self):
        """
        Return QuiltNet-specific image preprocessing.

        Returns:
            Preprocessing transform
        """
        # Return the preprocessor we created during model loading
        return self.preprocess

    def get_attention_maps(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract attention maps for visualization.

        Args:
            images: Batch of images [B, C, H, W]

        Returns:
            Attention maps [B, H, W]

        Note:
            This is a placeholder. Actual attention extraction depends on
            QuiltNet's internal architecture.
        """
        if images.device != self.device:
            images = images.to(self.device)

        # Placeholder: return uniform attention maps
        # TODO: Extract actual attention weights from QuiltNet's vision encoder
        batch_size = images.shape[0]
        h, w = images.shape[2:4]

        # Create simple attention maps
        attention = torch.ones(batch_size, h, w).to(self.device)

        return attention

    def get_model_info(self):
        """Get QuiltNet model information."""
        info = super().get_model_info()
        info.update({
            'model_name': 'QuiltNet',
            'model_id': self.model_id,
            'embedding_dim': self.embedding_dim,
            'architecture': 'ViT-B/32',
            'framework': 'open_clip',
            'training_data': 'Quilt-1M (1M histopathology image-text pairs)'
        })
        return info


def create_quiltnet_model(config: dict) -> QuiltNetModel:
    """
    Factory function to create QuiltNet model from config.

    Args:
        config: Configuration dictionary

    Returns:
        QuiltNetModel instance

    Example:
        >>> config = {
        ...     'model_id': 'hf-hub:wisdomik/QuiltNet-B-32',
        ...     'device': 'cuda'
        ... }
        >>> model = create_quiltnet_model(config)
    """
    return QuiltNetModel(
        model_id=config.get('model_id', 'hf-hub:wisdomik/QuiltNet-B-32'),
        device=config.get('device', 'cuda'),
        embedding_dim=config.get('embedding_dim')
    )
