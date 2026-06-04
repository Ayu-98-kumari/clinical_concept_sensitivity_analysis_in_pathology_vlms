"""
PLIP (Pathology Language-Image Pre-training) model wrapper.

PLIP is a vision-language model specifically designed for pathology image analysis.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from .base_model import BaseVLModel


class PLIPModel(BaseVLModel):
    """
    PLIP model wrapper.

    PLIP uses a dual-encoder architecture with vision and text encoders
    trained via contrastive learning on pathology images.

    Args:
        model_id: HuggingFace model identifier (default: "vinid/plip")
        device: Device to load model on
        embedding_dim: Embedding dimension (inferred from model)

    Example:
        >>> model = PLIPModel(device='cuda')
        >>> images = torch.randn(4, 3, 224, 224).cuda()
        >>> features = model.encode_image(images)
        >>> features.shape
        torch.Size([4, 512])
    """

    def __init__(
        self,
        model_id: str = "vinid/plip",
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
        Load PLIP model from HuggingFace.

        PLIP uses AutoModelForZeroShotImageClassification from transformers.
        """
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
            from torchvision import transforms

            print(f"Loading PLIP model from: {self.model_id}")

            # Load processor (handles both image and text preprocessing)
            self.processor = AutoProcessor.from_pretrained(self.model_id)

            # Load model
            self.model = AutoModelForZeroShotImageClassification.from_pretrained(
                self.model_id
            )

            # Move model to device
            self.model = self.model.to(self.device)

            # Set model to eval mode
            self.model.eval()

            # Set up image preprocessing compatible with our pipeline
            # PLIP expects 224x224 images with standard ImageNet normalization
            self.preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])

            # Infer embedding dimension if not provided
            if self.embedding_dim is None:
                # Get embedding dimension from model config
                if hasattr(self.model.config, 'projection_dim'):
                    self.embedding_dim = self.model.config.projection_dim
                elif hasattr(self.model.config, 'hidden_size'):
                    self.embedding_dim = self.model.config.hidden_size
                else:
                    # Test with dummy input to infer dimension
                    with torch.no_grad():
                        dummy_img = torch.randn(1, 3, 224, 224).to(self.device)
                        dummy_features = self.model.get_image_features(pixel_values=dummy_img)
                        self.embedding_dim = dummy_features.shape[-1]

            print(f"PLIP model loaded successfully on {self.device}")
            print(f"Embedding dimension: {self.embedding_dim}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import required libraries for PLIP: {e}\n"
                "Please install: pip install transformers torchvision pillow"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load PLIP model: {e}")

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
            # Use PLIP's get_image_features method
            image_features = self.model.get_image_features(pixel_values=images)

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
            # Tokenize texts using PLIP's processor
            text_inputs = self.processor(
                text=texts,
                return_tensors='pt',
                padding=True,
                truncation=True
            )

            # Move tokens to device
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

            # Encode text using PLIP's get_text_features method
            text_features = self.model.get_text_features(**text_inputs)

            # Normalize features for zero-shot classification
            text_features = torch.nn.functional.normalize(text_features, dim=-1)

        return text_features

    def get_preprocessor(self):
        """
        Return PLIP-specific image preprocessing.

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
            PLIP's internal architecture.
        """
        if images.device != self.device:
            images = images.to(self.device)

        # Placeholder: return uniform attention maps
        # TODO: Extract actual attention weights from PLIP's vision encoder
        batch_size = images.shape[0]
        h, w = images.shape[2:4]

        # Create simple attention maps
        attention = torch.ones(batch_size, h, w).to(self.device)

        return attention

    def get_model_info(self):
        """Get PLIP model information."""
        info = super().get_model_info()
        info.update({
            'model_name': 'PLIP',
            'model_id': self.model_id,
            'embedding_dim': self.embedding_dim,
            'framework': 'transformers',
            'architecture': 'Dual-Encoder (Vision-Language)'
        })
        return info


def create_plip_model(config: dict) -> PLIPModel:
    """
    Factory function to create PLIP model from config.

    Args:
        config: Configuration dictionary

    Returns:
        PLIPModel instance

    Example:
        >>> config = {
        ...     'model_id': 'vinid/plip',
        ...     'device': 'cuda'
        ... }
        >>> model = create_plip_model(config)
    """
    return PLIPModel(
        model_id=config.get('model_id', 'vinid/plip'),
        device=config.get('device', 'cuda'),
        embedding_dim=config.get('embedding_dim')
    )
