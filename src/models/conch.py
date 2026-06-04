"""
CONCH (CONtrastive learning from Captions for Histopathology) model wrapper.

CONCH is a vision-language model specifically pretrained on histopathology images.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from .base_model import BaseVLModel


class ConchModel(BaseVLModel):
    """
    CONCH model wrapper.

    CONCH uses contrastive learning on histopathology image-caption pairs.
    Loaded via timm library with HuggingFace hub.

    Args:
        model_id: HuggingFace model identifier
        device: Device to load model on
        requires_auth: Whether HuggingFace authentication is required

    Example:
        >>> model = ConchModel(
        ...     model_id="hf_hub:MahmoodLab/CONCH",
        ...     device="cuda"
        ... )
        >>> images = torch.randn(4, 3, 224, 224).cuda()
        >>> features = model.encode_image(images)
        >>> features.shape
        torch.Size([4, 512])
    """

    def __init__(
        self,
        model_id: str = "hf_hub:MahmoodLab/CONCH",
        device: str = "cuda",
        requires_auth: bool = True,
        **kwargs  # Accept and ignore extra config keys
    ):
        super().__init__(device=device)

        self.model_id = model_id
        self.requires_auth = requires_auth
        self.embedding_dim = kwargs.get('embedding_dim', 512)

        # Load the model
        self._load_model()

    def _load_model(self):
        """
        Load CONCH model from HuggingFace.

        Note: Requires HuggingFace authentication token if model is gated.
        Set token via environment variable HUGGING_FACE_HUB_TOKEN or HF_TOKEN.
        """
        try:
            import os
            from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

            # Get HF token from environment
            hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')

            if self.requires_auth and hf_token is None:
                raise ValueError(
                    "HuggingFace authentication required for CONCH. "
                    "Please set HUGGING_FACE_HUB_TOKEN or HF_TOKEN environment variable."
                )

            print(f"Loading CONCH model from: {self.model_id}")

            # Load model using CONCH's official API
            self.model, self.preprocess = create_model_from_pretrained(
                'conch_ViT-B-16',
                self.model_id,
                hf_auth_token=hf_token
            )

            # Load tokenizer for text encoding
            self.tokenizer = get_tokenizer()

            # Move model to device
            self.model = self.model.to(self.device)

            # Set model to eval mode
            self.model.eval()

            # Update embedding dimension (CONCH uses 512 for ViT-B/16)
            self.embedding_dim = 512

            print(f"CONCH model loaded successfully on {self.device}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import CONCH library: {e}\n"
                "Please install: pip install git+https://github.com/Mahmoodlab/CONCH.git"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load CONCH model: {e}")

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
            # Use CONCH's encode_image with projection and normalization for zero-shot
            # proj_contrast=True: Use projection head for contrastive learning
            # normalize=True: Normalize embeddings for similarity computation
            features = self.model.encode_image(images, proj_contrast=True, normalize=True)

        return features

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text prompts into feature embeddings.

        Args:
            texts: List of text strings

        Returns:
            Text embeddings [N, D]
        """
        with torch.no_grad():
            # Tokenize texts with padding
            text_tokens = self.tokenizer(texts, padding=True, return_tensors='pt')

            # Move to device
            text_tokens = text_tokens['input_ids'].to(self.device)

            # Encode text using CONCH's text encoder
            # The model already applies normalization internally
            text_embeddings = self.model.encode_text(text_tokens)

        return text_embeddings

    def get_preprocessor(self):
        """
        Return CONCH-specific image preprocessing.

        Returns:
            Preprocessing transform
        """
        # Return the preprocessor loaded with the model
        # CONCH uses 448x448 images with specific normalization
        return self.preprocess

    def get_attention_maps(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract attention maps for GradCAM visualization.

        Args:
            images: Batch of images [B, C, H, W]

        Returns:
            Attention maps [B, H, W]

        Note:
            This is a placeholder implementation.
            Actual implementation would extract attention from vision transformer.
        """
        if images.device != self.device:
            images = images.to(self.device)

        # PLACEHOLDER: Return dummy attention maps
        batch_size = images.shape[0]
        h, w = images.shape[2:]

        # Create simple attention maps (center-focused)
        attention = torch.zeros(batch_size, h, w).to(self.device)
        center_h, center_w = h // 2, w // 2
        for i in range(h):
            for j in range(w):
                dist = ((i - center_h) ** 2 + (j - center_w) ** 2) ** 0.5
                attention[:, i, j] = 1.0 / (1.0 + dist / 10.0)

        return attention

    def get_model_info(self):
        """Get CONCH model information."""
        info = super().get_model_info()
        info.update({
            'model_name': 'CONCH',
            'model_id': self.model_id,
            'embedding_dim': self.embedding_dim,
            'requires_auth': self.requires_auth,
            'framework': 'timm'
        })
        return info


def create_conch_model(config: dict) -> ConchModel:
    """
    Factory function to create CONCH model from config.

    Args:
        config: Configuration dictionary

    Returns:
        ConchModel instance

    Example:
        >>> config = {
        ...     'model_id': 'hf_hub:MahmoodLab/CONCH',
        ...     'device': 'cuda',
        ...     'requires_hf_auth': True
        ... }
        >>> model = create_conch_model(config)
    """
    return ConchModel(
        model_id=config.get('model_id', 'hf_hub:MahmoodLab/CONCH'),
        device=config.get('device', 'cuda'),
        requires_auth=config.get('requires_hf_auth', True)
    )
