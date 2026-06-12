"""
KEEP (Knowledge-Enhanced Embedding for Pathology) model wrapper.

KEEP is a vision-language model specifically designed for pathology image analysis.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from .base_model import BaseVLModel


class KEEPModel(BaseVLModel):
    """
    KEEP model wrapper.

    KEEP uses transformers architecture with custom vision-language encoding
    for histopathology images.

    Args:
        model_id: HuggingFace model identifier (default: "Astaxanthin/KEEP")
        device: Device to load model on
        embedding_dim: Embedding dimension (inferred from model)
        trust_remote_code: Whether to trust remote code (required for KEEP)

    Example:
        >>> model = KEEPModel(device='cuda')
        >>> images = torch.randn(4, 3, 224, 224).cuda()
        >>> features = model.encode_image(images)
        >>> features.shape
        torch.Size([4, 512])
    """

    def __init__(
        self,
        model_id: str = "Astaxanthin/KEEP",
        device: str = "cuda",
        embedding_dim: Optional[int] = None,
        trust_remote_code: bool = True,
        **kwargs  # Accept and ignore extra config keys
    ):
        super().__init__(device=device)

        self.model_id = model_id
        self.trust_remote_code = trust_remote_code
        self.embedding_dim = embedding_dim  # Will be set after loading

        # Load the model
        self._load_model()

    def _load_model(self):
        """
        Load KEEP model from HuggingFace.

        Note: Requires trust_remote_code=True as KEEP uses custom model code.
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            from torchvision import transforms

            print(f"Loading KEEP model from: {self.model_id}")

            if not self.trust_remote_code:
                raise ValueError(
                    "KEEP model requires trust_remote_code=True. "
                    "Set trust_remote_code=True in config or model initialization."
                )

            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )

            # Move model to device
            self.model = self.model.to(self.device)

            # Set model to eval mode
            self.model.eval()

            # Set up image preprocessing (as specified in KEEP documentation)
            self.preprocess = transforms.Compose([
                transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

            # Infer embedding dimension if not provided
            if self.embedding_dim is None:
                # Try to get embedding dim from model
                # For KEEP, we can test with a dummy input
                with torch.no_grad():
                    dummy_img = torch.randn(1, 3, 224, 224).to(self.device)
                    dummy_features = self.model.encode_image(dummy_img)
                    self.embedding_dim = dummy_features.shape[-1]

            print(f"KEEP model loaded successfully on {self.device}")
            print(f"Embedding dimension: {self.embedding_dim}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import required libraries for KEEP: {e}\n"
                "Please install: pip install transformers torchvision pillow"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load KEEP model: {e}") from e

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
            # Use KEEP's encode_image method
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
            # Tokenize texts using KEEP's tokenizer
            # KEEP uses max_length=256 with padding
            token_input = self.tokenizer(
                texts,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Move tokens to device
            token_input = {k: v.to(self.device) for k, v in token_input.items()}

            # Encode text using KEEP's encode_text method
            text_features = self.model.encode_text(token_input)

            # Normalize features for zero-shot classification
            text_features = torch.nn.functional.normalize(text_features, dim=-1)

        return text_features

    def get_preprocessor(self):
        """
        Return KEEP-specific image preprocessing.

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
            KEEP's internal architecture.
        """
        if images.device != self.device:
            images = images.to(self.device)

        # Placeholder: return uniform attention maps
        # TODO: Extract actual attention weights from KEEP's vision encoder
        batch_size = images.shape[0]
        h, w = images.shape[2:]

        # Create simple attention maps
        attention = torch.ones(batch_size, h, w).to(self.device)

        return attention

    def get_model_info(self):
        """Get KEEP model information."""
        info = super().get_model_info()
        info.update({
            'model_name': 'KEEP',
            'model_id': self.model_id,
            'embedding_dim': self.embedding_dim,
            'trust_remote_code': self.trust_remote_code,
            'framework': 'transformers'
        })
        return info


def create_keep_model(config: dict) -> KEEPModel:
    """
    Factory function to create KEEP model from config.

    Args:
        config: Configuration dictionary

    Returns:
        KEEPModel instance

    Example:
        >>> config = {
        ...     'model_id': 'Astaxanthin/KEEP',
        ...     'device': 'cuda',
        ...     'trust_remote_code': True
        ... }
        >>> model = create_keep_model(config)
    """
    return KEEPModel(
        model_id=config.get('model_id', 'Astaxanthin/KEEP'),
        device=config.get('device', 'cuda'),
        embedding_dim=config.get('embedding_dim'),
        trust_remote_code=config.get('trust_remote_code', True)
    )
