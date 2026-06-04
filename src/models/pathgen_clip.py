"""
PathGen-CLIP model wrapper.

PathGen-CLIP is a CLIP model fine-tuned on pathology images.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from pathlib import Path
from .base_model import BaseVLModel


class PathGenCLIPModel(BaseVLModel):
    """
    PathGen-CLIP model wrapper.

    PathGen-CLIP is based on OpenAI CLIP but fine-tuned on pathology data.

    Args:
        model_id: CLIP architecture (e.g., 'ViT-B-16')
        checkpoint_path: Path to PathGen-CLIP checkpoint (.pt file)
        device: Device to load model on
        embedding_dim: Embedding dimension (default: 512 for ViT-B-16)

    Example:
        >>> model = PathGenCLIPModel(
        ...     model_id='ViT-B-16',
        ...     checkpoint_path='./downloaded_models/pathgenclip.pt',
        ...     device='cuda'
        ... )
        >>> images = torch.randn(4, 3, 224, 224).cuda()
        >>> features = model.encode_image(images)
        >>> features.shape
        torch.Size([4, 512])
    """

    def __init__(
        self,
        model_id: str = "ViT-B-16",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        embedding_dim: int = 512,
        **kwargs  # Accept and ignore extra config keys
    ):
        super().__init__(device=device)

        self.model_id = model_id
        self.checkpoint_path = checkpoint_path
        self.embedding_dim = embedding_dim

        # Load the model
        self._load_model()

    def _load_model(self):
        """
        Load PathGen-CLIP model from checkpoint.

        PathGen-CLIP uses OpenCLIP architecture with custom weights.
        """
        try:
            import open_clip

            print(f"Loading PathGen-CLIP model: {self.model_id}")

            # Create base CLIP model architecture
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_id,
                pretrained=None,  # Don't load pretrained weights yet
                device=self.device
            )

            # Load PathGen-CLIP checkpoint if provided
            if self.checkpoint_path:
                checkpoint_path = Path(self.checkpoint_path)

                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"PathGen-CLIP checkpoint not found at: {self.checkpoint_path}\n"
                        f"Please download the checkpoint and update the path in config/model_configs.yaml"
                    )

                print(f"Loading checkpoint from: {self.checkpoint_path}")

                # Load checkpoint
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location=self.device
                )

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    # Check for common checkpoint keys
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        # Assume the checkpoint itself is the state dict
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # Load weights into model
                # Handle potential key mismatches
                try:
                    self.model.load_state_dict(state_dict, strict=True)
                except RuntimeError as e:
                    print(f"Warning: Strict loading failed, trying non-strict: {e}")
                    self.model.load_state_dict(state_dict, strict=False)

                print(f"✓ PathGen-CLIP checkpoint loaded successfully")
            else:
                print("Warning: No checkpoint path provided, using base CLIP weights")
                # Load standard OpenAI CLIP weights as fallback
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    self.model_id,
                    pretrained='openai',
                    device=self.device
                )

            # Get tokenizer
            self.tokenizer = open_clip.get_tokenizer(self.model_id)

            # Set model to eval mode
            self.model.eval()

            print(f"PathGen-CLIP model loaded successfully on {self.device}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import open_clip: {e}\n"
                "Please install: pip install open-clip-torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load PathGen-CLIP model: {e}")

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
            # Encode images using CLIP's vision encoder
            image_features = self.model.encode_image(images)

            # Normalize features
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
            # Tokenize texts
            text_tokens = self.tokenizer(texts).to(self.device)

            # Encode text using CLIP's text encoder
            text_features = self.model.encode_text(text_tokens)

            # Normalize features
            text_features = torch.nn.functional.normalize(text_features, dim=-1)

        return text_features

    def get_preprocessor(self):
        """
        Return PathGen-CLIP image preprocessing.

        Returns:
            Preprocessing transform
        """
        # Return the preprocessor loaded with the model
        return self.preprocess

    def get_attention_maps(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract attention maps from vision transformer.

        Args:
            images: Batch of images [B, C, H, W]

        Returns:
            Attention maps [B, H, W]

        Note:
            This extracts attention from the last layer of the ViT.
        """
        if images.device != self.device:
            images = images.to(self.device)

        # For ViT models, we can extract attention from transformer layers
        # This is a placeholder - actual implementation depends on CLIP version

        # Get image features (this processes through the vision encoder)
        with torch.no_grad():
            # Simplified: return uniform attention maps
            # TODO: Extract actual attention weights from ViT layers
            batch_size = images.shape[0]
            h, w = images.shape[2:]

            # Create dummy attention maps for now
            attention = torch.ones(batch_size, h, w).to(self.device)

        return attention

    def get_model_info(self):
        """Get PathGen-CLIP model information."""
        info = super().get_model_info()
        info.update({
            'model_name': 'PathGen-CLIP',
            'model_id': self.model_id,
            'checkpoint_path': self.checkpoint_path,
            'embedding_dim': self.embedding_dim,
            'framework': 'open_clip'
        })
        return info


def create_pathgen_clip_model(config: dict) -> PathGenCLIPModel:
    """
    Factory function to create PathGen-CLIP model from config.

    Args:
        config: Configuration dictionary

    Returns:
        PathGenCLIPModel instance

    Example:
        >>> config = {
        ...     'model_id': 'ViT-B-16',
        ...     'checkpoint_path': './downloaded_models/pathgenclip.pt',
        ...     'device': 'cuda',
        ...     'embedding_dim': 512
        ... }
        >>> model = create_pathgen_clip_model(config)
    """
    return PathGenCLIPModel(
        model_id=config.get('model_id', 'ViT-B-16'),
        checkpoint_path=config.get('checkpoint_path'),
        device=config.get('device', 'cuda'),
        embedding_dim=config.get('embedding_dim', 512)
    )
