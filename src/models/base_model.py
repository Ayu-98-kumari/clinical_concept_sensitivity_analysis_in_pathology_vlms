"""
Abstract base class for all vision-language models.

This module defines the interface that all VLM wrappers must implement
to ensure consistent API across different model architectures.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseVLModel(ABC, nn.Module):
    """
    Abstract base class ensuring consistent API across all VLMs.

    All model wrappers (CONCH, PathGen-CLIP, KEEP, QuiltNet, LLaVA-Med)
    must inherit from this class and implement the abstract methods.

    Args:
        device: Device to load the model on ('cuda' or 'cpu')

    Attributes:
        device: Device where the model is loaded
        model: The underlying model instance
        temperature: Temperature parameter for similarity scaling
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.temperature = 1.0

    @abstractmethod
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images into feature embeddings.

        Args:
            images: Batch of images [B, C, H, W]

        Returns:
            Image embeddings [B, D] where D is embedding dimension

        Example:
            >>> model = ConchModel()
            >>> images = torch.randn(4, 3, 224, 224)
            >>> embeddings = model.encode_image(images)
            >>> embeddings.shape
            torch.Size([4, 512])
        """
        pass

    @abstractmethod
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text prompts into feature embeddings.

        Args:
            texts: List of text strings

        Returns:
            Text embeddings [N, D] where N is number of texts

        Example:
            >>> model = ConchModel()
            >>> texts = ["cancer", "normal"]
            >>> embeddings = model.encode_text(texts)
            >>> embeddings.shape
            torch.Size([2, 512])
        """
        pass

    @abstractmethod
    def get_preprocessor(self):
        """
        Return model-specific image preprocessing function.

        Returns:
            Preprocessing transform function

        Example:
            >>> model = ConchModel()
            >>> preprocess = model.get_preprocessor()
            >>> from PIL import Image
            >>> img = Image.open("image.png")
            >>> processed = preprocess(img)
        """
        pass

    def compute_similarity(
        self,
        img_feat: torch.Tensor,
        txt_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between image and text features.

        Args:
            img_feat: Image features [B, D]
            txt_feat: Text features [N, D]

        Returns:
            Similarity matrix [B, N] with temperature scaling

        Example:
            >>> img_feat = torch.randn(4, 512)
            >>> txt_feat = torch.randn(2, 512)
            >>> sim = model.compute_similarity(img_feat, txt_feat)
            >>> sim.shape
            torch.Size([4, 2])
        """
        # Normalize features
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)

        # Compute cosine similarity
        similarity = torch.matmul(img_feat, txt_feat.t())

        # Apply temperature scaling
        similarity = similarity / self.temperature

        return similarity

    def zero_shot_predict(
        self,
        images: torch.Tensor,
        text_prompts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform zero-shot prediction using text prompts.

        Args:
            images: Batch of images [B, C, H, W]
            text_prompts: List of text prompts for each class

        Returns:
            Tuple of (predictions, probabilities)
                - predictions: Class predictions [B]
                - probabilities: Class probabilities [B, N]

        Example:
            >>> images = torch.randn(4, 3, 224, 224)
            >>> prompts = ["cancer", "normal"]
            >>> preds, probs = model.zero_shot_predict(images, prompts)
            >>> preds.shape, probs.shape
            (torch.Size([4]), torch.Size([4, 2]))
        """
        with torch.no_grad():
            # Encode images and text
            img_features = self.encode_image(images)
            txt_features = self.encode_text(text_prompts)

            # Compute similarities
            similarity = self.compute_similarity(img_features, txt_features)

            # Get probabilities using softmax
            probabilities = F.softmax(similarity, dim=-1)

            # Get predictions
            predictions = torch.argmax(probabilities, dim=-1)

        return predictions, probabilities

    @abstractmethod
    def get_attention_maps(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract attention maps for GradCAM visualization.

        Args:
            images: Batch of images [B, C, H, W]

        Returns:
            Attention maps [B, H, W] or feature maps for visualization

        Note:
            Implementation depends on model architecture
        """
        pass

    def freeze_encoder(self):
        """
        Freeze all model parameters.

        Used for linear probing where only the classifier is trained.

        Example:
            >>> model = ConchModel()
            >>> model.freeze_encoder()
            >>> # Now model parameters won't be updated during training
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """
        Unfreeze all model parameters.

        Example:
            >>> model = ConchModel()
            >>> model.unfreeze_encoder()
        """
        for param in self.parameters():
            param.requires_grad = True

    def count_parameters(self) -> Dict[str, int]:
        """
        Count total and trainable parameters.

        Returns:
            Dictionary with parameter counts

        Example:
            >>> model = ConchModel()
            >>> counts = model.count_parameters()
            >>> print(counts)
            {'total': 86000000, 'trainable': 86000000, 'frozen': 0}
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }

    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model metadata

        Example:
            >>> model = ConchModel()
            >>> info = model.get_model_info()
            >>> print(info['model_name'])
            'CONCH'
        """
        param_counts = self.count_parameters()

        return {
            'model_name': self.__class__.__name__,
            'device': str(self.device),
            'parameters': param_counts,
            'temperature': self.temperature
        }

    def set_temperature(self, temperature: float):
        """
        Set temperature parameter for similarity scaling.

        Args:
            temperature: Temperature value (higher = softer probabilities)

        Example:
            >>> model = ConchModel()
            >>> model.set_temperature(0.07)
        """
        self.temperature = temperature

    def to_device(self, device: str):
        """
        Move model to specified device.

        Args:
            device: Target device ('cuda' or 'cpu')

        Example:
            >>> model = ConchModel()
            >>> model.to_device('cuda')
        """
        self.device = device
        self.to(device)

    def eval_mode(self):
        """
        Set model to evaluation mode.

        Example:
            >>> model = ConchModel()
            >>> model.eval_mode()
        """
        self.eval()

    def train_mode(self):
        """
        Set model to training mode.

        Example:
            >>> model = ConchModel()
            >>> model.train_mode()
        """
        self.train()
