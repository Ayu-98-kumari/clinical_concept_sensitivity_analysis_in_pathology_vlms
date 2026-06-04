"""
Model factory for creating vision-language models.

This module provides a factory pattern for creating model instances
from configuration dictionaries.
"""

from typing import Dict, Optional
import yaml
from pathlib import Path
from .base_model import BaseVLModel
from .conch import ConchModel
from .pathgen_clip import PathGenCLIPModel
from .keep import KEEPModel
from .quiltnet import QuiltNetModel
from .plip import PLIPModel


class ModelFactory:
    """
    Factory class for creating vision-language models.

    Supports creating models from configuration dictionaries or YAML files.

    Example:
        >>> factory = ModelFactory()
        >>> config = {'model_id': 'conch', 'device': 'cuda'}
        >>> model = factory.create('conch', config)

        >>> # Or from config file
        >>> model = factory.create_from_config('conch', 'config/model_configs.yaml')
    """

    # Registry of model creators
    _model_registry = {
        'conch': ConchModel,
        'pathgen_clip': PathGenCLIPModel,
        'keep': KEEPModel,
        'quiltnet': QuiltNetModel,
        'plip': PLIPModel,
        # Add more models here as they are implemented:
        # 'llava_med': LLaVAMedModel,
    }

    @classmethod
    def register_model(cls, name: str, model_class: type):
        """
        Register a new model class.

        Args:
            name: Model identifier
            model_class: Model class (must inherit from BaseVLModel)

        Example:
            >>> ModelFactory.register_model('custom_model', CustomModel)
        """
        if not issubclass(model_class, BaseVLModel):
            raise ValueError(
                f"Model class must inherit from BaseVLModel, "
                f"got {model_class}"
            )
        cls._model_registry[name.lower()] = model_class

    @classmethod
    def create(
        cls,
        model_name: str,
        config: Optional[Dict] = None,
        **kwargs
    ) -> BaseVLModel:
        """
        Create a model instance.

        Args:
            model_name: Name of the model to create
            config: Configuration dictionary
            **kwargs: Additional keyword arguments for model initialization

        Returns:
            Model instance

        Raises:
            ValueError: If model name is not registered

        Example:
            >>> model = ModelFactory.create('conch', {'device': 'cuda'})
        """
        model_name = model_name.lower()

        if model_name not in cls._model_registry:
            available = ', '.join(cls._model_registry.keys())
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {available}"
            )

        model_class = cls._model_registry[model_name]

        # Merge config and kwargs
        model_config = config or {}
        model_config.update(kwargs)

        # Create model instance
        try:
            model = model_class(**model_config)
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to create model {model_name}: {e}"
            )

    @classmethod
    def create_from_config(
        cls,
        model_name: str,
        config_path: str,
        **kwargs
    ) -> BaseVLModel:
        """
        Create a model from a YAML config file.

        Args:
            model_name: Name of the model to create
            config_path: Path to YAML configuration file
            **kwargs: Additional keyword arguments (override config)

        Returns:
            Model instance

        Example:
            >>> model = ModelFactory.create_from_config(
            ...     'conch',
            ...     'config/model_configs.yaml'
            ... )
        """
        # Load config file
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)

        # Extract model-specific config
        models_config = full_config.get('models', {})
        model_config = models_config.get(model_name.lower(), {})

        if not model_config:
            raise ValueError(
                f"Model '{model_name}' not found in config file {config_path}"
            )

        # Merge with kwargs (kwargs take precedence)
        model_config.update(kwargs)

        # Create model
        return cls.create(model_name, model_config)

    @classmethod
    def list_models(cls) -> list:
        """
        List all registered models.

        Returns:
            List of model names

        Example:
            >>> ModelFactory.list_models()
            ['conch', 'pathgen_clip', 'keep', 'quiltnet', 'llava_med']
        """
        return list(cls._model_registry.keys())

    @classmethod
    def create_all_models(
        cls,
        config_path: str,
        model_names: Optional[list] = None,
        **kwargs
    ) -> Dict[str, BaseVLModel]:
        """
        Create multiple models from config file.

        Args:
            config_path: Path to YAML configuration file
            model_names: List of model names to create (None = all models)
            **kwargs: Additional keyword arguments for all models

        Returns:
            Dictionary mapping model names to model instances

        Example:
            >>> models = ModelFactory.create_all_models(
            ...     'config/model_configs.yaml',
            ...     model_names=['conch', 'quiltnet']
            ... )
            >>> print(models.keys())
            dict_keys(['conch', 'quiltnet'])
        """
        # Load config file
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)

        models_config = full_config.get('models', {})

        # Determine which models to create
        if model_names is None:
            model_names = list(models_config.keys())

        # Create models
        models = {}
        for model_name in model_names:
            model_name = model_name.lower()
            if model_name in cls._model_registry:
                try:
                    model = cls.create_from_config(
                        model_name,
                        config_path,
                        **kwargs
                    )
                    models[model_name] = model
                except Exception as e:
                    print(f"Warning: Failed to create model {model_name}: {e}")
            else:
                print(f"Warning: Model {model_name} not implemented yet")

        return models


# Convenience function
def create_model(
    model_name: str,
    config: Optional[Dict] = None,
    config_path: Optional[str] = None,
    **kwargs
) -> BaseVLModel:
    """
    Convenience function to create a model.

    Args:
        model_name: Name of the model to create
        config: Configuration dictionary (optional)
        config_path: Path to config file (optional)
        **kwargs: Additional keyword arguments

    Returns:
        Model instance

    Example:
        >>> # From dict
        >>> model = create_model('conch', config={'device': 'cuda'})

        >>> # From file
        >>> model = create_model('conch', config_path='config/model_configs.yaml')
    """
    if config_path is not None:
        return ModelFactory.create_from_config(model_name, config_path, **kwargs)
    else:
        return ModelFactory.create(model_name, config, **kwargs)
