"""
Download and cache vision-language models for offline use.

This script downloads and caches models so they're available for evaluation
without needing to download them again during experiments.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_conch(hf_token=None):
    """
    Download CONCH model from HuggingFace.

    Args:
        hf_token: HuggingFace authentication token
    """
    print("=" * 80)
    print("Downloading CONCH Model")
    print("=" * 80)

    if hf_token:
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
        os.environ['HF_TOKEN'] = hf_token

    try:
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

        print("Loading CONCH model (this will download ~400MB)...")
        model, preprocess = create_model_from_pretrained(
            'conch_ViT-B-16',
            'hf_hub:MahmoodLab/conch',
            hf_auth_token=hf_token
        )

        print("✓ CONCH model downloaded and cached successfully!")
        print(f"  Model parameters: ~395M")
        print(f"  Model location: ~/.cache/huggingface/hub/models--MahmoodLab--CONCH")

        return True

    except Exception as e:
        print(f"✗ Failed to download CONCH: {e}")
        return False


def download_clip(model_name='ViT-B-16', pretrained='openai'):
    """
    Download OpenAI CLIP or PathGen-CLIP model.

    Args:
        model_name: CLIP architecture (e.g., 'ViT-B-16', 'ViT-L-14')
        pretrained: Pretrained weights ('openai' or custom)
    """
    print("=" * 80)
    print(f"Downloading CLIP Model: {model_name} ({pretrained})")
    print("=" * 80)

    try:
        import open_clip

        print(f"Loading {model_name} with {pretrained} weights...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )

        tokenizer = open_clip.get_tokenizer(model_name)

        print(f"✓ CLIP model downloaded and cached successfully!")
        print(f"  Model: {model_name}")
        print(f"  Pretrained: {pretrained}")
        print(f"  Cache location: ~/.cache/clip")

        return True

    except Exception as e:
        print(f"✗ Failed to download CLIP: {e}")
        print("  Note: Install open_clip_torch if not available:")
        print("  pip install open-clip-torch")
        return False


def download_pathgen_clip(hf_token=None):
    """
    Download PathGen-CLIP model.

    Note: This is a placeholder. Update with actual PathGen-CLIP source when available.
    """
    print("=" * 80)
    print("Downloading PathGen-CLIP Model")
    print("=" * 80)

    print("PathGen-CLIP download location:")
    print("  Option 1: Via HuggingFace Hub (if available)")
    print("  Option 2: Direct download from authors")
    print("  Option 3: Via timm/open_clip registry")
    print()
    print("Status: Placeholder - awaiting model access details")
    print("  Please contact authors or check paper for model weights")

    # TODO: Add actual download code when PathGen-CLIP is publicly available
    return False


def download_quilt_net(hf_token=None):
    """
    Download QuiltNet model from HuggingFace.

    QuiltNet is available via OpenCLIP HuggingFace Hub integration.
    """
    print("=" * 80)
    print("Downloading QuiltNet-B-32 Model")
    print("=" * 80)

    try:
        import open_clip

        model_name = 'hf-hub:wisdomik/QuiltNet-B-32'

        print(f"Loading QuiltNet model from: {model_name}")
        print("This will download the model to HuggingFace cache...\n")

        # Download model and transforms
        print("Downloading model and preprocessing transforms...")
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)

        print("✓ Model downloaded")

        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = open_clip.get_tokenizer(model_name)

        print("✓ Tokenizer downloaded")

        # Set to eval mode
        model.eval()

        print("\n✓ QuiltNet model downloaded and cached successfully!")
        print(f"  Model ID: {model_name}")
        print(f"  Architecture: ViT-B/32")
        print(f"  Cache location: ~/.cache/huggingface/hub/models--wisdomik--QuiltNet-B-32")
        print("\nUsage:")
        print("  import open_clip")
        print(f"  model, _, preprocess = open_clip.create_model_and_transforms('{model_name}')")
        print(f"  tokenizer = open_clip.get_tokenizer('{model_name}')")

        return True

    except ImportError as e:
        print(f"✗ Failed to import required libraries: {e}")
        print("\nPlease install open_clip_torch:")
        print("  pip install open_clip_torch")
        return False

    except Exception as e:
        print(f"✗ Failed to download QuiltNet: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure open_clip_torch is installed: pip install open_clip_torch")
        print("  2. Check internet connection")
        print("  3. Verify model is accessible on HuggingFace Hub")
        return False


def download_llava_med(hf_token=None):
    """
    Download LLaVA-Med model.

    LLaVA-Med is available via HuggingFace.
    """
    print("=" * 80)
    print("Downloading LLaVA-Med Model")
    print("=" * 80)

    if hf_token:
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
        os.environ['HF_TOKEN'] = hf_token

    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch

        print("Loading LLaVA-Med model (this may take several minutes)...")
        model_id = "microsoft/llava-med-v1.5-mistral-7b"

        # Download processor first
        print("Downloading processor...")
        processor = AutoProcessor.from_pretrained(model_id)
        print("✓ Processor downloaded")

        # Download model (this is large ~13GB)
        print("Downloading model weights (~13GB, this will take time)...")
        print("Note: Using CPU mode for download. Set device_map='auto' for GPU when using.")

        # Download to cache without loading to GPU (saves memory during download)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use float16 to save space
            device_map="cpu",  # Download to CPU first
            low_cpu_mem_usage=True
        )

        print("✓ LLaVA-Med model downloaded and cached successfully!")
        print(f"  Model ID: {model_id}")
        print(f"  Size: ~13GB")
        print(f"  Cache location: ~/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b")
        print("\nUsage:")
        print("  from transformers import AutoModelForCausalLM, AutoProcessor")
        print(f"  model = AutoModelForCausalLM.from_pretrained('{model_id}', torch_dtype='auto')")
        print(f"  processor = AutoProcessor.from_pretrained('{model_id}')")

        return True

    except Exception as e:
        print(f"✗ Failed to download LLaVA-Med: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure you have enough disk space (~15GB free)")
        print("  2. Check internet connection")
        print("  3. Try manual download:")
        print("     from transformers import AutoModelForCausalLM")
        print(f"     model = AutoModelForCausalLM.from_pretrained('microsoft/llava-med-v1.5-mistral-7b')")
        return False


def download_keep(hf_token=None):
    """
    Download KEEP model.

    KEEP (Knowledge-Enhanced Embedding for Pathology) is available via HuggingFace.
    """
    print("=" * 80)
    print("Downloading KEEP Model")
    print("=" * 80)

    if hf_token:
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
        os.environ['HF_TOKEN'] = hf_token

    try:
        from transformers import AutoModel, AutoTokenizer

        model_id = "Astaxanthin/KEEP"

        print(f"Loading KEEP model from: {model_id}")
        print("Note: This model requires trust_remote_code=True\n")

        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded")

        # Download model
        print("Downloading model...")
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        model.eval()

        print("✓ KEEP model downloaded and cached successfully!")
        print(f"  Model ID: {model_id}")
        print(f"  Cache location: ~/.cache/huggingface/hub/models--Astaxanthin--KEEP")
        print("\nUsage:")
        print("  from transformers import AutoModel, AutoTokenizer")
        print(f"  model = AutoModel.from_pretrained('{model_id}', trust_remote_code=True)")
        print(f"  tokenizer = AutoTokenizer.from_pretrained('{model_id}', trust_remote_code=True)")

        return True

    except Exception as e:
        print(f"✗ Failed to download KEEP: {e}")
        print("\nNote: Requires trust_remote_code=True")
        print("  Install transformers: pip install transformers")
        return False


def download_plip(hf_token=None):
    """
    Download PLIP model.

    PLIP (Pathology Language-Image Pre-training) is available via HuggingFace.
    """
    print("=" * 80)
    print("Downloading PLIP Model")
    print("=" * 80)

    if hf_token:
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
        os.environ['HF_TOKEN'] = hf_token

    try:
        from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

        model_id = "vinid/plip"

        print(f"Loading PLIP model from: {model_id}")

        # Download processor
        print("Downloading processor...")
        processor = AutoProcessor.from_pretrained(model_id)
        print("✓ Processor downloaded")

        # Download model
        print("Downloading model...")
        model = AutoModelForZeroShotImageClassification.from_pretrained(model_id)
        model.eval()

        print("✓ Model downloaded")

        print("\n✓ PLIP model downloaded and cached successfully!")
        print(f"  Model ID: {model_id}")
        print(f"  Cache location: ~/.cache/huggingface/hub/models--vinid--plip")
        print("\nUsage:")
        print("  from transformers import AutoProcessor, AutoModelForZeroShotImageClassification")
        print(f"  processor = AutoProcessor.from_pretrained('{model_id}')")
        print(f"  model = AutoModelForZeroShotImageClassification.from_pretrained('{model_id}')")

        return True

    except Exception as e:
        print(f"✗ Failed to download PLIP: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure transformers is installed: pip install transformers")
        print("  2. Check internet connection")
        print("  3. Verify model is accessible on HuggingFace Hub")
        return False


def download_all_available(hf_token=None):
    """Download all available models."""
    print("\n" + "=" * 80)
    print("DOWNLOADING ALL AVAILABLE MODELS")
    print("=" * 80 + "\n")

    results = {}

    # Download CONCH
    results['conch'] = download_conch(hf_token)
    print()

    # Download OpenAI CLIP
    results['clip_vit_b16'] = download_clip('ViT-B-16', 'openai')
    print()

    # Download larger CLIP
    results['clip_vit_l14'] = download_clip('ViT-L-14', 'openai')
    print()

    # Try LLaVA-Med (large model, optional)
    print("LLaVA-Med is a large model (~13GB). Download? (y/n): ", end='')
    if input().lower().strip() == 'y':
        results['llava_med'] = download_llava_med(hf_token)
        print()

    # Show status for unavailable models
    results['pathgen_clip'] = download_pathgen_clip(hf_token)
    print()

    results['quiltnet'] = download_quilt_net(hf_token)
    print()

    results['keep'] = download_keep(hf_token)
    print()

    results['plip'] = download_plip(hf_token)
    print()

    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)

    for model, success in results.items():
        status = "✓ Downloaded" if success else "✗ Not available / Failed"
        print(f"  {model:20s}: {status}")

    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and cache vision-language models for evaluation"
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['all', 'conch', 'clip', 'pathgen_clip', 'quiltnet', 'llava_med', 'keep', 'plip'],
        default='all',
        help='Model to download (default: all available models)'
    )

    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace authentication token for gated models'
    )

    parser.add_argument(
        '--clip_model',
        type=str,
        default='ViT-B-16',
        help='CLIP model architecture (default: ViT-B-16)'
    )

    parser.add_argument(
        '--clip_pretrained',
        type=str,
        default='openai',
        help='CLIP pretrained weights (default: openai)'
    )

    args = parser.parse_args()

    # Get HF token from environment if not provided
    if args.hf_token is None:
        args.hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')

    print("\n" + "=" * 80)
    print("VISION-LANGUAGE MODEL DOWNLOADER")
    print("=" * 80 + "\n")

    if args.model == 'all':
        download_all_available(args.hf_token)
    elif args.model == 'conch':
        download_conch(args.hf_token)
    elif args.model == 'clip':
        download_clip(args.clip_model, args.clip_pretrained)
    elif args.model == 'pathgen_clip':
        download_pathgen_clip(args.hf_token)
    elif args.model == 'quiltnet':
        download_quilt_net(args.hf_token)
    elif args.model == 'llava_med':
        download_llava_med(args.hf_token)
    elif args.model == 'keep':
        download_keep(args.hf_token)
    elif args.model == 'plip':
        download_plip(args.hf_token)

    print("\nNote: Downloaded models are cached and will be available for offline use.")
    print("Cache locations:")
    print("  HuggingFace models: ~/.cache/huggingface/")
    print("  CLIP models: ~/.cache/clip/")
    print("  Timm models: ~/.cache/torch/hub/")


if __name__ == '__main__':
    main()
