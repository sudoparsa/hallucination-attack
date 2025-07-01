import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import open_clip
import random
import numpy as np


def get_llava_image_features(images, model, processor, device="cuda"):
    """
    Extract mean-pooled LLaVA features from a batch of images.

    Args:
        images: List of PIL images
        model: LlavaForConditionalGeneration
        processor: LlavaProcessor
        device: "cuda" or "cpu"

    Returns:
        Tensor of shape [B, D]: one feature vector per image
    """
    # Preprocess all images
    inputs = processor.image_processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)      # [B, N_crops, 3, H, W]
    image_sizes = inputs["image_sizes"].to(device)        # [B, 2]

    vision_feature_layer = model.config.vision_feature_layer
    vision_feature_select_strategy = model.config.vision_feature_select_strategy

    image_features = model.get_image_features(
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        vision_feature_layer=vision_feature_layer,
        vision_feature_select_strategy=vision_feature_select_strategy,
    )

    image_features, feature_lens = model.pack_image_features(
        image_features,
        image_sizes=image_sizes,
        vision_feature_select_strategy=vision_feature_select_strategy,
        image_newline=model.image_newline,
    )

    # Use `feature_lens` to split
    split_features = torch.split(image_features, list(feature_lens), dim=0)

    mean_features = torch.stack([f.mean(dim=0) for f in split_features], dim=0)  # [B, D]

    return mean_features  # shape: [B, D]


def get_target_dim(model_name):
    """
    Get the target dimension for the projector based on the model name.
    
    Args:
        model_name: Name of the model (e.g., "llava", "qwen")
    
    Returns:
        int: Target dimension for the projector
    """
    if model_name == "llava":
        return 4096  # LLaVA feature dimension
    elif model_name == "qwen":
        return 3584  # Qwen feature dimension
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_clip_image_features(images, clip_model, clip_preprocess):
    """
    Extract mean-pooled CLIP features from a batch of images.

    Args:
        images: List of PIL images
        clip_model: CLIP model
        clip_preprocess: Preprocessing function for CLIP

    Returns:
        Tensor of shape [B, D]: one feature vector per image
    """
    # Preprocess all images
    inputs = torch.stack([clip_preprocess(image) for image in images]).to("cuda")  # [B, 3, H, W]

    image_features = clip_model.encode_image(inputs)  # [B, D]

    return image_features # shape: [B, D]

def get_model(model_name, cache_path):
    if model_name == "llava":
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        processor = LlavaNextProcessor.from_pretrained(model_id, cache_dir=cache_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir=cache_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    elif model_name == "qwen":
        pass
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model, processor

def get_clip_model(cache_path):
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', cache_dir=cache_path)
    return clip_model, clip_preprocess


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you use multi-GPU