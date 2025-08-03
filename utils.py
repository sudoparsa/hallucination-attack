import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, MllamaForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import open_clip
import random
import numpy as np
from transformers.utils import is_torchdynamo_compiling
from diffusers import DiffusionPipeline
from transformers.image_processing_utils import select_best_resolution


def get_llava_image_features(images, model, processor, avg_pool=False, device="cuda"):
    """
    Extract mean-pooled LLaVA features from a batch of images.

    Args:
        images: List of PIL images
        model: LlavaForConditionalGeneration
        processor: LlavaProcessor
        device: "cuda" or "cpu"

    Returns:
        Tensor of shape [B, N_tokens, D] 
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
        image_newline=model.model.image_newline,
    )


    return image_features  # shape: [B, D]

def get_llama_image_features(images, model, processor, device="cuda"):
    """
    Extract Llama image features from a batch of images.
    Args:
        images: List of PIL images
        model: LlamaForConditionalGeneration
        processor: LlamaProcessor
        device: "cuda" or "cpu"
    Returns:
        Tensor of shape [B, N_tokens, D]
    """
    inputs = processor.image_processor(images=images, return_tensors="pt")
    vision_outputs = model.vision_model(
        pixel_values=inputs['pixel_values'].to(device),
        aspect_ratio_ids=inputs['aspect_ratio_ids'].to(device),
        aspect_ratio_mask=inputs['aspect_ratio_mask'].to(device),
        )
    cross_attention_states = vision_outputs.last_hidden_state 
    cross_attention_states = model.model.multi_modal_projector(cross_attention_states).reshape(
    -1, cross_attention_states.shape[-2]*cross_attention_states.shape[-3], model.model.hidden_size)
    return  cross_attention_states

def get_qwen_image_features(images, model, processor, device="cuda"):
    """
    Extract Qwen image features from a batch of images.
    Args:
        images: List of PIL images
        model: Qwen2_5_VLForConditionalGeneration
        processor: AutoProcessor
        device: "cuda" or "cpu"
    Returns:
        Tensor of shape [B, N_tokens, D]
    """
    B = len(images)
    inputs = processor.image_processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)      
    image_grid_thw = inputs["image_grid_thw"].to(device)        

    pixel_values = pixel_values.type(model.visual.dtype)
    image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
    image_embeds = image_embeds.view(B, image_embeds.shape[0] // B, -1)
    return image_embeds


def get_llava_inputs(inputs, model, image_features, device="cuda"):
    """
    Prepare inputs for LLaVA model using precomputed image features.

    Args:
        inputs: Preprocessed inputs from LlavaProcessor
        model: LlavaForConditionalGeneration
        image_features: Precomputed image features

    Returns:
        Dict: Inputs ready for model.forward()
    """
    inputs = inputs.to(device)
    inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
    special_image_mask = (inputs['input_ids'] == model.config.image_token_index).unsqueeze(-1)
    special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
    if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
        n_image_tokens = (inputs['input_ids'] == model.config.image_token_index).sum()
        n_image_features = image_features.shape[0]
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
        )
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    inputs['inputs_embeds'] = inputs_embeds
    inputs['pixel_values'] = None
    

    return inputs


def get_qwen_inputs(inputs, model, image_embeds, device="cuda"):
    """
    Prepare inputs for Qwen model using precomputed image features.
    Args:
        inputs: Preprocessed inputs from AutoProcessor
        model: Qwen2_5_VLForConditionalGeneration
        image_embeds: Precomputed image features
    Returns:
        Dict: Inputs ready for model.forward()
    """

    inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
    n_image_tokens = (inputs['input_ids'] == model.config.image_token_id).sum().item()
    n_image_features = image_embeds.shape[0]
    if n_image_tokens != n_image_features:
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
        )
    mask = inputs['input_ids'] == model.config.image_token_id
    mask_unsqueezed = mask.unsqueeze(-1)
    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
    image_mask = mask_expanded.to(inputs_embeds.device)

    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    inputs['inputs_embeds'] = inputs_embeds
    inputs['pixel_values'] = None
    

    return inputs

def get_llama_inputs(inputs, model, image_features, max_num_tokens=1,device="cuda"):
    """
    Prepare inputs for LLamA model using precomputed image features.

    Args:
        inputs: Preprocessed inputs from LlamaProcessor
        model: LlamaForConditionalGeneration
        image_features: Precomputed image features

    Returns:
        Dict: Inputs ready for model.forward()
    """
    inputs['cross_attention_states'] = image_features
    cross_attention_mask = inputs['cross_attention_mask']
    inputs['cross_attention_mask'] = cross_attention_mask.repeat(1,1,1,max_num_tokens)
    inputs['pixel_values'] = None
   
    return inputs

def get_model_inputs(model_name, inputs, model, image_features, max_num_tokens=1, device="cuda"):
    if model_name == "llava":
        return get_llava_inputs(inputs, model, image_features, device=device)
    elif model_name == "qwen":
        return get_qwen_inputs(inputs, model, image_features, device=device)
    elif model_name == "llama":
        return get_llama_inputs(inputs, model, image_features, max_num_tokens, device=device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")



def get_model_image_features(model_name, images, model, processor, device="cuda"):
    if model_name == "llava":
        return get_llava_image_features(images, model, processor, device=device)
    elif model_name == "qwen":
        return get_qwen_image_features(images, model, processor, device=device)
    elif model_name == "llama":
         return get_llama_image_features(images, model, processor, device=device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")



def llava_image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    """
    Calculate the number of patches after the preprocessing for images of any resolution.

    Args:
        image_size (`torch.LongTensor` or `np.ndarray` or `tuple[int, int]`):
            The size of the input image in the format (height, width). ?
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        int: the number of patches
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(f"image_size invalid type {type(image_size)} with value {image_size}")
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    height, width = best_resolution
    num_patches = 0
    # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    num_patches += 1
    return num_patches

def get_clip_image_features(images, clip_model, clip_preprocess):
    """
    Extract CLIP features from a batch of images.

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
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir=cache_path)
    elif model_name == "llama":
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        processor = AutoProcessor.from_pretrained(model_id''',token=""''')
        model = MllamaForConditionalGeneration.from_pretrained(model_id, '''token="",'''  torch_dtype=torch.float16,device_map="auto", low_cpu_mem_usage=True, cache_dir=cache_path)
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



def get_messages(prompt, image):
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
            ]}
    ]
    images = [image]
    return messages, images

def vllm_standard_preprocessing(processor, prompt, image, **processor_kwargs):
    messages, images = get_messages(prompt, image)
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt], images=images, padding=True, return_tensors="pt",
        **processor_kwargs
    ).to('cuda')
    return inputs

def vllm_decoding(inputs, output_ids, processor) -> str:
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text[0]


def get_vllm_output(model, processor, prompt, image, max_new_tokens=512):
    if model == 'gpt-4o':
        return
    inputs = vllm_standard_preprocessing(processor, prompt, image)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return vllm_decoding(inputs, output_ids, processor)



def get_num_tokens(model_name):
    if model_name == "llava":
        return 1176, 4096
    elif model_name == "qwen":
        return 144, 3584
    elif model_name == "llama":
        return 4*1601, 4096
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_diffusion_model(cache_path):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip",
        torch_dtype=torch.float16,
        cache_dir=cache_path,
    ).to("cuda")
    return pipe


def get_prompt_templates():
    templates = [
        "Do you see a {obj} in the image?",
        "Is there a {obj} here?",
        "Does the image contain a {obj}?",
        "Can you find a {obj} in this picture?",
        "Would you say there's a {obj} here?",
        "Is a {obj} present in this image?",
        ]
    return templates


