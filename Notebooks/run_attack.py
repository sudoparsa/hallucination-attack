#!/usr/bin/env python3
"""
Simplified Adversarial Attack Pipeline - Matching Notebook Functionality
Run from terminal: python run_enhanced_attack.py
"""

import os
import sys
import torch
import argparse
import json
from pathlib import Path

# Set environment variables
os.environ["HF_HOME"] = "./cache"

# Import all required libraries
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import open_clip
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from collections import defaultdict
import torch.nn.functional as F
import numpy as np

# Import our simplified attack system
from qwen_attack import run_attack

class COCO(Dataset):
    """COCO Dataset class (copied from notebook)"""
    def __init__(self, coco_dir, split='train', transform=None):
        self.image_dir = os.path.join(coco_dir, f"{split}2017/")
        with open(os.path.join(coco_dir, f"annotations/instances_{split}2017.json"), 'r') as file:
            coco = json.load(file)
        
        self.transform = transform
        self.annIm_dict = defaultdict(list)        
        self.cat_dict = {} 
        self.annId_dict = {}
        self.im_dict = {}

        for ann in coco['annotations']:           
            self.annIm_dict[ann['image_id']].append(ann) 
            self.annId_dict[ann['id']] = ann
        
        for img in coco['images']:
            self.im_dict[img['id']] = img
        
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat
    
    def __len__(self):
        return len(list(self.im_dict.keys()))
    
    def __getitem__(self, idx):
        img = self.im_dict[idx]
        image = default_loader(os.path.join(self.image_dir, img['file_name']))
        if self.transform is not None:
            image = self.transform(image)
        return image
        
    def get_targets(self, idx):
        return [self.cat_dict[ann['category_id']]['name'] for ann in self.annIm_dict[idx]]
    
    def get_categories(self, supercategory):
        return [self.cat_dict[cat_id]['name'] for cat_id in self.cat_dict.keys() if self.cat_dict[cat_id]['supercategory']==supercategory]
    
    def get_all_supercategories(self):
        return {self.cat_dict[cat_id]['supercategory'] for cat_id in self.cat_dict.keys()}
    
    def get_spurious_supercategories(self):
        return ['kitchen', 'food', 'vehicle',
                'furniture', 'appliance', 'indoor',
                'outdoor', 'electronic', 'sports',
                'accessory', 'animal']
    
    def get_no_classes(self, supercategories):
        return len([self.cat_dict[cat_id]['name'] for cat_id in self.cat_dict.keys() if self.cat_dict[cat_id]['supercategory'] in supercategories])
    
    def get_imgIds(self):
        return list(self.im_dict.keys())
    
    def get_all_targets_names(self):
        return [self.cat_dict[cat_id]['name'] for cat_id in self.cat_dict.keys()]
    
    def get_imgIds_by_class(self, present_classes=[], absent_classes=[]):
        ids = []
        for img_id in self.get_imgIds():
            targets = self.get_targets(img_id)
            flag = False
            for c in present_classes:
                if c in targets:
                    flag = True
                    break
            for c in absent_classes:
                if c in targets:
                    flag = False
                    break
            if flag:
                ids.append(img_id)
        return ids

class ClipToQwenProjector(nn.Module):
    """CLIP to Qwen projector (copied from notebook)"""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Linear(4096, 4096),
            nn.GELU(),
            nn.Linear(4096, 3584),
            nn.GELU(),    
            nn.LayerNorm(3584)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

def load_and_compute_similarity(clip_model, exclude_indices, object_text, 
                              embeddings_path="../clip_embeddings.pt", device="cuda"):
    """Load precomputed CLIP embeddings and compute similarities (from notebook)"""
    checkpoint = torch.load(embeddings_path)
    clip_embeds = checkpoint["clip_embeds"].to(device) 
    indices = checkpoint["indices"]         

    print(f"Loaded {clip_embeds.shape[0]} embeddings.")

    if exclude_indices is not None:
        mask = torch.isin(indices, torch.tensor(exclude_indices))
        clip_embeds = clip_embeds[mask]
        indices = indices[mask]
        print(f"Filtered down to {clip_embeds.shape[0]} embeddings after exclusion.")
    
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    # Use enhanced text embedding approach - sentence instead of raw token
    sentence = f"A photo of a {object_text}"
    tokens = tokenizer([sentence]).to(device)
    with torch.no_grad():
        text_embed = clip_model.encode_text(tokens).to('cuda')
        text_embed = F.normalize(text_embed, dim=-1)[0]  

    clip_embeds = F.normalize(clip_embeds, dim=-1).to('cuda')  
    similarities = torch.matmul(clip_embeds, text_embed)  
    sorted_similarities, sorted_idx = torch.sort(similarities, descending=True)
    sorted_indices = indices[sorted_idx.cpu()]

    return sorted_indices, sorted_similarities

def load_models(device='cuda'):
    """Load all required models"""
    print("🔄 Loading models...")
    
    # Load Qwen model
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    print("  Loading Qwen model...")
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load CLIP model
    print("  Loading CLIP model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    clip_model = clip_model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    
    # Load Stable Diffusion pipeline
    print("  Loading Stable Diffusion pipeline...")
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16).to(device)
    
    # Load OWLv2 models
    print("  Loading OWLv2 models...")
    processor_owl = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model_owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
    
    # Load projector model
    print("  Loading CLIP-to-Qwen projector...")
    projector_model = ClipToQwenProjector().to(device)
    if os.path.exists('../Reverese.pt'):
        checkpoint = torch.load('../Reverese.pt')
        projector_model.load_state_dict(checkpoint)
        print("  ✅ Loaded pretrained projector weights")
    else:
        print("  ⚠️  Warning: Projector weights not found, using random initialization")
    
    # Set models to eval mode and disable gradients for base models
    for model in [qwen, pipe.vae, pipe.unet, pipe.text_encoder, projector_model]:
        for param in model.parameters():
            param.requires_grad = False
    
    print("✅ All models loaded successfully!")
    
    return {
        'qwen': qwen,
        'clip_model': clip_model,
        'pipe': pipe,
        'projector': projector_model,
        'processor': processor,
        'tokenizer': tokenizer,
        'clip_preprocess': clip_preprocess,
        'processor_owl': processor_owl,
        'model_owl': model_owl
    }

def setup_dataset(coco_path, target_object):
    """Setup COCO dataset and filter images"""
    print("🔄 Setting up dataset...")
    
    dset = COCO(coco_path)
    supercategories = dset.get_spurious_supercategories()
    no_classes = dset.get_no_classes(supercategories)
    print(f"  Number of classes: {no_classes}")
    
    # Get images that don't contain the target object
    present = [x for cat in dset.get_all_supercategories() for x in dset.get_categories(cat)]
    cat_spur_all = dset.get_imgIds_by_class(present_classes=present, absent_classes=[target_object])
    print(f"  Found {len(cat_spur_all)} images without '{target_object}'")
    
    return dset, cat_spur_all

def main():
    parser = argparse.ArgumentParser(description='Adversarial Attack Pipeline - Notebook Compatible')
    parser.add_argument('--coco_path', type=str, default='/data/gpfs/datasets/COCO/', 
                       help='Path to COCO dataset')
    parser.add_argument('--target_object', type=str, default='boat', 
                       help='Target object to hallucinate')
    parser.add_argument('--num_images', type=int, default=10, 
                       help='Number of images to attack')
    parser.add_argument('--embeddings_path', type=str, default='../clip_embeddings.pt',
                       help='Path to precomputed CLIP embeddings')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    # Hyperparameters matching notebook exactly - all configurable
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (matches notebook)')
    parser.add_argument('--learning_rate', type=float, default=1.0,
                       help='Learning rate for SGD optimizer (matches notebook: lr=1)')
    parser.add_argument('--text_similarity_weight', type=float, default=5.0,
                       help='Weight for text embedding cosine similarity (matches notebook: 5 * sim1)')
    parser.add_argument('--clip_mse_weight', type=float, default=5.0,
                       help='Weight for CLIP embedding MSE loss (matches notebook: 5 * sim2)')
    parser.add_argument('--clip_gen_similarity_weight', type=float, default=5.0,
                       help='Weight for generated CLIP embedding similarity (matches notebook: 5 * sim3)')
    parser.add_argument('--guidance_scale', type=int, default=10,
                       help='Guidance scale for image generation (matches notebook)')
    parser.add_argument('--owlvit_threshold', type=float, default=0.3,
                       help='OWLv2 detection threshold (matches notebook: contains_obj_owlvit(generated, 0.3))')
    parser.add_argument('--use_enhanced_text', action='store_true', default=True,
                       help='Use enhanced text embeddings (compositional)')
    parser.add_argument('--coco_captions_path', type=str, 
                       default='/data/gpfs/datasets/COCO/annotations/captions_train2017.json',
                       help='Path to COCO captions for enhanced text embeddings')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Starting Adversarial Attack Pipeline (Notebook Compatible)")
    print(f"  Target object: {args.target_object}")
    print(f"  COCO path: {args.coco_path}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Number of images: {args.num_images}")
    print(f"  Device: {args.device}")
    print(f"  Loss weights: sim1 (text_similarity_weight)={args.text_similarity_weight}, sim2 (clip_mse_weight)={args.clip_mse_weight}, sim3 (clip_gen_similarity_weight)={args.clip_gen_similarity_weight}")
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    try:
        # Load models
        models = load_models(args.device)
        
        # Setup dataset
        dset, cat_spur_all = setup_dataset(args.coco_path, args.target_object)
        
        # Load similarities and get filtered indices
        print("🔄 Computing image similarities...")
        if os.path.exists(args.embeddings_path):
            exclude_indices = cat_spur_all
            filtered_indices, similarities = load_and_compute_similarity(
                models['clip_model'],
                exclude_indices=exclude_indices,
                object_text=args.target_object,
                embeddings_path=args.embeddings_path
            )
            print(f"  Found {len(filtered_indices)} candidate images")
            
            # Show top matches
            top5 = torch.topk(similarities, 5)
            print("  Top 5 most similar images:")
            for i, (score, idx) in enumerate(zip(top5.values, top5.indices)):
                print(f"    {i+1}. Index: {filtered_indices[idx]}, Similarity: {score:.4f}")
        else:
            print(f"⚠️  Embeddings file not found at {args.embeddings_path}")
            print("     Using first available images without similarity ranking")
            filtered_indices = torch.tensor(cat_spur_all[:args.num_images])
        
        # Configuration matching notebook exactly - all from args
        config = {
            'epochs': args.epochs,  # Match notebook variable name
            'learning_rate': args.learning_rate,  # Match notebook: lr=1
            'text_similarity_weight': args.text_similarity_weight,    # Match notebook: 5 * sim1
            'clip_mse_weight': args.clip_mse_weight,           # Match notebook: 5 * sim2  
            'clip_gen_similarity_weight': args.clip_gen_similarity_weight, # Match notebook: 5 * sim3
            'guidance_scale': args.guidance_scale,  # Match notebook
            'owlvit_threshold': args.owlvit_threshold,  # Match notebook
            'use_enhanced_text': args.use_enhanced_text,  # Our enhancement
            'coco_captions_path': args.coco_captions_path,  # For enhanced text
            'device': args.device
        }
        
        print(f"📋 Using configuration: {config}")
        
        # Run adversarial attack pipeline
        print(f"\n🎯 Running adversarial attack on {min(args.num_images, len(filtered_indices))} images...")
        print(f"📁 Generated images will be saved to: {os.path.join(args.output_dir, 'generated_images', args.target_object)}")
        
        results = run_attack(
            qwen_model=models['qwen'],
            clip_model=models['clip_model'],
            diffusion_pipe=models['pipe'],
            projector_model=models['projector'],
            processor=models['processor'],
            tokenizer=models['tokenizer'],
            clip_preprocess=models['clip_preprocess'],
            processor_owl=models['processor_owl'],
            model_owl=models['model_owl'],
            dataset=dset,
            image_indices=filtered_indices[:args.num_images].tolist(),
            target_object=args.target_object,
            config=config,
            output_dir=args.output_dir
        )
        
        # Save results
        results_file = os.path.join(args.output_dir, f'attack_results_{args.target_object}.json')
        
        # Save images and convert types for JSON serialization
        images_dir = os.path.join(args.output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        def save_image_and_get_path(img, prefix, index=None):
            """Save PIL image to disk and return relative path"""
            if hasattr(img, 'save'):  # It's a PIL Image
                if index is not None:
                    filename = f"{prefix}_{index}.png"
                else:
                    filename = f"{prefix}.png"
                filepath = os.path.join(images_dir, filename)
                img.save(filepath)
                return os.path.relpath(filepath, args.output_dir)  # Relative path
            return None
        
        def make_json_serializable(obj, path_prefix=""):
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif hasattr(obj, '__class__') and 'PIL' in str(type(obj)):
                # Save image and return path
                return save_image_and_get_path(obj, path_prefix)
            elif isinstance(obj, list):
                result = []
                for i, item in enumerate(obj):
                    if hasattr(item, '__class__') and 'PIL' in str(type(item)):
                        # For lists of images, use index in filename
                        saved_path = save_image_and_get_path(item, f"{path_prefix}_img", i)
                        result.append(saved_path)
                    else:
                        result.append(make_json_serializable(item, f"{path_prefix}_{i}"))
                return result
            elif isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    new_prefix = f"{path_prefix}_{k}" if path_prefix else str(k)
                    result[k] = make_json_serializable(v, new_prefix)
                return result
            else:
                return obj
        
        json_results = make_json_serializable(results, args.target_object)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✅ Attack completed successfully!")
        print(f"📊 Results summary:")
        print(f"  Overall Success Rate: {results['overall_success_rate']:.2%}")
        print(f"  Successful Attacks: {results['successful_attacks']}/{results['total_attempts']}")
        print(f"  Total Generated Images: {results['total_generated_images']}")
        print(f"  Results saved to: {results_file}")
        print(f"  Images directory: {results['images_directory']}")
        
        # Display some stats about the run
        if results['individual_results']:
            avg_epochs = np.mean([r['epochs'] for r in results['individual_results']])
            successful_results = [r for r in results['individual_results'] if r.get('success', False)]
            if successful_results:
                avg_success_epochs = np.mean([r['epochs'] for r in successful_results])
                print(f"  Average epochs (all): {avg_epochs:.1f}")
                print(f"  Average epochs (successful): {avg_success_epochs:.1f}")
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 