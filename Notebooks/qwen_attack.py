import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from torchvision import transforms
from qwen_vl_utils import process_vision_info
import open_clip

# Import only the text representations enhancement
from enhanced_text_representations import EnhancedTextRepresentations

class AdversarialAttack:
    """Adversarial attack matching notebook functionality exactly"""
    
    def __init__(self, models_dict: Dict, config: Dict = None):
        """
        Initialize adversarial attack system
        Args:
            models_dict: Dictionary containing all models (qwen, clip, pipe, etc.)
            config: Configuration dictionary
        """
        self.models = models_dict
        self.device = config.get('device', 'cuda')
        self.config = config or self.get_default_config()
        
        # Initialize only text enhancement (keep this improvement)
        self.text_enhancer = EnhancedTextRepresentations(
            self.models['clip_model'], 
            self.models['tokenizer'],
            self.config.get('coco_captions_path')
        )
        
    def get_default_config(self) -> Dict:
        """Get default configuration matching notebook"""
        return {
            'epochs': 100,  # Match notebook variable name
            'learning_rate': 1.0,  # Match notebook: lr=1
            'text_similarity_weight': 5.0,    # Match notebook: 5 * sim1
            'clip_mse_weight': 5.0,           # Match notebook: 5 * sim2  
            'clip_gen_similarity_weight': 5.0, # Match notebook: 5 * sim3
            'guidance_scale': 10,  # Match notebook
            'owlvit_threshold': 0.3,  # Match notebook: contains_obj_owlvit(generated, 0.3)
            'use_enhanced_text': True,  # Our only enhancement
            'device': 'cuda'
        }
    
    def get_enhanced_question_templates(self, target_object: str) -> List[str]:
        """Get enhanced question templates for hallucination testing"""
        # Core templates from notebook (exact match for compatibility)
        core_templates = [
            "Do you see a {obj} in the image?",
            "Is there a {obj} here?", 
            "Does the image contain a {obj}?",
            "Can you find a {obj} in this picture?",
            "Would you say there's a {obj} here?",
        ]
        
        # Enhanced contextual templates for better coverage
        contextual_templates = [
            "Looking at this image, can you identify a {obj}?",
            "What objects do you see? Is there a {obj}?",
            "Please examine the image carefully. Do you notice a {obj}?",
            "Among the items visible, is there a {obj}?",
            "Can you spot a {obj} anywhere in this scene?",
            "Are you able to detect a {obj} in this photograph?",
            "Would you confirm the presence of a {obj} in this image?",
            "Is a {obj} visible somewhere in this picture?",
        ]
        
        return core_templates + contextual_templates
    
    def _create_enhanced_text_embedding(self, target_object: str) -> torch.Tensor:
        """Create enhanced compositional text embedding for the target object"""
        if self.config['use_enhanced_text']:
            print(f"🎯 Creating enhanced compositional embedding for '{target_object}'")
            print("   - Using contextual templates, COCO captions, and weighted combinations")
            
            # Use the enhanced text representation system
            compositional_embedding = self.text_enhancer.get_compositional_embeddings(
                target_object, self.device
            ).detach()
            
            # Ensure proper tensor dimensions
            if compositional_embedding.dim() > 1:
                compositional_embedding = compositional_embedding.squeeze()
            
            return compositional_embedding
        else:
            print(f"📝 Using simple text embedding for '{target_object}'")
            
            # Fallback to simple sentence embedding (better than raw token)
            sentence = f"A photo of a {target_object}"
            tokenizer = open_clip.get_tokenizer('ViT-H-14')
            tokens = tokenizer([sentence]).to(self.device)
            simple_embedding = self.models['clip_model'].encode_text(tokens).detach()
            
            # Ensure proper tensor dimensions
            if simple_embedding.dim() > 1:
                simple_embedding = simple_embedding.squeeze()
            
            return simple_embedding

    def get_clip_embedding(self, image: Image.Image) -> torch.Tensor:
        """Get CLIP embedding - exact match to notebook"""
        inputs = torch.stack([self.models['clip_preprocess'](image)]).to('cuda')
        with torch.no_grad():
            embedding = self.models['clip_model'].encode_image(inputs)[0]
        return embedding
    
    def get_qwen_inputs(self, prompt: str, image: Image.Image, clip_embed: torch.Tensor) -> Dict:
        """Get Qwen inputs - exact match to notebook"""
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}
        ]
        mean = self.models['projector'](clip_embed)
        qwen_tokens = mean.repeat(64, 1)  # Match notebook exactly

        text = self.models['processor'].apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, _ = process_vision_info(messages)

        inputs = self.models['processor'](
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to('cuda')

        inputs_embeds = self.models['qwen'].get_input_embeddings()(inputs['input_ids'])

        n_image_tokens = (inputs['input_ids'] == self.models['qwen'].config.image_token_id).sum().item()
        n_image_features = qwen_tokens.shape[0]

        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens={n_image_tokens}, features={n_image_features}"
            )

        image_mask = (inputs['input_ids'] == self.models['qwen'].config.image_token_id)\
            .unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

        image_embeds = qwen_tokens.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        inputs['inputs_embeds'] = inputs_embeds

        return inputs

    def get_qwen_probabilities(self, inputs: Dict) -> torch.Tensor:
        """Get Qwen probabilities - exact match to notebook"""
        logits = self.models['qwen'](**inputs).logits.float()
        logits_step = logits[:, -1, :]
        probs = torch.softmax(logits_step, dim=-1)
        return probs

    def compute_loss(self, negative_log_prob_yes: torch.Tensor, optimizable_embedding: torch.Tensor, 
                    target_text_embedding: torch.Tensor, original_embedding_ref: torch.Tensor, 
                    generated_embedding: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute adversarial loss with three similarity terms - exact match to notebook"""
        
        # Ensure tensor dimensions are compatible for cosine similarity calculations
        if target_text_embedding.dim() > 1:
            target_text_embedding = target_text_embedding.squeeze()
        if optimizable_embedding.dim() > 1:
            optimizable_embedding = optimizable_embedding.squeeze()
            
        # Similarity term 1: Text-image alignment (encourage similarity to target object)
        text_similarity = F.cosine_similarity(
            optimizable_embedding.unsqueeze(0), 
            target_text_embedding.unsqueeze(0), 
            dim=1
        )
        
        # Similarity term 2: Original embedding preservation (MSE loss)
        clip_mse_loss = F.mse_loss(optimizable_embedding, original_embedding_ref[0])
        
        # Similarity term 3: Generated embedding similarity (if available)
        generated_similarity = torch.tensor(0.0, device=optimizable_embedding.device)
        if generated_embedding is not None:
            if generated_embedding.dim() > 1:
                generated_embedding = generated_embedding.squeeze()
            generated_similarity = F.cosine_similarity(
                optimizable_embedding.unsqueeze(0), 
                generated_embedding.unsqueeze(0), 
                dim=1
            )
        
        # Combine all loss terms with notebook weights
        total_loss = (
            negative_log_prob_yes + 
            self.config['text_similarity_weight'] * text_similarity + 
            self.config['clip_mse_weight'] * clip_mse_loss + 
            self.config['clip_gen_similarity_weight'] * generated_similarity
        )
        
        return total_loss, text_similarity, clip_mse_loss, generated_similarity

    def generate_and_validate_image(self, clip_embed: torch.Tensor, prompt: str) -> Tuple[Image.Image, str]:
        """Generate and validate image - exact match to notebook"""
        result = self.models['pipe'](
            negative_prompt="low quality, ugly, unrealistic",
            image_embeds=clip_embed.unsqueeze(0),
            guidance_scale=self.config['guidance_scale']
        )
        generated = result.images[0]
        torch.cuda.empty_cache()

        output = self.ask_qwen(prompt, generated)

        if output[0].lower().startswith("yes"):
            return generated, output
        return None, output

    def ask_qwen(self, prompt: str, img: Image.Image) -> List[str]:
        """Ask Qwen - exact match to notebook"""
        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img
                        },
                        {"type": "text", "text": prompt},
                    ],
                }  
        
            ]
        
            # Preparation for inference
            text = self.models['processor'].apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.models['processor'](
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            # Inference: Generation of the output
            generated_ids = self.models['qwen'].generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.models['processor'].batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
        return output_text

    def contains_obj_owlvit(self, image: Image.Image, score_threshold: float = 0.1) -> bool:
        """Check if object is in image using OWLv2 - exact match to notebook"""
        texts = [[self.target_object]]  # supports multiple labels
        inputs = self.models['owl_processor'](text=texts, images=image, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.models['owl_model'](**inputs)

        target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
        results = self.models['owl_processor'].post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=score_threshold
        )[0]

        for score, label in zip(results["scores"], results["labels"]):
            if label == 0 and score > score_threshold:  
                return True
        return False

    def get_obj_owlvit(self, image: Image.Image, score_threshold: float = 0.1) -> Image.Image:
        """Get object crop from image using OWLv2 - exact match to notebook"""
        texts = [[self.target_object]]  # supports multiple labels
        inputs = self.models['owl_processor'](text=texts, images=image, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.models['owl_model'](**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to("cuda")  
        results = self.models['owl_processor'].post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=score_threshold
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        if len(boxes) == 0:
            return None

        top_idx = scores.argmax().item()
        box = boxes[top_idx].tolist() 
        obj_crop = image.crop(box)
        return obj_crop

    def get_qwen_loss(self, probs: torch.Tensor, yes_id: int, no_id: int) -> torch.Tensor:
        """Get Qwen loss - exact match to notebook"""
        prob_yes = probs[0, yes_id]
        prob_no = probs[0, no_id]
        log_prob_yes = -torch.log(prob_yes + 1e-8)
        return log_prob_yes

    def attack_single_image(self, image: Image.Image, target_object: str, output_dir: str = "./") -> Dict:
        """Attack single image - exact match to notebook main_loop logic"""
        
        self.target_object = target_object  # Store for OWLv2 functions
        question_templates = self.get_enhanced_question_templates(target_object)
        
        print(f"Attacking image for object: {target_object}")
        
        # Initial hallucination check - exact match to notebook logic
        initial_question = random.choice(question_templates).format(obj=target_object)
        resized_image = image.resize((224, 224))
        
        original_clip_embedding = self.get_clip_embedding(resized_image)
        initial_response = self.ask_qwen(initial_question, resized_image)
        if initial_response[0].lower().startswith("yes"):
            print("✅ Image already triggers hallucination, skipping...")
            return {'success': True, 'epochs': 0, 'initial_success': True}

        # Setup adversarial optimization - exact match to notebook
        optimizable_clip_embedding = nn.Parameter(original_clip_embedding)
        sgd_optimizer = torch.optim.SGD([optimizable_clip_embedding], lr=self.config['learning_rate'], momentum=0.9, nesterov=True)
        original_clip_embedding_reference = optimizable_clip_embedding.clone().detach().unsqueeze(0)
        generated_clip_embedding = None

        # Create enhanced text representation for optimization target
        enhanced_text_embedding = self._create_enhanced_text_embedding(target_object)

        results = {
            'success': False,
            'epochs': self.config['epochs'],
            'generated_images': [],
            'responses': [],
            'image_paths': []
        }

        # Adversarial training loop - exact match to notebook logic
        for epoch in tqdm(range(self.config['epochs']), desc=f"Optimizing for {target_object}"):
            # Generate random question for this epoch
            current_question = random.choice(question_templates).format(obj=target_object)
            
            # Forward pass through Qwen model
            qwen_inputs = self.get_qwen_inputs(current_question, resized_image, optimizable_clip_embedding)
            qwen_probabilities = self.get_qwen_probabilities(qwen_inputs)
            
            # Get token IDs for "Yes" and "No" responses
            yes_token_id = self.models['processor'].tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
            no_token_id = self.models['processor'].tokenizer("No", add_special_tokens=False)["input_ids"][0]
            
            # Compute primary loss term (negative log probability of "Yes")
            negative_log_prob_yes = self.get_qwen_loss(qwen_probabilities, yes_token_id, no_token_id)
            
            # Compute total loss with similarity terms
            total_loss, text_similarity, clip_mse_loss, generated_similarity = self.compute_loss(
                negative_log_prob_yes, optimizable_clip_embedding, enhanced_text_embedding, 
                original_clip_embedding_reference, generated_clip_embedding
            )
            
            # Log similarity components for monitoring
            print(f"📊 Epoch {epoch}: text_sim={text_similarity.item():.4f}, "
                  f"clip_mse={clip_mse_loss.item():.4f}, gen_sim={generated_similarity.item():.4f}")

            # Backpropagation step
            sgd_optimizer.zero_grad()
            total_loss.backward()
            sgd_optimizer.step()

            # Check if model is ready to generate hallucination - exact match to notebook
            generation_test_outputs = self.models['qwen'].generate(
                **qwen_inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
            generation_probabilities = torch.softmax(generation_test_outputs.scores[0], dim=-1)

            yes_probability = generation_probabilities[0, yes_token_id].item()
            no_probability = generation_probabilities[0, no_token_id].item()
            
            print(f"🎲 Generation decision: Yes={yes_probability:.3f}, No={no_probability:.3f}")

            # Generate image if yes_prob > no_prob - exact match to notebook logic
            if yes_probability > no_probability:
                print(f"🎨 Generating image (Yes > No, epoch {epoch})...")
                generated_image, generation_response = self.generate_and_validate_image(
                    optimizable_clip_embedding.half(), current_question
                )
                if generated_image is not None:
                    # Save generated image with metadata
                    images_output_dir = os.path.join(output_dir, 'generated_images', target_object)
                    os.makedirs(images_output_dir, exist_ok=True)
                    
                    generated_image_path = os.path.join(
                        images_output_dir, 
                        f"epoch_{epoch:04d}_prob_{yes_probability:.3f}.png"
                    )
                    generated_image.save(generated_image_path)
                    print(f"💾 Saved generated image: {generated_image_path}")
                    
                    # Store results
                    results['generated_images'].append(generated_image)
                    results['responses'].append(generation_response)
                    results['image_paths'].append(generated_image_path)
                    
                    # Update generated clip embedding for sim3 - exact match to notebook
                    print(f"🔍 Checking if generated image contains {target_object}...")
                    object_found = False
                    if self.contains_obj_owlvit(generated_image, self.config['owlvit_threshold']):
                        object_crop = self.get_obj_owlvit(generated_image)
                        if object_crop is not None:
                            crop_inputs = torch.stack([self.models['clip_preprocess'](object_crop)]).to('cuda')
                            with torch.no_grad():
                                generated_clip_embedding = self.models['clip_model'].encode_image(crop_inputs).float()[0].detach()
                            print(f"✅ Generated image contains {target_object} - updated sim3 embedding")
                            object_found = True
                        else:
                            print(f"⚠️ Could not extract {target_object} crop from generated image")
                    else:
                        print(f"❌ Generated image does not contain detectable {target_object}")

                    
                    # Check if we achieved successful hallucination
                    if generation_response[0].lower().startswith("yes") and not object_found and yes_probability > 0.8:
                        print(f"🎉 SUCCESS! Generated hallucination image at epoch {epoch}!")
                        results['success'] = True
                        results['epochs'] = epoch
                        break # you can remove this break to continue optimization 

        return results
    
    def run_attack_on_images(self, dataset, image_indices: List[int], 
                           target_object: str, output_dir: str = "./") -> Dict:
        """Run attack on multiple images - simplified version matching notebook structure"""
        
        all_results = []
        successful_attacks = 0
        
        print(f"Starting attack on {len(image_indices)} images")
        print(f"Target object: {target_object}")
        print(f"Configuration: {self.config}")
        
        for i, img_idx in enumerate(image_indices):
            print(f"\n--- Image {i+1}/{len(image_indices)} (Index: {img_idx}) ---")
            
            image = dataset[img_idx]
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            elif not isinstance(image, Image.Image):
                if hasattr(image, 'convert'):
                    image = image.convert('RGB')
                else:
                    raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Attack single image
            result = self.attack_single_image(image, target_object, output_dir)
            all_results.append(result)
            
            if result.get('success', False):
                successful_attacks += 1
        
        # Calculate results
        overall_success_rate = successful_attacks / len(all_results) if all_results else 0.0
        avg_epochs = np.mean([r['epochs'] for r in all_results]) if all_results else 0
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS:")
        print(f"Overall Success Rate: {overall_success_rate:.2f}")
        print(f"Successful Attacks: {successful_attacks}/{len(all_results)}")
        print(f"Average Epochs: {avg_epochs:.1f}")
        print(f"{'='*60}")
        
        return {
            'overall_success_rate': overall_success_rate,
            'successful_attacks': successful_attacks,
            'total_attempts': len(all_results),
            'individual_results': all_results,
            'total_generated_images': sum(len(r.get('image_paths', [])) for r in all_results),
            'images_directory': os.path.join(output_dir, 'generated_images')
        }

# Convenience functions for notebook integration
def create_adversarial_attack_system(models_dict, config=None):
    """Create adversarial attack system matching notebook exactly"""
    return AdversarialAttack(models_dict, config)

def run_attack(qwen_model, clip_model, diffusion_pipe, projector_model, 
                                 processor, tokenizer, clip_preprocess, processor_owl, model_owl,
                                 dataset, image_indices, target_object, config=None, output_dir="./"):
    """
    Complete attack pipeline matching notebook functionality exactly
    
    This is a drop-in replacement for the notebook's main_loop function
    """
    
    # Build models dictionary
    models_dict = {
        'qwen': qwen_model,
        'clip_model': clip_model,
        'pipe': diffusion_pipe,
        'projector': projector_model,
        'processor': processor,
        'tokenizer': tokenizer,
        'clip_preprocess': clip_preprocess,
        'owl_processor': processor_owl,
        'owl_model': model_owl
    }
    
    # Create attack system
    attack_system = create_adversarial_attack_system(models_dict, config)
    
    # Run attack
    results = attack_system.run_attack_on_images(
        dataset=dataset,
        image_indices=image_indices,
        target_object=target_object,
        output_dir=output_dir
    )
    
    return results

def replace_main_loop_with_notebook_attack(dset, filtered_indices, obj_hallucination,
                                         qwen, clip_model, pipe, model, processor, 
                                         tokenizer, clip_preprocess, processor_owl, model_owl,
                                         config=None):
    """
    Exact drop-in replacement for the notebook's main_loop function
    
    Usage in notebook:
    # Instead of: main_loop(dset, obj_hallucination)  
    # Use: replace_main_loop_with_notebook_attack(dset, filtered_indices, obj_hallucination, ...)
    """
    
    print("🚀 Starting Notebook-Compatible Attack")
    print(f"Target object: {obj_hallucination}")
    print(f"Number of candidate images: {len(filtered_indices)}")
    
    # Default config matching notebook exactly
    notebook_config = {
        'epochs': 100,  
        'learning_rate': 1.0,  
        'text_similarity_weight': 5.0,    
        'clip_mse_weight': 5.0,           
        'clip_gen_similarity_weight': 5.0, 
        'guidance_scale': 10,  
        'owlvit_threshold': 0.3,  
        'use_enhanced_text': True,  
    }
    
    if config:
        notebook_config.update(config)
    
    results = run_attack(
        qwen_model=qwen,
        clip_model=clip_model,
        diffusion_pipe=pipe,
        projector_model=model,
        processor=processor,
        tokenizer=tokenizer,
        clip_preprocess=clip_preprocess,
        processor_owl=processor_owl,
        model_owl=model_owl,
        dataset=dset,
        image_indices=filtered_indices,
        target_object=obj_hallucination,
        config=notebook_config
    )
    
    print("✅ Notebook-Compatible Attack Complete!")
    return results 