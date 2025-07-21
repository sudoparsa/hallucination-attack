import torch
import random
from typing import List, Dict
import json
import open_clip

class EnhancedTextRepresentations:
    """Enhanced text representations for more effective adversarial attacks"""
    
    def __init__(self, clip_model, tokenizer, coco_captions_path: str = None):
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.coco_captions = self._load_coco_captions(coco_captions_path) if coco_captions_path else None
        
    def _load_coco_captions(self, captions_path: str) -> Dict:
        """Load COCO captions for mining contextual descriptions"""
        with open(captions_path, 'r') as f:
            return json.load(f)
    
    def get_contextual_templates(self, obj: str) -> List[str]:
        """Generate contextual templates based on object type"""
        object_contexts = {
            'boat': [
                "A {obj} sailing on the water",
                "A {obj} docked at the harbor", 
                "A {obj} floating in the lake",
                "People enjoying a ride on a {obj}",
                "A fishing {obj} in the ocean",
                "A small {obj} near the shore"
            ],
            'car': [
                "A {obj} parked on the street",
                "A {obj} driving down the highway",
                "A vintage {obj} in the garage",
                "A red {obj} in the parking lot"
            ],
            # Add more objects as needed
        }
        
        generic_templates = [
            "A scene featuring a {obj}",
            "An image showing a {obj}",
            "A photograph with a {obj}",
            "A picture containing a {obj}"
        ]
        
        return object_contexts.get(obj, []) + generic_templates
    
    def mine_captions_with_object(self, obj: str, max_captions: int = 50) -> List[str]:
        """Mine COCO captions that contain the target object"""
        if not self.coco_captions:
            return []
            
        matching_captions = []
        for annotation in self.coco_captions['annotations']:
            caption = annotation['caption'].lower()
            if obj.lower() in caption and len(matching_captions) < max_captions:
                matching_captions.append(annotation['caption'])
        
        return matching_captions
    
    def get_compositional_embeddings(self, obj: str, device: str = 'cuda') -> torch.Tensor:
        """Create compositional text embeddings combining multiple representations"""
        
        # 1. Simple sentence embedding (better than raw token)
        simple_sentence = f"A photo of a {obj}"
        simple_tokens = self.tokenizer([simple_sentence]).to(device)
        simple_embed = self.clip_model.encode_text(simple_tokens)
        
        # 2. Contextual template embeddings
        contextual_templates = self.get_contextual_templates(obj)
        context_embeds = []
        for template in contextual_templates[:3]:  # Use top 3 templates
            text = template.format(obj=obj)
            tokens = self.tokenizer([text]).to(device)
            embed = self.clip_model.encode_text(tokens)
            context_embeds.append(embed)
        
        # 3. COCO caption embeddings (if available)
        caption_embeds = []
        if self.coco_captions:
            captions = self.mine_captions_with_object(obj, max_captions=5)
            for caption in captions:
                tokens = self.tokenizer([caption]).to(device)
                embed = self.clip_model.encode_text(tokens)
                caption_embeds.append(embed)
        
        # 4. Combine embeddings with weighted average
        all_embeds = [simple_embed] + context_embeds + caption_embeds
        weights = torch.tensor([0.3] + [0.4/len(context_embeds)]*len(context_embeds) + 
                              [0.3/len(caption_embeds)]*len(caption_embeds)).to(device)
        
        if len(all_embeds) > 1:
            combined_embed = torch.stack(all_embeds).squeeze(1)
            weighted_embed = torch.sum(combined_embed * weights.unsqueeze(-1), dim=0)
            return weighted_embed
        else:
            return simple_embed.squeeze(0)

    def get_adaptive_text_embedding(self, obj: str, image_embed: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
        """Adaptively select best text representation based on image content"""
        
        # Get multiple text representations
        representations = {
            'simple': self.tokenizer([f"A photo of a {obj}"]).to(device),
            'contextual': [self.tokenizer([template.format(obj=obj)]).to(device) 
                          for template in self.get_contextual_templates(obj)[:3]]
        }
        
        # Compute similarities with image
        similarities = {}
        
        # Simple token similarity
        simple_embed = self.clip_model.encode_text(representations['simple'])
        similarities['simple'] = torch.cosine_similarity(image_embed.unsqueeze(0), simple_embed)
        
        # Contextual similarities
        for i, tokens in enumerate(representations['contextual']):
            context_embed = self.clip_model.encode_text(tokens)
            sim = torch.cosine_similarity(image_embed.unsqueeze(0), context_embed)
            similarities[f'contextual_{i}'] = sim
        
        # Select best representation
        best_key = max(similarities.keys(), key=lambda k: similarities[k])
        
        if best_key == 'simple':
            return simple_embed.squeeze(0)
        else:
            idx = int(best_key.split('_')[1])
            tokens = representations['contextual'][idx]
            return self.clip_model.encode_text(tokens).squeeze(0)

# Usage example for integration into main code:
def enhanced_compute_loss(log_prob_yes, clip_embed, obj_hallucination, clip_model, tokenizer, 
                         clip_embed_orig, clip_embed_gen=None, device='cuda'):
    """Enhanced loss function using improved text representations"""
    
    text_rep = EnhancedTextRepresentations(clip_model, tokenizer)
    
    # Get enhanced text embedding instead of simple token
    text_embedding = text_rep.get_compositional_embeddings(obj_hallucination, device)
    
    # Original loss terms
    sim1 = torch.nn.functional.cosine_similarity(clip_embed, text_embedding.unsqueeze(0))
    sim2 = torch.nn.functional.mse_loss(clip_embed, clip_embed_orig[0])
    sim3 = 0
    
    if clip_embed_gen is not None:
        sim3 = torch.nn.functional.cosine_similarity(clip_embed, clip_embed_gen.unsqueeze(0))
        return log_prob_yes + 5 * sim1 + 5 * sim2 + 5 * sim3, sim1, sim2, sim3
    else:
        return log_prob_yes + 5 * sim1 + 5 * sim2, sim1, sim2, sim3 