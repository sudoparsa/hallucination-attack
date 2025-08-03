import argparse
import os
from projector.projector import *
from data import COCO
from utils import *
from torch.utils.data import DataLoader, Subset
import torch
from tqdm import tqdm
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
import re
import subprocess


def get_args():
    parser = argparse.ArgumentParser(description="Hallucination Adversarial Attack")
    parser.add_argument("--model_name", type=str, required=True, choices=["llava", "qwen", "llama"], help="Name of the victim model")
    parser.add_argument("--projector_path", type=str, required=True, help="Path to the projector checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--cache_path", type=str, default="/fs/nexus-scratch/phoseini/cache/huggingface/hub", help="Path to cache directory for HF models")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save generated images")
    parser.add_argument("--target_object", type=str, required=True, help="Target object for the attack")

    # Attack parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate for the attack")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of optimization steps")
    parser.add_argument("--lambda_contrast", type=float, default=5.0, help="Encourages contrast against the target object in embedding space")
    parser.add_argument("--lambda_reg", type=float, default=5.0, help="Regularization term for the embedding space")

    # Others
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser.parse_args()

def get_log_name(args):
    return f"{args.target_object}_lr={args.lr}_steps={args.num_steps}_lambda_contrast={args.lambda_contrast}_lambda_reg={args.lambda_reg}_{''.join(os.path.basename(args.projector_path).split('.')[:-1])}"

def attack(args):
    logger.info("Loading model...")
    model, processor = get_model(args.model_name, args.cache_path)
    model.eval().cuda()
    logger.info("Loading CLIP Model...")
    clip_model, clip_preprocess = get_clip_model(args.cache_path)
    clip_model.eval().cuda()
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    logger.info("Loading Projector...")
    num_tokens, target_dim = get_num_tokens(args.model_name)
    context_dim = re.search(r"context_dim=(\d+)", args.projector_path)
    hidden_dim = re.search(r"hidden_dim=(\d+)", args.projector_path)

    context_dim = int(context_dim.group(1)) if context_dim else 4096
    hidden_dim = int(hidden_dim.group(1)) if hidden_dim else 4096

    checkpoint = torch.load(args.projector_path, map_location='cpu')
    projector = TokenMLP(num_tokens=num_tokens, context_dim=context_dim, clip_dim=1024, hidden_dim=hidden_dim, target_dim=target_dim)
    projector.load_state_dict(checkpoint)
    projector.eval().cuda()
    logger.info(f"Projector loaded from {args.projector_path} with context_dim={context_dim}, hidden_dim={hidden_dim}, target_dim={target_dim}")

    logger.info("Loading Diffusion Model...")
    pipe = get_diffusion_model(args.cache_path)

    logger.info("Loading Dataset...")
    dset = COCO(args.data_path, split='train', transform=(336, 336))
    present = [x  for cat in dset.get_all_supercategories() for x in dset.get_categories(cat) ]
    cat_spur_all = dset.get_imgIds_by_class(present_classes=present, absent_classes=[args.target_object])
    logger.info(f"Number of images without {args.target_object}: {len(cat_spur_all)}")

    prompts = get_prompt_templates()
    prompts = [p.format(obj=args.target_object) for p in prompts]
    logger.info("Using prompts: \n" + '\n'.join(prompts))

    yes_id = processor.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    no_id = processor.tokenizer("No", add_special_tokens=False)["input_ids"][0]
    text_tokens = tokenizer(args.target_object).to('cuda')

    logger.info(subprocess.check_output("nvidia-smi", text=True))
    asr = 0
    total_images_generated = 0
    total_images_optimized = 0
    for i, img_id in enumerate(cat_spur_all):
        total_images_optimized += 1
        if i >= 100:  # Limit to first 10 images for debugging
            break
        img_id = cat_spur_all[i]
        image, path = dset[img_id]
        logger.info(f"Processing image {i}/{len(cat_spur_all)} id={img_id}: {path}")

        prompt = random.choice(prompts)
        # output = get_vllm_output(model, processor, prompt, image, max_new_tokens=128)
        # logger.info(f"Prompt: {prompt} \nOutput: {output}")
        
        clip_emb = get_clip_image_features([image], clip_model, clip_preprocess)
        clip_emb = nn.Parameter(clip_emb).cuda()
        clip_emb.requires_grad = True
        optimizer = torch.optim.AdamW([clip_emb], lr=args.lr)
        clip_emb_initial = clip_emb.clone().detach()
        
        pbar = tqdm(range(args.num_steps), desc=f"Image {i+1}/{len(cat_spur_all)}")
        for step in pbar:
            prompt = random.choice(prompts)
            image_features = projector(clip_emb).half().squeeze(0)  # [num_tokens, target_dim]
            inputs = vllm_standard_preprocessing(processor, prompt, image)
            inputs = get_model_inputs(args.model_name, inputs, model, image_features)
            
            if arg.model_name != "llama":
                inputs['input_ids'] = None

            logits = model(**inputs).logits.float()
            logits_step = logits[:, -1, :]
            probs = torch.softmax(logits_step, dim=-1)
            
            # Calculate loss 
            prob_yes = probs[0, yes_id]
            prob_no = probs[0, no_id]
            log_prob_yes = -torch.log(prob_yes + 1e-8)

            text_embedding = clip_model.encode_text(text_tokens).detach()
            sim1 = F.cosine_similarity(clip_emb, text_embedding)
            sim2 = F.mse_loss(clip_emb, clip_emb_initial)

            loss = log_prob_yes + args.lambda_contrast * sim1 + args.lambda_reg * sim2

            pbar.set_postfix({"Loss": loss.item(), "Prob Yes": prob_yes.item(), "Prob No": prob_no.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True)
            
            gen_probs = torch.softmax(generated_ids.scores[0], dim=-1)

            gen_yes_prob = gen_probs[0, yes_id].item()
            gen_no_prob = gen_probs[0, no_id].item()
            pbar.set_postfix({"Gen Yes Prob": gen_yes_prob, "Gen No Prob": gen_no_prob})
            
            if gen_yes_prob > 0.95:
                # Generate image with diffusion model
                logger.info(f"Step={step}: Generating image for path={path}, id={img_id}, loss={gen_yes_prob}")
                result = pipe(
                negative_prompt="low quality, ugly, unrealistic",
                image_embeds=clip_emb.half(),
                guidance_scale=10
                )
                generated = result.images[0]
                torch.cuda.empty_cache()

                output = get_vllm_output(model, processor, prompt, generated, max_new_tokens=128)
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Output: {output}")
                output_path = f"logs/attack/{args.model_name}/{get_log_name(args)}/images/{i}_{img_id}_{step}.png"
                generated.save(output_path)
                logger.info(f"Saved generated image to {output_path}")
                if output.lower().startswith("yes"):
                    logger.info(f"Attack successful for image {img_id} at step {step}")
                    asr += 1
                else:
                    logger.info(f"Attack failed for image {img_id} at step {step}")
                total_images_generated += 1
                break
            if step == args.num_steps - 1:
                logger.info(f"Attack failed for image {img_id} after {args.num_steps} steps. Prob Yes: {gen_yes_prob}, Prob No: {gen_no_prob}")
    
    logger.info(f"Attack completed. Total images optimized: {total_images_generated}/{total_images_optimized} Total images generated: {asr}/{total_images_generated} ASR: {asr / total_images_generated if total_images_generated > 0 else 0:.2f} {total_images_generated/total_images_optimized}")






if __name__ == "__main__":
    args = get_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO

    # create folder
    os.makedirs(f"logs", exist_ok=True)
    os.makedirs(f"logs/attack", exist_ok=True)
    os.makedirs(f"logs/attack/{args.model_name}", exist_ok=True)
    os.makedirs(f"logs/attack/{args.model_name}/{get_log_name(args)}", exist_ok=True)
    os.makedirs(f"logs/attack/{args.model_name}/{get_log_name(args)}/images", exist_ok=True)

    logging.basicConfig(format="### %(message)s ###")

    logger = logging.getLogger("HallucinationAttack")
    logger.setLevel(level=logging_level)

    logger.addHandler(logging.FileHandler(f"logs/attack/{args.model_name}/{get_log_name(args)}/log.txt", mode='w'))

    # Setting Seed
    set_seed(args.seed)

    logger.info(get_log_name(args))
    logger.info(f"Arguments: {args}")

    attack(args)