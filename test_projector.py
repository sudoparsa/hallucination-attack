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
    parser.add_argument("--target_objects", type=str, required=True, help="Target object for the attack")

    # Others
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser.parse_args()

def get_log_name(args):
    return f"{args.target_objects}_{''.join(os.path.basename(args.projector_path).split('.')[:-1])}"

def test(args):
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
    logger.info("Loading Dataset...")
    dset = COCO(args.data_path, split='train', transform=(336, 336))
    present = [x  for cat in dset.get_all_supercategories() for x in dset.get_categories(cat) ]
    for target_object in args.target_objects.split(','):
        cat_spur_all = dset.get_imgIds_by_class(present_classes=present, absent_classes=[target_object])
        logger.info(f"Number of images without {target_object}: {len(cat_spur_all)}")

        prompts = get_prompt_templates()
        prompts = [p.format(obj=target_object) for p in prompts]
        logger.info("Using prompts: \n" + '\n'.join(prompts))

        yes_id = processor.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
        no_id = processor.tokenizer("No", add_special_tokens=False)["input_ids"][0]

        correct_p = 0
        correct_mllm = 0
        total = 0

        for i, img_id in enumerate(tqdm(cat_spur_all[:100])):  # Evaluate only first 100 images
            image, path = dset[img_id]
            prompt = random.choice(prompts)

            # Get projected tokens from CLIP
            clip_emb = get_clip_image_features([image], clip_model, clip_preprocess).clone().detach()
            image_features = projector(clip_emb).half().squeeze(0)
            

            # Prepare input and generate
            inputs = vllm_standard_preprocessing(processor, prompt, image)
            inputs = get_model_inputs(args.model_name, inputs, model, image_features)
            output = model.generate(**inputs, max_new_tokens=1)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)]
            decoded_p = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            if "no" in decoded_p.lower():
                correct_p += 1

            decoded_m = get_vllm_output(model, processor, prompt, image, max_new_tokens=1)
            if "no" in decoded_m.lower():
                correct_mllm += 1

            logger.info(f"[{i}] Prompt: '{prompt}' | Output projector: '{decoded_p}' | Output MLLM: '{decoded_m}'")
            
            total += 1

        acc_p = correct_p / total * 100
        logger.info(f"Object: {target_object} | Accuracy: {acc_p:.2f}% ({correct_p}/{total})")
        acc_mllm = correct_mllm / total * 100
        logger.info(f"Object: {target_object} | MLLM Accuracy: {acc_mllm:.2f}% ({correct_mllm}/{total})")
        

        correct_p = 0
        correct_mllm = 0
        total = 0

        cat_spur_all = dset.get_imgIds_by_class(present_classes=[target_object])
        logger.info(f"Number of images with {target_object}: {len(cat_spur_all)}")

        for i, img_id in enumerate(tqdm(cat_spur_all[:100])):  # Evaluate only first 100 images
            image, path = dset[img_id]
            prompt = random.choice(prompts)

            # Get projected tokens from CLIP
            clip_emb = get_clip_image_features([image], clip_model, clip_preprocess).clone().detach()
            image_features = projector(clip_emb).half().squeeze(0)

            # Prepare input and generate
            inputs = vllm_standard_preprocessing(processor, prompt, image)
            inputs = get_model_inputs(args.model_name, inputs, model, image_features)
            output = model.generate(**inputs, max_new_tokens=1)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)]
            decoded_p = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            

            if "yes" in decoded_p.lower():
                correct_p += 1
            

            decoded_m = get_vllm_output(model, processor, prompt, image, max_new_tokens=1)
            
            if "yes" in decoded_m.lower():
                correct_mllm += 1

            logger.info(f"[{i}] Prompt: '{prompt}' | Output projector: '{decoded_p}' | Output MLLM: '{decoded_m}'")
            

            total += 1

        acc_p = correct_p / total * 100
        logger.info(f"Object: {target_object} | Accuracy: {acc_p:.2f}% ({correct_p}/{total})")
        acc_mllm = correct_mllm / total * 100
        logger.info(f"Object: {target_object} | MLLM Accuracy: {acc_mllm:.2f}% ({correct_mllm}/{total})")



                





if __name__ == "__main__":
    args = get_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO

    # create folder
    os.makedirs(f"logs", exist_ok=True)
    os.makedirs(f"logs/projector", exist_ok=True)
    os.makedirs(f"logs/projector/test", exist_ok=True)
    os.makedirs(f"logs/projector/test/{args.model_name}/{get_log_name(args)}", exist_ok=True)

    logging.basicConfig(format="### %(message)s ###")

    logger = logging.getLogger("HallucinationAttack")
    logger.setLevel(level=logging_level)

    logger.addHandler(logging.FileHandler(f"logs/projector/test/{args.model_name}/{get_log_name(args)}/log.txt", mode='w'))

    # Setting Seed
    set_seed(args.seed)

    logger.info(get_log_name(args))
    logger.info(f"Arguments: {args}")

    test(args)