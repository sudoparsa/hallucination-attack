import argparse
import os
from projector.projector import *
from utils import *
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt





def get_args():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--model_name", type=str, default="llava", choices=["qwen", "llava"], help="model name")
    parser.add_argument("--coco_dir", type=str, default="/fs/cml-datasets/coco/images")
    parser.add_argument("--cache_path", type=str, default="/fs/nexus-scratch/phoseini/cache/huggingface/hub")
    parser.add_argument("--save_projector_dir", type=str, default="./projector/models2")
    parser.add_argument("--embeddings_dir", type=str, default="./projector/embeddings")

    # Training
    parser.add_argument("--save_emb_batch_size", type=int, default=64, help="Batch size for saving embeddings")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--online_training", action="store_true", help="If set, will train projector online instead of using precomputed embeddings")

    # Others
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    return parser.parse_args()


def get_name(args):
    return f"{args.model_name}_projector"

def get_log_name(args):
    return f"{args.model_name}_bs={args.batch_size}_lr={args.lr}_epochs={args.epochs}_{args.online_training}"


@torch.no_grad()
def save_embeddings(loader, model, processor, clip_model, clip_preprocess, output_path):
    all_model, all_clip, all_paths = [], [], []

    for images, paths in tqdm(loader, desc=f"Extracting embeddings"):
        clip_embs = get_clip_image_features(images, clip_model, clip_preprocess)
        llava_embs = get_llava_image_features(images, model, processor)
        all_clip.append(clip_embs)
        all_model.append(llava_embs)
        all_paths.extend(paths)
    
    all_clip_cat = torch.cat(all_clip)
    all_model_cat = torch.cat(all_model)
    logger.info(f"Clip embeddings shape: {all_clip_cat.shape}, Model embeddings shape: {all_model_cat.shape}")
    out = {
        "clip": all_clip_cat,
        "model": all_model_cat,
        "paths": all_paths
    }
    torch.save(out, output_path)

    

def train_projector(args):
    logger.info("Starting projector training...")
    logger.info(f"Loading {args.model_name}")
    model, processor = get_model(args.model_name, args.cache_path)
    model = model.to("cuda").eval()
    logger.info(f"Loading CLIP model")
    clip_model, clip_preprocess = get_clip_model(args.cache_path)
    clip_model = clip_model.to("cuda").eval()

    
    if not args.online_training:
        os.makedirs(args.embeddings_dir, exist_ok=True)

        train_embeddings_path = os.path.join(args.embeddings_dir, f"{args.model_name}_train_emb.pt")
        if not os.path.exists(train_embeddings_path):
            logger.info(f"Loading COCO train dataset from {args.coco_dir}")
            train_dataset = ImageDataset(os.path.join(args.coco_dir, "train2017"))
            train_loader = DataLoader(train_dataset, batch_size=args.save_emb_batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))
            save_embeddings(train_loader, model, processor, clip_model, clip_preprocess,
                            output_path=train_embeddings_path)
        
        val_embeddings_path = os.path.join(args.embeddings_dir, f"{args.model_name}_val_emb.pt")
        if not os.path.exists(val_embeddings_path):
            logger.info(f"Loading COCO val dataset from {args.coco_dir}")
            val_dataset = ImageDataset(os.path.join(args.coco_dir, "val2017"))
            val_loader = DataLoader(val_dataset, batch_size=args.save_emb_batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))
            save_embeddings(val_loader, model, processor, clip_model, clip_preprocess,
                            output_path=val_embeddings_path)
        
        train_dataset = CachedEmbeddingDataset(train_embeddings_path)
        val_dataset = CachedEmbeddingDataset(val_embeddings_path)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        logger.info(f"Using online training, no precomputed embeddings will be used.")
        train_dataset = ImageDataset(os.path.join(args.coco_dir, "train2017"), resize=(336, 336))
        val_dataset = ImageDataset(os.path.join(args.coco_dir, "val2017"), resize=(336, 336))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))
    
    logger.info(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
    projector = ClipProjector(1176, get_target_dim(args.model_name)).cuda()

    best_loss = float("inf")
    train_losses = []
    val_losses = []
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr)

    logger.info(f"Training projector with model: {args.model_name}, batch size: {args.batch_size}, learning rate: {args.lr}, epochs: {args.epochs}")
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.epochs}")
        projector.train()
        total_loss = 0.0
        pbar = tqdm(train_loader)
        for images, path in pbar:
            clip_embs = get_clip_image_features(images, clip_model, clip_preprocess).cuda().float()
            model_embs = get_llava_image_features(images, model, processor).cuda().float()
            projected_embs = projector(clip_embs)
            loss = F.mse_loss(projected_embs, model_embs)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch+1} Train Loss: {loss.item():.4f}")
            total_loss += loss.item()
    
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_loss:.4f}")
        
        projector.eval()
        val_loss = 0
        pbar = tqdm(val_loader)
        logger.info(f"Validating projector...")
        with torch.no_grad():
            for images, path in pbar:
                clip_embs = get_clip_image_features(images, clip_model, clip_preprocess).cuda().float()
                model_embs = get_llava_image_features(images, model, processor).cuda().float()
                projected_embs = projector(clip_embs)
                loss = F.mse_loss(projected_embs, model_embs)
        
                pbar.set_description(f"Epoch {epoch+1} Val Loss: {loss.item():.4f}")
                val_loss += loss.item()
    
            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)
            logger.info(f"Epoch {epoch+1}/{args.epochs}, Val Loss: {val_loss:.4f}")
            
                
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(projector.state_dict(), os.path.join(args.save_projector_dir, get_name(args))+".pt")
                logger.info(f"Saved projector model with loss {best_loss:.4f} to {os.path.join(args.save_projector_dir, get_name(args))}.pt")

    plot_path = os.path.join(args.save_projector_dir, get_name(args) + "_loss_curve.png")
    logger.info(f"Train losses: {train_losses}")
    logger.info(f"Val losses: {val_losses}")
    plt.figure(figsize=(16, 9))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Projector Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"Saved loss curve to {plot_path}")



if __name__ == "__main__":
    args = get_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO

    # create folder
    os.makedirs(f"logs", exist_ok=True)
    os.makedirs(f"logs/projector", exist_ok=True)
    os.makedirs(args.save_projector_dir, exist_ok=True)

    logging.basicConfig(format="### %(message)s ###")

    logger = logging.getLogger("HallucinationAttack")
    logger.setLevel(level=logging_level)

    logger.addHandler(logging.FileHandler(f"logs/projector/{get_log_name(args)}.txt", mode='w'))

    # Setting Seed
    set_seed(args.seed)

    logger.info(f"Arguments: {args}")

    train_projector(args)
