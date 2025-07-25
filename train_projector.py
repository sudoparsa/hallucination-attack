import argparse
import os
from projector.projector import *
from utils import *
from torch.utils.data import DataLoader, Subset
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
    parser.add_argument("--save_projector_dir", type=str, default="./projector/models")
    parser.add_argument("--embeddings_dir", type=str, default="./projector/embeddings")

    # Training
    parser.add_argument("--save_emb_batch_size", type=int, default=64, help="Batch size for saving embeddings")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--context_dim", type=int, default=1024, help="Context dimension for the projector")
    parser.add_argument("--hidden_dim", type=int, default=2048, help="Hidden dimension for the projector")
    parser.add_argument("--no_selected_tokens", type=int, default=32, help="Number of tokens to select for training")
    parser.add_argument("--coco_subset", type=float, default=0.1, help="Fraction of COCO dataset to use for training (1.0 means full dataset)")

    # Others
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    return parser.parse_args()


def get_name(args):
    return f"{args.model_name}_projector"

def get_log_name(args):
    return f"{args.model_name}_bs={args.batch_size}_lr={args.lr}_epochs={args.epochs}_context_dim={args.context_dim}_hidden_dim={args.hidden_dim}_coco_subset={args.coco_subset}"


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
    num_tokens, target_dim = get_num_tokens(args.model_name)
    decoder = TokenMLP(context_dim=args.context_dim, hidden_dim=args.hidden_dim, num_tokens=num_tokens, target_dim=target_dim).cuda()
    logger.info(f"Decoder: {decoder}")

    logger.info(f"Loading {args.model_name}")
    model, processor = get_model(args.model_name, args.cache_path)
    model = model.to("cuda").eval()
    logger.info(f"Loading CLIP model")
    clip_model, clip_preprocess = get_clip_model(args.cache_path)
    clip_model = clip_model.to("cuda").eval()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
    train_dataset = ImageDataset(os.path.join(args.coco_dir, "train2017"), resize=(336, 336))
    indices = random.sample(range(len(train_dataset)), int(args.coco_subset * len(train_dataset)))  # Use subset of the dataset for training
    train_dataset = Subset(train_dataset, indices)
    logger.info(f"Train dataset size: {len(train_dataset)}")
    val_dataset = ImageDataset(os.path.join(args.coco_dir, "val2017"), resize=(336, 336))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=lambda x: list(zip(*x)))

    criterion = nn.MSELoss()   
    
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    for epoch in range(1, args.epochs+1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        decoder.train()
        total_tr_loss = 0
        epoch_losses = []
        for i, (images, _) in enumerate(pbar):
            with torch.no_grad():
                cls_h = get_clip_image_features(images, clip_model, clip_preprocess)
                tokens_l = get_model_image_features(args.model_name, images, model, processor)
            pred_tokens = decoder(cls_h).half()
            loss = criterion(pred_tokens, tokens_l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_tr_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "total_loss": total_tr_loss / (i + 1)})
            epoch_losses.append(loss.item())
        avg_train_loss = total_tr_loss / len(train_loader)
        logger.info(f"Epoch {epoch} - Train Loss: {sum(epoch_losses) / len(epoch_losses)}")
        train_losses.append(avg_train_loss)

        total_val_loss = 0.0
        decoder.eval()
        pbar = tqdm(val_loader, desc=f"Validating epoch {epoch}")
        for i, (images, _) in enumerate(pbar):   
            with torch.no_grad():
                cls_h = get_clip_image_features(images, clip_model, clip_preprocess)
                tokens_l = get_model_image_features(args.model_name, images, model, processor)
                pred_tokens = decoder(cls_h).half()
                # B, T, D = pred_tokens.shape
                # token_idx = torch.randint(0, T, (B, args.no_selected_tokens), device=pred_tokens.device)  # [B, 32]
                # batch_idx = torch.arange(B, device=pred_tokens.device).unsqueeze(1)  # [B, 1]
                # pred_sample = pred_tokens[batch_idx, token_idx]  # [B, 32, D]
                # target_sample = tokens_l[batch_idx, token_idx]   # [B, 32, D]
                # val_loss = criterion(pred_sample, target_sample)
                val_loss = criterion(pred_tokens, tokens_l)
                total_val_loss += val_loss.item()
            
            pbar.set_postfix({"val_loss": val_loss.item(), "total_val_loss": total_val_loss / (i + 1)})
        
        logger.info(f"Epoch {epoch} - Validation Loss: {total_val_loss / len(val_loader)}")
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(decoder.state_dict(), os.path.join(args.save_projector_dir, get_log_name(args))+".pt")
            logger.info(f"Saved projector model with loss {best_val_loss} to {os.path.join(args.save_projector_dir, get_log_name(args))}.pt")
        
    plot_path = os.path.join(args.save_projector_dir, get_log_name(args) + "_loss_curve.png")
    logger.info(f"Train losses: {train_losses}")
    logger.info(f"Val losses: {val_losses}")

    plt.figure(figsize=(18, 6))

    # Train Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Train Loss")
    plt.grid(True)

    # Val Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Validation Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved loss curve to {plot_path}")
    logger.info(f"Training completed. Best validation loss: {best_val_loss}")


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
