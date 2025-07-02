import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch



class ClipProjector(nn.Module):
    def __init__(self, target_dim, clip_dim=1024):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(clip_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Linear(4096, 4096),
            nn.GELU(),
            nn.Linear(4096, target_dim),
            nn.GELU(),    
            nn.LayerNorm(target_dim)
            )

    def forward(self, x):
        x = self.mlp(x)
        return x
    

class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_paths = sorted([p for p in Path(image_folder).glob("*.jpg")])
        if transform is None:
            self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        # image = self.transform(image)
        return image, path
    

class CachedEmbeddingDataset(Dataset):
    def __init__(self, embedding_path):
        data = torch.load(embedding_path, weights_only=False)
        self.clip_embs = data["clip"]      # shape: [N, D]
        self.model_embs = data["model"]    # shape: [N, D]
        self.paths = data["paths"]          # shape: [N]

        assert self.clip_embs.shape[0] == self.model_embs.shape[0], "Mismatched embedding sizes"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.clip_embs[idx], self.model_embs[idx]
