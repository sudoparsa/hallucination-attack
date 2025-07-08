import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch


class CLS2TokensDecoder(nn.Module):
    def __init__(self, output_tokens=576, dim=1024, num_layers=4, num_heads=8, hidden_dim=2048):
        super().__init__()       
        self.output_tokens = output_tokens
        self.dim = dim       
        # self.queries = nn.Parameter(torch.randn(output_tokens, dim))
        self.queries = nn.Parameter(torch.zeros(1, output_tokens, dim))
        # nn.init.trunc_normal_(self.queries, std=0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True),
            num_layers=num_layers
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, output_tokens, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        
    def forward(self, cls):
        """
        cls: [B, dim]
        """
        B = cls.size(0)
        memory = cls.unsqueeze(1)             # [B, 1, D]
        queries = self.queries.repeat(B, 1, 1) + self.pos_emb  # [B, T, D]
        out = self.decoder(queries, memory)   # [B, T, D]
        return out


class ClipProjector(nn.Module):
    def __init__(self, num_tokens, target_dim, hidden_dim=512, clip_dim=1024):
        super().__init__()
        self.num_tokens = num_tokens
        self.target_dim = target_dim
        self.hidden_mlp = nn.Sequential(
            nn.Linear(clip_dim, clip_dim),
            nn.GELU(),
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, target_dim),
    
            )
        
        # Learnable positional embeddings [1, num_tokens, target_dim]
        self.pos_emb = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)  # Optional: initialization


    def forward(self, x):
        x = self.hidden_mlp(x) # [B, hidden_dim]
        x = x.unsqueeze(1).repeat(1, self.num_tokens, 1)  # [B, num_tokens, hidden_dim]
        x = x + self.pos_emb
        x = self.mlp(x)  # [B, num_tokens, target_dim]
        return x
    

class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None, resize=None):
        self.image_paths = sorted([p for p in Path(image_folder).glob("*.jpg")])
        if transform is None:
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((336, 336)),
        ])
        self.resize = resize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        # image = self.transform(image)
        if self.resize is not None:
            image = image.resize(self.resize, Image.LANCZOS)
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
