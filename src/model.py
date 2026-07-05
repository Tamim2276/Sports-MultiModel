import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPModel


class VideoEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        
        # Pretrained ResNet-18
        resnet = models.resnet18(weights='DEFAULT')
        
        # Remove final classification layer
        # We want 512-dim features not 1000 class scores
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1]
        )
        
        # Project 512 -> output_dim
        self.projection = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Freeze ResNet don't update during training
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, frames):
        # frames: [batch, 8, 3, 224, 224]
        B, T, C, H, W = frames.shape
        
        # Process all frames together
        frames = frames.view(B * T, C, H, W)
        
        # Through ResNet
        feat = self.backbone(frames)          # [B*T, 512, 1, 1]
        feat = feat.squeeze(-1).squeeze(-1)   # [B*T, 512]
        feat = feat.view(B, T, -1)            # [B, T, 512]
        
        # Average across frames
        feat = feat.mean(dim=1)               # [B, 512]
        
        return self.projection(feat)          # [B, output_dim]


class TextEncoder(nn.Module):

    #Turns text into feature vectors using CLIP.

    def __init__(self, output_dim=256):
        super().__init__()
        
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        
        self.projection = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Freeze CLIP
        for param in self.clip.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        # input_ids: [batch, 4, 77]
        B, N, L = input_ids.shape
        
        # Flatten
        ids   = input_ids.view(B * N, L)
        mask  = attention_mask.view(B * N, L)
        
        # CLIP text features
        feat = self.clip.get_text_features(
            input_ids=ids, attention_mask=mask
        )

        # Make sure we have a plain tensor
        if hasattr(feat, 'pooler_output'):
            feat = feat.pooler_output
        elif hasattr(feat, 'last_hidden_state'):
            feat = feat.last_hidden_state[:, 0, :]

        # Now reshape
        feat = feat.view(B, N, -1)            # [B, N, 512]
        return self.projection(feat)          # [B, N, output_dim]


class SportsMultimodalModel(nn.Module):

    #Full model: video + text → answer

    def __init__(self, feature_dim=256):
        super().__init__()
        
        self.video_encoder = VideoEncoder(feature_dim)
        self.text_encoder  = TextEncoder(feature_dim)
        
        # Combine video + text → score
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, frames, input_ids, attention_mask):
        # Encode
        video = self.video_encoder(frames)
        # video: [B, feature_dim]
        
        text = self.text_encoder(input_ids, attention_mask)
        # text: [B, 4, feature_dim]
        
        B, N, D = text.shape
        
        # Expand video to match 4 options
        video = video.unsqueeze(1).expand(B, N, D)
        # video: [B, 4, feature_dim]
        
        # Concatenate
        combined = torch.cat([video, text], dim=-1)
        # combined: [B, 4, feature_dim*2]
        
        # Score each option
        combined = combined.view(B * N, -1)
        scores   = self.fusion(combined)
        scores   = scores.view(B, N)
        # scores: [B, 4]
        
        return scores