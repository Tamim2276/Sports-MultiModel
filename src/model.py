"""
ViT + RoBERTa multimodal model

Architecture:
Video frames -> ViT      → 768-dim visual features
Text options -> RoBERTa  → 768-dim text features
Both         -> Attentive Cross-Attention Fusion
             -> Score each of 4 answer options
             -> Highest score = predicted answer
"""

import torch
import torch.nn as nn
from transformers import ViTModel
from transformers import RobertaModel


class VideoEncoder(nn.Module):

    def __init__(self, output_dim=768):
        super().__init__()

        # Pretrained ViT trained on ImageNet-21k
        self.vit = ViTModel.from_pretrained(
            'google/vit-base-patch16-224'
        )

        # ViT outputs 768-dim CLS token
        # We match this to output_dim
        self.projection = nn.Sequential(
            nn.Linear(768, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.1)
        )

        # Freeze all layers first
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last 4 layers + layernorm
        for param in self.vit.encoder.layer[-4:].parameters():
            param.requires_grad = True
        for param in self.vit.layernorm.parameters():
            param.requires_grad = True

    def forward(self, frames):
        """
        Input:  frames [batch, num_frames, 3, 224, 224]
        Output: video_features [batch, output_dim]
        """
        B, T, C, H, W = frames.shape

        # Flatten batch and time dims
        # Process all frames together: [B*T, 3, 224, 224]
        frames = frames.view(B * T, C, H, W)

        # Frozen layers — no gradients, saves memory
        with torch.no_grad():
            hidden = self.vit.embeddings(frames)
            for layer in self.vit.encoder.layer[:-4]:
                hidden = layer(hidden)[0]

        # Detach so frozen layers don't accumulate grad history
        hidden = hidden.detach()

        # Unfrozen last 4 layers — gradients flow here
        for layer in self.vit.encoder.layer[-4:]:
            hidden = layer(hidden)[0]

        hidden = self.vit.layernorm(hidden)

        # CLS token: [B*T, 768]
        features = hidden[:, 0, :]

        # Restore batch and time: [B, T, 768]
        features = features.view(B, T, -1)

        # Average across frames: [B, 768]
        features = features.mean(dim=1)

        # Project: [B, output_dim]
        return self.projection(features)


class TextEncoder(nn.Module):
    """
    Encodes question + answer options using RoBERTa
    """

    def __init__(self, output_dim=768):
        super().__init__()

        # Pretrained RoBERTa-base
        self.roberta = RobertaModel.from_pretrained('roberta-base')

        self.projection = nn.Sequential(
            nn.Linear(768, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.1)
        )

        # Freeze all layers first
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Unfreeze last 4 layers
        for param in self.roberta.encoder.layer[-4:].parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        B, N, L = input_ids.shape

        ids  = input_ids.view(B * N, L)
        mask = attention_mask.view(B * N, L)

        # Frozen layers — no gradients
        # Note: embeddings does not take attention_mask
        with torch.no_grad():
            hidden = self.roberta.embeddings(input_ids=ids)
            extended_mask = self.roberta.get_extended_attention_mask(
                mask, ids.shape
            )
            for layer in self.roberta.encoder.layer[:-4]:
                hidden = layer(hidden, extended_mask)[0]

        # Detach so frozen layers don't accumulate grad history
        hidden = hidden.detach()

        # Unfrozen last 4 layers — gradients flow here
        for layer in self.roberta.encoder.layer[-4:]:
            hidden = layer(hidden, extended_mask)[0]

        # CLS token: [B*N, 768]
        features = hidden[:, 0, :]

        # Reshape: [B, N, 768]
        features = features.view(B, N, -1)

        return self.projection(features)


class AttentiveFusion(nn.Module):
    """
    Bidirectional cross-attention fusion.
    Text attends to image, image attends to text.
    Concatenates original text + both attended outputs -> [B, 4, D*3]
    """

    def __init__(self, feature_dim=768, num_heads=8):
        super().__init__()

        self.feature_dim = feature_dim
        self.scale = feature_dim ** 0.5

        # Text attends to image (text as query)
        self.text_query   = nn.Linear(feature_dim, feature_dim)
        self.visual_key   = nn.Linear(feature_dim, feature_dim)
        self.visual_value = nn.Linear(feature_dim, feature_dim)

        # Image attends to text (image as query)
        self.visual_query = nn.Linear(feature_dim, feature_dim)
        self.text_key     = nn.Linear(feature_dim, feature_dim)
        self.text_value   = nn.Linear(feature_dim, feature_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, video_features, text_features):
        B, N, D = text_features.shape

        # Keep video as [B, 1, D] for key/value
        video_single = video_features.unsqueeze(1)         # [B, 1, D]

        # Text attends to Image
        Q_text = self.text_query(text_features)            # [B, 4, D]
        K_vis  = self.visual_key(video_single)             # [B, 1, D]
        V_vis  = self.visual_value(video_single)           # [B, 1, D]

        attn_text2vis = self.softmax(
            torch.bmm(Q_text, K_vis.transpose(1, 2)) / self.scale
        )                                                  # [B, 4, 1]
        attended_vis = self.dropout(
            torch.bmm(attn_text2vis, V_vis)                # [B, 4, D]
        )

        # Image attends to Text
        Q_vis  = self.visual_query(video_single)           # [B, 1, D]
        K_text = self.text_key(text_features)              # [B, 4, D]
        V_text = self.text_value(text_features)            # [B, 4, D]

        attn_vis2text = self.softmax(
            torch.bmm(Q_vis, K_text.transpose(1, 2)) / self.scale
        )                                                  # [B, 1, 4]
        attended_text = self.dropout(
            torch.bmm(attn_vis2text, V_text)               # [B, 1, D]
        )

        # Expand to match N options
        attended_text = attended_text.expand(B, N, D)      # [B, 4, D]

        # Concatenate: [B, 4, D*3]
        fused = torch.cat([text_features, attended_vis, attended_text], dim=-1)
        return fused


class SportsMultimodalModel(nn.Module):

    def __init__(self, feature_dim=768, dropout=0.2):
        super().__init__()

        self.video_encoder = VideoEncoder(output_dim=feature_dim)
        self.text_encoder  = TextEncoder(output_dim=feature_dim)
        self.fusion        = AttentiveFusion(feature_dim=feature_dim)

        # Classifier
        # Input: feature_dim * 3 (from AttentiveFusion output)
        # Output: 1 score per option
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 3, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(256, 1)         # one score per option
        )

    def forward(self, frames, input_ids, attention_mask):

        # Encode video frames: [B, 768]
        video = self.video_encoder(frames)

        # Encode text options: [B, 4, 768]
        text = self.text_encoder(input_ids, attention_mask)

        # Attentive Fusion: [B, 4, 768*3]
        fused = self.fusion(video, text)

        # Score each option
        B, N, D = fused.shape

        # Flatten for classifier
        fused  = fused.view(B * N, D)    # [B*N, D]
        scores = self.classifier(fused)  # [B*N, 1]
        scores = scores.view(B, N)       # [B, 4]

        return scores