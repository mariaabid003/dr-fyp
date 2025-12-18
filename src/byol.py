"""
BYOL (Bootstrap Your Own Latent) implementation
for self-supervised learning on fundus images.

This module defines:
- Online encoder
- Target encoder (EMA)
- Projection head
- Prediction head
- BYOL loss

Training loop is handled separately in train_byol.py
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


# --------------------------------------------------
# Utility: MLP Head
# --------------------------------------------------
class MLP(nn.Module):
    """
    Simple 2-layer MLP used for projection and prediction heads
    """

    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------
# Utility: Cosine similarity loss
# --------------------------------------------------
def byol_loss(p, z):
    """
    BYOL loss: negative cosine similarity
    p: prediction from online network
    z: projection from target network (stop-gradient)
    """
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=1).mean()


# --------------------------------------------------
# BYOL Model
# --------------------------------------------------
class BYOL(nn.Module):
    """
    BYOL main model class
    """

    def __init__(
        self,
        backbone_name="resnet50",
        image_size=512,
        proj_dim=256,
        hidden_dim=4096,
        momentum=0.996
    ):
        super().__init__()

        self.momentum = momentum

        # --------------------------
        # Online network
        # --------------------------
        self.online_encoder = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,      # remove classifier
            global_pool="avg"
        )

        encoder_out_dim = self.online_encoder.num_features

        self.online_projector = MLP(
            in_dim=encoder_out_dim,
            hidden_dim=hidden_dim,
            out_dim=proj_dim
        )

        self.online_predictor = MLP(
            in_dim=proj_dim,
            hidden_dim=hidden_dim,
            out_dim=proj_dim
        )

        # --------------------------
        # Target network (EMA)
        # --------------------------
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # target network does not receive gradients
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    # --------------------------------------------------
    # EMA update for target network
    # --------------------------------------------------
    @torch.no_grad()
    def update_target_network(self):
        """
        Exponential Moving Average update
        """
        for online_p, target_p in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_p.data = (
                self.momentum * target_p.data
                + (1 - self.momentum) * online_p.data
            )

        for online_p, target_p in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters()
        ):
            target_p.data = (
                self.momentum * target_p.data
                + (1 - self.momentum) * online_p.data
            )

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------
    def forward(self, view1, view2):
        """
        Input:
            view1, view2: two augmented views of the same image
        Output:
            scalar BYOL loss
        """

        # ---- online network ----
        o1 = self.online_encoder(view1)
        o2 = self.online_encoder(view2)

        z1 = self.online_projector(o1)
        z2 = self.online_projector(o2)

        p1 = self.online_predictor(z1)
        p2 = self.online_predictor(z2)

        # ---- target network ----
        with torch.no_grad():
            t1 = self.target_encoder(view1)
            t2 = self.target_encoder(view2)

            t1 = self.target_projector(t1)
            t2 = self.target_projector(t2)

        # ---- BYOL loss ----
        loss = byol_loss(p1, t2) + byol_loss(p2, t1)

        return loss
