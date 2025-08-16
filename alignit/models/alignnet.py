import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, resnet18
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights


class AlignNet(nn.Module):
    def __init__(
        self,
        backbone_name="efficientnet_b0",
        backbone_weights="DEFAULT",
        use_vector_input=True,
        use_depth_input=True,
        fc_layers=[256, 128],
        vector_hidden_dim=64,
        depth_hidden_dim=128,
        output_dim=7,
        feature_agg="mean",
        dropout: float | int = 0.0,
    ):
        """
        :param backbone_name: 'efficientnet_b0' or 'resnet18'
        :param backbone_weights: 'DEFAULT' or None
        :param use_vector_input: whether to accept a vector input
        :param fc_layers: list of hidden layer sizes for the fully connected head
        :param vector_hidden_dim: output dim of the vector MLP
        :param output_dim: final output vector size
        :param feature_agg: 'mean' or 'max' across image views
        :param use_depth_input: whether to accept depth input
        :param depth_hidden_dim: output dim of the depth MLP
        """
        super().__init__()
        self.use_vector_input = use_vector_input
        self.use_depth_input = use_depth_input
        self.feature_agg = feature_agg

        self.backbone, self.image_feature_dim = self._build_backbone(
            backbone_name, backbone_weights
        )

        self.image_fc = nn.Sequential(
            nn.Linear(self.image_feature_dim, fc_layers[0]), nn.ReLU()
        )

        # Optional attention vectors for view aggregation
        if self.feature_agg == "attn":
            # One learnable attention vector per modality
            self.attn_vector_img = nn.Parameter(torch.randn(self.image_feature_dim))
            # Depth features before FC are 16-dim in the tiny depth CNN below
            self.attn_vector_depth = nn.Parameter(torch.randn(16))
        else:
            self.attn_vector_img = None
            self.attn_vector_depth = None

        if use_depth_input:
            # Lightweight depth encoder with BatchNorm for stability
            self.depth_cnn = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.depth_fc = nn.Sequential(nn.Linear(16, depth_hidden_dim), nn.ReLU())
            input_dim = fc_layers[0] + depth_hidden_dim
        else:
            input_dim = fc_layers[0]

        # Optional vector input processing
        if use_vector_input:
            self.vector_fc = nn.Sequential(nn.Linear(1, vector_hidden_dim), nn.ReLU())
            input_dim += vector_hidden_dim

        # Fully connected layers
        layers = []
        in_dim = input_dim
        for out_dim in fc_layers[1:]:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout and float(dropout) > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))  # Final output layer
        self.head = nn.Sequential(*layers)

    def _build_backbone(self, name, weights):
        if name == "efficientnet_b0":
            model = efficientnet_b0(
                weights=(
                    EfficientNet_B0_Weights.DEFAULT if weights == "DEFAULT" else None
                )
            )
            model.classifier = nn.Identity()
            return model, 1280
        elif name == "resnet18":
            model = resnet18(
                weights=ResNet18_Weights.DEFAULT if weights == "DEFAULT" else None
            )
            model.fc = nn.Identity()
            return model, 512
        else:
            raise ValueError(f"Unsupported backbone: {name}")

    def aggregate_image_features(self, feats, modality: str = "img"):
        """Aggregate view features across the view dimension.

        feats: (B, N, D)
        modality: 'img' or 'depth' (relevant for attention pooling)
        """
        if self.feature_agg == "mean":
            return feats.mean(dim=1)
        elif self.feature_agg == "max":
            return feats.max(dim=1)[0]
        elif self.feature_agg == "attn":
            if modality == "img":
                attn_vec = self.attn_vector_img
            else:
                attn_vec = self.attn_vector_depth
            # Compute attention weights: (B, N)
            # Normalize attn vector for stability
            norm_attn = attn_vec / (attn_vec.norm(p=2) + 1e-6)
            scores = torch.einsum("bnd,d->bn", feats, norm_attn)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, N, 1)
            return (feats * weights).sum(dim=1)
        else:
            raise ValueError("Invalid aggregation type")

    def forward(self, rgb_images, vector_inputs=None, depth_images=None):
        """
        :param rgb_images: Tensor of shape (B, N, 3, H, W)
        :param vector_inputs: List[Tensor(L_i,)] or Tensor(B, L) or Tensor(B, L, 1)
        :param depth_images: Tensor of shape (B, N, 1, H, W) or None
        :return: Tensor of shape (B, output_dim)
        """
        B, N, C, H, W = rgb_images.shape
        images = rgb_images.reshape(B * N, C, H, W)
        # Use channels_last to improve convolution performance
        images = images.contiguous(memory_format=torch.channels_last)
        feats = self.backbone(images).reshape(B, N, -1)
        image_feats = self.aggregate_image_features(feats, modality="img")
        image_feats = self.image_fc(image_feats)

        features = [image_feats]

        if self.use_depth_input and depth_images is not None:
            depth = depth_images.view(B * N, 1, H, W)
            depth_feats = self.depth_cnn(depth).view(B, N, -1)
            depth_feats = self.aggregate_image_features(depth_feats, modality="depth")
            depth_feats = self.depth_fc(depth_feats)
            features.append(depth_feats)

        if self.use_vector_input and vector_inputs is not None:
            vec_feats = []
            for vec in vector_inputs:
                vec = vec.unsqueeze(1)  # (L, 1)
                pooled = self.vector_fc(vec).mean(dim=0)  # (D,)
                vec_feats.append(pooled)
            vec_feats = torch.stack(vec_feats, dim=0)
            features.append(vec_feats)

        fused = torch.cat(features, dim=1)
        return self.head(fused)  # (B, output_dim)
