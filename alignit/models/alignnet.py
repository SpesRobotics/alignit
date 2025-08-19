import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, resnet18
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights
import math


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
        feature_agg="attn",
        dropout: float | int = 0.0,
    ):
        """
        :param backbone_name: 'efficientnet_b0' or 'resnet18'
        :param backbone_weights: 'DEFAULT' or None
        :param use_vector_input: whether to accept a vector input
        :param fc_layers: list of hidden layer sizes for the fully connected head
        :param vector_hidden_dim: output dim of the vector MLP
        :param output_dim: final output vector size
        :param feature_agg: 'mean', 'max', or 'attn' across image/depth views
        :param use_depth_input: whether to accept depth input
        :param depth_hidden_dim: output dim of the depth MLP
        """
        super().__init__()
        self.use_vector_input = use_vector_input
        self.use_depth_input = use_depth_input
        self.feature_agg = feature_agg
        self._dropout = float(dropout) if dropout else 0.0

        self.backbone, self.image_feature_dim = self._build_backbone(
            backbone_name, backbone_weights
        )

        # Stronger image feature head for better separability
        self.image_fc = nn.Sequential(
            nn.Linear(self.image_feature_dim, fc_layers[0]),
            nn.GELU(),
            nn.LayerNorm(fc_layers[0]),
            nn.Dropout(p=self._dropout) if self._dropout > 0 else nn.Identity(),
        )

        # Configure depth encoder output channels upfront for attention sizing
        self.depth_cnn_out_channels = 32 if use_depth_input else None

        # Optional attention vectors for view aggregation
        if self.feature_agg == "attn":
            # One learnable attention vector per modality
            self.attn_vector_img = nn.Parameter(torch.randn(self.image_feature_dim))
            # Depth features before FC (match depth_cnn_out_channels when depth is enabled)
            self.attn_vector_depth = (
                nn.Parameter(torch.randn(self.depth_cnn_out_channels))
                if use_depth_input
                else None
            )
        else:
            self.attn_vector_img = None
            self.attn_vector_depth = None

        if use_depth_input:
            self.depth_cnn = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, self.depth_cnn_out_channels, 3, padding=1),
                nn.BatchNorm2d(self.depth_cnn_out_channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.depth_fc = nn.Sequential(
                nn.Linear(self.depth_cnn_out_channels, depth_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(depth_hidden_dim),
                nn.Dropout(p=self._dropout) if self._dropout > 0 else nn.Identity(),
            )
            input_dim = fc_layers[0] + depth_hidden_dim
        else:
            input_dim = fc_layers[0]

        # Optional vector input processing
        if use_vector_input:
            self.vector_fc = nn.Sequential(
                nn.Linear(1, vector_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(vector_hidden_dim),
                nn.Dropout(p=self._dropout) if self._dropout > 0 else nn.Identity(),
            )
            input_dim += vector_hidden_dim

        # Normalize fused features before final head
        self.fuse_norm = nn.LayerNorm(input_dim)

        # Fully connected layers
        layers = []
        in_dim = input_dim
        for out_dim in fc_layers[1:]:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            if self._dropout > 0:
                layers.append(nn.Dropout(p=self._dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))  # Final output layer
        self.head = nn.Sequential(*layers)

        # Initialize only newly created layers (avoid touching backbone weights)
        self._init_module(self.image_fc)
        if use_depth_input:
            self._init_module(self.depth_cnn)
            self._init_module(self.depth_fc)
        if use_vector_input:
            self._init_module(self.vector_fc)
        self._init_module(self.head)
        self._init_module(self.fuse_norm)

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
            # Support list-of-tensors (length B) or batched tensor (B, L[, 1])
            if torch.is_tensor(vector_inputs):
                v = vector_inputs
                if v.dim() == 2:
                    v = v.unsqueeze(-1)  # (B, L, 1)
                vec_feats = self.vector_fc(v).mean(dim=1)  # (B, D)
            else:
                vec_feats_list = []
                for vec in vector_inputs:
                    vec = vec.unsqueeze(1)  # (L, 1)
                    pooled = self.vector_fc(vec).mean(dim=0)  # (D,)
                    vec_feats_list.append(pooled)
                vec_feats = torch.stack(vec_feats_list, dim=0)
            features.append(vec_feats)

        fused = torch.cat(features, dim=1)
        fused = self.fuse_norm(fused)
        return self.head(fused)  # (B, output_dim)

    def _init_module(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
