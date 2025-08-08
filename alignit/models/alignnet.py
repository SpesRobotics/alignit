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
        fc_layers=[256, 128],
        vector_hidden_dim=64,
        output_dim=7,
        feature_agg="mean",
    ):
        """
        :param backbone_name: 'efficientnet_b0' or 'resnet18'
        :param backbone_weights: 'DEFAULT' or None
        :param use_vector_input: whether to accept a vector input
        :param fc_layers: list of hidden layer sizes for the fully connected head
        :param vector_hidden_dim: output dim of the vector MLP
        :param output_dim: final output vector size
        :param feature_agg: 'mean' or 'max' across image views
        """
        super().__init__()
        self.use_vector_input = use_vector_input
        self.feature_agg = feature_agg

        # CNN backbone
        self.backbone, self.image_feature_dim = self._build_backbone(
            backbone_name, backbone_weights
        )

        # Linear projection of image features
        self.image_fc = nn.Sequential(
            nn.Linear(self.image_feature_dim, fc_layers[0]), nn.ReLU()
        )

        # Optional vector input processing
        if use_vector_input:
            self.vector_fc = nn.Sequential(nn.Linear(1, vector_hidden_dim), nn.ReLU())
            input_dim = fc_layers[0] + vector_hidden_dim
        else:
            input_dim = fc_layers[0]

        # Fully connected layers
        layers = []
        in_dim = input_dim
        for out_dim in fc_layers[1:]:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
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

    def aggregate_image_features(self, feats):
        if self.feature_agg == "mean":
            return feats.mean(dim=1)
        elif self.feature_agg == "max":
            return feats.max(dim=1)[0]
        else:
            raise ValueError("Invalid aggregation type")

    def forward(self, rgb_images, vector_inputs=None):
        """
        :param rgb_images: Tensor of shape (B, N, 3, H, W)
        :param vector_inputs: List of tensors of shape (L_i,) or None
        :return: Tensor of shape (B, output_dim)
        """
        B, N, C, H, W = rgb_images.shape
        images = rgb_images.view(B * N, C, H, W)
        feats = self.backbone(images).view(B, N, -1)
        image_feats = self.aggregate_image_features(feats)
        image_feats = self.image_fc(image_feats)

        if self.use_vector_input and vector_inputs is not None:
            vec_feats = []
            for vec in vector_inputs:
                vec = vec.unsqueeze(1)  # (L, 1)
                pooled = self.vector_fc(vec).mean(dim=0)  # (D,)
                vec_feats.append(pooled)
            vec_feats = torch.stack(vec_feats, dim=0)
            fused = torch.cat([image_feats, vec_feats], dim=1)
        else:
            fused = image_feats

        return self.head(fused)  # (B, output_dim)
