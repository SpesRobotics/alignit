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
        use_depth_input=True,  # NEW
        fc_layers=[256, 128],
        vector_hidden_dim=64,
        depth_hidden_dim=128,  # NEW
        output_dim=7,
        feature_agg="mean",
    ):
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

        if use_depth_input:
            self.depth_cnn = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.depth_fc = nn.Sequential(
                nn.Linear(16, depth_hidden_dim), nn.ReLU()
            )
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

    def forward(self, rgb_images, vector_inputs=None, depth_images=None):
        """
        :param rgb_images: Tensor of shape (B, N, 3, H, W)
        :param vector_inputs: List of tensors of shape (L_i,) or None
        :param depth_images: Tensor of shape (B, N, 1, H, W) or None
        :return: Tensor of shape (B, output_dim)
        """
        B, N, C, H, W = rgb_images.shape
        images = rgb_images.view(B * N, C, H, W)
        feats = self.backbone(images).view(B, N, -1)
        image_feats = self.aggregate_image_features(feats)
        image_feats = self.image_fc(image_feats)

        features = [image_feats]

        if self.use_depth_input and depth_images is not None:
            depth = depth_images.view(B * N, 1, H, W)
            depth_feats = self.depth_cnn(depth).view(B, N, -1)
            depth_feats = self.aggregate_image_features(depth_feats)
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


if __name__ == "__main__":
    import time

    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlignNet(backbone_name="efficientnet_b0", use_vector_input=True, use_depth_input=True).to(device)

    rgb_images = torch.randn(batch_size, 4, 3, 224, 224).to(device)
    depth_images = torch.randn(batch_size, 4, 1, 224, 224).to(device)
    vector_inputs = [torch.randn(10).to(device) for _ in range(batch_size)]

    output = None
    start_time = time.time()
    for i in range(100):
        output = model(rgb_images, vector_inputs, depth_images)
    end_time = time.time()
    duration_ms = ((end_time - start_time) / 100) * 1000
    print(f"Inference time: {duration_ms:.3f} ms")
    print(f"Optimal for {1000 / duration_ms:.2f} fps")
    print("Output shape:", output.shape)