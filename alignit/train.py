import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from datasets import load_from_disk
from torchvision import transforms
import draccus
import numpy as np

from alignit.config import TrainConfig
from alignit.models.alignnet import AlignNet
from alignit.losses import InversePredictionWeightedLoss
#import mse loss for comparison
from torch.nn import MSELoss


def collate_fn(batch):
    images = [item["images"] for item in batch]
    actions = [item["action"] for item in batch]
    return {
        "images": images,
        "action": torch.tensor(actions, dtype=torch.float32),
    }


@draccus.wrap()
def main(cfg: TrainConfig):
    """Train AlignNet model using configuration parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_from_disk(cfg.dataset.path)
    net = AlignNet(
        backbone_name=cfg.model.backbone,
        backbone_weights=cfg.model.backbone_weights,
        use_vector_input=cfg.model.use_vector_input,
        fc_layers=cfg.model.fc_layers,
        vector_hidden_dim=cfg.model.vector_hidden_dim,
        output_dim=cfg.model.output_dim,
        feature_agg=cfg.model.feature_agg,
        use_depth_input=cfg.model.use_depth_input,
    ).to(device)

    train_dataset = dataset.train_test_split(
        test_size=cfg.test_size, seed=cfg.random_seed
    )
    train_loader = DataLoader(
        train_dataset["train"],
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = Adam(net.parameters(), lr=cfg.learning_rate)
    criterion = InversePredictionWeightedLoss(epsilon=0.01) 
    net.train()

    for epoch in range(cfg.epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch["images"]
            actions = batch["action"].to(device)

            batch_rgb_tensors = []
            rgb_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

            for image_sequence in images:
                tensor_sequence_rgb = [
                    rgb_transform(img.convert("RGB")) for img in image_sequence
                ]
                stacked_tensors_rgb = torch.stack(tensor_sequence_rgb, dim=0)
                batch_rgb_tensors.append(stacked_tensors_rgb)

            batch_rgb_tensors = torch.stack(batch_rgb_tensors, dim=0).to(device)

            optimizer.zero_grad()
            outputs = net(batch_rgb_tensors)

            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        tqdm.write(f"Loss: {total_loss / len(train_loader):.4f}")
        torch.save(net.state_dict(), cfg.model.path)
        tqdm.write(f"Model saved as {cfg.model.path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
