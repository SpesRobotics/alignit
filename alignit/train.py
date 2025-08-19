import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm
from datasets import load_from_disk
from torchvision import transforms
import draccus
import numpy as np

from alignit.config import TrainConfig
from alignit.models.alignnet import AlignNet


def collate_fn(batch):
    images = [item["images"] for item in batch]
    depth_images = [item.get("depth", None) for item in batch]
    actions = [item["action"] for item in batch]
    return {
        "images": images,
        "depth_images": depth_images,
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
    criterion = MSELoss()
    net.train()

    for epoch in range(cfg.epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch["images"]
            depth_images_pil = batch["depth_images"]
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

            batch_depth_tensors = None
            if cfg.model.use_depth_input:
                batch_depth_tensors = []

                # print min and max in depth_images_pil
                print("Depth images min and max values:")
                print([
                    (np.array(d_img).min(), np.array(d_img).max())
                    for d_img in depth_images_pil
                ])

                for depth_sequence in depth_images_pil:
                    if depth_sequence is None:
                        raise ValueError(
                            "Depth images expected but not found when use_depth_input=True"
                        )

                    depth_sequence_processed = []
                    for d_img in depth_sequence:
                        depth_array = np.array(d_img)
                        depth_tensor = torch.from_numpy(depth_array).float()
                        depth_tensor = depth_tensor.unsqueeze(0)
                        depth_sequence_processed.append(depth_tensor)

                    stacked_depth = torch.stack(depth_sequence_processed, dim=0)
                    batch_depth_tensors.append(stacked_depth)

                batch_depth_tensors = torch.stack(batch_depth_tensors, dim=0).to(device)

            optimizer.zero_grad()
            if cfg.model.use_depth_input:
                outputs = net(batch_rgb_tensors, depth_images=batch_depth_tensors)
            else:
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
