import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm
from datasets import load_from_disk
from torchvision import transforms
import draccus

from alignit.config import TrainConfig
from alignit.models.alignnet import AlignNet


def collate_fn(batch):
    images = [item["images"] for item in batch]
    actions = [item["action"] for item in batch]
    return {"images": images, "action": torch.tensor(actions, dtype=torch.float32)}


@draccus.wrap()
def main(cfg: TrainConfig):
    """Train AlignNet model using configuration parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset from disk
    dataset = load_from_disk(cfg.dataset.path)

    # Create model using config parameters
    net = AlignNet(
        backbone_name=cfg.model.backbone,
        backbone_weights=cfg.model.backbone_weights,
        use_vector_input=cfg.model.use_vector_input,
        fc_layers=cfg.model.fc_layers,
        vector_hidden_dim=cfg.model.vector_hidden_dim,
        output_dim=cfg.model.output_dim,
        feature_agg=cfg.model.feature_agg,
    ).to(device)

    # Split dataset
    train_dataset = dataset.train_test_split(
        test_size=cfg.test_size, seed=cfg.random_seed
    )

    # Create data loader
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
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch["images"]
            actions = batch["action"].to(device)

            # Convert PIL Images to tensors and stack them properly
            # images is a list of lists of PIL Images
            batch_images = []
            transform = transforms.Compose([transforms.ToTensor()])

            for image_sequence in images:
                tensor_sequence = [
                    transform(img.convert("RGB")) for img in image_sequence
                ]
                stacked_tensors = torch.stack(tensor_sequence, dim=0)
                batch_images.append(stacked_tensors)

            # Stack all batches to get shape (B, N, 3, H, W)
            batch_images = torch.stack(batch_images, dim=0).to(device)

            optimizer.zero_grad()
            outputs = net(batch_images)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            tqdm.write(f"Loss: {loss.item():.4f}")

        # Save the trained model
        torch.save(net.state_dict(), cfg.model.path)
        tqdm.write(f"Model saved as {cfg.model.path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
