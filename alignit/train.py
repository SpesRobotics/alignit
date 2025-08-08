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
from alignit.utils.dataset import load_dataset_smart

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def collate_fn(batch):
    images = [item["images"] for item in batch]
    actions = [item["action"] for item in batch]
    return {"images": images, "action": torch.tensor(actions, dtype=torch.float32)}


@draccus.wrap()
def main(cfg: TrainConfig):
    """Train AlignNet model using configuration parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Weights & Biases if configured
    use_wandb = WANDB_AVAILABLE and cfg.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            tags=cfg.wandb_tags,
            config=cfg.__dict__
        )
        print(f"Initialized Weights & Biases project: {cfg.wandb_project}")
    elif cfg.wandb_project and not WANDB_AVAILABLE:
        print("Warning: wandb project specified but wandb not installed. Install with: pip install wandb")

    # Load the dataset from disk or HuggingFace Hub
    if cfg.dataset.hf_dataset_name:
        print(f"Loading dataset from HuggingFace Hub: {cfg.dataset.hf_dataset_name}")
        dataset_path = cfg.dataset.hf_dataset_name
    else:
        print(f"Loading dataset from disk: {cfg.dataset.path}")
        dataset_path = cfg.dataset.path
    
    dataset = load_dataset_smart(dataset_path)

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

    # Watch model with wandb if enabled
    if use_wandb:
        wandb.watch(net, log_freq=100)

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
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
            
            epoch_loss += loss.item()
            num_batches += 1
            
            tqdm.write(f"Loss: {loss.item():.4f}")

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_epoch_loss,
                "learning_rate": cfg.learning_rate
            })

        # Save the trained model
        torch.save(net.state_dict(), cfg.model.path)
        tqdm.write(f"Model saved as {cfg.model.path}")

    print("Training complete.")
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
