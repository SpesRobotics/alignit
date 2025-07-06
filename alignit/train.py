from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from torch.nn import MSELoss
from tqdm import tqdm
from datasets import load_from_disk
from alignit.models.alignnet import AlignNet
from torchvision import transforms
from PIL import Image


def collate_fn(batch):
    images = [item["images"] for item in batch]
    actions = [item["action"] for item in batch]
    return {"images": images, "action": torch.tensor(actions, dtype=torch.float32)}


def main():
    # Load the dataset from disk
    dataset = load_from_disk("data/duck")
    net = AlignNet(
        output_dim=9,  # 3 for translation + 6 for rotation in sixd format
        use_vector_input=False,  # Disable vector input since we're not using it
    )

    # train
    train_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # implement a training loop here
    train_loader = DataLoader(
        train_dataset["train"], batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    optimizer = Adam(net.parameters(), lr=1e-4)
    criterion = MSELoss()
    net.train()
    for epoch in range(1):  # number of epochs
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch["images"]
            actions = batch["action"]

            # Convert PIL Images to tensors and stack them properly
            # images is a list of lists of PIL Images
            batch_images = []
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),  # Resize if needed
                    transforms.ToTensor(),
                ]
            )

            for image_sequence in images:
                tensor_sequence = [
                    transform(img.convert("RGB")) for img in image_sequence
                ]
                stacked_tensors = torch.stack(tensor_sequence, dim=0)
                batch_images.append(stacked_tensors)

            # Stack all batches to get shape (B, N, 3, H, W)
            batch_images = torch.stack(batch_images, dim=0)

            optimizer.zero_grad()
            outputs = net(batch_images)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            tqdm.write(f"Loss: {loss.item():.4f}")

    # Save the trained model
    torch.save(net.state_dict(), "alignnet_model.pth")
    tqdm.write("Model saved as alignnet_model.pth")

    print("Training complete.")


if __name__ == "__main__":
    main()
