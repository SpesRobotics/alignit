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
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the dataset from disk
    dataset = load_from_disk("data/duck")
    net = AlignNet(
        output_dim=9,  
        use_vector_input=False,  
    ).to(device)

    train_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    train_loader = DataLoader(
        train_dataset["train"], batch_size=4, shuffle=True, collate_fn=collate_fn
    )
    optimizer = Adam(net.parameters(), lr=1e-4)
    criterion = MSELoss()
    net.train()
    for epoch in range(100):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch["images"]
            depth_images_pil = batch["depth_images"]
            actions = batch["action"].to(device)

            batch_rgb_tensors = []
            batch_depth_tensors = []
            rgb_transform = transforms.Compose([transforms.ToTensor()])
            depth_transform = transforms.Compose([transforms.ToTensor()])

            for i, image_sequence in enumerate(images):
                # Process RGB
                tensor_sequence_rgb = [
                    rgb_transform(img.convert("RGB")) for img in image_sequence
                ]
                stacked_tensors_rgb = torch.stack(tensor_sequence_rgb, dim=0)
                batch_rgb_tensors.append(stacked_tensors_rgb)

                # Process Depth
                if depth_images_pil[i] is not None: # Check if depth is actually present for this item
                    depth_sequence = depth_images_pil[i]
                    tensor_sequence_depth = [
                        depth_transform(d_img.convert("L")) for d_img in depth_sequence
                    ] 
                    stacked_tensors_depth = torch.stack(tensor_sequence_depth, dim=0)
                    batch_depth_tensors.append(stacked_tensors_depth)
                else:
                    print(f"Warning: Depth image missing for sample {i} in batch.")
                    raise ValueError("Depth images expected but not found for a sample when use_depth is True.")


            batch_rgb_tensors = torch.stack(batch_rgb_tensors, dim=0).to(device)
            batch_depth_tensors = torch.stack(batch_depth_tensors, dim=0).to(device)

            optimizer.zero_grad()
            outputs = net(batch_rgb_tensors, depth_images=batch_depth_tensors) # <--- Pass depth_images
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            tqdm.write(f"Loss: {loss.item():.4f}")
        torch.save(net.state_dict(), "alignnet_model.pth")
        tqdm.write("Model saved as alignnet_model.pth")

    print("Training complete.")


if __name__ == "__main__":
    main()

