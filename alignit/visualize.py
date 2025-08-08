import gradio as gr
import draccus
from PIL import Image
import numpy as np

from alignit.utils.dataset import load_dataset_smart
from alignit.utils.zhou import sixd_se3
from alignit.utils.tfs import get_pose_str
from alignit.config import VisualizeConfig


@draccus.wrap()
def visualize(cfg: VisualizeConfig):
    # Load the dataset from disk or HuggingFace Hub
    if cfg.dataset.hf_dataset_name:
        print(f"Loading dataset from HuggingFace Hub: {cfg.dataset.hf_dataset_name}")
        dataset_path = cfg.dataset.hf_dataset_name
    else:
        print(f"Loading dataset from disk: {cfg.dataset.path}")
        dataset_path = cfg.dataset.path
    
    dataset = load_dataset_smart(dataset_path)

    def get_data(index):
        item = dataset[index]
        image = item["images"][0]  # Should now be a PIL Image for both local and Hub datasets
        action_sixd = item["action"]
        action = sixd_se3(action_sixd)
        label = get_pose_str(action, degrees=True)
        return image, label

    gr.Interface(
        fn=get_data,
        inputs=gr.Slider(0, len(dataset) - 1, step=1, label="Index", interactive=True),
        outputs=[gr.Image(type="pil", label="Image"), gr.Text(label="Label")],
        title="Dataset Image Viewer",
        live=True,
    ).launch(
        share=cfg.share,
        server_name=cfg.server_name,
        server_port=cfg.server_port
    )


if __name__ == "__main__":
    visualize()
