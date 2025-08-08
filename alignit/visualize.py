import gradio as gr
import draccus

from alignit.utils.dataset import load_dataset
from alignit.utils.zhou import sixd_se3
from alignit.utils.tfs import get_pose_str
from alignit.config import VisualizeConfig


@draccus.wrap()
def visualize(cfg: VisualizeConfig):
    dataset = load_dataset(cfg.dataset.path)

    def get_data(index):
        item = dataset[index]
        image = item["images"][0]
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
