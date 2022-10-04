import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Tuple

import gradio as gr
import hydra
import torch
from omegaconf import DictConfig

from src import utils

log = utils.get_pylogger(__name__)


def demo_gradio(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info("Running Demo with gradio")

    log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)

    log.info(f"Loaded Model: {model}")

    categories = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def recognize_cifar10(image):
        if image is None:
            return None
        image = torch.tensor(image.reshape((1, 3, 28, 28)), dtype=torch.float32)
        preds = model.pass_jit(image)
        preds = preds[0].tolist()
        return {categories[i]: preds[i] for i in range(10)}

    im = gr.Image(shape=(28, 28), image_mode="RGB", source="upload")

    demo = gr.Interface(
        fn=recognize_cifar10,
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
    )

    demo.launch(server_name=cfg.server_name, server_port=8080)


@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="demo_cifar10_scripted.yaml"
)
def main(cfg: DictConfig) -> None:
    demo_gradio(cfg)


if __name__ == "__main__":
    main()
