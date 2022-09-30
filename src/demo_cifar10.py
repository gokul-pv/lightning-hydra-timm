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
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

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

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    ckpt = torch.load(cfg.ckpt_path)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    log.info(f"Loaded Model: {model}")

    transforms = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

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
        image = transforms(image).unsqueeze(0)
        logits = model(image)
        preds = F.softmax(logits, dim=1).squeeze(0).tolist()
        return {categories[i]: preds[i] for i in range(10)}

    im = gr.Image(shape=(28, 28), type="pil", image_mode="RGB", source="upload")

    demo = gr.Interface(
        fn=recognize_cifar10,
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
    )

    demo.launch()


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="demo_cifar10.yaml")
def main(cfg: DictConfig) -> None:
    demo_gradio(cfg)


if __name__ == "__main__":
    main()
