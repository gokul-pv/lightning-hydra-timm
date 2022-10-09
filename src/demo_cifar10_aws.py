import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
from typing import List, Tuple

import boto3
import gradio as gr
import hydra
import torch
from omegaconf import DictConfig

from utils import CSVLoggerS3, get_pylogger

log = get_pylogger(__name__)


def demo_gradio(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    model_env = os.environ.get("model")
    upload_dir_env = os.environ.get("flagged_dir")
    if model_env:
        log.info(f"Model will be downloaded from {model_env}")
        os.system(f"aws s3 cp {model_env} .")
        model_name = model_env.split("/")[-1]
    else:
        log.info(
            "Model will be downloaded from default path s3://myemlobucket/models/model_s3.traced.pt"
        )
        os.system("aws s3 cp s3://myemlobucket/models/model_s3.traced.pt .")
        model_name = "model_s3.traced.pt"
        # s3 = boto3.client("s3")
        # s3.download_file("myemlobucket", "models/model_s3.traced.pt", "model_s3.traced.pt")
    if upload_dir_env:
        log.info(f"Logs will be saved to {upload_dir_env}")

    assert os.path.exists(model_name)

    log.info("Running Demo with gradio")

    log.info(f"Instantiating scripted model <{model_name}>")
    model = torch.jit.load(model_name)

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

    if upload_dir_env:
        demo = gr.Interface(
            fn=recognize_cifar10,
            inputs=[im],
            outputs=[gr.Label(num_top_classes=10)],
            allow_flagging="manual",
            flagging_dir="flagged",
            flagging_callback=CSVLoggerS3(s3_dir=upload_dir_env, to_s3=True),
        )
    else:
        demo = gr.Interface(
            fn=recognize_cifar10,
            inputs=[im],
            outputs=[gr.Label(num_top_classes=10)],
        )

    demo.launch(server_name=cfg.server_name, server_port=80)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="demo_cifar10_aws.yaml")
def main(cfg: DictConfig) -> None:
    demo_gradio(cfg)


if __name__ == "__main__":
    main()
