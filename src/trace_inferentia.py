import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import hydra
import torch
import torch.nn.functional as F
import torch_neuron
import torchvision.transforms as T
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from src import utils

log = utils.get_pylogger(__name__)


def trace_neuron(cfg: DictConfig):
    """Function to load a model and trace it for aws neuron.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    assert cfg.ckpt_path

    log.info("Running...")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    ckpt = torch.load(cfg.ckpt_path, map_location=torch.device("cpu"))

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    log.info(f"Loaded Model: {model}")

    # Create an example input for compilation
    image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)

    torch.neuron.analyze_model(model.net.eval(), example_inputs=[image])

    model_neuron = torch.neuron.trace(model.net.eval(), example_inputs=[image])

    model_neuron.save("/opt/src/cifar10-net.neuron.traced.pt")


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="demo_cifar10.yaml")
def main(cfg: DictConfig) -> None:
    trace_neuron(cfg)


if __name__ == "__main__":
    main()
