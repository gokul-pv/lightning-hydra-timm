import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import urllib
from typing import List, Tuple

import albumentations as A
import hydra
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from captum.attr import FeatureAblation
from captum.robust import FGSM, PGD, MinParamPerturbation
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning import LightningModule

from utils import get_pylogger

log = get_pylogger(__name__)


def get_prediction(model: timm, image: torch.Tensor) -> Tuple[str, float, int]:
    """Function to return the model prediction, confidence and label index."""

    # Download human-readable labels for ImageNet and get the class names
    # url, filename = (
    #     "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    #     "imagenet_classes.txt",
    # )
    # urllib.request.urlretrieve(url, filename)

    with open("imagenet_classes.txt") as f:
        categories = [s.strip() for s in f.readlines()]

    with torch.no_grad():
        output = model(image)

    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = categories[pred_label_idx.item()]

    return predicted_label, prediction_score.squeeze().item(), pred_label_idx.item()


def image_show(img: torch.Tensor, pred: str) -> None:
    """Function to display the image with prediction."""

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_transform = T.Compose(
        [
            T.Normalize(
                mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                std=(1 / np.array(std)).tolist(),
            )
        ]
    )

    npimg = inv_transform(img).squeeze().permute(1, 2, 0).detach().numpy()

    plt.imshow(npimg)
    plt.title("prediction: %s" % pred)
    plt.show()


def get_pgd(model: timm, image_tensor: torch.Tensor, target_index: int) -> None:
    """Function to create a targeted PGD adversarial image."""

    pgd = PGD(
        model, torch.nn.CrossEntropyLoss(reduction="none"), lower_bound=-1, upper_bound=1
    )  # construct the PGD attacker
    perturbed_image_pgd = pgd.perturb(
        inputs=image_tensor,
        radius=0.13,
        step_size=0.02,
        step_num=7,
        target=torch.tensor([target_index]),
        targeted=True,
    )

    new_pred_pgd, score_pgd, _ = get_prediction(model, perturbed_image_pgd)

    image_show(perturbed_image_pgd, new_pred_pgd + " " + str(score_pgd))


def get_fgsm(model: timm, image_tensor: torch.Tensor, target_index: int) -> None:
    """Function to create a non-targeted FGSM adversarial image."""

    # Construct FGSM attacker
    fgsm = FGSM(model, lower_bound=-1, upper_bound=1)
    perturbed_image_fgsm = fgsm.perturb(image_tensor, epsilon=0.16, target=target_index)

    new_pred_fgsm, score_fgsm, _ = get_prediction(model, perturbed_image_fgsm)

    image_show(perturbed_image_fgsm, new_pred_fgsm + " " + str(score_fgsm))


def get_pixel_dropout(model: timm, image_tensor: torch.Tensor, target_index: int) -> None:
    """Function to check model robustness by pixel dropout."""

    feature_mask = (
        torch.arange(64 * 7 * 7)
        .reshape(8 * 7, 8 * 7)
        .repeat_interleave(repeats=4, dim=1)
        .repeat_interleave(repeats=4, dim=0)
        .reshape(1, 1, 224, 224)
    )

    ablator = FeatureAblation(model)
    attr = ablator.attribute(image_tensor, target=target_index, feature_mask=feature_mask)

    # Choose single channel, all channels have same attribution scores
    pixel_attr = attr[:, 0:1]

    def pixel_dropout(image, dropout_pixels):
        keep_pixels = image[0][0].numel() - int(dropout_pixels)
        vals, _ = torch.kthvalue(pixel_attr.flatten(), keep_pixels)
        return (pixel_attr < vals.item()) * image

    min_pert_attr = MinParamPerturbation(
        forward_func=model,
        attack=pixel_dropout,
        arg_name="dropout_pixels",
        mode="linear",
        arg_min=0,
        arg_max=1024,
        arg_step=16,
        preproc_fn=None,
        apply_before_preproc=True,
    )

    pixel_dropout_im, pixels_dropped = min_pert_attr.evaluate(
        image_tensor, target=target_index, perturbations_per_eval=10
    )
    log.info(f"Minimum Pixels Dropped:{pixels_dropped}")

    new_pred_dropout, score_dropout, _ = get_prediction(model, pixel_dropout_im)

    image_show(pixel_dropout_im, new_pred_dropout + " " + str(score_dropout))


def get_random_noise(model: timm, image: Image):
    """Function to check model robustness by adding random Gaussian noise."""

    transforms = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.GaussNoise(p=0.35),
            ToTensorV2(),
        ]
    )
    image = np.array(image)
    image_tensor = transforms(image=image)["image"]
    image_tensor = image_tensor.unsqueeze(0)

    pred, score, index = get_prediction(model, image_tensor)
    log.info(f"Predicted: {pred} (confidence = {score}, index = {index})")
    image_show(image_tensor, pred + " " + str(score))


def get_random_brightness(model: timm, image: Image):
    """Function to check model robustness by adding random brightness and contrast."""

    transforms = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.RandomBrightnessContrast(p=0.5),
            ToTensorV2(),
        ]
    )
    image = np.array(image)
    image_tensor = transforms(image=image)["image"]
    image_tensor = image_tensor.unsqueeze(0)

    pred, score, index = get_prediction(model, image_tensor)
    log.info(f"Predicted: {pred} (confidence = {score}, index = {index})")
    image_show(image_tensor, pred + " " + str(score))


def model_robustness(cfg: DictConfig) -> None:
    """Function to check the robustness of any pre-trained timm model
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    log.info("Running Model explanation")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.eval()
    log.info(f"Loaded Model: {model}")

    transforms = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )
    transform_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    image = Image.open(cfg.input_image)
    transformed_img = transforms(image)
    image_tensor = transform_normalize(transformed_img)
    image_tensor = image_tensor.unsqueeze(0)

    predicted_label, prediction_score, pred_label_idx = get_prediction(model, image_tensor)

    log.info(
        f"Predicted: {predicted_label} (confidence = {prediction_score}, index = {pred_label_idx})"
    )

    if 1 in cfg.model_robusness:
        log.info("Projected Gradient Descent :")
        image_tensor_grad = image_tensor
        image_tensor_grad.requires_grad = True
        get_pgd(model, image_tensor_grad, cfg.target_index)

    if 2 in cfg.model_robusness:
        log.info("Fast Gradient Sign Method :")
        image_tensor_grad = image_tensor
        image_tensor_grad.requires_grad = True
        get_fgsm(model, image_tensor_grad, pred_label_idx)

    if 3 in cfg.model_robusness:
        log.info("Pixel Dropout :")
        image_tensor_grad = image_tensor
        image_tensor_grad.requires_grad = True
        get_pixel_dropout(model, image_tensor_grad, pred_label_idx)

    if 4 in cfg.model_robusness:
        log.info("Random Noise :")
        get_random_noise(model, image)

    if 5 in cfg.model_robusness:
        log.info("Random Brightness :")
        get_random_brightness(model, image)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="robustness.yaml")
def main(cfg: DictConfig) -> None:
    model_robustness(cfg)


if __name__ == "__main__":
    main()
