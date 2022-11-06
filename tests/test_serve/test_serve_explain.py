import matplotlib.pyplot as plt
import numpy as np
import requests
import torchvision.transforms as T
from captum.attr import visualization as viz
from PIL import Image


def test_serve_explain(cifar10_images, public_ip, model):

    for image in cifar10_images:
        print(f"Testing image: {image}")
        res = requests.post(
            f"http://{public_ip}:8080/predictions/{model}/1.0", files={"data": open(image, "rb")}
        )
        top_output = res.json()
        print(f"Response: {top_output}")

        res = requests.post(
            f"http://{public_ip}:8080/explanations/{model}/1.0", files={"data": open(image, "rb")}
        )
        ig = res.json()

        inp_image = Image.open(image)
        to_tensor = T.Compose(
            [
                T.Resize(224),
                T.ToTensor(),
                # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        inp_image = to_tensor(inp_image)
        inp_image = inp_image.numpy()

        attributions = np.array(ig)
        inp_image, attributions = inp_image.transpose(1, 2, 0), attributions.transpose(1, 2, 0)

        print(inp_image.shape, attributions.shape)

        plt.imshow(inp_image, cmap="inferno")
        plt.savefig(f"{image.parent.parent}/original_image.png")

        viz.visualize_image_attr(
            attributions,
            inp_image,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title="Overlaid Integrated Gradients",
        )
        plt.savefig(f"{image.parent.parent}/model_explanation.png")

        break
