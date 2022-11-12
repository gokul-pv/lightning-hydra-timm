import json
import subprocess

from tests.helpers.run_if import RunIf


@RunIf(torchserve=True)
def test_serve_grpc_inference(cifar10_images, grpc_client, model):

    for image in cifar10_images:
        print(f"Testing image: {image}")
        proc = subprocess.run(
            ["python3", grpc_client, "infer", model, image],
            capture_output=True,
            text=True,
            shell=False,
        )

        top_output = json.loads(proc.stdout)
        print(f"Response: {top_output}")

        predicted_label = list(top_output.keys())[0]
        print(f"Predicted label: {predicted_label} and Confidence: {top_output[predicted_label]}")
        assert predicted_label == image.stem[2:]
        print("=======================================================================\n")
