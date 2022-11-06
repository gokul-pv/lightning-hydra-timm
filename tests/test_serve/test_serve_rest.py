import requests


def test_serve_inference(cifar10_images, public_ip, model):

    for image in cifar10_images:
        print(f"Testing image: {image}")
        res = requests.post(
            f"http://{public_ip}:8080/predictions/{model}/1.0", files={"data": open(image, "rb")}
        )
        top_output = res.json()
        print(f"Response: {top_output}")

        predicted_label = list(top_output.keys())[0]
        print(f"Predicted label: {predicted_label} and Confidence: {top_output[predicted_label]}")
        assert predicted_label == image.stem[2:]
        print("=======================================================================\n")
