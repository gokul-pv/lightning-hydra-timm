## Model Explainability

Model explainability refers to the concept of being able to understand the machine learning model. For example – if a healthcare model is predicting whether a patient is suffering from a particular disease or not. The medical practitioners need to know what parameters the model is taking into account or if the model contains any bias.

### Why is Model Explainability required?

- Being able to interpret a model increases trust in a machine-learning model. This becomes all the more important in scenarios involving life-and-death situations like healthcare, law, credit lending, etc.
- Once we understand a model, we can detect if there is any bias present in the model.
- Model explainability becomes important while debugging a model during the development phase.

### Description

In this experiment, I try to explain pre-trained model from timm using the following methods

- [Integrated Gradients](https://arxiv.org/abs/1703.01365) (Click [here](https://erdem.pl/2022/04/xai-methods-integrated-gradients) for more)
- Noise Tunnel
- Saliency
- [Occlusion](https://arxiv.org/abs/1311.2901)
- SHAP
- [GradCAM](https://arxiv.org/abs/1610.02391)
- [GradCAM++](https://sites.math.northwestern.edu/~mlerma/papers/Grad_CAM___is_equivalent_to_Grad_CAM_with_positive_gradients.pdf)

I also try to test the robustness of model by creating some adversarial samples and augmented images.

- [Projected Gradient Descent](https://arxiv.org/abs/1706.06083)
- [Fast Gradient Sign Method](https://arxiv.org/abs/1412.6572)
- Pixel Dropout
- Random Noise
- Random Brightness

### How to run

```python
# For model explanation, run
python src/explain.py

# For testing model robustness, run
python src/robustness.py
```

### Results

The results obtained from the above-mentioned methods for model explainability are shown below:

| Original Image                           | Integrated Gradients                       | Noise Tunnel                               | Saliency                                   | Occlusion                                  | SHAP                                       | GradCAM                                    | GradCAM++                                  |
| ---------------------------------------- | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ |
| ![](./logBook/resources/imagenet/1.jpg)  | ![](./logBook/resources/imagenet/1_1.jpg)  | ![](./logBook/resources/imagenet/1_2.jpg)  | ![](./logBook/resources/imagenet/1_3.jpg)  | ![](./logBook/resources/imagenet/1_4.jpg)  | ![](./logBook/resources/imagenet/1_5.jpg)  | ![](./logBook/resources/imagenet/1_6.jpg)  | ![](./logBook/resources/imagenet/1_7.jpg)  |
| ![](./logBook/resources/imagenet/2.jpg)  | ![](./logBook/resources/imagenet/2_1.jpg)  | ![](./logBook/resources/imagenet/2_2.jpg)  | ![](./logBook/resources/imagenet/2_3.jpg)  | ![](./logBook/resources/imagenet/2_4.jpg)  | ![](./logBook/resources/imagenet/2_5.jpg)  | ![](./logBook/resources/imagenet/2_6.jpg)  | ![](./logBook/resources/imagenet/2_7.jpg)  |
| ![](./logBook/resources/imagenet/3.jpg)  | ![](./logBook/resources/imagenet/3_1.jpg)  | ![](./logBook/resources/imagenet/3_2.jpg)  | ![](./logBook/resources/imagenet/3_3.jpg)  | ![](./logBook/resources/imagenet/3_4.jpg)  | ![](./logBook/resources/imagenet/3_5.jpg)  | ![](./logBook/resources/imagenet/3_6.jpg)  | ![](./logBook/resources/imagenet/3_7.jpg)  |
| ![](./logBook/resources/imagenet/4.jpg)  | ![](./logBook/resources/imagenet/4_1.jpg)  | ![](./logBook/resources/imagenet/4_2.jpg)  | ![](./logBook/resources/imagenet/4_3.jpg)  | ![](./logBook/resources/imagenet/4_4.jpg)  | ![](./logBook/resources/imagenet/4_5.jpg)  | ![](./logBook/resources/imagenet/4_6.jpg)  | ![](./logBook/resources/imagenet/4_7.jpg)  |
| ![](./logBook/resources/imagenet/5.jpg)  | ![](./logBook/resources/imagenet/5_1.jpg)  | ![](./logBook/resources/imagenet/5_2.jpg)  | ![](./logBook/resources/imagenet/5_3.jpg)  | ![](./logBook/resources/imagenet/5_4.jpg)  | ![](./logBook/resources/imagenet/5_5.jpg)  | ![](./logBook/resources/imagenet/5_6.jpg)  | ![](./logBook/resources/imagenet/5_7.jpg)  |
| ![](./logBook/resources/imagenet/6.jpg)  | ![](./logBook/resources/imagenet/6_1.jpg)  | ![](./logBook/resources/imagenet/6_2.jpg)  | ![](./logBook/resources/imagenet/6_3.jpg)  | ![](./logBook/resources/imagenet/6_4.jpg)  | ![](./logBook/resources/imagenet/6_5.jpg)  | ![](./logBook/resources/imagenet/6_6.jpg)  | ![](./logBook/resources/imagenet/6_7.jpg)  |
| ![](./logBook/resources/imagenet/7.jpg)  | ![](./logBook/resources/imagenet/7_1.jpg)  | ![](./logBook/resources/imagenet/7_2.jpg)  | ![](./logBook/resources/imagenet/7_3.jpg)  | ![](./logBook/resources/imagenet/7_4.jpg)  | ![](./logBook/resources/imagenet/7_5.jpg)  | ![](./logBook/resources/imagenet/7_6.jpg)  | ![](./logBook/resources/imagenet/7_7.jpg)  |
| ![](./logBook/resources/imagenet/8.jpg)  | ![](./logBook/resources/imagenet/8_1.jpg)  | ![](./logBook/resources/imagenet/8_2.jpg)  | ![](./logBook/resources/imagenet/8_3.jpg)  | ![](./logBook/resources/imagenet/8_4.jpg)  | ![](./logBook/resources/imagenet/8_5.jpg)  | ![](./logBook/resources/imagenet/8_6.jpg)  | ![](./logBook/resources/imagenet/8_7.jpg)  |
| ![](./logBook/resources/imagenet/9.jpg)  | ![](./logBook/resources/imagenet/9_1.jpg)  | ![](./logBook/resources/imagenet/9_2.jpg)  | ![](./logBook/resources/imagenet/9_3.jpg)  | ![](./logBook/resources/imagenet/9_4.jpg)  | ![](./logBook/resources/imagenet/9_5.jpg)  | ![](./logBook/resources/imagenet/9_6.jpg)  | ![](./logBook/resources/imagenet/9_7.jpg)  |
| ![](./logBook/resources/imagenet/10.jpg) | ![](./logBook/resources/imagenet/10_1.jpg) | ![](./logBook/resources/imagenet/10_2.jpg) | ![](./logBook/resources/imagenet/10_3.jpg) | ![](./logBook/resources/imagenet/10_4.jpg) | ![](./logBook/resources/imagenet/10_5.jpg) | ![](./logBook/resources/imagenet/10_6.jpg) | ![](./logBook/resources/imagenet/10_7.jpg) |

PGD was used to make the model predict Boston bull for all the above images. The images that made the model predict Boston bull are shown below:

| Original Image                           | PGD Image                                  |
| ---------------------------------------- | ------------------------------------------ |
| ![](./logBook/resources/imagenet/1.jpg)  | ![](./logBook/resources/imagenet/1_8.jpg)  |
| ![](./logBook/resources/imagenet/2.jpg)  | ![](./logBook/resources/imagenet/2_8.jpg)  |
| ![](./logBook/resources/imagenet/3.jpg)  | ![](./logBook/resources/imagenet/3_8.jpg)  |
| ![](./logBook/resources/imagenet/4.jpg)  | ![](./logBook/resources/imagenet/4_8.jpg)  |
| ![](./logBook/resources/imagenet/5.jpg)  | ![](./logBook/resources/imagenet/5_8.jpg)  |
| ![](./logBook/resources/imagenet/6.jpg)  | ![](./logBook/resources/imagenet/6_8.jpg)  |
| ![](./logBook/resources/imagenet/7.jpg)  | ![](./logBook/resources/imagenet/7_8.jpg)  |
| ![](./logBook/resources/imagenet/8.jpg)  | ![](./logBook/resources/imagenet/8_8.jpg)  |
| ![](./logBook/resources/imagenet/9.jpg)  | ![](./logBook/resources/imagenet/9_8.jpg)  |
| ![](./logBook/resources/imagenet/10.jpg) | ![](./logBook/resources/imagenet/10_8.jpg) |

The above images were also tested for model robustness and results are as follows:

| Original Image                           | FGSM Image                                 | Pixel Dropout                               | Random Noise                                | Random Brightness                           |
| ---------------------------------------- | ------------------------------------------ | ------------------------------------------- | ------------------------------------------- | ------------------------------------------- |
| ![](./logBook/resources/imagenet/1.jpg)  | ![](./logBook/resources/imagenet/1_9.jpg)  | ![](./logBook/resources/imagenet/1_10.jpg)  | ![](./logBook/resources/imagenet/1_11.jpg)  | ![](./logBook/resources/imagenet/1_12.jpg)  |
| ![](./logBook/resources/imagenet/2.jpg)  | ![](./logBook/resources/imagenet/2_9.jpg)  | ![](./logBook/resources/imagenet/2_10.jpg)  | ![](./logBook/resources/imagenet/2_11.jpg)  | ![](./logBook/resources/imagenet/2_12.jpg)  |
| ![](./logBook/resources/imagenet/3.jpg)  | ![](./logBook/resources/imagenet/3_9.jpg)  | ![](./logBook/resources/imagenet/3_10.jpg)  | ![](./logBook/resources/imagenet/3_11.jpg)  | ![](./logBook/resources/imagenet/3_12.jpg)  |
| ![](./logBook/resources/imagenet/4.jpg)  | ![](./logBook/resources/imagenet/4_9.jpg)  | ![](./logBook/resources/imagenet/4_10.jpg)  | ![](./logBook/resources/imagenet/4_11.jpg)  | ![](./logBook/resources/imagenet/4_12.jpg)  |
| ![](./logBook/resources/imagenet/5.jpg)  | ![](./logBook/resources/imagenet/5_9.jpg)  | ![](./logBook/resources/imagenet/5_10.jpg)  | ![](./logBook/resources/imagenet/5_11.jpg)  | ![](./logBook/resources/imagenet/5_12.jpg)  |
| ![](./logBook/resources/imagenet/6.jpg)  | ![](./logBook/resources/imagenet/6_9.jpg)  | ![](./logBook/resources/imagenet/6_10.jpg)  | ![](./logBook/resources/imagenet/6_11.jpg)  | ![](./logBook/resources/imagenet/6_12.jpg)  |
| ![](./logBook/resources/imagenet/7.jpg)  | ![](./logBook/resources/imagenet/7_9.jpg)  | ![](./logBook/resources/imagenet/7_10.jpg)  | ![](./logBook/resources/imagenet/7_11.jpg)  | ![](./logBook/resources/imagenet/7_12.jpg)  |
| ![](./logBook/resources/imagenet/8.jpg)  | ![](./logBook/resources/imagenet/8_9.jpg)  | ![](./logBook/resources/imagenet/8_10.jpg)  | ![](./logBook/resources/imagenet/8_11.jpg)  | ![](./logBook/resources/imagenet/8_12.jpg)  |
| ![](./logBook/resources/imagenet/9.jpg)  | ![](./logBook/resources/imagenet/9_9.jpg)  | ![](./logBook/resources/imagenet/9_10.jpg)  | ![](./logBook/resources/imagenet/9_11.jpg)  | ![](./logBook/resources/imagenet/9_12.jpg)  |
| ![](./logBook/resources/imagenet/10.jpg) | ![](./logBook/resources/imagenet/10_9.jpg) | ![](./logBook/resources/imagenet/10_10.jpg) | ![](./logBook/resources/imagenet/10_11.jpg) | ![](./logBook/resources/imagenet/10_12.jpg) |
