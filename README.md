<div align="center">

## Lightning-Hydra-Timm

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

- A convenient all-in-one technology stack for deep learning prototyping - allows you to rapidly iterate over new models provided with timm, datasets and tasks on different hardware accelerators like CPUs, multi-GPUs or TPUs.
- A collection of best practices for efficient workflow and reproducibility.
- Thoroughly commented - you can use this repo as a reference and educational resource.

## Main Technologies

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - a lightweight PyTorch wrapper for high-performance AI research. Think of it as a framework for organizing your PyTorch code.

[Hydra](https://github.com/facebookresearch/hydra) - a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

[Pytorch Image Models](https://rwightman.github.io/pytorch-image-models/) - Py**T**orch **Im**age **M**odels (`timm`) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.

Click on the links below for more

- [Distributed Training](./logBook/DistributedTraining.md)
- [Model explainability and robustness](./logBook/ModelExplainaibilty.md)
- [Model serving with TorchServe](./logBook/TorchServe.md)
- [Training on Habana chips and deployment on AWS Accelerators](./logBook/AWSAccelerators.md)
- [Train CIFAR10 on Resnet](#cifar10-training)
- [Hyper-parameter sweep using optuna](#hyperparameter-search)
- [Demo webapp using Gradio, TorchScript and AWS](#deploy-using-gradio-and-torchscript)

## How to run

- Install dependencies

```bash
# clone project
git clone https://github.com/gokul-pv/lightning-hydra-timm
cd lightning-hydra-timm

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

- Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

- Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

- You can override any parameter from the command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

#### CIFAR10 Training

- For training CIFAR10 dataset using Resnet18 from timm

```bash
# For training
python src/train.py experiment=cifar10_example

# To build docker image run
make build-train
# or
docker build -t gokulpv/lighteninghydratimm -f dockers/train/Dockerfile .
# or pull from Docker hub
docker pull gokulpv/lighteninghydratimm:latest

# To run the docker image. NOTE: To get the trained model checkpoint and all the logs on the host machine, you'll have to volume mount your directory inside docker.
docker run -it --volume /workspace/lightning-hydra-timm/dockerMount:/opt/src/logs gokulpv/lighteninghydratimm python src/train.py experiment=cifar10_example
```

#### Hyperparameter Search

- To do a Hyperparam sweep for CIFAR10 dataset with Resnet18 from timm,

```bash
# Hyperparam sweep using optuna
python src/train.py -m experiment=cifar10_example hparams_search=cifar10_optuna

# Sample data and logs can be pulled from google drive using dvc
dvc pull -r gdrive

# Logs are also available at
https://tensorboard.dev/experiment/4Eki3IjVTGaSGtgE4HnCGg/#scalars
```

#### Deploy using Gradio and TorchScript

- To build a demo app for deployment using gradio,

```bash
# Train cifar10 using resnet18 from timm and save model using torch trace
python src/train.py experiment=cifar10_example trace=True

# Launch demo using checkpoint
python src/demo_cifar10_scripted.py ckpt_path=logs/<run folder>/model.traced.pt

# Build docker using
make build-demo-traced
# or
docker build -t gokulpv/demogradiotraced -f dockers/demo-gradio-traced/Dockerfile .
# Pull image from docker hub
docker pull gokulpv/demogradiotraced:latest
# Run the docker using
docker run -p 8080:8080 -it gokulpv/demogradiotraced:latest


# To save trained model using torch script, switch to commit 68a338b and run
python src/train.py experiment=cifar10_example script=True
# To build docker image for scripted model,
make build-demo
# or pull the image
docker pull gokulpv/demogradio:latest
```

- To download a scripted model from AWS S3 and run the demo application, do the following:

```bash
# Train cifar10 using resnet18 from timm and save scripted model to S3
# Refer s3://myemlobucket/models/model_s3.scripted.pt

# To build docker image, run
make build-demo-aws
# or pull the image from docker hub
docker pull gokulpv/demogradioaws:latest
# or from AWS ECR
docker pull public.ecr.aws/j7y6i3l0/demogradio:latest
# Run the docker using
docker run -p 80:80 -it gokulpv/demogradioaws:latest

# To download the model from custom S3 path and log the result to custom S3 path
make build-demo-aws-env
# or pull the image
docker pull gokulpv/demogradioawsenv:latest
# or
docker pull public.ecr.aws/j7y6i3l0/demogradio:envvar
# Run the docker using
docker run -p 80:80 -it -e "model=s3://myemlobucket/models/model_s3.traced.pt" -e "flagged_dir=s3://myemlobucket/logs/" gokulpv/demogradioawsenv:latest
```
