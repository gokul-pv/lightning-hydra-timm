FROM python:3.9.14-slim-bullseye

WORKDIR /opt/src

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt && rm -rf /root/.cache/pip

COPY src/demo_cifar10_scripted.py src/
COPY src/utils/ src/utils/
COPY configs/demo_cifar10_scripted.yaml configs/
COPY configs/hydra/ configs/hydra
COPY configs/paths/ configs/paths
COPY logs/train/runs/2022-10-04_10-13-22/model.scripted.pt pyproject.toml ./

CMD ["python", "src/demo_cifar10_scripted.py", "ckpt_path=model.scripted.pt" ]
