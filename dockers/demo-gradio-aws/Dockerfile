FROM python:3.9.14-slim-bullseye

WORKDIR /opt/src

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt && rm -rf /root/.cache/pip

COPY src/demo_cifar10_aws.py src/
COPY src/utils/ src/utils/
COPY configs/demo_cifar10_aws.yaml configs/
COPY configs/hydra/ configs/hydra
COPY configs/paths/ configs/paths
COPY pyproject.toml ./

CMD ["python", "src/demo_cifar10_aws.py"]
