<div  align="center">

## Distributed Training

</div>

**Description**
A pre-trained  transformer model (`vit_base_patch32_224`) from timm was trained on CIFAR10 Dataset using  4 nodes each with single NVIDIA T4 GPU and Distributed DataParallel ([DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html?highlight=distributed)). Refer this [link](https://pytorch.org/docs/stable/notes/ddp.html) to understand how gradients are applied in the backward pass. Read about all reduce algorithm [here](https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/).

**How to run**

```bash
# On master node
MASTER_PORT=29500 MASTER_ADDR=172.31.10.234 WORLD_SIZE=4 NODE_RANK=0 python src/train.py experiment=cifar10_ddp

# On worker node
MASTER_PORT=29500 MASTER_ADDR=172.31.10.234 WORLD_SIZE=4 NODE_RANK=1 python src/train.py experiment=cifar10_ddp

# MASTER_PORT - required; has to be a free port on machine with NODE_RANK 0
# MASTER_ADDR - required (except for NODE_RANK 0); address of NODE_RANK 0 node
# WORLD_SIZE - required; how many nodes are in the cluster
# NODE_RANK - required; id of the node in the cluster
```

**Results**
The best model checkpoint for the above training can be found [here](https://myemlobucket.s3.ap-south-1.amazonaws.com/DistributedTrainingLogs/ddp_logs/checkpoints/epoch_024.ckpt).

- Model: `vit_base_patch32_224`
- Dataset: `CIFAR10`
- Epochs: `25`
- Effective batch size: `1280 (=320*4*1)`
- Test Accuracy: `90.13` ([link](./resources/ddp_test_metric.txt))
- Test Loss: `0.36529`
- Training time: `1 hour 22 minutes`

<p align="center" style="padding: 10px">
<img alt="Forwarding" src="https://github.com/gokul-pv/lightning-hydra-timm/blob/main/logBook/resources/ddp_gpu_util.png?raw=true" width =500><br>
<em style="color: grey">GPU utilization </em>
</p>

## Sharded Training using FSDP

Sharded Training still utilizes Data Parallel Training under the hood, except optimizer states and gradients are sharded across GPUs. This means the memory overhead per GPU is lower, as each GPU only has to maintain a partition of your optimizer state and gradients. Refer the [link](https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html).

**How to run**

```bash
# On master node
MASTER_PORT=29500 MASTER_ADDR=172.31.10.234 WORLD_SIZE=4 NODE_RANK=0 python src/train.py experiment=cifar10_fsdp

# On worker node
MASTER_PORT=29500 MASTER_ADDR=172.31.10.234 WORLD_SIZE=4 NODE_RANK=1 python src/train.py experiment=cifar10_fsdp
```

**Results**
The best model checkpoint for the above training can be found [here](https://myemlobucket.s3.ap-south-1.amazonaws.com/DistributedTrainingLogs/fsdp_logs/checkpoints/epoch_023.ckpt).

- Model: `vit_base_patch32_224`
- Dataset: `CIFAR10`
- Epochs: `25`
- Effective batch size: `1280 (=320*4*1)`
- Test Accuracy: `83.6799` ([link](./resources/fsdp_test_metric.txt))
- Test Loss: `0.52269`
- Training time: `35 minutes`

<p align="center" style="padding: 10px">
<img alt="Forwarding" src="https://github.com/gokul-pv/lightning-hydra-timm/blob/main/logBook/resources/fsdp_gpu_util.png?raw=true" width =500><br>
<em style="color: grey">GPU utilization </em>
</p>

## Training over the internet using Hivemind

Hivemind provides capabilities of decentralized Training over the Internet. Refer the [link](https://pytorch-lightning.readthedocs.io/en/stable/strategies/hivemind_basic.html).

**How to run**

```bash
# On master node
python src/train.py experiment=cifar10_hivemind

# On other node, provide initial peers as ENV var or pass it during instantiation and
# run above command
```

**Results**
The best model checkpoint for the above training can be found [here](https://myemlobucket.s3.ap-south-1.amazonaws.com/DistributedTrainingLogs/hivemind_logs/checkpoints/epoch_012.ckpt).

- Model: `vit_base_patch32_224`
- Dataset: `CIFAR10`
- Epochs: `25`
- Effective batch size: `640(=320*2*1)`
- Test Accuracy: `95.4299` ([link](./resources/hivemind_test_metric.txt))
- Test Loss: `0.22328`
- Training time: `3 hours 50 minutes`
- Tensorboard link : [click here](https://tensorboard.dev/experiment/3onn9vunQ8yTLbe07JQKag/#scalars)
