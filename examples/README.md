## Merak-Zeus examples

`run.sh` can run both single node training and multi-node training.

### Setting up the environment

First, spawn a Docker container for Merak.

```sh
docker run -dit \
    --gpus all \
    --name merak-zeus \
    -v $(pwd):/workspace/merak-zeus \
    --cap-add SYS_ADMIN \
    --net host \
    --ipc host \
    mlenergy/zeus:v0.7.1
```

- `$(pwd)` assumes that you're running the command in the root of this repository.
- `SYS_ADMIN` is required for Zeus to be able to change the GPU's frequency.
- If you have Infiniband, consider adding something like `-v /dev/infiniband:/dev/infiniband`.
- If you want to train ImageNet classification models, download imagenet and add `-v $IMAGENET_DIR:/data/imagenet`. Data for text models (Wikitext-103 as reference) will be automatically downloaded with Hugging Face datasets.

Then, install Merak inside the container.

```console
$ docker exec -it merak-zeus bash
# cd /workspace/merak-zeus
# pip install pybind11
# pip uninstall torch torchvision
# conda install -c pytorch -c conda-forge pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3
# pip install -e .
```

### Training

Single-node training:

```bash
bash run.sh [MODEL_NAME] [MICROBATCH_SIZE] [NUM_MICROBATCHES]
```

Multi-node training (two nodes in example below):

```bash
# Run on master node
NNODES=2 NODE_RANK=0 MASTER_ADDR=[MASTER_NODE_ADDR] bash run.sh [MODEL_NAME] [MICROBATCH_SIZE] [NUM_MICROBATCHES]

# Run on non-master node
NNODES=2 NODE_RANK=1 MASTER_ADDR=[MASTER_NODE_ADDR] bash run.sh [MODEL_NAME] [MICROBATCH_SIZE] [NUM_MICROBATCHES]
```

Using only a subset of a node's GPUs (two GPUs in example below):

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash run.sh [MODEL_NAME] [MICROBATCH_SIZE] [NUM_MICROBATCHES] --pp 2 --dp 1 --tp 1
```

Notable command line arguments that `run.sh` accepts are:
- `--partition_method`: Pipeline stage partition method to use.
- `--max_steps`: Maximum number of steps to run training.
- `--shard_count`: Maximum number of `GraphModule`s to create from the traced DNN computation graph. Merak tries to figure out the per-shard number of parameters by computing `per_shard_num_params = total_num_params / shard_count` and adds layers to a `GraphModule`. When the number of parameters of that `GraphModule` exceeds `per_shard_num_params`, the `GraphModule` is cut there and Merak starts a new layer from that `GraphModule`. Note that skip connection forces multiple layers between the skip connection to be treated as one large layer.
- `--profile`: Give `true` to run offline instruction profiling (which will invoke `MerakTrainer.profile` instead of `MerakTrainer.train`. When running profiling, the number of microbatches must be 1.
  - e.g. `bash run.sh gpt3-large 4 1 --profile true` will run offline profiling for microbatch size 4.
- `--num_warmup_steps`: How many steps to warmup for offline instruction profiling.
- `--num_prof_steps`: How many steps to measure for offline instruction profiling. When not profiling (i.e., training), how many steps to measure for the given `PowerStateSchedule` and report profiling results to the Perseus server.
- `--train_schedule`: Set to `early_recompute_1f1b` to use the alternative Merak 1F1B schedule that decouples activation recomputation and backward.

Supported models and their sizes:

| model name | parameter count | global batch size |
|---|---|---|
| `bert-base-uncased` | 109.43M | 256 |
| `bert-large-uncased` | 334.95M | 256 |
| `gpt3-small` | 124.44M | 512 |
| `gpt3-medium` | 354.82M | 512 |
| `gpt3-large` | 758.73M | 512 |
| `gpt3-xl` | 1313.63M | 1024 |
| `gpt3-2.7b` | 2648.93M | 1024 |
| `gpt3-6.7b` | 6654.21M | 2048 |
| `gpt3-13b` | 12848.14M | 2048 |
| `gpt3-175b` | 174591.68M | 3200 |
| `wide-resnet50_2` | 68.88M | 1536 |
| `wide-resnet50_4` | 223.44M | 1536 |
| `wide-resnet50_8` | 804.16M | 1536 |
| `wide-resnet101_2` | 126.89M | 1536 |
| `wide-resnet101_4` | 419.63M | 1536 |
| `wide-resnet101_8` | 1517.37M | 1536 |

Recommended batch sizes are from the original publication of the DNN model.
Wide-ResNet variants were created by changing the width (number of convolution channels) of the original Wide-ResNet implementation.
50 and 101 are the number of layers, and 2, 4, 6, 8 are the width increase factor (2 is the original Wide-ResNet model).
The number of layers must be either 50 or 101, but any other width factor should work fine.

### Instruction Profiling

For profiling, you *must* pass in 1 as the number of microbatches and add `--profile true`.
Tweak `--num_prof_steps` for the number of consecutive forward/backward computations to average over.

```bash
bash run.sh [MODEL_NAME] [MICROBATCH_SIZE] 1 --profile true
```

After profiling completes, you'll find the CSV file that records all profiling information inside either `language-modeling` (for text models like BERT and GPT3) or `image-classification` (for image models like Wide-ResNet).
