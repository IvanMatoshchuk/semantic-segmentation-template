<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
</div>

## Description

Template for training Semantic Segmentation models.
Main frameworks:

* [hydra](https://github.com/facebookresearch/hydra)
* [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

## How to run
Install dependencies
```yaml
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
bash bash/setup_conda.sh

# install requirements
pip install -r requirements.txt
```

Add your label-classes into `data/label_classes.json`.\
Specify number of classes in the selected `model` config.

Train model with default configuration:
```yaml
# default
python run.py

# train on CPU
python run.py trainer.gpus=0

# train on GPU
python run.py trainer.gpus=1

# train on multiple GPUs
python run.py trainer.gpus=[0,1,2,3]
```

You can override any parameter from the command line:
```yaml
python run.py trainer.max_epochs=20 datamodule.dataset_args.train.crop_size=416 model=unet
```

You can run hyperparameter search from the command line:
```yaml
# this will run 6 experiments one after the other,
# each with different combination of batch_size and learning rate
python run.py -m datamodule.dataloader_args.train.batch_size=32,64,128 optimizer.lr=0.001,0.0005
```

### How it works
By design, every run is initialized by [run.py](run.py) file. All PyTorch Lightning modules are dynamically instantiated from module paths specified in config. Example model config (unet.yaml):
```yaml
_target_: src.model.segmentation_model.HoneyBeeModel
_recursive_: False

model_cfg:
  _target_: segmentation_models_pytorch.Unet

  encoder_name: efficientnet-b0  # efficientnet-b0 timm-mobilenetv3_small_100 
  encoder_weights: imagenet
  encoder_depth: 5
  classes: 9
  in_channels: 1
```
Using this config we can instantiate the object with the following line:
```python
model = hydra.utils.instantiate(config.model)
```
This allows you to easily iterate over new models!<br>
Every time you create a new one, just specify its module path and parameters in appriopriate config file. <br>
The whole pipeline managing the instantiation logic is placed in [src/train.py](src/train.py).

<br>

## Main Project Configuration
Location: [configs/config.yaml](configs/config.yaml)<br>
Main project config contains default training configuration.<br>
It determines how config is composed when simply executing command `python run.py`.<br>
It also specifies everything that shouldn't be managed by experiment configurations.
<details>
<summary><b>Show main project configuration</b></summary>

```yaml
# specify here default training configuration
defaults:
  - _self_
  - logger: wandb
  - callbacks: wandb
  - datamodule: batch_datamodule
  - model: unet
  - trainer: default_trainer
  - optimizer: adam
  - scheduler: cosinewarm
  - loss: dice_with_ce

  # enable color logging
  - override hydra/hydra_logging: colorlog
  - override hydra/job_logging: colorlog
 
general:
  name: test  # name of the run, accessed by loggers
  seed: 123
  work_dir: ${hydra:runtime.cwd}

# print config at the start
print_config: True

# disable python warnings if they annoy you
ignore_warnings: False

# check performance on test set, using the best model achieved during training
# lightning chooses best model based on metric specified in checkpoint callback
test_after_training: False
```
</details>

## Other Repositories

<details>
<summary><b>Inspirations</b></summary>

This template was inspired by:
- [PyTorchLightning/deep-learninig-project-template](https://github.com/PyTorchLightning/deep-learning-project-template)
- [Erlemar/pytorch_tempest](https://github.com/Erlemar/pytorch_tempest)
- [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

</details>

<details>
<summary><b>Useful repositories</b></summary>

- [pytorch/hydra-torch](https://github.com/pytorch/hydra-torch) - resources for configuring PyTorch classes with Hydra,
- [qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) - pytorch-based models for semantic segmentation.

</details>
 
 <br>

## License
This project is licensed under the MIT License.
```
MIT License

Copyright (c) 2021 ashleve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


<br>
