# DPFL-Robustnes
This repository provides the code for our ACM CCS 2023 paper titled [Unraveling the Connections between Privacy and Certified Robustness in Federated Learning Against Poisoning Attacks](https://arxiv.org/abs/2209.04030).

# Installation:
Our implementation was based on `python=3.6.13`, `torch==1.10.1` and   `torchvision==0.11.2`.

```
conda create -n dpflrobust python=3.6
conda activate dpflrobust
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```


# Datasets:
MNIST and CIFAR-10 will be download into the dir 
`data`

# Run experiments: 
## 1. DPFL training under adversarial users/instances

####  Train User-level DP clean models for certified prediction:
```python
python main_userdp.py --config utils/mnist_params_ceracc.yaml  --fl_aggregation fedavg --dpfl max_model --n_runs 1

python main_userdp.py --config utils/cifar_params_ceracc.yaml  --fl_aggregation fedavg --dpfl max_model --n_runs 1
```
* `--dpfl` can be set as `median_per_layer, max_per_layer, max_model,median_model` for different clipping mechanisms.
* `--n_runs` is the number of runs for Monte Carlo Approximation
####  Train User-level DP models under $k$ adversarial users for certified attack inefficacy:

```python
python main_userdp.py --config utils/mnist_params_cerattack.yaml --is_poison --fl_aggregation fedavg --dpfl max_model  --adv_method labelflip --scale_factor 100 --num_adv 10 --n_runs 1

python main_userdp.py --config utils/cifar_params_cerattack.yaml --is_poison --fl_aggregation fedavg --dpfl max_model  --adv_method backdoor --scale_factor 100 --num_adv 10 --n_runs 1
```
* `--adv_method` can be set as `labelflip,backdoor`
* `--scale_factor` is the factor for scaling  the local update
* `--num_adv` is the number of adversarial users


####  Train Instance-level DP clean models for certified prediction:
```python
python main_insdp.py --config utils/mnist_params_insdp_ceracc.yaml --n_runs 1 

python main_insdp.py --config utils/cifar_params_insdp_ceracc.yaml --n_runs 1
```
* `--num_adv` is the number of adversarial instances

####  Train Instance-level DP models under $k$ adversarial instances for certified attack inefficacy:
```python
python main_insdp.py --config utils/mnist_params_insdp_cerattack.yaml --is_poison --adv_method labelflip --num_adv 5 --n_runs 1

python main_insdp.py --config utils/cifar_params_insdp_cerattack.yaml --is_poison --adv_method backdoor --num_adv 5 --n_runs 1
```
* `--adv_method` can be set as `labelflip,backdoor`

Hyperparameters can be changed according to our experiments setups (see appendix A.3 for paramerters details) in those yaml files to reproduce our experiments.

