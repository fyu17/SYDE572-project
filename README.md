This is my SYDE-572 final project code forked from [MOON](https://github.com/QinbinLi/MOON).

## Changelog: 
* 2022-12-3
Add FedAvgM to FedProx

## Dependencies
* PyTorch >= 1.0.0
* torchvision >= 0.2.1
* scikit-learn >= 0.23.1

## Usage

Here is an example to run FedProx with server momentum (FedAvgM) on CIFAR-10 with a simple CNN:
```
python main.py --dataset=cifar10 \
    --model=simple-cnn \
    --alg=fedprox \
    --lr=0.01 \
    --mu=0.01 \
    --epochs=10 \
    --comm_round=20 \
    --n_parties=4 \
    --partition=noniid \
    --beta=0.5 \
    --logdir='./logs/' \
    --datadir='./data/' \
```

## Hyperparameters
mu is for FedProx, and is usually {0.001, 0.01, 0.1, 1, 5, 10}. beta is for non-iid data partition, and is usually 0.5. server_momentum is for FedAvgM, and usually {0.5, 0.7, 0.9, 0.99}.
