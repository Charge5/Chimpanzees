# Impact of hyper parameters on Catastrophic forgetting in Continous Learning

## How to set it up ? 

Make sure you have python 3.10 or later installed in your virtual environnement.
That should be the case if your are using google colab. To check run the following in your environnement:

```
python --version
```
This is required to avoid the following error:

```
def register_network_fn(target: T | Callable) -> T:
TypeError: unsupported operand type(s) for |: 'TypeVar' and '_CallableType'
```

When done install the [requirements.txt](./requirements.txt) from the main directory:

```
pip install -r requirements.txt
```

To run mammoth, first go to the following directory: 
```
cd ./Code/src/mammoth
```

Then run:
```
python utils/main.py --dataset seq_mnist \ 
                      --model lwf \
                      --lr 0.001 \
                      --optimizer adam \
                      --seed 42
```