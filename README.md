# Impact of hyper parameters on Catastrophic forgetting in Continous Learning

## How to set it up ? 

Make sure you have python 3.10 or later installed in your virtual environment.
That should be the case if your are using google colab. To check run the following in your environment:

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
python utils/main.py --dataset seq-mnist --backbone mnistmlp --model lwf --lr 0.01 --seed 42

```

However it was noticed that with lwf the MLP can not generalize accross the different tasks but the following command can:
```
python utils/main.py --dataset seq-mnist --backbone mnistmlp --model agem --lr 0.01 --seed 42 --n_epochs 2 --buffer_size 256
```

## Mammoth for dummies

* Model: Model of continous learning
* Backbone: Network used for the training
* Dataset: Dataset used for the training for example mnist

### What does the results mean?

#### 0. [Class-IL] (Class Incremental Learning):
* In Class Incremental Learning, the model is evaluated on all classes encountered so far without access to task identifiers.
* This setting is more challenging because the model must learn to distinguish between classes from multiple tasks simultaneously, even if tasks are disjoint.
* A low percentage (e.g., 19.27%) indicates the model struggles to generalize across tasks, likely due to catastrophic forgetting.
#### 1. [Task-IL] (Task Incremental Learning):
* In Task Incremental Learning, the model is given the task ID during evaluation, meaning it knows which task (or subset of classes) the current data belongs to.
* This setting is easier because the model can limit its predictions to only the classes of the current task.
* A high percentage (e.g., 97.62%) suggests the model performs well when it knows the task context, demonstrating that it can still learn individual tasks effectively.

### Where to modify ?
* To modify the dataset and tweak the order [seq_mnist](./Code/src/mammoth/datasets/seq_mnist.py)
* To modify the network architecture [MNISTMLP](./Code/src/mammoth/backbone/MNISTMLP.py)
