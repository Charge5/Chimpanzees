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

## How to run Mammoth?  
First go to the following directory: 
```
cd ./Code/src/mammoth
```

Then run for example:
```
python utils/main.py --dataset seq-mnist --backbone mnistmlp --model lwf-mc --lr 0.01 --seed 42 --n_epochs 50 --mlp_hidden_size 100 --mlp_hidden_depth 2

```
The above use mammoth to train an MLP model of depth 2, width 100 on the sequential MNIST dataset using the LwF-MC algorithm.  
If you want to use another model, e.g. A-GEM, just change the '--model' parameter and add the required parameters for this model (for A-GEM, the buffer size is required).

```
python utils/main.py --dataset seq-mnist --backbone mnistmlp --model agem --lr 0.01 --seed 42 --n_epochs 50 --mlp_hidden_size 100 --mlp_hidden_depth 2 --buffer_size 500
```

## How to run the experiments?
To run the experiments, first go to the 'Code' directory:
```
cd ./Code
```
Each experiment corresponds to one python script. To reproduce an experiment, you just need to run the corresponding script. A description of the experiment is always given in the experiment script.  
- To run the experiment reproducing Figure 3 of https://arxiv.org/abs/1906.00904, run the following command
  ```
  python replication_paper.py
  ```
  This will run the experiment for a depth 4 width 16 MLP. To modify the size, you can directly edit the 'params' dictionnary in the .py file.
- To run the experiment where we count the number of activation regions of a model trained via continuous learning, run the following command
  ```
  python count_regions_during_CL.py
  ```
  This will run the experiment for a depth 2 width 20 MLP using the LwF-MC algorithm. To modify the size, model, learning rate and other parameters you can directly edit the 'params' dictionnary in the .py file.
- To run the experiment where we relate the number of activation regions to the accuracy of each task, you can run the following command
  ```
  python regions_and_accuracy.py
  ```
  This will run the experiment for a depth 2 width 70 MLP using the LwF-MC algorithm. Again you can modify the parameters by editing the 'params' dictionnary in the .py file.

When running all the above experiments, the results and logs will be saved in a directory with the same name as the experiment script.
For example the results of the experiment [`replication_paper.py`](./Code/replication_paper.py) are saved in the directory [`replication_paper`](./Code/replication_paper) located at
```
cd ./Code/replication_paper
```

## How the plots were created?
We ran all the above experiments and the results are availble in the corresponding directories. The final plots available in our report are then created by the notebook [`generate_figures.ipynb`](./Code/generate_figures.ipynb) located at
```
cd ./Code/generate_figures.ipynb
```
This notebook simply loads the results saved during the experiments and build the corresponding plot.  


## More details about the repository

### Software contributions
The main software contribution in our repository is the package [`activationregion`](./Code/src/activationregion) located at `./Code/src/activationregion`.  
It contains the function `exact_count_2D`, available at [`activationregion/core`](./Code/src/activationregion/core), our implementation of the exact counting of the number of activation regions, outlined in https://arxiv.org/abs/1906.00904.  

As a first try, we also implemented the counting method based on sampling the input space, e.g. described in https://arxiv.org/abs/1802.08760, but we didn't use it for our experiments as we prefer to have the exact number of activation regions. This function is `sample_count_2D` and is still availble in [`activationregion/core`](./Code/src/activationregion/core).

### Mammoth
The public repository Mammoth, publicly available at https://github.com/aimagelab/mammoth, is integrated in ours. The only modifications done to Mammoth are the following:
- Modified the `main.py` and `train.py` scripts to save the model after each task (always) and within each task at different epochs via the parameter `--save_model_within_tasks True` (optional). This was useful to then load each model and count the number of regions.  
- We further modified `main.py` and `train.py` via the parameter `--save_accuracy_within_tasks True` (optional). This was useful to save the model's accuracy within each task and not only after each task. 

### Mammoth for dummies

* Model: Model of continous learning
* Backbone: Network used for the training
* Dataset: Dataset used for the training for example mnist

### What do the results mean?

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
