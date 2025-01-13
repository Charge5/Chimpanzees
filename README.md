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

## How to run the experiments?
To run the experiments, first go to the 'Code' directory:
```
cd ./Code
```
Each experiment corresponds to one python script. To reproduce an experiment, you just need to run the corresponding script. A description of the experiment is always given in the experiment script.

#### Experiments corresponding to Figure 1
- To run the experiment corresponding to Figure 1, you will need to run:
  ```commandline
  python bulk_run_agem_hiof.py
  python bulk_run_lwfmc_hiof.py
  ```
  The setup used for the experience, i.e. the different hyperparameters, are already defined in the script.  
  When the two runs are done, run the following script to generate the plots:
  ```commandline
  python hyperparameters_impact_on_forgetting.py
  ```

#### Experiments corresponding to Figure 2 to 5
- To run the experiment corresponding to Figure 2, run the following command
  ```
  python count_regions_during_CL.py
  ```
  This will run the experiment for a depth 2 width 20 MLP using the LwF-MC algorithm. To modify the size, model, learning rate and other parameters you can directly edit the 'params' dictionnary in the .py file.
- To run the experiment corresponding to Figure 3, run the following command
  ```
  python regions_and_accuracy.py
  ```
  This will run the experiment for a depth 2 width 70 MLP using the LwF-MC algorithm. Again you can modify the parameters by editing the 'params' dictionnary in the .py file.
- To run the experiment corresponding to Figure 4, run the following command
  ```
  python replication_paper.py
  ```
  This will run the experiment for a depth 4 width 16 MLP. To modify the size, you can directly edit the 'params' dictionnary in the .py file.
- To run the experiment corresponding to Figure 5, run the following command
  ```
  python regions_density_after_tasks.py
  ```

When running all the above experiments, the results and logs will be saved in a directory with the same name as the experiment script.
For example the results of the experiment [`replication_paper.py`](./Code/replication_paper.py) are saved in the directory [`replication_paper`](./Code/replication_paper) located at
```
cd ./Code/replication_paper
```
##### How the plots corresponding to the above experiments were created?
The final plots available in our report are then created by the notebook [`generate_figures.ipynb`](./Code/generate_figures.ipynb) located at
```
cd ./Code/generate_figures.ipynb
```
This notebook simply loads the results saved during the experiments and build the corresponding plot.

We ran all the experiments locally. The running time goes from a few minutes to around 50 hours, mainly because experiments are averaged over 10 independent runs.

## More details about the repository

### Software contributions
The main software contribution in our repository is the package [`activationregion`](./Code/src/activationregion) located at `./Code/src/activationregion`.  
It contains the function `exact_count_2D`, available in[`activationregion/core`](./Code/src/activationregion/core.py). This function is our implementation of the exact counting of the number of activation regions, following https://arxiv.org/abs/1906.00904.  

As a first try, we also implemented the counting method based on sampling the input space, e.g. described in https://arxiv.org/abs/1802.08760, but we didn't use it for our experiments as we prefer to have the exact number of activation regions. This function is `sample_count_2D` and is still availble in [`activationregion/core`](./Code/src/activationregion/core.py).

### Mammoth
The repository Mammoth, publicly available at https://github.com/aimagelab/mammoth, is integrated in ours. The only modifications done to Mammoth are the following:
- Modified the `main.py` and `train.py` scripts to save the model after each task (always, and saved in `./Code/Data/Output/models`) and within each task at different epochs via the parameter `--save_model_within_tasks True` (optional). This was useful to then load each model and count the number of activation regions for our experiments.  
- We further modified `main.py` and `train.py` via the parameter `--save_accuracy_within_tasks True` (optional). This was useful to save the model's accuracy within each task and not only after each task. 
- We added the following parameters (required) `--mlp_hidden_depth 2` to allow users to chose a specific depth for the MLP (only chosing the width was possible).

### Mammoth for dummies

* Model: Model of continous learning
* Backbone: Network used for the training
* Dataset: Dataset used for the training for example mnist

### How to run Mammoth?  
First go to the following directory: 
```
cd ./Code/src/mammoth
```

Then run for example:
```
python utils/main.py --dataset seq-mnist --backbone mnistmlp --model lwf-mc --lr 0.01 --seed 42 --n_epochs 50 --mlp_hidden_size 100 --mlp_hidden_depth 2

```
The above use Mammoth to train a MLP model of depth 2, width 100 on the sequential MNIST dataset using the LwF-MC algorithm.  
If you want to use another model, e.g. A-GEM, just change the '--model' parameter and add the required parameters for this model (for A-GEM, the buffer size is required).

```
python utils/main.py --dataset seq-mnist --backbone mnistmlp --model agem --lr 0.01 --seed 42 --n_epochs 50 --mlp_hidden_size 100 --mlp_hidden_depth 2 --buffer_size 500
```