import os
import matplotlib.pyplot as plt
import numpy as np

env = os.environ
env['PATH'] = env['PATH'] + ";\\"

def benchmark():
    cmd = 'python utils/main.py'

    model_list = ['agem']
    #model_list = ['lwf_mc']
    epoch_list = [10, 20]
    lr_list = [0.01, 0.001]
    #mlp_hidden_size_list = [5, 10, 15, 20, 50, 75, 100]
    mlp_hidden_size_list = [64, 128, 256, 512]
    for model in model_list:
        for epoch in epoch_list:
            for lr in lr_list:
                for sz in mlp_hidden_size_list:
                    args = "--model {0} --dataset seq_mnist --backbone mnistmlp --n_epochs {1} --lr {2} --optimizer adam --seed 42 --mlp_hidden_size {3} --batch_size 64".format(model, epoch, lr, sz)
                    if model == 'agem':
                        args = args + " --buffer_size 50"
                    os.popen(cmd + ' ' + args).read()

if __name__ == "__main__":
    benchmark()