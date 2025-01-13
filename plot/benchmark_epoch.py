import os
import matplotlib.pyplot as plt
import numpy as np

env = os.environ
env['PATH'] = env['PATH'] + ";\\"


def benchmark():
    cmd = 'python utils/main.py'

    # model_list = ['lwf_mc', 'agem']
    # model_list = ['lwf']
    model_list = ['agem']
    epoch_list = [10, 20]
    # lr_list = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    lr_list = [0.01, 0.001]
    # mlp_hidden_size_list = [64, 128, 256, 512]
    mlp_hidden_size_list = [5, 10, 15, 20, 50, 75, 100]

    for model in model_list:
        for epoch in epoch_list:
            for lr in lr_list:
                for sz in mlp_hidden_size_list:
                    #args = "--dataset seq-mnist --model {0} --backbone mnistmlp --n_epochs {1} --lr {2} --optimizer adam --seed 42 --mlp_hidden_size {3} --batch_size 64".format(model, epoch, lr, sz)
                    args = "--model {0} --dataset seq-mnist --backbone mnistmlp --n_epochs {1} --lr {2} --optimizer adam --seed 42 --mlp_hidden_size {3} --batch_size 64".format(model, epoch, lr, sz)
                    if model == 'agem':
                        args = args + " --buffer_size 50"
                    os.popen(cmd + ' ' + args).read()


def plot_hidden_size(il, model, lr, epoch):
    log_file = 'data/results/{0}/seq-mnist/{1}/logs.pyd'.format(il, model)
    task_types = ("task1", "task2", "task3", "task4", "task5")

    with open(log_file) as file:
        for line in file:
            metric = eval(line.replace("'", "\"").replace("device(type=\"cpu\")", "\"cpu\""))
            if metric['lr'] == lr and metric['n_epochs'] == epoch:
                accuracy = {
                    'task1': (int(metric['accuracy_1_task1']), int(metric['accuracy_1_task2']), int(metric['accuracy_1_task3']), int(metric['accuracy_1_task4']), int(metric['accuracy_1_task5'])),
                    'task2': (0,                               int(metric['accuracy_2_task2']), int(metric['accuracy_2_task3']), int(metric['accuracy_2_task4']), int(metric['accuracy_2_task5'])),
                    'task3': (0,                               0,                               int(metric['accuracy_3_task3']), int(metric['accuracy_3_task4']), int(metric['accuracy_3_task5'])),
                    'task4': (0,                               0,                               0,                               int(metric['accuracy_4_task4']), int(metric['accuracy_4_task5'])),
                    'task5': (0,                               0,                               0,                               0,                               int(metric['accuracy_5_task5']))}


                x = np.arange(len(task_types))  # the label locations
                width = 0.15  # the width of the bars
                multiplier = 0

                fig, ax = plt.subplots(layout='constrained')

                for attribute, measurement in accuracy.items():
                    offset = width * multiplier
                    rects = ax.bar(x + offset, measurement, width, label=attribute)
                    ax.bar_label(rects, padding=3)
                    multiplier += 1

                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel('accuracy (%)')
                ax.set_title('hidden size={0}: {1}, {2}, lr={3}, epochs={4}'.format(metric['mlp_hidden_size'], il, model, lr, epoch))
                ax.set_xticks(x + width, task_types)
                ax.set_ylim(0, 200)
                ax.legend()

                #plt.show()
                png_file = '{0}_{1}_{2}.png'.format(il, model, metric['mlp_hidden_size'])
                plt.savefig(png_file)

        file.close()

def plot_task(il, model, lr, epoch):
    log_file = 'data/results/{0}/seq-mnist/{1}/logs.pyd'.format(il, model)
    hidden_size = ("20", "50", "75", "100")

    accuracy = {
        'accuracy_1': (0, 0, 0, 0),
        'accuracy_2': (0, 0, 0, 0),
        'accuracy_3': (0, 0, 0, 0),
        'accuracy_4': (0, 0, 0, 0),
        'accuracy_5': (0, 0, 0, 0),
        'mean_accuracy': (0, 0, 0, 0)
    }

    with open(log_file) as file:
        for line in file:
            metric = eval(line.replace("'", "\"").replace("device(type=\"cpu\")", "\"cpu\""))
            if metric['lr'] == lr and metric['n_epochs'] == epoch:
                if metric['mlp_hidden_size'] == 20: i = 0
                elif metric['mlp_hidden_size'] == 50: i = 1
                elif metric['mlp_hidden_size'] == 75: i = 2
                elif metric['mlp_hidden_size'] == 100: i = 3
                else:continue

                l = list(accuracy['accuracy_1'])
                l[i] = int(metric['accuracy_1_task5'])
                accuracy['accuracy_1'] = tuple(l)

                l = list(accuracy['accuracy_2'])
                l[i] = int(metric['accuracy_2_task5'])
                accuracy['accuracy_2'] = tuple(l)

                l = list(accuracy['accuracy_3'])
                l[i] = int(metric['accuracy_3_task5'])
                accuracy['accuracy_3'] = tuple(l)

                l = list(accuracy['accuracy_4'])
                l[i] = int(metric['accuracy_4_task5'])
                accuracy['accuracy_4'] = tuple(l)

                l = list(accuracy['accuracy_5'])
                l[i] = int(metric['accuracy_5_task5'])
                accuracy['accuracy_5'] = tuple(l)

                l = list(accuracy['mean_accuracy'])
                l[i] = int(metric['accmean_task5'])
                accuracy['mean_accuracy'] = tuple(l)

        x = np.arange(4)  # the label locations
        width = 0.13  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for attribute, measurement in accuracy.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('accuracy (%)')
        ax.set_xlabel('hidden size')
        ax.set_title('task: {0}, {1}, lr={2}, epochs={3}'.format(il, model, lr, epoch))
        ax.set_xticks(x + width, hidden_size)
        ax.set_ylim(0, 200)
        ax.legend()

        #plt.show()
        png_file = '{0}_{1}_epoch {2}_final.png'.format(il, model, epoch)
        plt.savefig(png_file)
        file.close()


def plot():
    #plot_hidden_size('class-il', 'agem', 0.01)
    #plot_hidden_size('class-il', 'lwf_mc', 0.01)
    #plot_hidden_size('task-il', 'agem', 0.01)
    #plot_hidden_size('task-il', 'lwf_mc', 0.01)
    plot_hidden_size('class-il', 'agem', 0.001, 10)
    plot_hidden_size('class-il', 'agem', 0.001, 20)
    #plot_task('class-il', 'agem', 0.01)
    #plot_task('class-il', 'lwf_mc', 0.01)
    #plot_task('task-il', 'agem', 0.01)
    #plot_task('task-il', 'lwf_mc', 0.01)
    plot_hidden_size('task-il', 'agem', 0.001, 10)
    plot_hidden_size('task-il', 'agem', 0.001, 20)
    plot_task('class-il', 'agem', 0.001, 10)
    plot_task('class-il', 'agem', 0.001, 20)
    plot_task('task-il', 'agem', 0.001, 10)
    plot_task('task-il', 'agem', 0.001, 20)


if __name__ == "__main__":
    #benchmark()
    plot()