"""
This script is the main entry point for the Mammoth project. It contains the main function `main()` that orchestrates the training process.

The script performs the following tasks:
- Imports necessary modules and libraries.
- Sets up the necessary paths and configurations.
- Parses command-line arguments.
- Initializes the dataset, model, and other components.
- Trains the model using the `train()` function.

To run the script, execute it directly or import it as a module and call the `main()` function.
"""
# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
run_on_colab = True

# needed (don't change it)
import logging
import numpy  # noqa
import os
import sys
import time
import importlib
import socket
import datetime
import uuid
from argparse import ArgumentParser, Namespace
import torch
import csv

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')




from utils import setup_logging
import datetime
setup_logging()
def get_output_dir():
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    code_dir = script_dir.parent.parent.parent
    output_dir = os.path.join(code_dir, "Data/Output")
    return output_dir

def create_model_dir():
    current_time = datetime.datetime.now().isoformat()
    current_time = current_time.replace(":", "_").split(".")[0]
    model_output_path = os.path.join(get_output_dir(), "models")
    if not os.path.exists(model_output_path):
        os.mkdir(model_output_path)
    model_output_path = os.path.join(model_output_path, current_time)
    logging.info(f"Creating {model_output_path}")
    os.mkdir(model_output_path)
    return model_output_path
def create_eth_dir(with_seconds=True):
    if with_seconds:
        current_time = datetime.datetime.now().isoformat()
    else:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    current_time = current_time.replace(":", "_").split(".")[0]
    eth_output_path = os.path.join(mammoth_path, "data/results/ETH")
    if not os.path.exists(eth_output_path):
        os.mkdir(eth_output_path)
    eth_output_path = os.path.join(eth_output_path, current_time)
    logging.info(f"Creating {eth_output_path}")
    os.mkdir(eth_output_path)
    return eth_output_path


if __name__ == '__main__':
    logging.info(f"Running Mammoth! on {socket.gethostname()}. (if you see this message more than once, you are probably importing something wrong)")

    from utils.conf import warn_once
    try:
        if os.getenv('MAMMOTH_TEST', '0') == '0':
            from dotenv import load_dotenv
            load_dotenv()
        else:
            warn_once("Running in test mode. Ignoring .env file.")
    except ImportError:
        warn_once("Warning: python-dotenv not installed. Ignoring .env file.")


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def check_args(args, dataset=None):
    """
    Just a (non complete) stream of asserts to ensure the validity of the arguments.
    """
    assert args.label_perc_by_class == 1 or args.label_perc == 1, "Cannot use both `label_perc_by_task` and `label_perc_by_class`"

    if args.joint:
        assert args.start_from is None and args.stop_after is None, "Joint training does not support start_from and stop_after"
        assert not args.enable_other_metrics, "Joint training does not support other metrics"
        assert not args.eval_future, "Joint training does not support future evaluation (what is the future?)"

    assert 0 < args.label_perc <= 1, "label_perc must be in (0, 1]"

    if args.savecheck:
        assert not args.inference_only, "Should not save checkpoint in inference only mode"

    assert (args.noise_rate >= 0.) and (args.noise_rate <= 1.), "Noise rate must be in [0, 1]"

    if dataset is not None:
        from datasets.utils.gcl_dataset import GCLDataset, ContinualDataset

        if isinstance(dataset, GCLDataset):
            assert args.n_epochs == 1, "GCLDataset is not compatible with multiple epochs"
            assert args.enable_other_metrics == 0, "GCLDataset is not compatible with other metrics (i.e., forward/backward transfer and forgetting)"
            assert args.eval_future == 0, "GCLDataset is not compatible with future evaluation"
            assert args.noise_rate == 0, "GCLDataset is not compatible with automatic noise injection"

        assert issubclass(dataset.__class__, ContinualDataset) or issubclass(dataset.__class__, GCLDataset), "Dataset must be an instance of `ContinualDataset` or `GCLDataset`"


def load_configs(parser: ArgumentParser) -> dict:
    from models import get_model_class
    from models.utils import load_model_config

    from datasets import get_dataset_class
    from datasets.utils import get_default_args_for_dataset, load_dataset_config
    from utils.args import fix_model_parser_backwards_compatibility, get_single_arg_value

    args = parser.parse_known_args()[0]

    # load the model configuration
    # - get the model parser and fix the get_parser function for backwards compatibility
    model_parser = get_model_class(args).get_parser(parser)
    parser = fix_model_parser_backwards_compatibility(parser, model_parser)
    is_rehearsal = any([p for p in parser._actions if p.dest == 'buffer_size'])
    buffer_size = None
    if is_rehearsal:  # get buffer size
        buffer_size = get_single_arg_value(parser, 'buffer_size')
        assert buffer_size is not None, "Buffer size not found in the arguments. Please specify it with --buffer_size."
        try:
            buffer_size = int(buffer_size)  # try convert to int, check if it is a valid number
        except ValueError:
            raise ValueError(f'--buffer_size must be an integer but found {buffer_size}')

    # - get the defaults that were set with `set_defaults` in the parser
    base_config = parser._defaults.copy()

    # - get the configuration file for the model
    model_config = load_model_config(args, buffer_size=buffer_size)

    # update the dataset class with the configuration
    dataset_class = get_dataset_class(args)

    # load the dataset configuration. If the model specified a dataset config, use it. Otherwise, use the dataset configuration
    base_dataset_config = get_default_args_for_dataset(args.dataset)
    if 'dataset_config' in model_config:  # if the dataset specified a dataset config, use it
        cnf_file_dataset_config = load_dataset_config(model_config['dataset_config'], args.dataset)
    else:
        cnf_file_dataset_config = load_dataset_config(args.dataset_config, args.dataset)

    dataset_config = {**base_dataset_config, **cnf_file_dataset_config}
    dataset_config = dataset_class.set_default_from_config(dataset_config, parser)  # the updated configuration file is cleaned from the dataset-specific arguments

    # - merge the dataset and model configurations, with the model configuration taking precedence
    config = {**dataset_config, **base_config, **model_config}

    return config


def parse_args():
    """
    Parse command line arguments for the mammoth program and sets up the `args` object.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    from utils import create_if_not_exists
    from utils.conf import warn_once
    from utils.args import add_initial_args, add_management_args, add_experiment_args, add_configuration_args, clean_dynamic_args, \
        check_multiple_defined_arg_during_string_parse, add_dynamic_parsable_args, update_cli_defaults, get_single_arg_value

    from models import get_all_models

    check_multiple_defined_arg_during_string_parse()

    parser = ArgumentParser(description='Mammoth - An Extendible (General) Continual Learning Framework for Pytorch', allow_abbrev=False)

    # 1) add arguments that include model, dataset, and backbone. These define the rest of the arguments.
    #   the backbone is optional as may be set by the dataset or the model. The dataset and model are required.
    add_initial_args(parser)
    args = parser.parse_known_args()[0]
    parser.add_argument("--mlp_hidden_depth", type=int, required=False, help="Size of the input layer")
    parser.add_argument("--save_models_within_tasks", type=bool, required=False, help="Save the models at specific epochs during training")
    parser.add_argument("--save_accuracy_within_tasks", type=bool, required=False, help="Save the accuracy at specific epochs during training")

    if args.backbone is None:
        logging.warning('No backbone specified. Using default backbone (set by the dataset).')

    # 2) load the configuration arguments for the dataset and model
    add_configuration_args(parser, args)

    config = load_configs(parser)

    # 3) add the remaining arguments

    # - get the chosen backbone. The CLI argument takes precedence over the configuration file.
    backbone = args.backbone
    if backbone is None:
        if 'backbone' in config:
            backbone = config['backbone']
        else:
            backbone = get_single_arg_value(parser, 'backbone')
    assert backbone is not None, "Backbone not found in the arguments. Please specify it with --backbone or in the model or dataset configuration file."

    # - add the dynamic arguments defined by the chosen dataset and model
    add_dynamic_parsable_args(parser, args.dataset, backbone)

    # - add the main Mammoth arguments
    add_management_args(parser)
    add_experiment_args(parser)

    # 4) Once all arguments are in the parser, we can set the defaults using the loaded configuration
    update_cli_defaults(parser, config)

    # 5) parse the arguments
    if args.load_best_args:
        from utils.best_args import best_args

        warn_once("The `load_best_args` option is untested and not up to date.")

        is_rehearsal = any([p for p in parser._actions if p.dest == 'buffer_size'])  # check if model has a buffer

        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if is_rehearsal:
            best = best[args.buffer_size]
        else:
            best = best[-1]

        to_parse = sys.argv[1:] + ['--' + k + '=' + str(v) for k, v in best.items()]
        to_parse.remove('--load_best_args')
        args = parser.parse_args(to_parse)
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
    else:
        args = parser.parse_args()

    # 6) clean dynamically loaded args
    args = clean_dynamic_args(args)

    # 7) final checks and updates to the arguments
    models_dict = get_all_models()
    args.model = models_dict[args.model]

    if args.lr_scheduler is not None:
        logging.info('`lr_scheduler` set to {}, overrides default from dataset.'.format(args.lr_scheduler))

    if args.seed is not None:
        from utils.conf import set_random_seed

        set_random_seed(args.seed)

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()


    # Add the current git commit hash to the arguments if available
    try:
        import git
        repo = git.Repo(path=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args.conf_git_hash = repo.head.object.hexsha
    except Exception:
        logging.error("Could not retrieve git hash.")
        args.conf_git_hash = None

    if args.savecheck:
        if not os.path.isdir('checkpoints'):
            create_if_not_exists("checkpoints")

        now = time.strftime("%Y%m%d-%H%M%S")
        uid = args.conf_jobnum.split('-')[0]
        extra_ckpt_name = "" if args.ckpt_name is None else f"{args.ckpt_name}_"
        args.ckpt_name = f"{extra_ckpt_name}{args.model}_{args.dataset}_{args.dataset_config}_{args.buffer_size if hasattr(args, 'buffer_size') else 0}_{args.n_epochs}_{str(now)}_{uid}"
        print("Saving checkpoint into", args.ckpt_name, file=sys.stderr)

    check_args(args)

    if args.validation is not None:
        logging.info(f"Using {args.validation}% of the training set as validation set.")
        logging.info(f"Validation will be computed with mode `{args.validation_mode}`.")

    return args


def extend_args(args, dataset):
    """
    Extend the command-line arguments with the default values from the dataset and the model.
    """
    from datasets import ContinualDataset
    dataset: ContinualDataset = dataset  # noqa, used for type hinting

    if hasattr(args, 'num_classes') and args.num_classes is None:
        args.num_classes = dataset.N_CLASSES

    if args.fitting_mode == 'epochs' and args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    elif args.fitting_mode == 'iters' and args.n_iters is None and isinstance(dataset, ContinualDataset):
        args.n_iters = dataset.get_iters()

    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
        if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and (not hasattr(args, 'minibatch_size') or args.minibatch_size is None):
            args.minibatch_size = dataset.get_minibatch_size()
    else:
        args.minibatch_size = args.batch_size

    if args.validation:
        if args.validation_mode == 'current':
            assert dataset.SETTING in ['class-il', 'task-il'], "`current` validation modes is only supported for class-il and task-il settings (requires a task division)."

    if args.debug_mode:
        print('Debug mode enabled: running only a few forward steps per epoch with W&B disabled.')
        # set logging level to debug
        args.nowand = 1

    if args.wandb_entity is None:
        args.wandb_entity = os.getenv('WANDB_ENTITY', None)
    if args.wandb_project is None:
        args.wandb_project = os.getenv('WANDB_PROJECT', None)

    if args.wandb_entity is None or args.wandb_project is None:
        logging.info('`wandb_entity` and `wandb_project` not set. Disabling wandb.')
        args.nowand = 1
    else:
        print('Logging to wandb: {}/{}'.format(args.wandb_entity, args.wandb_project))
        args.nowand = 0


def main(args=None):
    from utils.conf import base_path, get_device
    from models import get_model
    from datasets import get_dataset
    from utils.training import train
    from models.utils.future_model import FutureModel
    from backbone import get_backbone

    lecun_fix()
    if args is None:
        args = parse_args()

    device = get_device(avail_devices=args.device)
    args.device = device

    # set base path
    base_path(args.base_path)

    if args.code_optimization != 0:
        torch.set_float32_matmul_precision('high' if args.code_optimization == 1 else 'medium')
        logging.info(f"Code_optimization is set to {args.code_optimization}")
        logging.info(f"Using {torch.get_float32_matmul_precision()} precision for matmul.")

        if args.code_optimization == 2:
            if not torch.cuda.is_bf16_supported():
                raise NotImplementedError('BF16 is not supported on this machine.')

    dataset = get_dataset(args)

    extend_args(args, dataset)

    check_args(args, dataset=dataset)

    backbone = get_backbone(args)
    logging.info(f"Using backbone: {args.backbone}")

    if args.code_optimization == 3:
        # check if the model is compatible with torch.compile
        # from https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
        if torch.cuda.get_device_capability()[0] >= 7 and os.name != 'nt':
            print("================ Compiling model with torch.compile ================")
            logging.warning("`torch.compile` may break your code if you change the model after the first run!")
            print("This includes adding classifiers for new tasks, changing the backbone, etc.")
            print("ALSO: some models CHANGE the backbone during initialization. Remember to call `torch.compile` again after that.")
            print("====================================================================")
            backbone = torch.compile(backbone)
        else:
            if torch.cuda.get_device_capability()[0] < 7:
                raise NotImplementedError('torch.compile is not supported on this machine.')
            else:
                raise Exception(f"torch.compile is not supported on Windows. Check https://github.com/pytorch/pytorch/issues/90768 for updates.")

    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform(), dataset=dataset)
    assert isinstance(model, FutureModel) or not args.eval_future, "Model does not support future_forward."

    if args.distributed == 'dp':
        from utils.distributed import make_dp

        if args.batch_size < torch.cuda.device_count():
            raise Exception(f"Batch too small for DataParallel (Need at least {torch.cuda.device_count()}).")

        model.net = make_dp(model.net)
        model.to('cuda:0')
        args.conf_ngpus = torch.cuda.device_count()
    elif args.distributed == 'ddp':
        # DDP breaks the buffer, it has to be synchronized.
        raise NotImplementedError('Distributed Data Parallel not supported yet.')

    try:
        import setproctitle
        # set job name
        setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))
    except Exception:
        pass


    if args.save_models_within_tasks or args.save_accuracy_within_tasks:
        eth_output_path = create_eth_dir(with_seconds=False)
    else:
        eth_output_path = create_model_dir()


    training_settings = open(os.path.join(eth_output_path,"training_settings.txt"),"w")
    # training_settings.write(sys.argv[:])
    for i in sys.argv[1:]:
        training_settings.write(i)
    training_settings.close()

    if args.save_models_within_tasks:
        train(model, dataset, args,eth_output_path, save_within_tasks=True)
    else:
        train(model, dataset, args,eth_output_path, save_within_tasks=False)

    def append_to_csv(file_path, data, headers=None):
        # Check if the file exists
        file_exists = os.path.exists(file_path)

        # Open the file in append mode
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Write headers only if the file does not exist and headers are provided
            if not file_exists and headers:
                writer.writerow(headers)

            # Write the data
            writer.writerow(data)

    # Example usage
    results_path =  os.path.join(mammoth_path, "data/results/ETH")




    file_path = os.path.join(get_output_dir(),f"results-{args.model}.csv")


    import ast

    logs_path = ""
    if args.model == "agem":
        logs_path = os.path.join(mammoth_path, "data/results/class-il/seq-mnist/agem/logs.pyd")
    elif args.model == "lwf_mc":
        logs_path = os.path.join(mammoth_path, "data/results/class-il/seq-mnist/lwf_mc/logs.pyd")

    f = open(logs_path, "r")
    results = f.readlines()
    last_results = results[-1]
    last_results = last_results.replace("np.float64(", "").replace(")", "").replace("device(type=", "").replace("index=0",'"index":0')

    # Convert the string to a dictionary
    last_results = ast.literal_eval(last_results)

    start_date = os.path.split(eth_output_path)[-1]
    headers = ["Date","Seed","Epochs","Learning Rate" ,"Depth", "Width","Mean","Mean_array","Forgetting"]
    mean_array = [last_results[f"accuracy_{x+1}_task5"] for x in range(5)]
    data = [start_date,args.seed,args.n_epochs,args.lr, args.mlp_hidden_depth, args.mlp_hidden_size,last_results[f'accmean_task{5}'],mean_array,last_results['forgetting']]

    append_to_csv(file_path, data, headers=headers)


if __name__ == '__main__':
    main()



