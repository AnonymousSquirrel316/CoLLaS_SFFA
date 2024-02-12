from datetime import datetime

from torch.utils.tensorboard.writer import SummaryWriter

from ff_mod.trainer import Trainer
from ff_mod.utils import create_network, save_experiment_info
from ff_mod.callbacks.accuracy_writer import AccuracyWriter
from ff_mod.callbacks.best_model_saver import BestModelSaver

for dataset in ["mnist", "fashion_mnist", "kmnist", "emnist"]:
    args = {
        "neurons_per_layer": 1400,
        "pattern_size": 100,
        "num_vectors": 1,
        "p": 0.1,
        "loss": "ProbabilityBCELoss",
        "threshold": 2,
        "alpha": 1,
        "beta": 1,
        "negative_threshold": 2,
        "num_steps": 20,
        "internal_epoch": 1,
        "input_size": 784 if dataset != "cifar10" else 3072,
        "num_layers": 1,
        "bounded_goodness": False,
        "lr": 0.0001,
        "greedy_goodness": False,
        "batch_size": 512,
        "epochs": 100,
        "device": "cuda:0",
        "use_snn": False,
        "network_type": "double",
        "dataset": dataset,
        "dataset_resize": 28 if dataset != "cifar10" else 32,
        "dataset_subset": None,
        "residual_connections" : True,
        "normalize_activity": True,
    }

    EXPERIMENTAL_FOLDER = "experiments/train/"

    CURRENT_TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

    save_experiment_info(args, EXPERIMENTAL_FOLDER, CURRENT_TIMESTAMP)
    network = create_network(args)
    trainer = Trainer(device = 'cuda:0', greedy_goodness = args["greedy_goodness"])
    trainer.load_data_loaders(args["dataset"], batch_size = args["batch_size"], test_batch_size = args["batch_size"], resize=(args["dataset_resize"], args["dataset_resize"]), subset=args["dataset_subset"])
    trainer.set_network(network)

    experiment_name = f"{args['dataset']}_{'SNN' if args['use_snn'] else 'ANN'}_({args['neurons_per_layer']})_{CURRENT_TIMESTAMP}/"
            
    writer = SummaryWriter(f"{EXPERIMENTAL_FOLDER}/{experiment_name}/summary/" )
    network.layers[0].loss_function.writer = writer
    
    trainer.add_callback(AccuracyWriter(tensorboard=writer))
    trainer.add_callback(BestModelSaver(f"{EXPERIMENTAL_FOLDER}/{experiment_name}/", network))
    
    trainer.train(epochs=args["epochs"], verbose=1)
    network.save_network(f"{EXPERIMENTAL_FOLDER}/{experiment_name}/final_model")
