"""Main file performing training with labels (semi-supervised)."""

import os
import warnings
import logging
import argparse
import wandb

import torch

import datasets
import default_config
import evaluate.generate_f as generate_f
import models.dgm_lstm_vae as dgm_lstm_vae
import train

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = default_config.which_device

parser = argparse.ArgumentParser()
# parser.add_argument('-d',  '--data',    default='data')
parser.add_argument('-ld', '--loaddir', default='models')
parser.add_argument('-td', '--traindir', default='train')
parser.add_argument('-lr', '--loadraw', default='data/mariel_*.npy')
parser.add_argument('-ls', '--labelpath', default='data/labels_from_app.csv')
a = parser.parse_args()

logging.info(f"Using PyTorch version: {torch. __version__}")
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

def main():
    config = {}
    for name in dir(default_config):
        if not (name.startswith('__') or name.endswith('__') or name=='torch'):
            config[name] = getattr(default_config, name)

    a.traindir = os.path.join(a.traindir, config['run_name'])
    os.makedirs(a.traindir, exist_ok=True)

    for k in vars(a):
        config[k] = vars(a)[k]

    wandb.init(project=default_config.project, entity=default_config.entity, config=config, dir=a.traindir)
    config = wandb.config
    # wandb.run.name = default_config.run_name

    logging.info(f"Config: {config}")
    logging.info(f"---> Using device {config.device}")
    logging.info("Initialize model")

    model = dgm_lstm_vae.DeepGenerativeModel(
        n_layers = config.n_layers,
        input_dim = config.input_dim,
        h_dim = config.h_dim,
        latent_dim = config.latent_dim,
        output_dim = config.input_dim,
        seq_len = config.seq_len,
        neg_slope = config.neg_slope,
        label_dim = config.label_dim,
        batch_size = config.batch_size,
        h_dim_classif = config.h_dim_classif,
        neg_slope_classif = config.neg_slope_classif,
        n_layers_classif = config.n_layers_classif,
    ).to(config.device)

    logging.info("Get data")
    if config.train_ratio and config.train_lab_frac is not None:
        labelled_data_train, labels_train, unlabelled_data_train, labelled_data_val, labels_val, _, _, _ = datasets.get_model_data(config)
        # labelled_data_test, labels_test, unlabelled_data_test = not used
    if config.fraction_label is not None:
        labelled_data_train, labels_train, unlabelled_data_train, labelled_data_val, labels_val, _, _, _ = datasets.get_model_specific_data(config)
        # labelled_data_test, labels_test, unlabelled_data_test = not used

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))

    logging.info("Train")
    train.run_train_dgm(model, labelled_data_train, labels_train, unlabelled_data_train, labelled_data_val, labels_val, optimizer, config=config,
                        loaddir=a.loaddir, savedir=a.traindir)
    wandb.finish()

if __name__ == '__main__':
    main()
