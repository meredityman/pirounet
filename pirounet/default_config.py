"""Default configuration parameters for PirouNet dance.
If using for training: modify lines 10 and 11 for appropriate wandb.
"""
import torch

run_name = "test5"
load_from_checkpoint = None

# # Wandb
project = "pir-2"
entity = "eps"

# Hardware
which_device = "0"
device = torch.device("cuda:" + str(which_device)) if torch.cuda.is_available() else torch.device("cpu")

# Training
epochs = 500
learning_rate = 3e-4
batch_size = 80
with_clip = False

# Input data
seq_len = 40
input_dim = 159
# label_dim = 3
label_dim = 2
amount_of_labels = 1
effort = "time"

shuffle_data = False
train_ratio = None
train_lab_frac = None
fraction_label = 0.92 # dance 0.789, watch 0.92
# shuffle_data = True
# train_ratio = 0.9
# train_lab_frac = 0.21
# fraction_label = None

# LSTM VAE architecture
kl_weight = 1
neg_slope = 0
n_layers = 5
h_dim = 100
latent_dim = 256

# Classifier architecture
h_dim_classif = 100
neg_slope_classif = 0
# neg_slope_classif = 0.4295
n_layers_classif = 2
    # Classifier separate
    # h_dim_classif = 250
    # neg_slope_classif = 0.42950429805116874  # 0.5 #0.1 # 0.05
    # n_layers_classif = 13

# Tiling hyperparameters
step_size = [0.1, 0.2, 0.4]
# height/width of tiles for labels 0, 1, 2
# dances_per_tile = [3, 3, 1]
dances_per_tile = [1, 2, 1]
# minimum dances required to be in a high density neighborhood for labels 0, 1, 2
# density_thresh = [0.8, 0.75, 0.75]
density_thresh = [0.7, 0.75, 0.75]
# minimum percentage of dances required to share the label for labels 0, 1, 2
