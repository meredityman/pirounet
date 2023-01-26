"""Default configuration parameters for PirouNet dance.
If using for training: modify lines 10 and 11 for appropriate wandb.
"""
import torch

run_name = "PirouNet_dance"
load_from_checkpoint = None

# # Wandb
project = "pirounet"
entity = "meredityman"


# Hardware
which_device = "0"
device = (
    torch.device("cuda:" + str(which_device))
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# Training
epochs = 500
learning_rate = 3e-4
batch_size = 80
with_clip = False

# Input data
seq_len = 40
input_dim = 159
label_dim = 3
amount_of_labels = 1
effort = "time"
fraction_label = 0.789
shuffle_data = False
train_ratio = None
train_lab_frac = None

# LSTM VAE architecture
kl_weight = 1
neg_slope = 0
n_layers = 5
h_dim = 100
latent_dim = 256

# Classifier architecture
h_dim_classif = 100
neg_slope_classif = 0
n_layers_classif = 2


# """Default configuration parameters for PirouNet dance.
# If using for training: modify lines 10 and 11 for appropriate wandb.
# """
# import torch

# run_name = "check_data_sizes"
# load_from_checkpoint = None  # "checkpoint_pirounet_dance"

# # Wandb
# project = "pirounet"
# entity = "bioshape-lab"

# # Hardware
# which_device = "1"
# device = (
#     torch.device("cuda:" + str(which_device))
#     if torch.cuda.is_available()
#     else torch.device("cpu")
# )

# # Training
# epochs = 500
# learning_rate = 3e-4
# batch_size = 80
# with_clip = False

# # Input data
# seq_len = 40
# input_dim = 159
# label_dim = 3
# amount_of_labels = 1
# effort = "time"
# shuffle_data = True
# train_ratio = 0.9
# train_lab_frac = 0.21
# fraction_label = None

# # LSTM VAE architecture
# kl_weight = 1
# neg_slope = 0
# n_layers = 5
# h_dim = 100
# latent_dim = 256

# # Classifier architecture
# h_dim_classif = 100
# neg_slope_classif = 0
# n_layers_classif = 2
