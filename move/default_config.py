"""Default configuration parameters.

From Pettee 2019, Beyond Imitation.
The final architecture for the sequence VAE also
comprises an encoder and a decoder, each with
- 3 LSTM layers with 384 nodes and
- 1 dense layer with 256 nodes and
- a ReLU activation function,
where 256 represents the dimensionality of the latent space.

The model was compiled with the Adam optimizer.

The VAE maps:

- inputs of shape (53 x 3 x l), where l is the fixed length of the movement sequence,
- to the (256 x l)-dimensional latent space
- and then back to their original dimensionality.

We used input sequences of length l = 128, which corresponds
to about 4 seconds of continuous movement.
"""
import torch

run_name = 'semi_sup_study10' # "hdim100_hclass100_batch40_lr3e4"
load_from_checkpoint = (
    "checkpoint_15_perc_labelled_epoch489"
)

# ablation study
# checkpoint_hdim100_10_epoch183
# checkpoint_hdim100_10_epoch305
# checkpoint_ablation_100_80_epoch262
# checkpoint_ablation_100_80_epoch313
# checkpoint_15_perc_labelled_epoch233
# checkpoint_15_perc_labelled_epoch489
# checkpoint_ablation_sweep_epoch289
# checkpoint_ablation_sweep_epoch376
# checkpoint_hdim100_150_epoch363
# checkpoint_hdim100_150_epoch492

# for semi-supervised study
# checkpoint_perc_labelled_sweep_epoch85
# checkpoint_perc_labelled_3_epoch483
# checkpoint_15_perc_labelled_epoch30
# checkpoint_smaller_lstm_epoch108
# EXCLUDED checkpoint_10_perc_labelled_epoch599

# models in paper
# checkpoint_smaller_lstm_contd_epoch144
# checkpoint_15_perc_labelled_epoch489


# Hardware
which_device = "0" #CHANGE BACK TO 1 FOR TRAINING (0 for metrics)
device = (
    torch.device("cuda:"+str(which_device)) if torch.cuda.is_available() else torch.device("cpu")
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
effort = 'time'
fraction_label = 0.789

# LSTM VAE
kl_weight = 1
neg_slope = 0 
n_layers = 5 
h_dim = 100
latent_dim = 256

# Classifier
h_dim_classif = 100
neg_slope_classif = 0 
n_layers_classif = 2

# Artifacts to produce
generate_after_training = False