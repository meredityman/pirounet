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
which_device = "1"
run_name = "FIXEDvalid"
label_features = 4
amount_of_labels = 1

batch_size = 20
learning_rate = 3e-6
epochs = 400
seq_len = 40
negative_slope = 0  # LeakyRelu
kl_weight = 0
n_layers = 4
h_features_loop = 384
latent_dim = 256
