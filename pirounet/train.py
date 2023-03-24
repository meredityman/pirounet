"""Defines the training function for the deep generative model."""

import os
import itertools
import logging
import wandb

import torch
import torch.autograd
import torch.nn
from torch.autograd import Variable

import default_config
import evaluate.generate_f as generate_f
import models.dgm_lstm_vae as dgm_lstm_vae
import models.utils as utils

from progress_bar import ProgressBar as progbar

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = default_config.which_device

def run_train_dgm(model, labelled_data_train, labels_train, unlabelled_data_train, labelled_data_valid, labels_valid, optimizer, config, loaddir, savedir):
    """Run training and track it with wandb.

    The loss given is averaged on the number of examples (i.e. sequences) that have been seen until the current minibatch.

    Parameters
    ----------
    model :                 class,
                            Deep generative Torch model.
    labelled_data_train :   DataLoader iterator
                            Shape = [n_seq_train, seq_len, input_dim]
                            Batched sequences that have a label associated and that have been reserved for training.
    labels_train :          DataLoader iterator
                            Shape = [n_seq_train, 1]
                            Batched labels associated to labelled_data_train.
    labelled_data_valid :   DataLoader iterator
                            Shape = [n_seq_valid, seq_len, input_dim]
                            Batched sequences that have a label associated and that have been reserved for validation.
    labels_valid :          DataLoader iterator
                            Shape = [n_seq_valid, 1]
                            Batched labels associated to labelled_data_valid.
    optimizer :             class
                            Implementation of optimizer algorithm.
    config :                dict
                            Configuration for the run as inherited from wandb.config.
    """

    elbo = dgm_lstm_vae.SVI(model)
    alpha = 0.1 * len(unlabelled_data_train) / len(labelled_data_train)

    # graph_constraint = dgm_lstm_vae.graph_constraint(model)

    if config.load_from_checkpoint is not None:
        old_checkpoint_filepath = os.path.join(os.path.abspath(os.getcwd()), loaddir, config.load_from_checkpoint + ".pt")
        checkpoint = torch.load(old_checkpoint_filepath)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        latest_epoch = checkpoint["epoch"]

    else:
        latest_epoch = 0

    # filepath_for_artifacts = os.path.join(os.path.abspath(os.getcwd()), "animations/" + config.run_name)
    # os.makedirs(filepath_for_artifacts, exist_ok=True)

# !!!
    log_freq_train = 50
    log_freq_val   = 5
    train_count = max(max(len(labelled_data_train), len(labels_train)), len(unlabelled_data_train)) // log_freq_train # 92, 92, 455 or 90, 90, 431
    valid_count = max(len(labelled_data_valid), len(labels_valid)) // log_freq_val # 5, 5 or 8, 8
    epoch_count = config.epochs - latest_epoch
    count = (train_count + valid_count) * epoch_count # (9+5)*500
    print(' train ::', len(labelled_data_train), len(labels_train), len(unlabelled_data_train))
    print(' valid ::', len(labelled_data_valid), len(labels_valid))
    print(' iters ::', '(%d + %d) * %d = %d' % (train_count, valid_count, epoch_count, count))
    tmp = None
    pbar = progbar(count)

    for epoch in range(config.epochs - latest_epoch):

        # Train
        model.train()
        total_loss, accuracy = (0, 0)
        labloss, unlabloss, class_loss = (0, 0, 0)

        batches_seen = 0
        n_batches = len(unlabelled_data_train)
        n_batches_valid = len(labelled_data_valid)

        for i_batch, (x, y, u) in enumerate(zip(itertools.cycle(labelled_data_train), itertools.cycle(labels_train), unlabelled_data_train)):

            # Wrap in variables
            x, y, u = Variable(x), Variable(y), Variable(u)
            x, y = x.to(config.device), y.to(config.device)
            u = u.to(config.device)

            batches_seen += 1

            batch_one_hot = utils.batch_one_hot(y, config.label_dim)
            print(config.label_dim, y.shape, batch_one_hot.shape)
            y = batch_one_hot.to(config.device)

            if i_batch == 0:
                log_init = '%s' % str(x.detach().cpu().numpy().shape)
                # logging.info(f"Train minibatch x of shape: {x.shape}")

            L = -elbo(x, y)
            labloss += L
            U = -elbo(u)

            logits = model.classify(x)
            classification_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()
            class_loss += classification_loss

            # graph_loss = graph_constraint(u)

            J_alpha = L - alpha * classification_loss + U  # + config.beta * graph_loss

            J_alpha.backward()

            # Gradient clipping
            if config.with_clip:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

            optimizer.step()
            optimizer.zero_grad()

            total_loss += J_alpha.item()

            if (total_loss / batches_seen) > 1e20:
                logging.info(f"Loss exploded, skipping batch {batches_seen}")
                continue

            y_like_logits = y.reshape(y.shape[0], y.shape[-1])
            accuracy += torch.mean((torch.max(logits, 1).indices == torch.max(y_like_logits, 1).indices).float())

            if (i_batch+1) % log_freq_train == 0:
                log = '%s epoch %d, train %d/%d .. loss %.4g, accuracy %.4g, rec-lab-loss %.4g' % \
                        (log_init, epoch + latest_epoch, i_batch+1, n_batches, total_loss / batches_seen, accuracy / batches_seen, labloss / batches_seen)
                if tmp is not None: log += ' .. BAD all_high ' + tmp
                pbar.upd(log)
                # logging.info(f"Batch {i_batch}/{n_batches} at loss {total_loss / (batches_seen)}, accuracy {accuracy / (batches_seen)}")
                # logging.info(f"        Recon labeled-loss {labloss / (batches_seen)}")
                # logging.info(f"        Recon unlabeled-loss {unlabloss / (batches_seen)}")

        # logging.info(f"Epoch: {epoch + latest_epoch}")
        # logging.info("[Train]\t\t J_a: {:.2f}, mean accuracy on epoch: {:.2f}".format(total_loss / batches_seen, accuracy / batches_seen))

        # wandb.log({"epoch": epoch + latest_epoch, "loss": total_loss / batches_seen}, step=epoch)
        # wandb.log({"epoch": epoch + latest_epoch, "labelled_recon_loss": labloss / batches_seen}, step=epoch)
        # wandb.log({"epoch": epoch + latest_epoch, "unlabelled_recon_loss": unlabloss / batches_seen}, step=epoch)
        # wandb.log({"epoch": epoch + latest_epoch, "classification_loss": class_loss / batches_seen}, step=epoch)
        # wandb.log({"epoch": epoch + latest_epoch, "accuracy": accuracy / batches_seen}, step=epoch)

        wandb.log({"loss": total_loss / batches_seen, "labelled_recon_loss": labloss / batches_seen, 
                  "accuracy": accuracy / batches_seen, "classification_loss": class_loss / batches_seen}, step=epoch)
        # wandb.log({"unlabelled_recon_loss": unlabloss / batches_seen}, step=epoch)

        # Validation
        total_loss_valid, accuracy_valid = (0, 0)
        model.eval()

        batches_v_seen = 0
        total_v_batches = len(labelled_data_valid)

        for i_batch, (x, y) in enumerate(zip(labelled_data_valid, labels_valid)):

            x, y = Variable(x), Variable(y)
            x, y = x.to(config.device), y.to(config.device)

            batches_v_seen += 1

            batch_one_hot = utils.batch_one_hot(y, config.label_dim)
            y = batch_one_hot.to(config.device)

            L = -elbo(x, y)
            U = -elbo(x)  # MAKE IT into unlabelled data u (enumerate)

            logits_v = model.classify(x)
            classification_loss_v = torch.sum(y * torch.log(logits_v + 1e-8), dim=1).mean()

            J_alpha_v = L - alpha * classification_loss_v + U

            total_loss_valid += J_alpha_v.item()

            y_like_logits = y.reshape(y.shape[0], y.shape[-1])
            accuracy_valid += torch.mean((torch.max(logits_v, 1).indices == torch.max(y_like_logits, 1).indices).float())

            # if n_batches_valid <= 5 and i_batch != 0:
                # logging.info(f"Valid batch {i_batch}/{total_v_batches} at loss {total_loss_valid / batches_v_seen}, accuracy {accuracy_valid / batches_v_seen}")
# !!!
            # if i_batch > 0 and (n_batches_valid <= 5 or (n_batches_valid > 5 and i_batch % log_freq_val == 0)):
            if (i_batch+1) % log_freq_val == 0:
                log = '%s epoch %d, valid %d/%d .. loss %.4g, accuracy %.4g' % \
                        (log_init, epoch + latest_epoch, i_batch+1, total_v_batches, total_loss_valid / batches_v_seen, accuracy_valid / batches_v_seen)
                pbar.upd(log)

        # logging.info(f"Epoch: {epoch + latest_epoch}")
        # logging.info("[Validate]\t\t J_a: {:.2f}, mean accuracy on epoch: {:.2f}".format(total_loss_valid / batches_v_seen, accuracy_valid / batches_v_seen))

        # wandb.log({"epoch": epoch + latest_epoch, "valid_loss": total_loss_valid / batches_v_seen}, step=epoch)
        # wandb.log({"epoch": epoch + latest_epoch, "valid_accuracy": accuracy_valid / batches_v_seen}, step=epoch)

        wandb.log({"valid_loss": total_loss_valid / batches_v_seen, "valid_accuracy": accuracy_valid / batches_v_seen}, step=epoch)

        tmp = generate_f.generate_and_save(model=model, config=config, epoch = epoch + latest_epoch, results_path=savedir, num_artifacts=1, 
                                           type="cond", encoded_data=labelled_data_valid, encoded_labels=labels_valid, log_to_wandb=True)

        # logging.info("Save a checkpoint.")

        checkpoint_filepath = os.path.join(savedir, 'models')
        os.makedirs(checkpoint_filepath, exist_ok=True)
        checkpoint_filepath = os.path.join(checkpoint_filepath, 'ckpt-%s-%03d.pt' % (config.run_name, epoch + latest_epoch + 1))
        torch.save({
                "epoch": epoch + latest_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": (total_loss / batches_seen),
                "accuracy": (accuracy / batches_seen),
                "valid_loss": (total_loss_valid / batches_v_seen),
                "valid_accuracy": (accuracy_valid / batches_v_seen),
            }, checkpoint_filepath)
