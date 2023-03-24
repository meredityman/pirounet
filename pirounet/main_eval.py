"""Run this file to obtain all qualitative and quantitative metrics from a run.
Specify desired metrics in eval_config.py.
"""
import os
import logging
import numpy as np
import argparse

import torch

import classifier_config
import datasets
import default_config
import eval_config
import evaluate.confusion_plot as confusion_plot
import evaluate.generate_f as generate_f
import evaluate.metrics as metrics
import models.classifiers as classifiers
import models.dgm_lstm_vae as dgm_lstm_vae

logging.basicConfig(level=logging.INFO)

# !!!
parser = argparse.ArgumentParser()
parser.add_argument('-m',  '--model',    default=None)
parser.add_argument('-ld', '--loaddir', default='models')
parser.add_argument('-o', '--outdir', default='_out')
# from main.py
parser.add_argument('-lr', '--loadraw', default='data/mariel_*.npy')
parser.add_argument('-ls', '--labelpath', default='data/labels_from_app.csv')
a = parser.parse_args()

default_config.load_from_checkpoint = eval_config.load_from_checkpoint if a.model is None else a.model 
for k in vars(a):
    setattr(default_config, k, vars(a)[k])

# out_dir = os.path.join(os.path.abspath(os.getcwd()), "_out", eval_config.run_name)
out_dir = os.path.join(a.outdir, default_config.load_from_checkpoint)

def main():
    quanti_recon_dir = os.path.join(out_dir, "quanti_gen_recon_metrics")
    os.makedirs(quanti_recon_dir, exist_ok=True)

    logging.info("Load classifier.")
    fid_classifier_model = classifiers.LinearClassifier(
        input_dim=classifier_config.input_dim,
        h_dim=classifier_config.h_dim_class,
        label_dim=classifier_config.label_dim,
        seq_len=classifier_config.seq_len,
        neg_slope=classifier_config.neg_slope_classif,
        n_layers=classifier_config.n_layers_class,
    ).to(eval_config.device)

    classif_checkpoint_filepath = os.path.join(os.path.abspath(os.getcwd()), a.loaddir, classifier_config.load_from_checkpoint + ".pt")
    classif_checkpoint = torch.load(classif_checkpoint_filepath)
    fid_classifier_model.load_state_dict(classif_checkpoint["model_state_dict"])

    logging.info("Load model.")
    model = dgm_lstm_vae.DeepGenerativeModel(
        n_layers=default_config.n_layers,
        input_dim=default_config.input_dim,
        h_dim=default_config.h_dim,
        latent_dim=default_config.latent_dim,
        output_dim=default_config.input_dim,
        seq_len=default_config.seq_len,
        neg_slope=default_config.neg_slope,
        label_dim=default_config.label_dim,
        batch_size=default_config.batch_size,
        h_dim_classif=default_config.h_dim_classif,
        neg_slope_classif=default_config.neg_slope_classif,
        n_layers_classif=default_config.n_layers_classif,
    ).to(eval_config.device)

    dgm_checkpoint_filepath = os.path.join(os.path.abspath(os.getcwd()), a.loaddir, default_config.load_from_checkpoint + ".pt")
    checkpoint = torch.load(dgm_checkpoint_filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    latest_epoch = checkpoint["epoch"]

    logging.info("Load data.")
    if default_config.train_ratio and default_config.train_lab_frac is not None:
        labelled_data_train, labels_train, unlabelled_data_train, labelled_data_valid, labels_valid, labelled_data_test, labels_test, unlabelled_data_test \
            = datasets.get_model_data(default_config) # labels_train_true ??
    if default_config.fraction_label is not None:
        labelled_data_train, labels_train, unlabelled_data_train, labelled_data_valid, labels_valid, labelled_data_test, labels_test, unlabelled_data_test \
            = datasets.get_model_specific_data(default_config)

    if eval_config.quali_gen_metrics:
        logging.info("Qualitative generation metrics :")

        curpath = os.path.join(out_dir, "quali_gen_metrics/random_gen")
        os.makedirs(curpath, exist_ok=True)
        generate_f.generate_and_save(model=model, config=eval_config, epoch=latest_epoch, results_path=curpath, 
            num_artifacts=eval_config.num_random_artifacts,
            type="random",
            comic=False,
            npy_output=eval_config.npy_output,
        )
        logging.info("       - random: done.")
        
        curpath = os.path.join(out_dir, "quali_gen_metrics/cond_gen")
        os.makedirs(curpath, exist_ok=True)
        generate_f.generate_and_save(model=model, config=eval_config, epoch=latest_epoch, results_path=curpath, 
            num_artifacts=eval_config.num_cond_artifacts_per_lab,
            type="cond",
            encoded_data=labelled_data_train,
            encoded_labels=labels_train,
            comic=True,
            npy_output=eval_config.npy_output,
        )
        logging.info("       - conditional: done.")

    if eval_config.quali_recon_metrics:
        logging.info("Qualitative reconstruction metrics :")
        curpath = os.path.join(out_dir, "quali_recon_metrics")
        os.makedirs(curpath, exist_ok=True)
        generate_f.reconstruct(model, config=eval_config, epoch=latest_epoch, results_path=curpath, 
            input_data=labelled_data_valid,
            input_label=labels_valid,
            purpose="valid",
            comic=False,
        )
        logging.info("       - on valid data: done.")

    if eval_config.test_entanglement:
        logging.info("Testing latent space entanglement :")
        curpath = os.path.join(out_dir, "test_entanglement")
        os.makedirs(curpath, exist_ok=True)
        generate_f.generate_and_save_one_move(model, config=eval_config, path=curpath)
        logging.info("       - variations on one-move: done.")
        generate_f.plot_dist_one_move(model, config=eval_config, path=curpath, n_one_moves=30)
        logging.info("       - plot for one-moves: done.")

    if eval_config.generate_for_blind_labeling:
        logging.info(" Generating sequences for blind human labeling :")
        os.makedirs(os.path.join(out_dir, "to_be_labeled"), exist_ok=True)
        gen_dance, gen_label = generate_f.generate_cond(model, config=eval_config,
            num_gen_cond_lab=10,
            encoded_data=labelled_data_train,
            encoded_labels=labels_train,
            shuffle=True,
        )
        path_blind_gen = os.path.join(out_dir, "to_be_labeled")
        os.makedirs(path_blind_gen, exist_ok=True)
        np.save(f"{path_blind_gen}/dance_{default_config.load_from_checkpoint}", gen_dance)
        logging.info("       - npy dance file: done.")
        np.save(f"{path_blind_gen}/labels_{default_config.load_from_checkpoint}", gen_label)
        logging.info("       - npy label file: done.")

    if eval_config.plot_recognition_accuracy:
        logging.info("Making confusion matrix for recognition accuracy :")
        confusion_plot.plot_recognition_accuracy(
            human_labels=eval_config.human_labels,
            pirounet_labels=eval_config.pirounet_labels,
            path=quanti_recon_dir,
        )
        logging.info("done.")

    if eval_config.plot_classification_accuracy:
        logging.info("Making confusion matrix for classification accuracy :")
        confusion_plot.plot_classification_accuracy(model, eval_config, labelled_data_valid, labels_valid, purpose="valid", path=quanti_recon_dir)
        logging.info("       - on valid data: done.")
        confusion_plot.plot_classification_accuracy(model, eval_config, labelled_data_test, labels_test, purpose="test", path=quanti_recon_dir)
        logging.info("       - on test data: done.")

    if eval_config.plot_latent_space:
        logging.info("Plotting the latent space :")
        os.makedirs(os.path.join(out_dir, "latent_space"), exist_ok=True)
        generate_f.plot_latentspace(model, eval_config, labelled_data_train, labels_train, path=out_dir + "/latent_space")
        logging.info("done.")

    if eval_config.quanti_gen_recon_metrics:
        logging.info("Quantitative metrics on generation and reconstruction :")
        # BOOTSTRAP LOOP TO TAKE DISTRIBUTION ON STATS
        all_gen_diversity = []
        all_gen_multimodality = []
        all_gt_diversity = []
        all_gt_multimodality = []
        all_fid = []
        for i in range(eval_config.stat_sampling_size):
            # MAKE DANCE
            gen_dance, gen_labels = generate_f.generate_cond(model, eval_config,
                eval_config.num_gen_cond_lab,
                encoded_data=labelled_data_train,
                encoded_labels=labels_train,
                shuffle=True,
            )

            # LABELS
            # gen_labels = gen_labels.to(eval_config.device)
            gt_labels = np.squeeze(labels_train.dataset).astype("int")

            # ACTIVATIONS
            _, gen_activations = fid_classifier_model.forward(torch.tensor(gen_dance).to(eval_config.device))
            _, gt_activations  = fid_classifier_model.forward(torch.tensor(labelled_data_train.dataset).to(eval_config.device))

            # STATISTICS
            gen_statistics = metrics.calculate_activation_statistics(gen_activations.cpu().detach().numpy())
            gt_statistics  = metrics.calculate_activation_statistics(gt_activations.cpu().detach().numpy())

            # FID
            fid = metrics.calculate_frechet_distance(
                gt_statistics[0],
                gt_statistics[1],
                gen_statistics[0],
                gen_statistics[1],
            )

            # GEN AND MULTIMOD
            gen_diversity, gen_multimodality = metrics.calculate_diversity_multimodality(gen_activations, torch.tensor(gen_labels).cpu().detach().numpy(), 
                                                                                         eval_config.label_dim)
            gt_diversity,  gt_multimodality  = metrics.calculate_diversity_multimodality(gt_activations,  torch.tensor(gt_labels).cpu().detach().numpy(),  
                                                                                         eval_config.label_dim)

            all_gen_diversity.append(gen_diversity.cpu().data.numpy())
            all_gen_multimodality.append(gen_multimodality.cpu().data.numpy())
            all_gt_diversity.append(gt_diversity.cpu().data.numpy())
            all_gt_multimodality.append(gt_multimodality.cpu().data.numpy())
            all_fid.append(fid)

        logging.info("       - bootstrap iterations: done.")
        gen_diversity     = np.mean(np.array(all_gen_diversity))
        gen_multimodality = np.mean(np.array(all_gen_multimodality))
        gt_diversity      = np.mean(np.array(all_gt_diversity))
        gt_multimodality  = np.mean(np.array(all_gt_multimodality))
        fid               = np.mean(np.array(all_fid))

        siggen_diversity     = np.sqrt(np.var(np.array(all_gen_diversity)))
        siggen_multimodality = np.sqrt(np.var(np.array(all_gen_multimodality)))
        siggt_diversity      = np.sqrt(np.var(np.array(all_gt_diversity)))
        siggt_multimodality  = np.sqrt(np.var(np.array(all_gt_multimodality)))
        sigfid               = np.sqrt(np.var(np.array(all_fid)))

        # AJD
        ajd_valid = metrics.ajd(     model, eval_config.device, labelled_data_valid, labels_valid, eval_config.label_dim)
        ajd_train = metrics.ajd(     model, eval_config.device, labelled_data_train, labels_train, eval_config.label_dim)
        ajd_test  = metrics.ajd_test(model, eval_config.device, labelled_data_test,  labels_test,  eval_config.label_dim)
        logging.info("       - avg joint distance: done.")
        # ACCURACY
        acc_valid = metrics.calc_accuracy(model, eval_config.device, labelled_data_valid, labels_valid)
        acc_test  = metrics.calc_accuracy(model, eval_config.device, labelled_data_test,  labels_test)
        logging.info("       - accuracy: done.")
        ####################################################
        # Log everything
        evaluate_file = os.path.join(quanti_recon_dir, f"logfile_{default_config.load_from_checkpoint}.txt")
        with open(evaluate_file, "w") as f:
            f.write(f"========== Metrics for checkpoint {default_config.load_from_checkpoint} ========== \n")
            f.write(f"Classif Accuracy (Valid): {acc_valid} \n")
            f.write(f"Classif Accuracy (Test): {acc_test} \n")
            f.write(f"------------------------------ \n")
            f.write(f"Stats \n")
            f.write(f"FID: {fid} +/- {sigfid} \n")
            f.write(f"Diversity: {gt_diversity} +/- {siggt_diversity}\n")
            f.write(f"Multimodality: {gt_multimodality} +/- {siggt_multimodality} \n")
            f.write(f"------------------------------ \n")
            f.write(f"Gen Diversity: {gen_diversity} +/- {siggen_diversity} \n")
            f.write(f"Gen Multimodality: {gen_multimodality} +/- {siggen_multimodality} \n")
            f.write(f"------------------------------ \n")
            f.write(f"AJD valid (recon loss): {ajd_valid}\n")
            f.write(f"AJD train (recon loss): {ajd_train}\n")
            f.write(f"AJD test (recon loss): {ajd_test}")
            f.write(f"------------------------------ \n")
        logging.info("       - txt log file: done.")


if __name__ == '__main__':
    main()
