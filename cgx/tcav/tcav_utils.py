# imports

import numpy as np
import os#, glob

import matplotlib.pyplot as plt

#from PIL import Image

import scipy.stats as stats
from scipy.stats import ttest_ind

# ..........torch imports............
import torch
#import torchvision

from torch.utils.data import IterableDataset, DataLoader
#from torchvision import transforms

#.... Captum imports..................
from captum.attr import LayerGradientXActivation, LayerIntegratedGradients

from captum.concept import TCAV
from captum.concept import Concept

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str

# ..........new imports............
import glob

from ..concepts.concept_generation import get_concept_files

import h5py
import tqdm

from matplotlib.container import BarContainer
from scipy.stats import ttest_ind, mannwhitneyu
import pandas as pd

from sklearn.linear_model import SGDClassifier
from captum.concept import Classifier
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import itertools
from matplotlib.patches import Patch


# functions

def hdf5_paths(data_path):
    """Sorted .hdf5 files in a preprocessed dataset folder."""
    return sorted(glob.glob(os.path.join(data_path, '*.hdf5')))

class TensorIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __iter__(self):
        for idx in range(len(self.data_tensor)):
            yield self.data_tensor[idx]

def make_input_tensor(data_path, class_of_interest=1):
    data_tensor = []
    for hf_path in hdf5_paths(data_path):
        with h5py.File(hf_path, 'r') as f:
            files = [s.decode('utf-8') if isinstance(s, bytes) else str(s)
                    for s in f.attrs['files']]
            labels = f['labels'][:]  # shape (n_samples,)
            
            # Find the indices of the desired files
            desired_indices = [idx for idx, label in enumerate(labels) if label == class_of_interest]
            # Gather the data for the desired files
            desired_data = f['data'][desired_indices]
            hf_tensor = torch.tensor(desired_data)
            data_tensor.append(hf_tensor)
    data_tensor = torch.cat(data_tensor, dim=0)
    return data_tensor


def get_dataset_from_examples(data_path, examples, device, t_starts=None):
    data_tensor = []

    if t_starts is not None:
        assert len(examples) == len(t_starts), "examples and t_starts must have the same length"
        start_pairs = set(zip(examples, t_starts))
    else:
        valid_filenames = set(examples)

    for hf_path in hdf5_paths(data_path):
        with h5py.File(hf_path, 'r') as f:
            files = [s.decode('utf-8') if isinstance(s, bytes) else str(s)
                     for s in f.attrs['files']]
            all_file_idxs = f['file_idxs'][:]
            all_t_starts = f['time_slices'][:][:, 0]  # Assuming this dataset exists

            matched_indices = []
            seen_pairs = set()

            for i, file_idx in enumerate(all_file_idxs):
                filename = files[file_idx]
                t_start = all_t_starts[i]
                pair = (filename, t_start)

                if t_starts is not None:
                    if pair in start_pairs and pair not in seen_pairs:
                        matched_indices.append(i)
                        seen_pairs.add(pair)
                else:
                    if filename in valid_filenames and filename not in seen_pairs:
                        matched_indices.append(i)
                        seen_pairs.add(filename)

            if matched_indices:
                desired_data = f['data'][matched_indices]
                hf_tensor = torch.tensor(desired_data)
                data_tensor.append(hf_tensor)

    if not data_tensor:
        raise ValueError("No matching data found for the provided examples.")

    data_tensor = torch.cat(data_tensor, dim=0).to(device)
    dataset = TensorIterableDataset(data_tensor)

    return dataset


def assemble_concept(name, concepts_path, data_path, device, events=False):
    examples, id, t_starts = get_concept_files(concepts_path, name, return_type='name', events=events)
    dataset = get_dataset_from_examples(data_path, examples, device, t_starts=t_starts)
    concept_iter = dataset_to_dataloader(dataset)
    return Concept(id=id, name=name, data_iter=concept_iter)

def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def assemble_scores(scores, experimental_sets, idx, score_layer, score_type):
    score_list = []
    for concepts in experimental_sets:
        key = "-".join([str(c.id) for c in concepts])
        score_tensor = scores[key][score_layer][score_type][idx]
        score_list.append(score_tensor.detach().cpu().item())  # convert to Python float
    return score_list

def assemble_stats(stats, experimental_sets, score_layer):
    stat_list = []
    for concepts in experimental_sets:
        key = "-".join([str(c.id) for c in concepts])
        stat_val = stats[key][score_layer]['test_acc']
        stat_list.append(stat_val) 
    return stat_list


def get_random_concepts(concepts_path, data_path, M, device):
    random_concepts = []
    for i in tqdm.tqdm(range(M), desc='Loading random concepts'):
        random_concepts.append(assemble_concept(name=f'random_{1000 + i}', concepts_path=concepts_path, data_path=data_path, device=device))
    return random_concepts

def get_experimental_set(positive_concept, random_concepts):
    experimental_set = [[positive_concept, random_concept] for random_concept in random_concepts]
    return experimental_set

def get_pval(scores, experimental_sets, score_layer, score_type, alpha=0.05, print_ret=False):
    
    P1 = assemble_scores(scores, experimental_sets, 0, score_layer, score_type)
    P2 = assemble_scores(scores, experimental_sets, 1, score_layer, score_type)
    
    if print_ret:
        print('P1[mean, std]: ', format_float(np.mean(P1)), format_float(np.std(P1)))
        print('P2[mean, std]: ', format_float(np.mean(P2)), format_float(np.std(P2)))

    _, pval = ttest_ind(P1, P2)

    if print_ret:
        print("p-values:", format_float(pval))

    if pval < alpha:    # alpha value is 0.05 or 5%
        relation = "Disjoint"
        if print_ret:
            print("Disjoint")
    else:
        relation = "Overlap"
        if print_ret:
            print("Overlap")
        
    return P1, P2, format_float(pval), relation

def calculate_confidence_intervals(data, confidence=0.95):
    n = len(data)  # Number of observations
    mean = np.mean(data, axis=0)  # Mean along the 0th axis
    sem = np.std(data, axis=0) / np.sqrt(n)  # Standard error of the mean

    # Critical t-value for a 95% confidence interval
    t_critical = stats.t.ppf((1 + confidence) / 2., n-1)

    # Margin of error
    margin_of_error = sem * t_critical

    return mean, margin_of_error

def plot_tcav_scores(experimental_sets, tcav_scores, layers, score_type='sign_count'):

    n_comp = min(len(experimental_sets), 16)
    if n_comp < 6:
        n_cols=n_comp
        n_rows=1
    else:
        n_cols = int(np.ceil(np.sqrt(n_comp)))
        n_rows = int(np.ceil(n_comp / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols, figsize = (25, 7))
    if n_comp > 1: ax = ax.flatten()

    barWidth = 1 / (len(experimental_sets[0]) + 1)

    all_vals = []

    for idx_es, concepts in enumerate(experimental_sets):

        concepts_vals = []

        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)

        pos = [np.arange(len(layers))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i-1]])
        
        if idx_es < 16: _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
        for i in range(len(concepts)):
            val = [format_float(scores[score_type][i]) for layer, scores in tcav_scores[concepts_key].items()]
            concepts_vals.append(val)
            if idx_es < 16: _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)

        if idx_es < 16:
            # Add xticks on the middle of the group bars
            _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=16)
            _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
            _ax.set_xticklabels(layers, fontsize=16)
            _ax.set_ylim([0,1])


            # Create legend & Show graphic
            # _ax.legend(fontsize=16)

        all_vals.append(concepts_vals)

    fig.suptitle(f'{idx_es+1} comparisons for layers:\n{layers}', fontsize=18)
    plt.show()

    all_vals = np.asarray(all_vals)
    mean_vals, margin_of_error = calculate_confidence_intervals(all_vals)

    fig, ax = plt.subplots(1, 1, figsize=(25, 7))
    pos = [np.arange(len(layers))]
    for i in range(1, mean_vals.shape[0]):
        pos.append([(x + barWidth) for x in pos[i-1]])

    for i in range(mean_vals.shape[0]):
        val = mean_vals[i]
        moe = margin_of_error[i]

        concept_name = experimental_sets[0][i].name
        if concept_name[:6] == 'random': concept_name = 'random'

        ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concept_name)
        # Plot confidence interval error bars
        ax.errorbar(pos[i], val, yerr=moe, fmt='none', ecolor='black', capsize=5)

    new_labels = []
    for layer in layers:
        _, _, p_vals, _ = get_pval(tcav_scores, experimental_sets, layer, score_type, alpha=0.05, print_ret=False)
        new_labels.append(f"{layer}\n(p={str(p_vals)})")

    ax.set_xlabel('Layers', fontweight='bold', fontsize=16)
    ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
    ax.set_xticklabels(new_labels, fontsize=16)
    ax.set_ylim([0,1])
    ax.set_xlim(ax.get_xlim())
    ax.hlines(y=0.5, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors='k', linestyles=':', label='Non-significant value')
    ax.legend(fontsize=16)
    ax.grid(True)
    plt.show()

def get_experimental_stats(scores, cavs):
    stats_dict = {}
    for cav_key in scores.keys():
        exp_stats = {}
        for layer_key in cavs[cav_key].keys():
            exp_stats[layer_key] = {"train_acc": cavs[cav_key][layer_key].stats["train_acc"],
                                    "test_acc": cavs[cav_key][layer_key].stats["test_acc"],
                                    "best_params": cavs[cav_key][layer_key].stats["best_params"]
                                    }
        stats_dict[cav_key] = exp_stats

    return stats_dict

def plot_multiple_tcav_scores(
    all_experimental_sets,
    all_tcav_scores,
    all_stats,
    score_type="sign_count",
    alpha=0.05,
    only_significant=False,
    with_error=True,
    plt_name="plot",
    file_type="svg",
    bonferroni_correction=750,
    min_accuracy = 0.5
):


    tcav_score0 = next(iter(all_tcav_scores[0].values()))
    layers = list(tcav_score0.keys())
    scores_mean = {layer: [] for layer in layers}
    scores_ci = {layer: [] for layer in layers}
    stats_mean = {layer: [] for layer in layers}
    pvals = {layer: [] for layer in layers}
    concept_short_names = []

    corrected_alpha = alpha / bonferroni_correction

    for experimental_sets, tcav_scores, stats_dict in zip(all_experimental_sets, all_tcav_scores, all_stats):
        concept_short_names.append(experimental_sets[0][0].name)

        for layer in layers:
            pos_scores = assemble_scores(tcav_scores, experimental_sets, 0, layer, score_type)
            neg_scores = assemble_scores(tcav_scores, experimental_sets, 1, layer, score_type)
            #_, pval = ttest_ind(pos_scores, neg_scores, alternative = 'two-sided')
            _, pval = mannwhitneyu(pos_scores, neg_scores, alternative = 'two-sided')
            pvals[layer].append(pval)

            mean_score, ci_score = calculate_confidence_intervals(pos_scores)
            scores_mean[layer].append(mean_score)
            scores_ci[layer].append(ci_score)

            stat_list = assemble_stats(stats_dict, experimental_sets, layer)
            #mean_stat, _ = calculate_confidence_intervals(stat_list)
            mean_stat = np.median(stat_list, axis=0)
            stats_mean[layer].append(mean_stat)

    scores_mean_df = pd.DataFrame(data=scores_mean, index=concept_short_names)
    scores_ci_df = pd.DataFrame(data=scores_ci, index=concept_short_names)


    fig = plt.figure(figsize=(20, 4))
    ax = plt.subplot()

    # Plot without capturing return value
    scores_mean_df.plot.bar(ax=ax, rot=0, yerr=scores_ci_df if with_error else None)

    # Add horizontal line at y=0
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    ax.annotate("H0=0.5", (ax.get_xlim()[0]+0.01, 0.51), ha="left", fontsize=8, color="gray")


    # Extract containers after plotting
    bar_containers = [c for c in ax.containers if isinstance(c, BarContainer)]

    for layer_idx, (layer, container) in enumerate(zip(layers, bar_containers)):
        for bar_idx, bar in enumerate(container):
            height = bar.get_height()
            pval = pvals[layer][bar_idx]
            is_significant = pval < corrected_alpha

            accuracy_score = stats_mean[layer][bar_idx]
            is_accurate = accuracy_score > min_accuracy

            if not (is_significant and is_accurate):
                bar.set_hatch("////")
                bar.set_alpha(0.2)
                bar.set_edgecolor("gray")
            if is_significant:
                x = bar.get_x() + bar.get_width() * 0.30
                y = height
                ax.annotate("*", (x - 0.01, min(y, 0.92)), ha="center", fontsize=14, color="black")
            if is_accurate:
                x = bar.get_x() + bar.get_width() * 0.85
                y = height
                ax.annotate("#", (x - 0.01, min(y, 0.92)+0.02), ha="center", fontsize=10, color="black")

    # Assume 'original_colors' is a dict mapping layer names to their color
    original_colors = {
        layer: bar_containers[i][0].get_facecolor()[:3]  # remove alpha if faded
        for i, layer in enumerate(layers)
    }

    custom_patches_layer = [
        Patch(facecolor=original_colors[layer], label=layer.split('.')[-1])
        for layer in layers
    ]

    # custom_patches_text = [
    #     Patch(facecolor='none', edgecolor='none', label='Significant (*)'),
    #     Patch(facecolor='none', edgecolor='none', label='Accurate (#)')
    # ]

    # # Combine handles
    # all_handles = custom_patches_layer + custom_patches_text

    legend_fontsize = 14
    label_fontsize = 16
    tick_fontsize = 14



    ax.legend(
        handles=custom_patches_layer,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(custom_patches_layer),
        title="Network layer",
        frameon=False,
        fontsize=legend_fontsize,          # Legend item font size
        title_fontsize=legend_fontsize     # Legend title font size
    )

    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_ylabel("TCAV score", fontsize=label_fontsize)
    ax.set_xlabel("Concept", fontsize=label_fontsize)  # If applicable

    # Increase tick label size
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    plt.tight_layout()
    # plt.savefig(f"{plt_name}.{file_type}")
    plt.show()
    plt.close()

    





class TCAVClassifier(Classifier):
    def __init__(self,
                 alphas=[1e-20, 1e-2, 1e0, 1e2],                               #[1e-10, 1e-7, 1e-4, 1e-2, 1e0, 1e1, 1e2],                         #[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
                 losses=["hinge", "log_loss"],                                  #["hinge", "log_loss"],
                 penalties=["l2"],                                  #["l1", "l2"],
                 tol=1e-4,
                 shuffle=True,
                 n_jobs=-1,
                 learning_rate="optimal"):
        self.alphas = alphas
        self.losses = losses
        self.penalties = penalties
        self.tol = tol
        self.shuffle = shuffle
        self.n_jobs = n_jobs
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_params = None

    def train_and_eval(self, dataloader):
        inputs = []
        labels = []
        for input, label in dataloader:
            inputs.append(input)
            labels.append(label)

        x = torch.cat(inputs).detach().cpu().numpy()
        y = torch.cat(labels).detach().cpu().numpy()

        x_scaled = self.scaler.fit_transform(x)

        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled, y, random_state=42, stratify=y, test_size=0.1
        )


        #print("#################################")
        class_counts = Counter(y_train.tolist())
        #print("Class distribution (train):", class_counts)
        print("Input shape:", x.shape)

        best_acc = -1
        best_model = None
        best_params = None

        for alpha, loss, penalty in itertools.product(self.alphas, self.losses, self.penalties):
            try:
                model = SGDClassifier(
                    alpha=alpha,
                    loss=loss,
                    penalty=penalty,
                    tol=self.tol,
                    max_iter=1000,
                    random_state=42,
                    shuffle=self.shuffle,
                    n_jobs=self.n_jobs,
                    learning_rate=self.learning_rate
                )
                model.fit(x_train, y_train)
                preds_val = model.predict(x_test)
                acc = accuracy_score(y_test, preds_val)

                #print(f"Alpha: {alpha}, Loss: {loss}, Penalty: {penalty}, Test Accuracy: {acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_params = {
                        'alpha': alpha,
                        'loss': loss,
                        'penalty': penalty
                    }

            except Exception as e:
                print(f"Skipped combination alpha={alpha}, loss={loss}, penalty={penalty} due to error: {e}")

        self.lm = best_model
        self.best_model = best_model
        self.best_params = best_params

        preds_train = torch.tensor(self.lm.predict(x_train))
        preds_test = torch.tensor(self.lm.predict(x_test))

        train_acc = (preds_train == torch.tensor(y_train)).float().mean().item()
        test_acc = (preds_test == torch.tensor(y_test)).float().mean().item()

        print(f"Best Parameters: {best_params}")
        #print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'class_balance': class_counts,
            'best_params': best_params
        }

    def weights(self):
        if len(self.lm.coef_) == 1:
            return torch.tensor([-1 * self.lm.coef_[0], self.lm.coef_[0]])
        else:
            return torch.tensor(self.lm.coef_)

    def classes(self):
        return self.lm.classes_
