import matplotlib.pyplot as plt
import numpy as np
from .dataset import EEG_BANDS, RATIOS_OF_INTEREST, TOPOGRAPHICAL_MAP, ELECTRODES_TUH, ELECTRODES_BC, FEATURES, FEATURE_UNIT, get_biomarker_data
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from statsmodels.stats import diagnostic
from scipy.stats import ttest_ind, mannwhitneyu, anderson# , kstest
from IPython.display import display
from collections import Counter
import os
import mne
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.preprocessing import StandardScaler

import networkx as nx
from sklearn.feature_selection import mutual_info_classif

import shap

def plot_biomarkers(df):
    print(f'------------------------------------------------\n------ EEG set of {df.shape[0]} recording(s)\n------------------------------------------------')
    plot_psd(df)
    plot_aperiodic(df)
    plot_complexity(df)

def compare_biomarkers(df_base, df_comp, statistical_testing=True):
    print(f'------------------------------------------------\n------ [Base.] EEG set of {df_base.shape[0]} recording(s)\n------ [Comp.] EEG set of {df_comp.shape[0]} recording(s)\n------------------------------------------------')
    tests = None
    if statistical_testing: tests = {'metric': [],
                                     'statistics: [(mean1, median1, std1), (mean2, median2, std2)]': [],
                                     'normality: [(stat1, norm1), (stat2, norm2)]': [],
                                     'ttest: (stat, p)': [],
                                     'mannwhitneyu: (stat, p)': []} 
    tests = compare_psd(df_base, df_comp, tests)
    tests = compare_aperiodic(df_base, df_comp, tests)
    tests = compare_complexity(df_base, df_comp, tests)

    if tests: display(pd.DataFrame(tests).set_index('metric'))

def plot_psd(df):

    freqs = df['freqs'].iloc[0]  # Shared frequency axis

    psd_stack = np.stack(df['full_fit'].values, axis=0)  # shape: (n_rec, n_ch, n_freqs)
    ap_stack = np.stack(df['aperiodic_fit'].values, axis=0)  # shape: (n_rec, n_ch, n_freqs)
    
    n_rec, n_ch, n_freqs = psd_stack.shape

    single_file = n_rec == 1

    if single_file:
        psd_data = psd_stack.reshape((n_ch, n_freqs)) # shape (n_ch, n_freqs)
        ap_data  = ap_stack.reshape((n_ch, n_freqs)) # shape (n_ch, n_freqs)

    else:
        psd_data = np.mean(psd_stack, axis=1) # shape (n_rec, n_freqs)
        ap_data  = np.mean(ap_stack, axis=1) # shape (n_rec, n_freqs)


    # Compute statistics across recordings => shape (n_freqs,)
    psd_mean, psd_median, psd_std = compute_statistics(psd_data, axis=0)
    ap_mean, ap_median, ap_std = compute_statistics(ap_data, axis=0)

    # Find the maximum median value within [5, 80] Hz range
    valid_idx = np.where((freqs >= 7) & (freqs <= 15))[0]
    max_mean_value = np.max(psd_mean[valid_idx])
    max_mean_local_idx = np.argmax(psd_mean[valid_idx])
    max_idx = valid_idx[max_mean_local_idx]
    peak_freq = freqs[max_idx]


    band_names = list(EEG_BANDS.keys())  # e.g. ['delta','theta','alpha','beta','gamma']
    band_power_data = []
    categories = []  # Category labels for violin plot

    for band in band_names:
        # df[band] => each row is an array of shape (n_ch,) for that band's power
        band_stack = np.stack(df[band + '_rel'].values, axis=0)  # shape: (n_rec, n_ch)

        if single_file:
            band_rec = band_stack.reshape((n_ch,))
        else:
            # Average across channels => shape: (n_rec,)
            band_rec = np.mean(band_stack, axis=1)
        # Mean and std across recordings
        band_power_data.append(band_rec)  # Store each band's power for violin plot
        categories.extend([band + '_rel'] * len(band_rec))
    band_df = pd.DataFrame({"Band": categories, "Power": np.concatenate(band_power_data)})

    ratio_names = list(RATIOS_OF_INTEREST.keys())  # e.g. ['delta/alpha','theta/beta']
    ratio_power_data = []
    categories = []  # Category labels for violin plot

    for ratio in ratio_names:
        # df[band] => each row is an array of shape (n_ch,) for that band's power
        ratio_stack = np.stack(df[ratio].values, axis=0)  # shape: (n_rec, n_ch)

        if single_file:
            ratio_rec = ratio_stack.reshape((n_ch,))
        else:
            # Average across channels => shape: (n_rec,)
            ratio_rec = np.mean(ratio_stack, axis=1)
        # Mean and std across recordings
        ratio_power_data.append(ratio_rec)  # Store each band's power for violin plot
        categories.extend([ratio] * len(ratio_rec))
    ratio_df = pd.DataFrame({"Ratio": categories, "Value": np.concatenate(ratio_power_data)})



    fig, (ax_psd, ax_pb, ax_rb) = plt.subplots(ncols=3, figsize=(15, 4),
                                         gridspec_kw={'width_ratios': [2, 1, 1]})

    ax_psd.plot(freqs, psd_median, color='b', linestyle='--', label='Median PSD')
    ax_psd.plot(freqs, psd_mean,   color='b', linestyle='-',  label='Mean PSD')
    ax_psd.fill_between(freqs, psd_mean - psd_std, psd_mean + psd_std,
                        alpha=0.1, color='b', label='SD PSD')
    
    ax_psd.plot(freqs, ap_median, color='r', linestyle='--', label='Median Aperiodic')
    ax_psd.plot(freqs, ap_mean,   color='r', linestyle='-',  label='Mean Aperiodic')
    ax_psd.fill_between(freqs, ap_mean - ap_std, ap_mean + ap_std,
                        alpha=0.1, color='r', label='SD Aperiodic')

    # Vertical line for peak freq
    ax_psd.axvline(x=peak_freq, color='k', linestyle='-', linewidth=0.5, label=f"Peak Frequency:\n{max_mean_value:.2f} @ {peak_freq:.1f}Hz")
    # Set up axis limits
    ax_psd.set_xlim([0, freqs[-1]+1])
    y_bottom = ax_psd.get_ylim()[0]
    y_top = ax_psd.get_ylim()[1]

    # Band boundaries
    for band_name, (f_low, f_high) in EEG_BANDS.items():
        ax_psd.axvline(x=f_high, color='black', linestyle=':', alpha=1, linewidth=0.5)
        ax_psd.text(f_high, y_bottom*0.998, band_name,
                    ha='right', va='bottom', rotation=90, color='grey')

    if single_file:ax_psd.set_title(f"PSD and Aperiodic Analysis of a Recording with {n_ch} Channels")
    else: ax_psd.set_title(f"PSD and Aperiodic Analysis of {n_rec} Recordings Averaged Across {n_ch} Channels")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("log10(Power)")
    ax_psd.legend(loc='upper right')

    # Scatter points overlay
    jitter, size = 0.1, 10

    # n_samp = 10
    # x_samp, y_samp = [], []
    # Add jitter for spread
    if single_file: x = np.random.normal(0, jitter, size=n_ch)  
    else: x = np.random.normal(0, jitter, size=n_rec) 
    
    for idx, band in enumerate(band_names):
        y = np.asarray(band_df[band_df['Band'] == band + '_rel']['Power'])
        # Assign color based on the category (default black)
        #ax_pb.scatter(x+idx, y, alpha=0.2, s=size, color='k', edgecolors='k')
        ax_pb.scatter(x+idx, y, s=size, facecolors=(0, 0, 0, 0.1), edgecolors=(0, 0, 0, 0.5))

        # x_samp.append(x[np.linspace(0,x.shape[0]-1,n_samp).astype(int)]+idx), y_samp.append(y[np.linspace(0,x.shape[0]-1,n_samp).astype(int)])

    # ax_pb.plot(x_samp, y_samp, marker='o', markersize=1, linestyle='-', color='r', linewidth=0.2, alpha=0.8, label="Some recordings")

    sns.violinplot(x="Band", y="Power", data=band_df, color='b',
                   ax=ax_pb, linewidth=2, fill=False,
                   inner_kws=dict(box_width=8, whis_width=2, color="0.7"))
    
    ax_pb.set_xlabel("EEG Bands")
    ax_pb.set_ylabel("Relative Band Power")
    if single_file: ax_pb.set_title(f"Band Power Analysis of a Recording\nwith {n_ch} Channels")
    else: ax_pb.set_title(f"Band Power Analysis of {n_rec} Recordings\nAveraged Across {n_ch} Channels")


    # Scatter points overlay
    jitter, size = 0.1, 10

    # # Add jitter for spread
    if single_file: x = np.random.normal(0, jitter, size=n_ch)  
    else: x = np.random.normal(0, jitter, size=n_rec) 
    
    for idx, ratio in enumerate(ratio_names):
        y = np.asarray(ratio_df[ratio_df['Ratio'] == ratio]['Value'])
        # Assign color based on the category (default black)
        # ax_rb.scatter(x+idx, y, alpha=0.3, s=size, color='k')
        ax_rb.scatter(x+idx, y, s=size, facecolors=(0, 0, 0, 0.1), edgecolors=(0, 0, 0, 0.5))

    sns.violinplot(x="Ratio", y="Value", data=ratio_df, log_scale=True,
                ax=ax_rb, linewidth=2, fill=False, color='b',
                inner_kws=dict(box_width=8, whis_width=2, color="0.7"))

    ax_rb.axhline(y=1, color='k', linestyle='--', linewidth=0.5)

    ax_rb.set_xticks(ax_rb.get_xticks())
    labels = [tick_label.get_text() for tick_label in ax_rb.get_xticklabels()]
    labels = [label.replace('delta', 'd').replace('theta', 't').replace('alpha', 'a').replace('beta', 'b').replace('gamma', 'g') for label in labels]
    ax_rb.set_xticklabels(labels)
    ax_rb.set_xlabel("EEG Bands Ratios")
    ax_rb.set_ylabel("Value")
    ax_rb.set_yscale("log")
    if single_file: ax_rb.set_title(f"Band Power Ratio Analysis of\na Recording with {n_ch} Channels")
    else: ax_rb.set_title(f"Band Power Ratio Analysis of\n{n_rec} Recordings Averaged Across {n_ch} Channels")
    
    fig.tight_layout()
    plt.show()

def plot_aperiodic(df):
    """
    Plots histograms for 'offset' and 'exponent' values across recordings,
    with vertical lines indicating mean and median, and annotations for standard deviation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'offset' and 'exponent' columns, each with arrays of shape (n_channels,).
    """

    # Extract 'offset' and 'exponent' values across all recordings and channels
    offsets = np.stack(df['offset'].values, axis=0)
    exponents = np.stack(df['exponent'].values, axis=0)

    n_rec, n_ch = offsets.shape

    single_file = n_rec == 1

    if single_file: 
        fg = df['fg'].iloc[0]
        fg.print_results()
        fg.plot()
    else:
        offsets_mean = np.mean(offsets, axis=1) # average per channel
        exponents_mean = np.mean(exponents, axis=1) # average per channel

        # Create a new figure with two subplots
        fig, (ax_offset, ax_exponent) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        # Plot histogram for 'offset' f"PSD and Aperiodic Analysis of {n_rec} Recordings Averaged Across {n_ch} Channels"
        plot_histogram_with_stats(ax_offset, offsets_mean, f'Aperiodic Offset Distribution of\n{n_rec} Recordings Averaged Across {n_ch} Channels', color='darkmagenta')

        # Plot histogram for 'exponent'
        plot_histogram_with_stats(ax_exponent, exponents_mean, f'Aperiodic Exponent Distribution of\n{n_rec} Recordings Averaged Across {n_ch} Channels', color='darkgreen')

        plt.tight_layout()
        plt.show()

def plot_complexity(df):
    
    n_rec = df.shape[0]
    n_ch = df['n_channels'].iloc[0]

    single_file = n_rec == 1

    complexity_names = ["dfa", "lzc", "hfd", "SampEn", "PermEn_5"] # "PermEn_3", "PermEn_4", "PermEn_5"]
    complexity_data = []
    categories = []  # Category labels for violin plot

    for complexity in complexity_names:
        complexity_stack = np.stack(df[complexity].values, axis=0)  # shape: (n_rec, n_ch)

        if single_file:
            complexity_rec = complexity_stack.reshape((n_ch,))
        else:
            # Average across channels => shape: (n_rec,)
            complexity_rec = np.mean(complexity_stack, axis=1)
        # Mean and std across recordings
        complexity_data.append(complexity_rec)  # Store each band's power for violin plot
        categories.extend([complexity] * len(complexity_rec))
    complexity_df = pd.DataFrame({"Complexity": categories, "Value": np.concatenate(complexity_data)})

    plt.figure(figsize=(8,5))

    # Scatter points overlay
    jitter, size = 0.1, 10

    # # Add jitter for spread
    if single_file: x = np.random.normal(0, jitter, size=n_ch)  
    else: x = np.random.normal(0, jitter, size=n_rec) 
    
    for idx, complexity in enumerate(complexity_names):
        y = np.asarray(complexity_df[complexity_df['Complexity'] == complexity]['Value'])
        # Assign color based on the category (default black)
        #plt.scatter(x+idx, y, alpha=0.3, s=size, color='k')
        plt.scatter(x+idx, y, s=size, facecolors=(0, 0, 0, 0.1), edgecolors=(0, 0, 0, 0.5))

    sns.violinplot(x="Complexity", y="Value", data=complexity_df,
                   linewidth=2, fill=False, color='b',
                   inner_kws=dict(box_width=8, whis_width=2, color="0.7"))

    plt.axhline(y=1, color='k', linestyle='--', linewidth=0.5)

    plt.xlabel("Complexity Metric")
    plt.ylabel("Value")
    plt.ylim([0, 2.3])
    if single_file: plt.title(f"Complexity Analysis of\na Recording with {n_ch} Channels")
    else: plt.title(f"Complexity Analysis of\n{n_rec} Recordings Averaged Across {n_ch} Channels")

    plt.show()

def plot_lateralization(df):
    return

def compare_psd(df_base, df_comp, tests=None):
    """
    df_base is a group of recordings
    df_comp is either a group of recordings or a single recording
    """

    freqs_base = df_base['freqs'].iloc[0]
    freqs_comp = df_comp['freqs'].iloc[0]

    psd_stack_base = np.stack(df_base['full_fit'].values, axis=0)  # shape: (n_rec, n_ch, n_freqs)
    ap_stack_base = np.stack(df_base['aperiodic_fit'].values, axis=0)  # shape: (n_rec, n_ch, n_freqs)
    psd_stack_comp = np.stack(df_comp['full_fit'].values, axis=0)  # shape: (n_rec, n_ch, n_freqs)
    ap_stack_comp = np.stack(df_comp['aperiodic_fit'].values, axis=0)  # shape: (n_rec, n_ch, n_freqs)
    
    n_rec_base, n_ch_base, n_freqs_base = psd_stack_base.shape
    n_rec_comp, n_ch_comp, n_freqs_comp = psd_stack_comp.shape

    single_file = n_rec_comp == 1

    psd_data_base = np.mean(psd_stack_base, axis=1) # shape (n_rec, n_freqs)
    ap_data_base  = np.mean(ap_stack_base, axis=1) # shape (n_rec, n_freqs)
    
    if single_file:
        psd_data_comp = psd_stack_comp.reshape((n_ch_comp, n_freqs_comp)) # shape (n_ch, n_freqs)
        ap_data_comp  = ap_stack_comp.reshape((n_ch_comp, n_freqs_comp)) # shape (n_ch, n_freqs)

    else:
        psd_data_comp = np.mean(psd_stack_comp, axis=1) # shape (n_rec, n_freqs)
        ap_data_comp  = np.mean(ap_stack_comp, axis=1) # shape (n_rec, n_freqs)


    # Compute statistics across recordings => shape (n_freqs,)
    psd_mean_base, psd_median_base, psd_std_base = compute_statistics(psd_data_base, axis=0)
    ap_mean_base, ap_median_base, ap_std_base = compute_statistics(ap_data_base, axis=0)
    psd_mean_comp, psd_median_comp, psd_std_comp = compute_statistics(psd_data_comp, axis=0)
    ap_mean_comp, ap_median_comp, ap_std_comp = compute_statistics(ap_data_comp, axis=0)


    band_names = list(EEG_BANDS.keys())  # e.g. ['delta','theta','alpha','beta','gamma']
    band_power_data_base, band_power_data_comp = [], []
    categories_base, categories_comp = [], []  # Category labels for violin plot

    for band in band_names:
        # df[band] => each row is an array of shape (n_ch,) for that band's power
        band_stack_base = np.stack(df_base[band + '_rel'].values, axis=0)  # shape: (n_rec, n_ch)
        band_stack_comp = np.stack(df_comp[band + '_rel'].values, axis=0)

        band_rec_base = np.mean(band_stack_base, axis=1)        
        if single_file:
            band_rec_comp = [np.mean(band_stack_comp)]
        else:
            # Average across channels => shape: (n_rec,)
            band_rec_comp = np.mean(band_stack_comp, axis=1)
        # Mean and std across recordings
        band_power_data_base.append(band_rec_base)  # Store each band's power for violin plot
        categories_base.extend([band + '_rel'] * len(band_rec_base))
        band_power_data_comp.append(band_rec_comp)  # Store each band's power for violin plot
        categories_comp.extend([band + '_rel'] * len(band_rec_comp))

        if tests and not single_file: tests = compute_statistical_tests(band_rec_base, band_rec_comp, band, tests)

    band_df_base = pd.DataFrame({"Band": categories_base, "Power": np.concatenate(band_power_data_base)})
    band_df_comp = pd.DataFrame({"Band": categories_comp, "Power": np.concatenate(band_power_data_comp)})

    if not single_file:
        band_df_base['dataset'], band_df_comp['dataset'] = 'base', 'comp'
        band_df = pd.concat([band_df_base, band_df_comp])

    
    ratio_names = list(RATIOS_OF_INTEREST.keys())  # e.g. ['delta/alpha','theta/beta']
    ratio_power_data_base, ratio_power_data_comp = [], []
    categories_base, categories_comp = [], []  # Category labels for violin plot

    for ratio in ratio_names:
        # df[ratio] => each row is an array of shape (n_ch,) for that ratio's power
        ratio_stack_base = np.stack(df_base[ratio].values, axis=0)  # shape: (n_rec, n_ch)
        ratio_stack_comp = np.stack(df_comp[ratio].values, axis=0)

        ratio_rec_base = np.mean(ratio_stack_base, axis=1)        
        if single_file:
            ratio_rec_comp = [np.mean(ratio_stack_comp)]
        else:
            # Average across channels => shape: (n_rec,)
            ratio_rec_comp = np.mean(ratio_stack_comp, axis=1)
        # Mean and std across recordings
        ratio_power_data_base.append(ratio_rec_base)  # Store each ratio's power for violin plot
        categories_base.extend([ratio] * len(ratio_rec_base))
        ratio_power_data_comp.append(ratio_rec_comp)  # Store each ratio's power for violin plot
        categories_comp.extend([ratio] * len(ratio_rec_comp))

        if tests and not single_file: tests = compute_statistical_tests(ratio_rec_base, ratio_rec_comp, ratio, tests)

    ratio_df_base = pd.DataFrame({"Ratio": categories_base, "Power": np.concatenate(ratio_power_data_base)})
    ratio_df_comp = pd.DataFrame({"Ratio": categories_comp, "Power": np.concatenate(ratio_power_data_comp)})

    if not single_file:
        ratio_df_base['dataset'], ratio_df_comp['dataset'] = 'base', 'comp'
        ratio_df = pd.concat([ratio_df_base, ratio_df_comp])



    fig, (ax_psd, ax_pb, ax_rb) = plt.subplots(ncols=3, figsize=(15, 4),
                                         gridspec_kw={'width_ratios': [2, 1, 1]})

    ax_psd.plot(freqs_base, psd_mean_base,   color='b', linestyle='-',  label='Mean PSD Base.')
    ax_psd.fill_between(freqs_base, psd_mean_base - psd_std_base, psd_mean_base + psd_std_base,
                        alpha=0.1, color='b', label='SD PSD Base.')
    ax_psd.plot(freqs_base, ap_mean_base,   color='b', linestyle='--',  label='Mean Aperiodic Base.')
    # ax_psd.fill_between(freqs_base, ap_mean_base - ap_std_base, ap_mean_base + ap_std_base,
    #                     alpha=0.1, color='b', label='SD Aperiodic Base.')
    
    ax_psd.plot(freqs_comp, psd_mean_comp,   color='r', linestyle='-',  label='Mean PSD Comp.')
    ax_psd.fill_between(freqs_comp, psd_mean_comp - psd_std_comp, psd_mean_comp + psd_std_comp,
                        alpha=0.1, color='r', label='SD PSD Comp.')
    ax_psd.plot(freqs_comp, ap_mean_comp,   color='r', linestyle='--',  label='Mean Aperiodic Comp.')
    # ax_psd.fill_between(freqs_comp, ap_mean_comp - ap_std_comp, ap_mean_comp + ap_std_comp,
    #                     alpha=0.1, color='r', label='SD Aperiodic Comp.')

    # Set up axis limits
    ax_psd.set_xlim([0, freqs_base[-1]+1])
    y_bottom = ax_psd.get_ylim()[0]
    y_top = ax_psd.get_ylim()[1]

    # Band boundaries
    for band_name, (f_low, f_high) in EEG_BANDS.items():
        ax_psd.axvline(x=f_high, color='black', linestyle=':', alpha=1, linewidth=0.5)
        ax_psd.text(f_high, y_bottom*0.998, band_name,
                    ha='right', va='bottom', rotation=90, color='grey')
    
    ax_psd.set_title(f"PSD and Aperiodic Comparison of\n[Base.] {n_rec_base} Recs with {n_ch_base} Chns vs.\n[Comp.] {n_rec_comp} Rec(s) with {n_ch_comp} Chns")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("log10(Power)")
    ax_psd.legend(loc='upper right')
        
    # # Scatter points overlay
    # jitter, size = 0.05, 10  

    # # Add jitter for spread
    # x_base = np.random.normal(-0.3, jitter, size=n_rec_base)
    # if single_file: x_comp = 0.3  
    # else: x_comp = np.random.normal(0.3, jitter, size=n_rec_comp) 
    
    # for idx, band in enumerate(band_names):
    #     y_base = np.asarray(band_df_base[band_df_base['Band'] == band]['Power'])
    #     y_comp = np.asarray(band_df_comp[band_df_comp['Band'] == band]['Power'])

    #     # Assign color based on the category (default black)
    #     ax_pb.scatter(x_base+idx, y_base, alpha=0.3, s=size, color='b')
    #     if single_file: ax_pb.scatter(x_comp+idx, y_comp, alpha=1, s=size*2, marker='x', color='r')
    #     else: ax_pb.scatter(x_comp+idx, y_comp, s=size, facecolors=(0, 0, 0, 0.1), edgecolors=(0, 0, 0, 0.5))

    if single_file:
        sns.violinplot(x="Band", y="Power", data=band_df_base, color='b',
                   ax=ax_pb, linewidth=2, fill=False,
                   inner_kws=dict(box_width=8, whis_width=2, color="0.7"))
        for idx, band in enumerate(band_names):
            y_comp = np.asarray(band_df_comp[band_df_comp['Band'] == band]['Power'])
            ax_pb.scatter(idx+0.2, y_comp, alpha=1, s=40, marker='x', color='r')
    else:
        sns.violinplot(x="Band", y="Power", data=band_df, hue='dataset', split=True, legend=None, gap=0.2,
                   ax=ax_pb, linewidth=2, fill=False, palette=['b', 'r'],
                   inner_kws=dict(box_width=6, whis_width=2, color="0.7")
                   )

    ax_pb.set_xlabel("EEG Bands")
    ax_pb.set_ylabel("Relative Band Power")
    ax_pb.set_title(f"Band Power Comparison of\n[Base.] {n_rec_base} Recs with {n_ch_base} Chns vs.\n[Comp.] {n_rec_comp} Rec(s) with {n_ch_comp} Chns")
        
    # # Scatter points overlay
    # jitter, size = 0.05, 2  

    # # Add jitter for spread
    # x_base = np.random.normal(-0.3, jitter, size=n_rec_base)
    # if single_file: x_comp = 0.3  
    # else: x_comp = np.random.normal(0.3, jitter, size=n_rec_comp) 
    
    # for idx, ratio in enumerate(ratio_names):
    #     y_base = np.asarray(ratio_df_base[ratio_df_base['Ratio'] == ratio]['Power'])
    #     y_comp = np.asarray(ratio_df_comp[ratio_df_comp['Ratio'] == ratio]['Power'])

    #     # Assign color based on the category (default black)
    #     ax_rb.scatter(x_base+idx, y_base, alpha=0.3, s=size, color='b')
    #     if single_file: ax_rb.scatter(x_comp+idx, y_comp, alpha=1, s=size*10, marker='x', color='r')
    #     else: ax_rb.scatter(x_comp+idx, y_comp, alpha=0.3, s=size, color='r')

    if single_file:
        sns.violinplot(x="Ratio", y="Power", data=ratio_df_base, color='b',
                   ax=ax_rb, linewidth=2, fill=False, log_scale=True,
                   inner_kws=dict(box_width=8, whis_width=2, color="0.7"))
        for idx, ratio in enumerate(ratio_names):
            y_comp = np.asarray(ratio_df_comp[ratio_df_comp['Ratio'] == ratio]['Power'])
            ax_rb.scatter(idx+0.2, y_comp, alpha=1, s=40, marker='x', color='r')
    else:
        sns.violinplot(x="Ratio", y="Power", data=ratio_df, hue='dataset', split=True, legend=None, gap=0.2,
                   ax=ax_rb, linewidth=2, fill=False, log_scale=True, palette=['b', 'r'],
                   inner_kws=dict(box_width=6, whis_width=2, color="0.7")
                   )

    ax_rb.axhline(y=1, color='k', linestyle='--', linewidth=0.5)

    ax_rb.set_xticks(ax_rb.get_xticks())
    labels = [tick_label.get_text() for tick_label in ax_rb.get_xticklabels()]
    labels = [label.replace('delta', 'd').replace('theta', 't').replace('alpha', 'a').replace('beta', 'b').replace('gamma', 'g') for label in labels]
    ax_rb.set_xticklabels(labels)
    ax_rb.set_xlabel("EEG Ratios")
    ax_rb.set_ylabel("Relative Ratio Power")
    ax_rb.set_yscale("log")
    ax_rb.set_title(f"Ratio Power Comparison of\n[Base.] {n_rec_base} Recs with {n_ch_base} Chns vs.\n[Comp.] {n_rec_comp} Rec(s) with {n_ch_comp} Chns")
    
    fig.tight_layout()
    plt.show()

    return tests

def compare_aperiodic(df_base, df_comp, tests=None):
    """
    Plots histograms for 'offset' and 'exponent' values across recordings,
    with vertical lines indicating mean and median, and annotations for standard deviation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'offset' and 'exponent' columns, each with arrays of shape (n_channels,).
    """

    # Extract 'offset' and 'exponent' values across all recordings and channels
    offsets_base = np.stack(df_base['offset'].values, axis=0)
    exponents_base = np.stack(df_base['exponent'].values, axis=0)
    offsets_comp = np.stack(df_comp['offset'].values, axis=0)
    exponents_comp = np.stack(df_comp['exponent'].values, axis=0)

    n_rec_base, n_ch_base = offsets_base.shape
    n_rec_comp, n_ch_comp = offsets_comp.shape

    single_file = n_rec_comp == 1

    offsets_mean_base = np.mean(offsets_base, axis=1) # average per channel
    exponents_mean_base = np.mean(exponents_base, axis=1) # average per channel
    offsets_mean_comp = np.mean(offsets_comp, axis=1) # average per channel
    exponents_mean_comp = np.mean(exponents_comp, axis=1) # average per channel

    if tests and not single_file:
        tests = compute_statistical_tests(offsets_mean_base, offsets_mean_comp, 'offset', tests)
        tests = compute_statistical_tests(exponents_mean_base, exponents_mean_comp, 'exponent', tests)


    # Create a new figure with two subplots
    fig, (ax_offset, ax_exponent) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Plot histogram for 'offset' f"PSD and Aperiodic Analysis of {n_rec} Recordings Averaged Across {n_ch} Channels"
    plot_histogram_with_stats(ax_offset, offsets_mean_base, '', color='blue')
    if single_file:
        ax_offset.axvline(x=offsets_mean_comp[0], color='red', linestyle='-', linewidth=3, label=f'Comp. Value: {offsets_mean_comp[0]:.2f}')
        ax_offset.legend()
    else: plot_histogram_with_stats(ax_offset, offsets_mean_comp, f"Aperiodic Offset Comparison of\n[Base.] {n_rec_base} Recs with {n_ch_base} Chns vs.\n[Comp.] {n_rec_comp} Rec(s) with {n_ch_comp} Chns", color='red')

    # Plot histogram for 'exponent'
    plot_histogram_with_stats(ax_exponent, exponents_mean_base, '', color='blue')
    if single_file:
        ax_exponent.axvline(x=exponents_mean_comp[0], color='red', linestyle='-', linewidth=3, label=f'Comp. Value: {exponents_mean_comp[0]:.2f}')
        ax_exponent.legend()
    else: plot_histogram_with_stats(ax_exponent, exponents_mean_comp, f"Aperiodic Exponent Comparison of\n[Base.] {n_rec_base} Recs with {n_ch_base} Chns vs.\n[Comp.] {n_rec_comp} Rec(s) with {n_ch_comp} Chns", color='red')

    plt.tight_layout()
    plt.show()

    return tests

def compare_complexity(df_base, df_comp, tests=None):
    
    n_rec_base = df_base.shape[0]
    n_ch_base = df_base['n_channels'].iloc[0]
    n_rec_comp = df_comp.shape[0]
    n_ch_comp = df_comp['n_channels'].iloc[0]

    single_file = n_rec_comp == 1

    complexity_names = ["dfa", "lzc", "hfd", "SampEn", "PermEn_5"]
    complexity_data_base, complexity_data_comp = [], []
    categories_base, categories_comp = [], []  # Category labels for violin plot

    for complexity in complexity_names:
        complexity_stack_base = np.stack(df_base[complexity].values, axis=0)  # shape: (n_rec, n_ch)
        complexity_stack_comp = np.stack(df_comp[complexity].values, axis=0)  # shape: (n_rec, n_ch)


        complexity_rec_base = np.mean(complexity_stack_base, axis=1) # shape (n_rec,)
        
        if single_file:
            complexity_rec_comp = [np.mean(complexity_stack_comp)]
        else:
            complexity_rec_comp = np.mean(complexity_stack_comp, axis=1) # shape (n_rec,)


        complexity_data_base.append(complexity_rec_base)  
        categories_base.extend([complexity] * len(complexity_rec_base))
        complexity_data_comp.append(complexity_rec_comp)  
        categories_comp.extend([complexity] * len(complexity_rec_comp))

        if tests and not single_file: tests = compute_statistical_tests(complexity_rec_base, complexity_rec_comp, complexity, tests)

    complexity_df_base = pd.DataFrame({"Complexity": categories_base, "Value": np.concatenate(complexity_data_base)})
    complexity_df_comp = pd.DataFrame({"Complexity": categories_comp, "Value": np.concatenate(complexity_data_comp)})

    if not single_file:
        complexity_df_base['dataset'], complexity_df_comp['dataset'] = 'base', 'comp'
        complexity_df = pd.concat([complexity_df_base, complexity_df_comp])

    plt.figure(figsize=(8,5))
        
    # # Scatter points overlay
    # jitter, size = 0.05, 2  

    # # Add jitter for spread
    # x_base = np.random.normal(-0.3, jitter, size=n_rec_base)
    # if single_file: x_comp = 0.3  
    # else: x_comp = np.random.normal(0.3, jitter, size=n_rec_comp) 
    
    # for idx, ratio in enumerate(complexity_names):
    #     y_base = np.asarray(complexity_df_base[complexity_df_base['Complexity'] == ratio]['Value'])
    #     y_comp = np.asarray(complexity_df_comp[complexity_df_comp['Complexity'] == ratio]['Value'])

    #     # Assign color based on the category (default black)
    #     plt.scatter(x_base+idx, y_base, alpha=0.3, s=size, color='b')
    #     if single_file: plt.scatter(x_comp+idx, y_comp, alpha=1, s=size*10, marker='x', color='r')
    #     else: plt.scatter(x_comp+idx, y_comp, alpha=0.3, s=size, color='r')

    if single_file:
        sns.violinplot(x="Complexity", y="Value", data=complexity_df_base, color='b',
                   linewidth=2, fill=False,
                   inner_kws=dict(box_width=8, whis_width=2, color="0.7")
                   )
        for idx, ratio in enumerate(complexity_names):
                y_comp = np.asarray(complexity_df_comp[complexity_df_comp['Complexity'] == ratio]['Value'])
                plt.scatter(idx+0.2, y_comp, alpha=1, s=40, marker='x', color='r')
    else:
        sns.violinplot(x="Complexity", y="Value", data=complexity_df, hue='dataset', split=True, legend=None, gap=0.2,
                   linewidth=2, fill=False, palette=['b', 'r'],
                   inner_kws=dict(box_width=6, whis_width=2, color="0.7")
                   )

    plt.axhline(y=1, color='k', linestyle='--', linewidth=0.5)

    plt.xlabel("Complexity Metric")
    plt.ylabel("Value")
    plt.ylim([0, 2.3])
    plt.title(f"Complexity Comparison of\n[Base.] {n_rec_base} Recs with {n_ch_base} Chns vs.\n[Comp.] {n_rec_comp} Rec(s) with {n_ch_comp} Chns")
    
    plt.show()

    return tests

def compute_statistics(data, axis=0):
    return np.mean(data, axis), np.median(data, axis), np.std(data, axis)

# Define a helper function to plot histograms with mean and median lines
def plot_histogram_with_stats(ax, data, title, color, log_scale=False, remove_outliers=False):

    if remove_outliers:
        Q1 = np.percentile(data, 5)
        Q3 = np.percentile(data, 95)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out outliers
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        data_to_plot = filtered_data
        title = title + f'\nRemoved {data.shape[0] - data_to_plot.shape[0]} ({100 * (data.shape[0] - data_to_plot.shape[0]) / data.shape[0]:.1f}%) outliers'
    else:
        data_to_plot = data

    mean_val, median_val, std_val = compute_statistics(data_to_plot)

    sns.histplot(data_to_plot, bins=30, kde=True, color=color, edgecolor='black', alpha=0.5, ax=ax, log_scale=log_scale)
    ax.axvline(mean_val, color=color, linestyle='dashed', linewidth=1, alpha=1, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color=color, linestyle='dotted', linewidth=1, alpha=1, label=f'Median: {median_val:.2f}')
    ax.axvline(mean_val, color=color, linestyle='dashed', alpha=0, label=f'Std Dev: {std_val:.2f}')
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    # ax.legend(loc='upper right')
    ax.legend(loc='best')
    ax.grid(True)

    # Annotate standard deviation
    # ax.annotate(f'Std Dev: {std_val:.2f}', xy=(0.8, 0.8), xycoords='axes fraction',
    #             horizontalalignment='left', verticalalignment='top', fontsize=10,
    #             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5))

def normal_test(x, alpha=1):
    if alpha < 1: alpha *= 100 
    statistic, crit_vals, signi_levels = anderson(x, 'norm')
    signi_level = np.where(signi_levels == alpha)[0][0]
    crit_val = crit_vals[signi_level]
    return np.round(statistic, 5), bool(statistic < crit_val)

def compute_statistical_tests(x1, x2, metric, tests, alpha=0.01):
    """
    Compare two distributions x1 and x2.
    
    The function performs:
      - Anderson-Darling tests for normality on each distribution.
      - A t-test (using ttest_ind with equal_var=False) if both distributions are normal.
      - A Mann-Whitney U test as a non-parametric alternative.
    
    Returns a dictionary where:
      - Keys are the names of the tests.
      - Values are lists [statistic, p_value]. 
      - If the assumptions for a test are not met, returns [np.nan, np.nan].
    
    Parameters:
      x1, x2 : 1D numpy arrays representing the samples.
      metric : str name of the metric being compared.
      tests : dictionnary of all tests.
      alpha  : significance level for the normality test (default=0.05).
    
    Returns:
      dict : Results for each statistical test.
    """

    #x1, x2 = np.array(x1), np.array(x2)

    tests['metric'].append(metric)

    mean1, median1, std1 = np.round(compute_statistics(x1), 3)
    mean2, median2, std2 = np.round(compute_statistics(x2), 3)
    tests['statistics: [(mean1, median1, std1), (mean2, median2, std2)]'].append([(mean1, median1, std1), (mean2, median2, std2)])
    
    # Normality tests using Anderson-Darling test for normality. If p<alpha, then not normal distribution
    # stat1, p1 = np.round(diagnostic.normal_ad(x1), 5)
    # stat2, p2 = np.round(diagnostic.normal_ad(x2), 5)

    # stat1, p1 = np.round(kstest(x1, 'norm'), 5)
    # stat2, p2 = np.round(kstest(x2, 'norm'), 5)
    stat1, norm1 = normal_test(x1, alpha)
    stat2, norm2 = normal_test(x2, alpha)


    tests['normality: [(stat1, norm1), (stat2, norm2)]'].append([(stat1, norm1),(stat2, norm2)])
    
    # t-test: only perform if both distributions pass the normality test. If p<alpha, then difference of distributions
    if norm1 and norm2:
        t_stat, t_p = np.round(ttest_ind(x1, x2, equal_var=False), 5)
        tests['ttest: (stat, p)'].append((t_stat, t_p))
    else:
        tests['ttest: (stat, p)'].append((np.nan, np.nan))
    
    # Mann-Whitney U test: non-parametric test that does not assume normality. If p<alpha, then difference of distributions
    try:
        mw_stat, mw_p = np.round(mannwhitneyu(x1, x2, alternative='two-sided'), 5)
    except Exception:
        mw_stat, mw_p = np.nan, np.nan
    tests['mannwhitneyu: (stat, p)'].append((mw_stat, mw_p))
    
    return tests

def compute_confusion_matrix(true, pred):                                                                                                                 
    TP = np.sum(np.logical_and(pred==1, true==1))
    TN = np.sum(np.logical_and(pred==0, true==0))
    FP = np.sum(np.logical_and(pred==1, true==0))
    FN = np.sum(np.logical_and(pred==0, true==1))

    confusion_matrix = np.asarray([[TP, FP],   # 0,0    0,1
                                   [FN, TN]])  # 1,0    1,1
    return confusion_matrix

def compute_metrics(true, pred):
    acc, prec, rec, f1 = accuracy_score(y_true=true, y_pred=pred), precision_score(y_true=true, y_pred=pred), recall_score(y_true=true, y_pred=pred), f1_score(y_true=true, y_pred=pred)
    return acc, prec, rec, f1

def plot_confusion_matrix(pkl_df_path):

    df = pd.read_pickle(pkl_df_path)
    print("Displaying dataframe first rows:")
    print(df.head(3))

    y_true = df['y_true']
    y_pred = df['y_pred']

    cf_matrix = compute_confusion_matrix(y_true, y_pred)
    [[TP, FP], [FN, TN]] = cf_matrix
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)

    label = np.asarray([['TP {}'.format(TP), 'FP {}'.format(FP)],
                        ['FN {}'.format(FN), 'TN {}'.format(TN)]])
    df_cm = pd.DataFrame(cf_matrix, index=['Yes', 'No'], columns=['Positive', 'Negative'])
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.imshow(df_cm, cmap='Reds', aspect='auto', vmin=0, vmax=y_true.shape[0])
    cbar = fig.colorbar(cax)
    cbar.set_label('Counts')

    for i in range(df_cm.shape[0]):
        for j in range(df_cm.shape[1]):
            text = label[i, j]
            ax.text(j, i, text, ha='center', va='center', color='black', fontsize=12)
    ax.set_title(f'Classifier performance\nAcc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}', pad=20)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(range(df_cm.shape[1]))
    ax.set_xticklabels(df_cm.columns)
    ax.set_yticks(range(df_cm.shape[0]))
    ax.set_yticklabels(df_cm.index)
    plt.tight_layout()
    plt.show()

def plot_training_results(csv_path):
    """
    Loads and visualizes training results from a CSV file.
    Generates multiple subplots summarizing training, validation, and test metrics.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing training results.
    """

    # Load CSV
    df = pd.read_csv(csv_path)

    # Convert all numerical columns to float, coerce errors for missing values
    df = df.apply(pd.to_numeric, errors='coerce')
    train_df = df[['step', 'train_loss']].dropna(axis=0)
    val_df = df[['step', 'val_loss', 'val_acc', 'val_f1']].dropna(axis=0)

    # Create Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  
    fig.suptitle("Cross Validation Summary", fontsize=16)

    # ----- Subplot 1: Train & Validation Loss -----
    ax = axes[0]
    ax.plot(train_df['step'], train_df['train_loss'], label="Train Loss", color='blue', linestyle='-', marker='o', markersize=3)
    ax.plot(val_df['step'], val_df['val_loss'], label="Validation Loss", color='red', linestyle='--', marker='s', markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True)

    # ----- Subplot 2: Validation Accuracy & F1-score -----
    ax = axes[1]
    ax.plot(val_df['step'], val_df['val_acc'], label="Validation Accuracy", color='green', linestyle='-', marker='o', markersize=3)
    ax.plot(val_df['step'], val_df['val_f1'], label="Validation F1-score", color='purple', linestyle='--', marker='s', markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Metric Value")
    ax.set_title("Validation Accuracy & F1-score")
    ax.legend()
    ax.grid(True)
    fig.show()


def count_channels(eeg_folder):
    standard_channels = [
        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF',
        'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF',
        'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
        'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF',
        'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF',
        'EEG PZ-REF', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF',
        'EMG-REF', 'EEG 26-REF', 'EEG 27-REF', 'EEG 28-REF',
        'EEG 29-REF', 'EEG 30-REF', 'EEG T1-REF', 'EEG T2-REF',
        'PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'ECG EKG-REF',
        'EEG 31-REF', 'EEG 32-REF', 'EEG C3P-REF', 'EEG C4P-REF',
        'EEG OZ-REF', 'EEG SP1-REF', 'EEG SP2-REF', 'PULSE RATE',
        'EEG PG1-REF', 'EEG PG2-REF'
    ]

    channel_counter = Counter()
    edf_files = [f for f in os.listdir(eeg_folder) if f.endswith('.edf')]
    total_recordings = len(edf_files)
    other_channels = set()
    
    for filename in edf_files:
        file_path = os.path.join(eeg_folder, filename)
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        for ch_name in raw.info['ch_names']:
            if ch_name in standard_channels:
                channel_counter[ch_name] += 1
            else:
                channel_counter['Other'] += 1
                other_channels.add(ch_name)

    # Calculate proportions
    channel_proportions = {ch: count / total_recordings for ch, count in channel_counter.items()}

    # Sort channels by proportion
    sorted_channels = sorted(channel_proportions.items(), key=lambda item: item[1])

    channels, proportions = zip(*sorted_channels)

    # Define colors based on proportion
    t = 0.999
    colors = ['lightgreen' if prop > t else 'skyblue' for prop in proportions]

    # Plotting
    plt.figure(figsize=(6, 7))
    plt.barh(channels, proportions, color=colors)
    plt.xlabel('Proportion of Recordings')
    plt.ylabel('Channel Names')
    plt.title(f'Proportion of Each Channel Across Recordings\nGreen: >{t*100}% recordings contain this channel')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    if 'Other' in channel_counter:
        print("Channels categorized under 'Other':")
        for ch in sorted(other_channels):
            print(f"- {ch}")


def visualize_reduced_features(
    df_features,
    n_pca=50,
    sample_size_3d=2000,
    tsne_pca_dims=50,
    umap_pca_dims=50,
    tsne_params=None,
    umap_params=None
):
    """
    Improved dimensionality reduction visualization:
    - 3D PCA plot with stratified subsampling to avoid overplotting
    - t-SNE and UMAP pre-reduced via PCA for speed
    - Hexbin density plots for t-SNE and UMAP instead of raw scatter
    """
    # Default parameters
    tsne_params = tsne_params or {"n_components": 2, "init": "pca", "perplexity": 30, "max_iter": 1000, "learning_rate": "auto", "random_state": 42}
    umap_params = umap_params or {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "random_state": 42}
    
    # Separate features and labels
    df = df_features.copy()
    y = df.pop("class").values
    X = df.values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca_full = PCA(n_components=n_pca, random_state=42)
    X_pca = pca_full.fit_transform(X_scaled)

    explained = pca_full.explained_variance_ratio_
    cum_explained = np.cumsum(explained)
    actual_components = X_pca.shape[1]
    
    # 3D PCA: stratified subsampling
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n_half = min(sample_size_3d, df_features.shape[0]) // 2
    sample0 = np.random.choice(idx0, min(len(idx0), n_half), replace=False)
    sample1 = np.random.choice(idx1, min(len(idx1), n_half), replace=False)
    sample_idx = np.concatenate([sample0, sample1])

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("PCA 3D Projection and Explained Variance", fontsize=16)

    # 1) 3D PCA scatter
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    sc = ax1.scatter(
        X_pca[sample_idx, 0],
        X_pca[sample_idx, 1],
        X_pca[sample_idx, 2],
        c=y[sample_idx],
        cmap="bwr",
        alpha=0.5,
        s=10
    )
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.set_title("3D PCA (Sub-sampled)")
    plt.colorbar(sc, ax=ax1, pad=0.1, label="Class")

    # 2) Per-component explained variance
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.bar(range(1, actual_components + 1), explained[:actual_components])
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_title("Per-Component Variance")
    ax2.set_ylim(0, 1)

    # 3) Cumulative explained variance
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(range(1, actual_components + 1), cum_explained[:actual_components], marker='o')
    ax3.set_xlabel("Principal Component")
    ax3.set_ylabel("Cumulative Explained Variance")
    ax3.set_title("Cumulative Variance")
    ax3.set_ylim(0, 1)
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # t-SNE (PCA pre-reduction)
    X_tsne_input = (PCA(n_components=tsne_pca_dims, random_state=42).fit_transform(X_scaled)
                    if X_scaled.shape[1] > tsne_pca_dims else X_scaled)
    X_tsne = TSNE(**tsne_params).fit_transform(X_tsne_input)
    
    # UMAP
    X_umap_input = (PCA(n_components=umap_pca_dims, random_state=42).fit_transform(X_scaled)
                    if X_scaled.shape[1] > umap_pca_dims else X_scaled)
    X_umap = UMAP(**umap_params).fit_transform(X_umap_input)

    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(10,4), sharex=True, sharey=True)
    ax0.hexbin(X_tsne[y==0,0], X_tsne[y==0,1], gridsize=50); ax0.set_title("Class 0")
    ax1.hexbin(X_tsne[y==1,0], X_tsne[y==1,1], gridsize=50); ax1.set_title("Class 1")
    plt.suptitle("t-SNE density (Dim 1 vs. Dim 2)")
    plt.show()

    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(10,4), sharex=True, sharey=True)
    ax0.hexbin(X_umap[y==0,0], X_umap[y==0,1], gridsize=50); ax0.set_title("Class 0")
    ax1.hexbin(X_umap[y==1,0], X_umap[y==1,1], gridsize=50); ax1.set_title("Class 1")
    plt.suptitle("UMAP density (Dim 1 vs. Dim 2)")
    plt.show()


def visualize_pca(df_features, n_components=3):
    """
    Perform PCA on the features (excluding 'class') and create subplots:
      - Figure 1 (3 subplots): PC1 vs PC2, PC1 vs PC3, PC2 vs PC3
      - Figure 2 (2 subplots): explained variance per component & cumulative variance

    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame with rows = recordings, columns = features + 'class'.
        'class' holds the label (0 or 1).
    n_components : int, optional
        Number of principal components to compute (default: 3).

    Returns
    -------
    None
        Displays two figures:
          1) Three scatter plots comparing pairs of principal components.
          2) A two-subplot figure showing explained variance ratio and cumulative variance.
    """

    # Separate features (X) and target (y)
    df_copy = df_features.copy()
    y = df_copy['class'].values
    X = df_copy.drop(columns=['class']).values

    # Scale features before PCA (helps avoid dominance by variables with large numeric ranges)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    cum_explained = np.cumsum(explained)

    # Determine how many components were actually computed
    actual_components = X_pca.shape[1]

    # -----------------------------
    # Figure 1: Scatter Plot Subplots
    # -----------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(12, 4))
    fig1.suptitle('PCA Scatter Plots')

    # 1) PC1 vs PC2
    if actual_components >= 2:
        sns.kdeplot(x=X_pca[y == 0, 0], y=X_pca[y == 0, 1], ax=axes1[0], levels=5, color='blue', alpha=0.2, fill=True)
        sns.kdeplot(x=X_pca[y == 1, 0], y=X_pca[y == 1, 1], ax=axes1[0], levels=5, color='red', alpha=0.2, fill=True)
        axes1[0].scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], facecolors=(0, 0, 1, 0.1), edgecolors=(0, 0, 1, 1), label='Class 0')
        axes1[0].scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], facecolors=(1, 0, 0, 0.1), edgecolors=(1, 0, 0, 1), label='Class 1')
        axes1[0].set_xlabel('PC1')
        axes1[0].set_ylabel('PC2')
        axes1[0].set_title('PC1 vs PC2')
        axes1[0].legend()

    # 2) PC1 vs PC3
    if actual_components >= 3:
        sns.kdeplot(x=X_pca[y == 0, 0], y=X_pca[y == 0, 2], ax=axes1[1], levels=5, color='blue', alpha=0.2, fill=True)
        sns.kdeplot(x=X_pca[y == 1, 0], y=X_pca[y == 1, 2], ax=axes1[1], levels=5, color='red', alpha=0.2, fill=True)
        axes1[1].scatter(X_pca[y == 0, 0], X_pca[y == 0, 2], facecolors=(0, 0, 1, 0.1), edgecolors=(0, 0, 1, 1), label='Class 0')
        axes1[1].scatter(X_pca[y == 1, 0], X_pca[y == 1, 2], facecolors=(1, 0, 0, 0.1), edgecolors=(1, 0, 0, 1), label='Class 1')
        axes1[1].set_xlabel('PC1')
        axes1[1].set_ylabel('PC3')
        axes1[1].set_title('PC1 vs PC3')
        axes1[1].legend()

    # 3) PC2 vs PC3
    if actual_components >= 3:
        sns.kdeplot(x=X_pca[y == 0, 1], y=X_pca[y == 0, 2], ax=axes1[2], levels=5, color='blue', alpha=0.2, fill=True)
        sns.kdeplot(x=X_pca[y == 1, 1], y=X_pca[y == 1, 2], ax=axes1[2], levels=5, color='red', alpha=0.2, fill=True)
        axes1[2].scatter(X_pca[y == 0, 1], X_pca[y == 0, 2], facecolors=(0, 0, 1, 0.1), edgecolors=(0, 0, 1, 1), label='Class 0')
        axes1[2].scatter(X_pca[y == 1, 1], X_pca[y == 1, 2], facecolors=(1, 0, 0, 0.1), edgecolors=(1, 0, 0, 1), label='Class 1')
        axes1[2].set_xlabel('PC2')
        axes1[2].set_ylabel('PC3')
        axes1[2].set_title('PC2 vs PC3')
        axes1[2].legend()

    plt.tight_layout()
    plt.show()

    # ------------------------------------------
    # Figure 2: Explained Variance Subplots
    # ------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(9, 4))
    fig2.suptitle('Explained Variance')

    # Left subplot: per-component explained variance
    axes2[0].bar(range(1, actual_components + 1), explained[:actual_components])
    axes2[0].set_xlabel('Principal Components')
    axes2[0].set_ylabel('Explained Variance Ratio')
    axes2[0].set_title('Per-Component Variance')
    axes2[0].set_ylim([0,1])

    # Right subplot: cumulative explained variance
    axes2[1].plot(range(1, actual_components + 1), cum_explained[:actual_components], marker='o')
    axes2[1].set_xlabel('Principal Components')
    axes2[1].set_ylabel('Cumulative Explained Variance')
    axes2[1].grid(True)
    axes2[1].set_title('Cumulative Variance')
    axes2[1].set_ylim([0,1])

    plt.tight_layout()
    plt.show()

    # ------------------------------------------
    # Figure 3: Non linear methods
    # ------------------------------------------
    fig3, axes3 = plt.subplots(1, 2, figsize=(9, 4))
    fig3.suptitle('Non linear methods')

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    sns.kdeplot(x=X_tsne[y == 0, 0], y=X_tsne[y == 0, 1], ax=axes3[0], levels=5, color='blue', alpha=0.2, fill=True)
    sns.kdeplot(x=X_tsne[y == 1, 0], y=X_tsne[y == 1, 1], ax=axes3[0], levels=5, color='red', alpha=0.2, fill=True)
    axes3[0].scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], facecolors=(0, 0, 1, 0.1), edgecolors=(0, 0, 1, 1), label='Class 0')
    axes3[0].scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], facecolors=(1, 0, 0, 0.1), edgecolors=(1, 0, 0, 1), label='Class 1')
    axes3[0].set_xlabel('Dim1')
    axes3[0].set_ylabel('Dim2')
    axes3[0].set_title('TSNE (2D): Dim1 vs Dim2')
    axes3[0].legend()

    umap_model = UMAP(n_components=2, random_state=42)
    X_umap = umap_model.fit_transform(X_scaled)
    sns.kdeplot(x=X_umap[y == 0, 0], y=X_umap[y == 0, 1], ax=axes3[1], levels=5, color='blue', alpha=0.2, fill=True)
    sns.kdeplot(x=X_umap[y == 1, 0], y=X_umap[y == 1, 1], ax=axes3[1], levels=5, color='red', alpha=0.2, fill=True)
    axes3[1].scatter(X_umap[y == 0, 0], X_umap[y == 0, 1], facecolors=(0, 0, 1, 0.1), edgecolors=(0, 0, 1, 1), label='Class 0')
    axes3[1].scatter(X_umap[y == 1, 0], X_umap[y == 1, 1], facecolors=(1, 0, 0, 0.1), edgecolors=(1, 0, 0, 1), label='Class 1')
    axes3[1].set_xlabel('Dim1')
    axes3[1].set_ylabel('Dim2')
    axes3[1].set_title('UMAP (2D): Dim1 vs Dim2')
    axes3[1].legend()

    plt.tight_layout()
    plt.show()


def plot_cf_matrix(y_true, y_pred):
    cf_matrix = compute_confusion_matrix(y_true, y_pred)
    [[TP, FP], [FN, TN]] = cf_matrix
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)

    label = np.asarray([['TP {}'.format(TP), 'FN {}'.format(FN)],
                        ['FP {}'.format(FP), 'TN {}'.format(TN)]])
    df_cm = pd.DataFrame(cf_matrix, index=['Yes', 'No'], columns=['Positive', 'Negative'])
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.imshow(df_cm, cmap='Reds', aspect='auto', vmin=0, vmax=y_true.shape[0])
    cbar = fig.colorbar(cax)
    cbar.set_label('Counts')

    for i in range(df_cm.shape[0]):
        for j in range(df_cm.shape[1]):
            text = label[i, j]
            ax.text(j, i, text, ha='center', va='center', color='black', fontsize=12)
    ax.set_title(f'Classifier performance\nAcc: {acc:.7f} | Prec: {prec:.7f} | Rec: {rec:.7f} | F1: {f1:.7f}', pad=20)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(range(df_cm.shape[1]))
    ax.set_xticklabels(df_cm.columns)
    ax.set_yticks(range(df_cm.shape[0]))
    ax.set_yticklabels(df_cm.index)
    plt.tight_layout()
    plt.show()

def format_concept_name(name: str) -> str:
    return (name.replace("_", " ")
                .replace("rel", "relative")
                .replace("abs", "absolute")
                .replace("(alpha+beta)/(delta+theta)", "(a+b)/(d+t)")
                .replace("60", "")
                .replace("clf", "classifier")
                .replace("enc augment", "encoder augment"))

def plot_importance(importance_values, feature_names, metric_name):
    """
    Plot the feature importances as a simple bar chart.
    Each chart is in its own figure, sorted by descending importance.
    """
    # Sort features by importance
    indices = np.argsort(importance_values)[:-min(50, len(importance_values)+1):-1]
    sorted_importances = importance_values[indices]
    sorted_features = [format_concept_name(feature_names[idx]) for idx in indices]
    
    plt.figure(figsize=(7,3))
    # plt.hlines(y=0, xmin=-1, xmax=len(sorted_features), colors='grey', linestyles='-')
    plt.bar(range(len(sorted_features)), sorted_importances)
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=25, ha='right')
    plt.xlabel("Features")
    plt.ylabel(f"{metric_name} Score")
    #plt.title(f"Feature Importances ({metric_name})")
    plt.tight_layout()
    plt.show()

# def plot_permutation_importance(perm_importances, feature_names):
#     """
#     Plot the permutation importances as a bar chart, sorted by descending importance.
#     """
#     # Sort features by importance
#     indices = np.argsort(perm_importances)[::-1]
#     sorted_importances = perm_importances[indices]
#     sorted_features = [feature_names[i] for i in indices]

#     plt.figure(figsize=(15,6))
#     plt.hlines(y=0, xmin=-1, xmax=len(sorted_features), colors='grey', linestyles='-')
#     plt.bar(range(len(sorted_features)), sorted_importances)
#     plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
#     plt.xlabel("Features")
#     plt.ylabel("Permutation Importance")
#     plt.title("Permutation Importance of Features")
#     plt.tight_layout()
#     plt.show()

def plot_shap_summary(explainer, shap_values, X_sample, feature_names, all_features=None, average_by=None):
    """
    Create a SHAP summary plot to visualize the distribution of feature importances
    across all samples.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        The SHAP explainer object.
    shap_values : list of np.ndarray or np.ndarray
        The SHAP values for each class, as returned by the explainer. 
        For binary classification, shap_values might be [array_for_class0, array_for_class1].
    X_sample : np.ndarray
        Corresponding sample of features (same shape as shap_values).
    feature_names : list of str
        Names of the features in the same order as X_sample's columns.
    """
    # If binary classification, shap_values is typically a list of two arrays 
    # (one for each class). 
    # You can use shap_values[1] for the "positive" class's perspective.
    # if isinstance(shap_values, list) and len(shap_values) == 2:
    #     shap_values_to_plot = shap_values[..., 1]  
    # else:
    #     shap_values_to_plot = shap_values
    shap_mean = np.mean(np.abs(shap_values[..., 0]), axis=0)
    plot_importance(shap_mean, feature_names, 'SHAP')
    shap.summary_plot(
        shap_values, 
        features=X_sample,
        feature_names=feature_names,
        show=True
    )
    
    if (all_features is not None) & (average_by is None):
        shap_df = pd.DataFrame(0, index=[0,], columns=all_features, dtype='float')
        shap_partial_df = pd.DataFrame(shap_mean.reshape(1, -1), columns=feature_names)
        shap_df.update(shap_partial_df)
        plot_topographical_features(shap_df, is_shap=True)

    # top_indices = np.argsort(np.abs(shap_values[sample,:,0]))[-3:]
    # shap.initjs()
    # shap.force_plot(explainer.expected_value[1], shap_values[sample,top_indices,1], X_sample.iloc[sample, top_indices], matplotlib=True)


def plot_rf_cv(results, param_grid):
    params = list(param_grid.keys())
    if len(params) == 2:
        plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
        results_pt = results.pivot_table(values='mean_test_score', index='param_'+params[0], columns='param_'+params[1])
        sns.heatmap(results_pt, cmap='Reds', annot=True, fmt=".3f", linewidths=.5)
        plt.title('Grid search')
        plt.xlabel(params[1])
        plt.ylabel(params[0])
        
    else:
        for header in param_grid[params[2]]:
            plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
            results_pt = results[results['param_'+params[2]] == header].pivot_table(values='mean_test_score', index='param_'+params[0], columns='param_'+params[1])
            sns.heatmap(results_pt, cmap='Reds', annot=True, fmt=".3f", linewidths=.5)
            plt.title(f'Grid search ({params[2]} = {header})')
            plt.xlabel(params[1])
            plt.ylabel(params[0])

def plot_feature_pruning(log):
    num_features = log['features'].apply(len).values
    f1_vals, f1_tests = log['f1_val'].values, log['f1_test'].values

    plt.figure(figsize=(6,4))
    plt.plot(num_features, f1_vals, c='r', label='Validation F1')
    plt.plot(num_features, f1_tests, c='b', label='Test F1')
    plt.vlines(x=num_features[np.argmax(f1_vals)], ymin=0.49, ymax=1.01, linestyle=':', color='k', label='Best Validation F1')
    plt.xlabel('Number of features')
    plt.ylabel('F1 Score')
    plt.title('F1 Score wrt Number of Features')
    plt.ylim([0.49, 1.01])
    plt.legend()
    plt.show()


def plot_metrics_by_region(df, columns, region_type='cortical'):
    """
    Plots distributions of EEG metrics based on brain regions.

    Parameters:
    - df: pandas.DataFrame containing EEG metrics and channel names.
    - columns: list of strings representing the metrics of interest.
    - region_type: 'cortical' to group by cortical regions, 'lateralization' to group by lateralization.
    """
    if region_type not in ['cortical', 'lateralization']:
        raise ValueError("region_type must be 'cortical' or 'lateralization'")

    # Select appropriate regions based on the region_type
    if region_type == 'cortical':
        regions = ['pre-frontal', 'frontal', 'temporal', 'parietal', 'occipital', 'central']
    else:  # lateralization
        regions = ['left', 'right', 'zero']

    # Initialize a dictionary to hold data for plotting
    aggregated_data = {metric: {region: [] for region in regions} for metric in columns}



    # Iterate over each recording
    for idx, row in df.iterrows():
        ch_names = row['ch_names']
        # Iterate over each metric
        for metric in columns:
            metric_values = row[metric]
            # Map each channel to its region and append the metric value
            for region in regions:
                region_values = [value for ch_name, value in zip(ch_names, metric_values)
                                 if ch_name in TOPOGRAPHICAL_MAP[region]]
                if region_values:
                    aggregated_data[metric][region].append(np.mean(region_values))
                else:
                    aggregated_data[metric][region].append(np.nan)  # Handle cases with no channels in the region

    # Convert lists to numpy arrays for each metric and region
    for metric in columns:
        for region in regions:
            aggregated_data[metric][region] = np.array(aggregated_data[metric][region])


    num_metrics = len(columns)
    fig, axes = plt.subplots(2, (num_metrics+1)//2, figsize=(num_metrics,8), sharex=True)
    x_pos = np.random.normal(0, 0.03, size=df.shape[0])
    x_regions = np.linspace(0, 1, len(regions))
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    cmap = ListedColormap(colors)
    for ax, metric in zip(axes.flatten(), columns):
        for x_region, region, i in zip(x_regions, regions, range(len(regions))):
            ax.scatter(x_pos+x_region, aggregated_data[metric][region], alpha=0.3, s=10, color=cmap(i))
            ax.set_ylabel(metric)
    plt.tight_layout()
    plt.show()


def plot_topographical_features(biomarkers, electrodes=ELECTRODES_TUH, features=FEATURES, is_shap=False, n_cols=None, save_as=None): #, scaler_path=None):
    """
    biomarkers is either a string (path to pkl dataframe) or a dataframes.
    """

    if isinstance(biomarkers, str): biomarkers = pd.read_pickle(biomarkers)
    if is_shap: df = biomarkers
    else: df = get_biomarker_data(df_0=biomarkers, df_1=None, features=features, df_age=None, average_by=None) #.drop('class', axis=1)
    
    # if scaler_path is not None: # for visualization purpose only
    #     import pickle
    #     sc = pickle.load(open(scaler_path, 'rb'))
    #     df = pd.DataFrame(sc.transform(df), columns=df.columns)
    
    # 4) Create MNE Info & get standard 10-20 montage
    info = mne.create_info(ch_names=electrodes, sfreq=256, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # 5) Get the 3D positions of each electrode, then reduce to 2D for topomap
    ch_positions = montage.get_positions()['ch_pos']  # dict of {ch_name: (x,y,z)}
    coords_3d = np.array([ch_positions[ch] for ch in electrodes])  # shape (21, 3)
    coords_2d = coords_3d[:, :2]  # shape (21, 2), for 2D topomaps

    # 6) Prepare figure with 2 subplots (one for each feature)
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    n_features = len(features)

    if n_cols is not None:
        n_rows = int(np.ceil(n_features / n_cols))
    elif n_features < 5:
        n_cols=n_features
        n_rows=1
    else:
        n_cols = int(np.ceil(np.sqrt(n_features)))
        n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        # Extract columns for this feature
        feat_cols = [f"{feat}_{elec}" for elec in electrodes]
        avg_values = df[feat_cols].mean(axis=0).values
        if 'abs' in feat: avg_values *= 1e12 # V² --> µV²
        elif 'rel' in feat: avg_values *= 1e2 # proportion --> %
        
        if is_shap: lmin, lmax = df.iloc[0].values.min(), df.iloc[0].values.max()
        elif 'has_alpha_peak' in feat: lmin, lmax = 0, 1
        else: lmin, lmax = avg_values.min(), avg_values.max()

        # Plot topomap directly from the array of values + 2D electrode positions
        im, _ = mne.viz.plot_topomap(
            data=avg_values,
            pos=coords_2d,
            axes=axes[i],
            show=False,
            cmap='RdBu_r',
            names=electrodes,
            outlines='head',         # Draw a head outline
            image_interp='cubic', # Smooth interpolation
            vlim=(lmin, lmax)
        )

        # Optionally add a colorbar for each subplot
        # fig.colorbar(im, ax=axes[i], orientation='vertical')
        cbar = fig.colorbar(im, ax=axes[i], location='bottom', orientation='horizontal', shrink=0.5, pad=0.05)
        cbar.set_label(label=FEATURE_UNIT[feat], size=17)
        cbar.ax.tick_params(labelsize=15)
        axes[i].set_title(feat, fontsize=19, weight='bold')

    plt.tight_layout()
    
    if save_as is not None:
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        plt.savefig(save_as)
        plt.close()
    else: plt.show()
    


def compare_topographical_features(biomarkers=[], labels=[], electrodes=ELECTRODES_TUH, features=FEATURES, n_cols=None, method='mean'): #, scaler_path=None):
    """
    biomarkers is either a list of strings (path to pkl dataframes) or a list of dataframes.
    """

    df = []
    for biom in biomarkers:
        if isinstance(biom, str): biom = pd.read_pickle(biom)
        biom = get_biomarker_data(df_0=biom, df_1=None, features=features, df_age=None, average_by=None) #.drop('class', axis=1)[features]
        
        # if scaler_path is not None: # for visualization purpose only
        #     import pickle
        #     sc = pickle.load(open(scaler_path, 'rb'))
        #     biom = pd.DataFrame(sc.transform(biom), columns=biom.columns)
        
        if method == 'median': df.append(biom.median(axis=0))
        else: df.append(biom.mean(axis=0))

    df = pd.DataFrame(df, index=labels)

    if "normal" in labels and "abnormal" in labels:
        info = mne.create_info(ch_names=ELECTRODES_TUH, sfreq=256, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)

        ch_positions = montage.get_positions()['ch_pos']  # dict of {ch_name: (x,y,z)}
        coords_3d_tuh = np.array([ch_positions[ch] for ch in ELECTRODES_TUH])  # shape (21, 3)
        coords_2d_tuh = coords_3d_tuh[:, :2]  # shape (21, 2), for 2D topomaps

    info = mne.create_info(ch_names=electrodes, sfreq=256, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    ch_positions = montage.get_positions()['ch_pos']  # dict of {ch_name: (x,y,z)}
    coords_3d = np.array([ch_positions[ch] for ch in electrodes])  # shape (21, 3)
    coords_2d = coords_3d[:, :2]  # shape (21, 2), for 2D topomaps


    for feat in features:

        n_comp = df.shape[0]
        if n_cols is not None:
            n_rows = int(np.ceil(n_comp / n_cols))
        elif n_comp < 5:
            n_cols=n_comp
            n_rows=1
        else:
            n_cols = int(np.ceil(np.sqrt(n_comp)))
            n_rows = int(np.ceil(n_comp / n_cols))

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4*n_cols, 4*n_rows))
        axes = axes.flatten()

        feat_cols = [f"{feat}_{elec}" for elec in electrodes]

        if "normal" in labels and "abnormal" in labels:
            feat_cols_tuh = [f"{feat}_{elec}" for elec in ELECTRODES_TUH]
            idx = [i for i, lab in enumerate(labels) if lab in ['normal', 'abnormal']]
            min_val, max_val = df[feat_cols_tuh].iloc[idx].values.min(), df[feat_cols_tuh].iloc[idx].values.max()
        else:
            min_val, max_val = df[feat_cols].values.min(), df[feat_cols].values.max()


        for row in range(n_comp):

            # Plot topomap directly from the array of values + 2D electrode positions
            if labels[row] in ['normal', 'abnormal']:
                row_values = df[feat_cols_tuh].iloc[row].values
                im, _ = mne.viz.plot_topomap(
                    data=row_values,
                    pos=coords_2d_tuh,
                    axes=axes[row],
                    show=False,
                    cmap='RdBu_r',
                    names=ELECTRODES_TUH,
                    outlines='head',         # Draw a head outline
                    image_interp='cubic', # Smooth interpolation
                    vlim=(min_val, max_val)
                )
            else:
                row_values = df[feat_cols].iloc[row].values
                im, _ = mne.viz.plot_topomap(
                    data=row_values,
                    pos=coords_2d,
                    axes=axes[row],
                    show=False,
                    cmap='RdBu_r',
                    names=electrodes,
                    outlines='head',         # Draw a head outline
                    image_interp='cubic', # Smooth interpolation
                    vlim=(min_val, max_val)
                )

            axes[row].set_title(df.index[row])
            fig.colorbar(im, ax=axes[row], orientation='vertical')


        plt.suptitle(feat, fontsize=18)
        plt.tight_layout()
        plt.show()


def plot_remaining_feature_proportions(feature_names, features=FEATURES, electrodes=ELECTRODES_TUH):
    """
    Efficiently plot proportions of remaining features per feature type and electrode.
    Uses robust string parsing based on the assumption that electrodes contain no underscores.
    """
    # Parse feature/electrode components from feature_names
    parsed = [fn.rsplit('_', 1) for fn in feature_names if '_' in fn]
    feature_only = [f[0] for f in parsed]
    electrode_only = [f[1] for f in parsed]

    print(feature_only)
    print(electrode_only)

    # Convert to pandas Series for easy counting
    feature_series = pd.Series(feature_only)
    electrode_series = pd.Series(electrode_only)

    # Count and normalize
    feature_counts = feature_series.value_counts().reindex(features, fill_value=0) / len(electrodes)
    electrode_counts = electrode_series.value_counts().reindex(electrodes, fill_value=0) / len(features)

    # Sort
    feature_counts = feature_counts.sort_values(ascending=False)
    electrode_counts = electrode_counts.sort_values(ascending=False)

    # -------- Plotting --------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    feature_counts.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title("Proportion of Remaining Features per Feature Type")
    ax1.set_ylabel("Proportion (relative to #electrodes)")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    electrode_counts.plot(kind='bar', ax=ax2, color='salmon')
    ax2.set_title("Proportion of Remaining Features per Electrode")
    ax2.set_ylabel("Proportion (relative to #features)")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_feature_correlations(df, features=FEATURES, method='pearson', threshold=None, tick_spacing=None):
    """
    Plots a correlation matrix with feature names shown only for category delimiters.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature matrix with shape (N, 463).
    features: list, optional
        list of features of interest (to define the tick positions).
    method : str, optional
        Correlation method: 'pearson', 'spearman', or 'kendall'.
    figsize : tuple
        Figure size for the plot.
    """

    n_features = df.shape[1]
    if tick_spacing is not None: category_size = tick_spacing
    else: category_size = int(np.floor(n_features / len(features)))
    n_categories = int(np.ceil(n_features / category_size))

    # Compute correlation matrix
    corr = df.corr(method=method)

    if threshold is not None: corr = corr.where(np.abs(corr) >= threshold)

    # Generate tick labels: one per category + the last feature
    feature_names = df.columns.to_list()
    tick_positions = [i * category_size for i in range(n_categories)] #+ [n_features - 1]
    tick_labels = [feature_names[i] for i in tick_positions]

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # Plot
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, 
                mask=mask, 
                cmap='RdBu_r', 
                vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False, 
                cbar_kws={"shrink": 0.6},
                square=True,
                linewidths=0)

    # Add tick labels
    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=90, fontsize=12)
    plt.yticks(ticks=tick_positions, labels=tick_labels, rotation=0, fontsize=12)
    plt.title(f"{method.capitalize()} Correlation Matrix (Threshold: {threshold})", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_feature_network_importance(
    df,
    target_col='class',
    score_fn=mutual_info_classif,
    score_kwargs=None,
    top_n=10,
    bottom_n=10
):
    """
    Compute a univariate score for each feature, then plot the top_n
    and bottom_n features in a side-by-side bar chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        Contains your features plus target_col.
    target_col : str
        Name of the binary target column.
    score_fn : callable
        Takes (X, y, **score_kwargs) and returns array of length n_features.
        By default, mutual_info_classif.
    score_kwargs : dict or None
        Extra args for score_fn (e.g. {'discrete_features': False, 'random_state': 42}).
    top_n : int
        Number of highest-scoring features to display.
    bottom_n : int
        Number of lowest-scoring (least predictive) features to display.
    figsize : tuple
        Matplotlib figure size.
    """
    # prepare
    score_kwargs = {} if score_kwargs is None else score_kwargs
    feats = [c for c in df.columns if c != target_col]
    X = df[feats].values
    y = df[target_col].values
    
    # compute scores
    scores = score_fn(X, y, **score_kwargs)
    feat_scores = sorted(zip(feats, scores), key=lambda x: x[1], reverse=True)
    
    top_feats  = feat_scores[:top_n]
    bot_feats  = feat_scores[-bottom_n:]
    
    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,8))
    
    # Top features (highest scores)
    names, vals = zip(*reversed(top_feats))  # reverse so highest is at top
    ax1.barh(names, vals)
    ax1.set_title(f"Top {top_n} Features by Score")
    ax1.set_xlabel("Score")
    
    # Bottom features (lowest scores)
    names, vals = zip(*bot_feats)
    ax2.barh(names, vals, color='gray')
    ax2.set_title(f"Bottom {bottom_n} Features")
    ax2.set_xlabel("Score")
    
    plt.tight_layout()
    plt.show()

def plot_nonredundant_feature_summary(
    df,
    target_col='class',
    corr_method='pearson',
    corr_thresh=0.8,
    score_fn=mutual_info_classif,
    score_kwargs=None,
    n_keep=1,
    display_n=10,
):
    """
    1) Cluster features by |corr| >= corr_thresh
    2) Score all features by score_fn
    3) Pick top n_keep per cluster
    4) Plot only the first `display_n` representative features:
       - Left: barh of selected features by score
       - Right: cluster_size vs. feature_score (point size=cluster_size)
    """
    score_kwargs = {} if score_kwargs is None else score_kwargs
    feats = [c for c in df.columns if c != target_col]

    # 1) Correlation matrix & graph
    corr = df[feats].corr(method=corr_method).abs()
    G = nx.Graph()
    G.add_nodes_from(feats)
    for i, f1 in enumerate(feats):
        for j in range(i+1, len(feats)):
            if corr.iloc[i,j] >= corr_thresh:
                G.add_edge(f1, feats[j])

    # 2) Score all features
    X = df[feats].values
    y = df[target_col].values
    scores = score_fn(X, y, **score_kwargs)
    feat2score = dict(zip(feats, scores))

    # 3) For each cluster, pick top-n_keep
    summary = []
    for cluster_id, comp in enumerate(nx.connected_components(G)):
        comp_list = sorted(comp, key=lambda f: feat2score[f], reverse=True)
        size = len(comp_list)
        for f in comp_list[:n_keep]:
            summary.append({
                'cluster_id': cluster_id,
                'cluster_size': size,
                'feature': f,
                'score': feat2score[f]
            })

    summary_df = pd.DataFrame(summary).sort_values('score', ascending=False)

    # 3b) Only keep first display_n rows
    display_df = summary_df.iloc[:display_n]

    # 4) Plotting
    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(14,6))

    # Left panel: horizontal bar of selected features
    ax_bar.barh(
        display_df['feature'][::-1],  # reverse so highest is on top
        display_df['score'][::-1],
        color='C0'
    )
    ax_bar.set_title(f"Top {display_n} Representative Features")
    ax_bar.set_xlabel("Relevance Score")
    ax_bar.set_ylabel("Feature")

    # Right panel: cluster_size vs. score
    sc = ax_scatter.scatter(
        display_df['cluster_size'],
        display_df['score'],
        s=display_df['cluster_size'] * 20,
        alpha=0.6,
        edgecolor='k'
    )
    for _, row in display_df.iterrows():
        ax_scatter.text(
            row['cluster_size'],
            row['score'],
            row['feature'],
            fontsize=8,
            va='center',
            ha='right'
        )
    ax_scatter.set_title(f"Top {display_n} Clusters: Size vs. Score")
    ax_scatter.set_xlabel("Cluster Size")
    ax_scatter.set_ylabel("Representative Feature Score")

    plt.tight_layout()
    plt.show()


def analyze_sample_representation(df_train, df_concepts, type='df'):
    """
    Analyze how many times each file from df_train appears in df_concepts,
    and plot the distribution of these counts.

    Parameters
    ----------
    df_train : pd.DataFrame
        Original dataframe with all files (index is filenames).
    df_concepts : pd.DataFrame
        Dataframe containing sampled concepts (index is filenames).

    Returns
    -------
    tuple
        (total_counts, positive_counts, negative_counts) Series with counts for each file.
    """
    # Get all unique filenames from df_train
    all_filenames = df_train.index.unique()

    # Calculate total occurrences of each file in df_concepts
    total_counts = (
        df_concepts.index
        .value_counts()
        .reindex(all_filenames, fill_value=0)
    )

    # Calculate occurrences as positive examples
    positive_counts = (
        df_concepts.index
        .value_counts()
        .reindex(all_filenames, fill_value=0)
    )

    # Plot the distribution of total counts
    plt.figure(figsize=(10, 6))
    plt.hist(total_counts, bins=range(0, total_counts.max() + 2), align='left', edgecolor='black')
    plt.xlabel(f'Number of occurrences in {type}')
    plt.ylabel('Number of files')
    plt.title(f'Distribution of file occurrences in {type}')
    plt.xticks(range(0, total_counts.max() + 1))
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # # Determine the maximum count for setting bins
    # max_count = max(positive_counts.max(), negative_counts.max())

    # # Create the figure and axis
    # fig, ax = plt.subplots(figsize=(10, 6))

    # # Plot positive counts with some transparency
    # ax.hist(positive_counts, bins=range(0, max_count + 2), align='left',
    #         edgecolor='black', alpha=0.3, label='Positive', color='green', log=True)

    # # Plot negative counts with some transparency
    # ax.hist(negative_counts, bins=range(0, max_count + 2), align='left',
    #         edgecolor='black', alpha=0.3, label='Negative', color='red', log=True)

    # ax.set_xlabel('Number of occurrences')
    # ax.set_ylabel('Number of files')
    # ax.set_title('Distribution of positive and negative occurrences')
    # ax.set_xticks(range(0, max_count + 1))
    # ax.grid(axis='y', alpha=0.75)
    # ax.legend()

    # plt.show()




    # Print some statistics
    print("Statistics about file occurrences:")
    print(f"Total files: {len(all_filenames)}")
    print(f"Files never selected: {(total_counts == 0).sum()}")
    print(f"Average occurrences per file: {total_counts.mean():.2f}")
    print(f"Max occurrences: {total_counts.max()}")
    print("\nTop 10 most frequent files:")
    print(total_counts.sort_values(ascending=False).head(10))


def plot_positive_negative_relationship(df_train, df_concepts):
    """
    Plot a heatmap showing the distribution of positive and negative occurrences.

    Parameters
    ----------
    df_train : pd.DataFrame
        Original dataframe with all files in its index.
    df_concepts : pd.DataFrame
        Dataframe containing sampled concepts.

    Returns
    -------
    None
    """
    # Get all unique filenames from df_train
    all_filenames = df_train.index.unique()

    # Calculate counts
    positive_counts = (
        df_concepts[df_concepts['example_type'] == 1].index
        .value_counts()
        .reindex(all_filenames, fill_value=0)
    )

    negative_counts = (
        df_concepts[df_concepts['example_type'] == -1].index
        .value_counts()
        .reindex(all_filenames, fill_value=0)
    )

    # Create a DataFrame with all counts
    counts_df = pd.DataFrame({
        'positive': positive_counts,
        'negative': negative_counts,
        'total': positive_counts + negative_counts
    }).reset_index().rename(columns={'index': 'filename'})

    # Create a cross-tabulation of positive and negative counts
    pivot_df = counts_df.groupby(['positive', 'negative']).size().reset_index(name='count')

    # Reshape the data into a matrix format suitable for a heatmap
    heatmap_data = pivot_df.pivot(index='negative', columns='positive', values='count').fillna(0)

    # Ensure all possible positive and negative counts are represented
    max_positive = heatmap_data.columns.max() if not heatmap_data.empty else 0
    max_negative = heatmap_data.index.max() if not heatmap_data.empty else 0

    # Create a full grid with all combinations
    all_positive = range(0, int(max_positive) + 1) if max_positive > 0 else [0]
    all_negative = range(0, int(max_negative) + 1) if max_negative > 0 else [0]
    heatmap_data = heatmap_data.reindex(index=all_negative, columns=all_positive, fill_value=0)

    # Mask for zero values
    mask = heatmap_data == 0

    # Custom colormap: map 0 to white
    cmap = sns.color_palette("Blues", as_cmap=True)
    cmap.set_under('white')

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        heatmap_data,
        annot=heatmap_data,                     # Only annotate non-zero values
        fmt='.0f',
        cmap=cmap,
        mask=mask,                       # Hide zero-value cells
        cbar_kws={'label': 'Number of files'},
        vmin=0.01                        # So 0 values use 'under' color (white)
    )

    plt.title('Distribution of positive vs negative occurrences')
    plt.xlabel('Positive occurrences')
    plt.ylabel('Negative occurrences')
    plt.tight_layout()
    plt.show()


def plot_stacked_counts(df_train, df_concepts):
    """
    Plot stacked histograms showing how total counts are composed of positive and negative counts.

    Parameters
    ----------
    df_train : pd.DataFrame
        Original dataframe with all files in its index.
    df_concepts : pd.DataFrame
        Dataframe containing sampled concepts.

    Returns
    -------
    None
    """
    all_filenames = df_train.index.unique()

    positive_counts = (
        df_concepts[df_concepts['example_type'] == 1].index
        .value_counts()
        .reindex(all_filenames, fill_value=0)
    )

    negative_counts = (
        df_concepts[df_concepts['example_type'] == -1].index
        .value_counts()
        .reindex(all_filenames, fill_value=0)
    )

    total_counts = positive_counts + negative_counts

    # Create a DataFrame with all counts
    counts_df = pd.DataFrame({
        'positive': positive_counts,
        'negative': negative_counts,
        'total': total_counts
    }).reset_index()
    counts_df = counts_df.rename(columns={'index': 'filename'})

    # For each total count value, we'll look at the distribution of positive and negative counts
    # But since total = positive + negative, it's more insightful to look at the distribution of positive
    # and negative counts for each total count value.

    # Let's do it differently: for each possible positive count, see how negative counts vary
    # Or better: for each total count, see how positive and negative counts are distributed

    # Create a new column that is the total count
    counts_df['total_count'] = counts_df['total']

    # Filter out files that never appear (total=0) for clarity if desired
    counts_df = counts_df[counts_df['total_count'] > 0]

    # Create a figure with subplots for different total counts
    # First, find the unique total counts and sort them
    unique_totals = sorted(counts_df['total_count'].unique())

    # Create a grid of plots for each total count value
    n_cols = 4
    n_rows = int(np.ceil(len(unique_totals) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()

    for i, total_val in enumerate(unique_totals):
        ax = axes[i]
        subset = counts_df[counts_df['total_count'] == total_val]

        # Create a bar plot showing the distribution of positive counts for this total value
        positive_dist = subset['positive'].value_counts().sort_index()
        negative_dist = subset['negative'].value_counts().sort_index()

        # Since total = positive + negative, and total is fixed for this subset,
        # positive and negative counts must sum to total_val
        # So we can plot the distribution of positive counts (negative will be total_val - positive)
        ax.bar(positive_dist.index, positive_dist.values, alpha=0.6, label='Positive')
        ax.set_title(f'Total count = {total_val}')
        ax.set_xlabel('Positive count')
        ax.set_ylabel('Number of files')
        ax.set_xlim(0, total_val)
        ax.set_xticks(range(0, total_val + 1))

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
