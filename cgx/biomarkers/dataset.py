import mne
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

import networkx as nx
from sklearn.feature_selection import mutual_info_classif

TUH_CHANNEL_MAP = {
    # REF
    'EEG FP1-REF': 'Fp1', 'EEG FP2-REF': 'Fp2', # Pre-frontal
    'EEG F3-REF': 'F3', 'EEG F4-REF': 'F4', 'EEG F7-REF': 'F7', 'EEG F8-REF': 'F8', 'EEG FZ-REF': 'Fz', # Frontal
    'EEG T3-REF': 'T7', 'EEG T4-REF': 'T8', # Temporal
    'EEG A1-REF': 'A1', 'EEG A2-REF': 'A2', # Ear
    'EEG P3-REF': 'P3', 'EEG P4-REF': 'P4', 'EEG P5-REF': 'P5', 'EEG P6-REF': 'P6', 'EEG T5-REF': 'P7', 'EEG T6-REF': 'P8', 'EEG PZ-REF': 'Pz', # Parietal
    'EEG O1-REF': 'O1', 'EEG O2-REF': 'O2', # Occipital
    'EEG C3-REF': 'C3', 'EEG C4-REF': 'C4', 'EEG CZ-REF': 'Cz', # Central
    # LE
    'EEG FP1-LE': 'Fp1', 'EEG FP2-LE': 'Fp2', # Pre-frontal
    'EEG F3-LE': 'F3', 'EEG F4-LE': 'F4', 'EEG F7-LE': 'F7', 'EEG F8-LE': 'F8', 'EEG FZ-LE': 'Fz', # Frontal
    'EEG T3-LE': 'T7', 'EEG T4-LE': 'T8', # Temporal
    'EEG A1-LE': 'A1', 'EEG A2-LE': 'A2', # Ear
    'EEG P3-LE': 'P3', 'EEG P4-LE': 'P4', 'EEG P5-LE': 'P5', 'EEG P6-LE': 'P6', 'EEG T5-LE': 'P7', 'EEG T6-LE': 'P8', 'EEG PZ-LE': 'Pz', # Parietal
    'EEG O1-LE': 'O1', 'EEG O2-LE': 'O2', # Occipital
    'EEG C3-LE': 'C3', 'EEG C4-LE': 'C4', 'EEG CZ-LE': 'Cz', # Central
    # BC-1
    'FP1': 'Fp1', 'FP2': 'Fp2',
    'F3': 'F3', 'F4': 'F4', 'F7': 'F7', 'F8': 'F8', 'F9': 'F9', 'F10': 'F10', 'Fz': 'Fz',
    'C3': 'C3', 'C4': 'C4', 'Cz': 'Cz',
    'P3': 'P3', 'P4': 'P4', 'P7': 'P7', 'P8': 'P8', 'P9': 'P9', 'P10': 'P10', 'Pz': 'Pz',
    'O1': 'O1', 'O2': 'O2',
    'T7': 'T7', 'T8': 'T8', 'T9': 'T9', 'T10': 'T10',
    'TP7': 'Tp7', 'TP8': 'Tp8',
    'R1': 'R1', 'R2': 'R2', 'TIP': 'TIP'
}

TOPOGRAPHICAL_MAP = {
    'pre-frontal': ['Fp1', 'Fp2'],
    'frontal': ['F3', 'F4', 'F7', 'F8', 'Fz'],
    'temporal': ['T3', 'T4', 'T7', 'T8'],
    #'ear': ['A1', 'A2'],
    'parietal': ['P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'Pz'],
    'occipital': ['O1', 'O2'],
    'central': ['C3', 'C4', 'Cz'],
    'left': ['Fp1', 'F3', 'F7', 'T3', 'T7', 'A1', 'P3', 'O1', 'C3'],
    'right': ['Fp2', 'F4', 'F8', 'T4', 'T8', 'A2', 'P4', 'O2', 'C4'],
    'zero': ['Fz', 'Pz', 'Cz']
}

# Standard EEG bands
EEG_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 70)
}

RATIOS_OF_INTEREST = {
    'delta/alpha': [['delta'],['alpha']], # stroke
    'theta/beta': [['theta'],['beta']], # ADHD and attentional processing, reflects aperiodic activity
    'delta/beta': [['delta'],['beta']], # sleep quality
    'theta/gamma': [['theta'],['gamma']], # learning
    '(alpha+beta)/(delta+theta)': [['alpha','beta'],['delta','theta']], # cognitive impairment and Alzheimer's disease
}

# ELECTRODES_TUH = [
#     'Fp1','Fp2','F3','F4','F7','F8','Fz','C3','C4','Cz','P3','P4','Pz','O1','O2','T3','T4','T5','T6','A1','A2' # 21 electrodes
# ]

ELECTRODES_TUH = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2' # 19 electrodes
]

ELECTRODES_BC = [
    'Fp1','Fp2','F3','F4','F7','F8','Fz','C3','C4','Cz','P3','P4','P7','P8','Pz','O1','O2','T7','T8'           # 19 electrodes
]

FEATURES = {
    'all':[
        'delta_abs', 'theta_abs', 'alpha_abs', 'beta_abs', 'gamma_abs',                         # absolute band powers (periodic components)
        'delta_rel', 'theta_rel', 'alpha_rel', 'beta_rel', 'gamma_rel',                         # relative band powers (periodic components)
        'delta/alpha', 'theta/beta', 'delta/beta', 'theta/gamma', '(alpha+beta)/(delta+theta)', # ratios of band powers
        'has_alpha_peak', 'alpha_peak_freq', 'alpha_peak_bw',                                   # alpha peak charateristics 
        'offset', 'exponent',                                                                   # aperiodic components
        'dfa', 'lzc', 'hfd', 'SampEn', 'PermEn_5'],                                             # complexity measures
    'RF':[
        'delta_rel', 'theta_rel', 'alpha_rel', 'beta_rel', 'gamma_rel',                         # relative band powers (periodic components)
        'delta/alpha', 'theta/beta', 'delta/beta', 'theta/gamma', '(alpha+beta)/(delta+theta)', # ratios of band powers
        'has_alpha_peak', 'alpha_peak_freq', 'alpha_peak_bw',                                   # alpha peak charateristics 
        'offset', 'exponent',                                                                   # aperiodic components
        'dfa', 'lzc', 'hfd', 'SampEn', 'PermEn_5'],                                             # complexity measures
    'qEEG_power':[
        'delta_abs', 'delta_rel', 'delta/alpha',                         
        'theta_abs', 'theta_rel', 'theta/beta',                       
        'alpha_abs', 'alpha_rel', 'delta/beta',
        'beta_abs', 'beta_rel', 'theta/gamma',
        'gamma_abs', 'gamma_rel', '(alpha+beta)/(delta+theta)'],
    'qEEG_other':[
        'has_alpha_peak', 'dfa',
        'alpha_peak_freq', 'lzc',
        'alpha_peak_bw', 'hfd',                              
        'offset', 'SampEn',
        'exponent', 'PermEn_5'],  
    }

FEATURE_DESC_EXT = [
    ("delta_abs", "Absolute power in the delta band (1-4 Hz); high amplitudes dominate deep sleep and indicate cortical slowing."),
    ("theta_abs", "Absolute power in the theta band (4-8 Hz); linked to drowsiness, memory encoding, and cognitive control."),
    ("alpha_abs", "Absolute power in the alpha band (8-13 Hz); prominent during relaxed wakefulness with eyes closed."),
    ("beta_abs", "Absolute power in the beta band (13-30 Hz); associated with active thinking, alertness, and motor activity."),
    ("gamma_abs", "Absolute power in the gamma band (30-50 Hz); involved in attention and perceptual binding."),
    ("delta_rel", "Relative power in the delta band (1-4 Hz); high amplitudes dominate deep sleep and indicate cortical slowing."),
    ("theta_rel", "Relative power in the theta band (4-8 Hz); linked to drowsiness, memory encoding, and cognitive control."),
    ("alpha_rel", "Relative power in the alpha band (8-13 Hz); prominent during relaxed wakefulness with eyes closed."),
    ("beta_rel", "Relative power in the beta band (13-30 Hz); associated with active thinking, alertness, and motor activity."),
    ("gamma_rel", "Relative power in the gamma band (30-50 Hz); involved in attention and perceptual binding."),
    ("delta/alpha", "Ratio of delta to alpha power; elevated values mark cortical slowing, e.g., after stroke."),
    ("theta/beta", "Ratio of theta to beta power; classic EEG index of attentional control—often elevated in ADHD."),
    ("delta/beta", "Ratio of delta to beta power; reflects arousal balance—higher in cognitive or affective dysregulation."),
    ("theta/gamma", "Ratio of theta to gamma power; growing marker for memory decline and concussion sensitivity."),
    ("(alpha+beta)/(delta+theta)", "Fast-over-slow power ratio: higher values signal alertness, lower values signal drowsiness or ischemia."),
    ("has_alpha_peak", "Binary flag (1/0) indicating whether a narrowband alpha peak (8-13 Hz) is detected by FOOOF."),
    ("alpha_peak_freq", "Central (peak) frequency of the detected alpha oscillation (individual alpha frequency, IAF)."),
    ("alpha_peak_bw", "Bandwidth (Hz) of the detected alpha peak, reflecting how sharp or broad the oscillation is."),
    ("offset", "FOOOF aperiodic offset (intercept) of the 1/f background; relates to broadband power level."),
    ("exponent", "FOOOF aperiodic exponent (slope) of the 1/f background; steeper (more negative) exponents indicate stronger low-frequency dominance."),
    ("dfa", "Detrended Fluctuation Analysis scaling exponent capturing long-range temporal correlations in EEG amplitude fluctuations."),
    ("lzc", "Lempel-Ziv Complexity measuring algorithmic complexity (novelty) of the EEG signal sequence."),
    ("hfd", "Higuchi Fractal Dimension estimating the fractal dimensionality (self-similarity) of the EEG time series."),
    ("SampEn", "Sample Entropy quantifying signal irregularity—lower values mean more regular, predictable patterns."),
    ("PermEn_5", "Permutation Entropy (order = 5) gauging complexity via ordinal pattern diversity across the EEG.")
]

# FEATURE_DESC = [
#     ("delta_abs", "Absolute power in the delta band (1-4 Hz)."),
#     ("theta_abs", "Absolute power in the theta band (4-8 Hz)."),
#     ("alpha_abs", "Absolute power in the alpha band (8-13 Hz)."),
#     ("beta_abs", "Absolute power in the beta band (13-30 Hz)."),
#     ("gamma_abs", "Absolute power in the gamma band (30-50 Hz)."),
#     ("delta_rel", "Relative power in the delta band (1-4 Hz)."),
#     ("theta_rel", "Relative power in the theta band (4-8 Hz)."),
#     ("alpha_rel", "Relative power in the alpha band (8-13 Hz)."),
#     ("beta_rel", "Relative power in the beta band (13-30 Hz)."),
#     ("gamma_rel", "Relative power in the gamma band (30-50 Hz)."),
#     ("delta/alpha", "Ratio of delta to alpha power."),
#     ("theta/beta", "Ratio of theta to beta power."),
#     ("delta/beta", "Ratio of delta to beta power."),
#     ("theta/gamma", "Ratio of theta to gamma power."),
#     ("(alpha+beta)/(delta+theta)", "Fast-over-slow power ratio."),
#     ("has_alpha_peak", "Binary flag (1/0) indicating whether an alpha peak is detected."),
#     ("alpha_peak_freq", "Central (peak) frequency of the detected alpha oscillation."),
#     ("alpha_peak_bw", "Bandwidth (Hz) of the detected alpha peak."),
#     ("offset", "FOOOF aperiodic offset (intercept) of the 1/f background."),
#     ("exponent", "FOOOF aperiodic exponent (slope) of the 1/f background."),
#     ("dfa", "Detrended Fluctuation Analysis scaling exponent (temporal correlations)."),
#     ("lzc", "Lempel-Ziv Complexity (algorithmic complexity, novelty)."),
#     ("hfd", "Higuchi Fractal Dimension (self-similarity)."),
#     ("SampEn", "Sample Entropy (predictability)."),
#     ("PermEn_5", "Permutation Entropy of order 5 (pattern diversity).")
# ]

FEATURE_DESC = [
    # ────────── Power (absolute & relative) ──────────
    ("delta_abs",  "µV²",          "Absolute power in the delta band (1 - 4 Hz)."),
    ("theta_abs",  "µV²",          "Absolute power in the theta band (4 - 8 Hz)."),
    ("alpha_abs",  "µV²",          "Absolute power in the alpha band (8 - 13 Hz)."),
    ("beta_abs",   "µV²",          "Absolute power in the beta  band (13 - 30 Hz)."),
    ("gamma_abs",  "µV²",          "Absolute power in the gamma band (30 - 50 Hz)."),

    ("delta_rel",  "%",      "Relative power in the delta band (1 - 4 Hz)."),
    ("theta_rel",  "%",      "Relative power in the theta band (4 - 8 Hz)."),
    ("alpha_rel",  "%",      "Relative power in the alpha band (8 - 13 Hz)."),
    ("beta_rel",   "%",      "Relative power in the beta band (13 - 30 Hz)."),
    ("gamma_rel",  "%",      "Relative power in the gamma band (30 - 50 Hz)."),

    # ────────── Band-power ratios (unit-less) ────────
    ("delta/alpha",               "", "Ratio of delta to alpha power."),
    ("theta/beta",                "", "Ratio of theta to beta  power."),
    ("delta/beta",                "", "Ratio of delta to beta  power."),
    ("theta/gamma",               "", "Ratio of theta to gamma power."),
    ("(alpha+beta)/(delta+theta)","", "Fast-over-slow power ratio."),

    # ────────── Alpha-peak characteristics ───────────
    ("has_alpha_peak", "",        "Binary flag (1/0) indicating whether an alpha peak is detected."),
    ("alpha_peak_freq", "Hz",     "Central (peak) frequency of the detected alpha oscillation."),
    ("alpha_peak_bw",   "Hz",     "Bandwidth (Hz) of the detected alpha peak."),

    # ────────── 1/f aperiodic background (FOOOF) ─────
    ("offset",   "log(µV²/Hz)",   "FOOOF aperiodic offset (intercept) of the 1/f background."),
    ("exponent", "",              "FOOOF aperiodic exponent (slope) of the 1/f background."),

    # ────────── Complexity & entropy metrics ─────────
    ("dfa",      "", "Detrended Fluctuation Analysis scaling exponent (temporal correlations)."),
    ("lzc",      "", "Lempel-Ziv Complexity (algorithmic novelty)."),
    ("hfd",      "", "Higuchi Fractal Dimension (self-similarity)."),
    ("SampEn",   "", "Sample Entropy (predictability)."),
    ("PermEn_5", "", "Permutation Entropy of order 5 (pattern diversity).")
]

FEATURE_UNIT = {name: unit for name, unit, _ in FEATURE_DESC}


def read_TUH_edf(file_path, electrodes, t_start=0, t_stop=None, high_pass=0.1, low_pass=100.0, notch=60.0, resampling_freq=256, proj = False):
    
    # Read the EDF file
    if isinstance(file_path, str):
        with warnings.catch_warnings(): #some edf files are anotated, prevent warnings while loading
            warnings.filterwarnings("ignore", message="Limited .* annotation")
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    else: raw = file_path

    # Filter the channel_map to include only the channels present in the raw data
    channel_map_sub = {k: v for k, v in TUH_CHANNEL_MAP.items() if (k in raw.ch_names) & (v in electrodes)}

    # Standardize the raw data
    # mne.datasets.eegbci.standardize(raw)
    
    # Rename the channels using the filtered channel_map
    raw = raw.rename_channels(channel_map_sub)

    # Create the standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')

    # Set the montage for the raw data, ignoring missing channels
    raw = raw.set_montage(montage, on_missing='ignore', verbose=False)
    
    # Pick only channels present in the filtered channel_map
    #raw = raw.pick_channels(list(channel_map_sub.values()))
    raw = raw.pick(list(channel_map_sub.values()))

    # Set the average reference for the raw data
    raw = raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)

    # Apply the average reference projection
    if proj:
        raw.apply_proj(verbose=False)

    # Resample the raw data to 256 Hz
    if raw.info['sfreq'] != resampling_freq:
        raw = raw.resample(resampling_freq)
    
    # Filter the data
    raw = raw.filter(high_pass, low_pass, fir_design='firwin', verbose=False)
    
    # Notch filter at 60 Hz
    raw = raw.notch_filter(notch, fir_design='firwin', verbose=False)

    missing_channels = [c for c in ELECTRODES_TUH if c not in channel_map_sub.values()]
    if missing_channels:
        raw = add_and_interpolate(raw, missing_channels, montage)

    # remove the first minute of recording to avoid artifacts
    raw = raw.crop(tmin=t_start, tmax=t_stop, verbose=False)

    return raw

def add_and_interpolate(raw, missing_channels, montage):
    # 1) add dummy channels filled with NaNs (so they stand out)
    n_times = raw.n_times
    data_nan = np.full((len(missing_channels), n_times), np.nan)
    info_nan = mne.create_info(missing_channels, sfreq=raw.info['sfreq'],
                               ch_types='eeg', verbose=False)
    raw_nan = mne.io.RawArray(data_nan, info_nan, verbose=False)

    # 2) merge into your existing Raw, set the montage
    raw.add_channels([raw_nan], force_update_info=True)
    raw.set_montage(montage, verbose=False)

    # 3) tell MNE which channels to interpolate…
    raw.info['bads'] = missing_channels

    # 4) …and run the spherical‐spline interpolation
    raw.interpolate_bads(reset_bads=True, mode='accurate', verbose=False)
    return raw


def get_biomarker_data(df_0, df_1=None, features=FEATURES, df_age=None, average_by=None):
    """
    Build a single DataFrame that stacks df_0 (class 0) and df_1 (class 1).
    Each row corresponds to one recording and the final column is y.
    
    Parameters
    ----------
    df_0 : pd.DataFrame
        DataFrame containing rows (recordings) of class 0.
        Each feature column is expected to be an array of shape (21,) (one per channel).
    df_1 : pd.DataFrame or None
        DataFrame containing rows (recordings) of class 1.
    features : list of str
        List of feature column names to extract (e.g. ['alpha', 'beta', 'gamma']).
    df_age : pd.Dataframe
        Dataframe containing the path to the files and the corresponding subject age.
    average_by : {None, 'region', 'lateralization', 'global'}, optional
        Method for averaging across channels.
        
    Returns
    -------
    df_out : pd.DataFrame
        A DataFrame where the first columns are the extracted features (optionally averaged)
        and the last column is the class label (y).
    """

    if isinstance(df_0, str): df_0 = pd.read_pickle(df_0)

    # Merge the two dataframes and create labels
    if df_1 is not None:
        if isinstance(df_1, str): df_1 = pd.read_pickle(df_1)
        df_all = pd.concat([df_0, df_1], axis=0, ignore_index=False)
        y = [0]*len(df_0) + [1]*len(df_1)
    else:
        df_all = df_0.copy()
        y = [0]*len(df_0)

    if df_age is not None:
        if isinstance(df_age, str): df_age = pd.read_csv(df_age, index_col=0)
        df_all_reset = df_all.reset_index().rename(columns={'index': 'filename'})
        df_all_merged = df_all_reset.merge(df_age[['path_local', 'age']], left_on='filename', right_on='path_local', how='left')
        df_all = df_all_merged.drop(columns='path_local').set_index('filename')

    #utility to average channels given a row's data
    def average_channels(values, ch_names, mode):
        """
        Given a 1D array of length 21 (values) and the channel list (ch_names),
        average them according to the specified mode.
        """
        if mode is None:
            # Return {channel_name: value} for each channel, with no averaging
            return dict(zip(ch_names, values))
        
        elif mode == 'region':
            # Average channels within each of the 7 main regions
            region_keys = ['pre-frontal','frontal','temporal','parietal','occipital','central']
            avg_dict = {}
            for reg in region_keys:
                ch_list = TOPOGRAPHICAL_MAP[reg]
                vals = [values[ch_names.index(ch)] for ch in ch_list if ch in ch_names]
                avg_dict[reg] = np.mean(vals) if len(vals) > 0 else np.nan
            return avg_dict
        
        elif mode == 'lateralization':
            # Average channels for left, right, zero
            side_keys = ['left', 'right', 'zero']
            avg_dict = {}
            for side in side_keys:
                ch_list = TOPOGRAPHICAL_MAP[side]
                vals = [values[ch_names.index(ch)] for ch in ch_list if ch in ch_names]
                avg_dict[side] = np.mean(vals) if len(vals) > 0 else np.nan
            return avg_dict
        
        elif mode == 'global':
            # Single global average
            return {'global': np.mean(values)}
    
    # ------------------------------------------------------
    # 2) Build rows (each row is a dict of {colName: value})
    # ------------------------------------------------------
    data_rows = []
    
    for idx, row in df_all.iterrows():
        ch_names = row['ch_names']  # e.g. ['Fp1','Fp2','F3',...,'O2']
        
        row_dict = {'filename': idx}
        # For each feature, retrieve the 21-value array and compute the averaging
        for feat in features:
            values = np.array(row[feat])  # shape (21,)
            feat_avg = average_channels(values, ch_names, average_by)
            
            # feat_avg is now a dict { key : value }, where key is either
            # channel names (None), region names, side, or 'global'
            for sub_name, val in feat_avg.items():
                col_name = f"{feat}_{sub_name}"
                row_dict[col_name] = val
        
        if df_age is not None: row_dict['age'] = row['age']

        data_rows.append(row_dict)
    
    # Create a DataFrame from these rows
    df_features = pd.DataFrame(data_rows)
    
    # Insert class labels as the last column
    df_features['class'] = y
    
    # Shuffle so that 0s and 1s are interspersed
    df_features = shuffle(df_features, random_state=42).set_index('filename')




    return df_features

def select_nonredundant_features(df, target_col='class', corr_method='pearson', corr_thresh=0.95, n_keep=1):
    """
    Returns a list of representative features, one (or n_keep) per correlated cluster.
    
    Parameters
    ----------
    df : DataFrame
        Must contain features + target_col.
    target_col : str
        Name of the binary target column.
    corr_method : str
        'pearson', 'spearman', etc.
    corr_thresh : float
        Absolute correlation threshold to consider an edge.
    n_keep : int
        Number of features to keep per cluster (usually 1).
    """
    # 1) Correlation matrix
    feats = df.columns.drop(target_col)
    corr = df[feats].corr(method=corr_method).abs()
    
    # 2) Build graph
    G = nx.Graph()
    G.add_nodes_from(feats)
    # add edges above threshold
    for i in range(len(feats)):
        for j in range(i+1, len(feats)):
            if corr.iloc[i,j] >= corr_thresh:
                G.add_edge(feats[i], feats[j])
    
    # 3) Score each feature by relevance to y
    X = df[feats].values
    y = df[target_col].values
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    score = dict(zip(feats, mi))
    
    # 4) For each cluster, pick top-scoring feature(s)
    selected_features = []
    for comp in nx.connected_components(G):
        # sort by score descending
        ranked = sorted(comp, key=lambda f: score[f], reverse=True)
        selected_features.extend(ranked[:n_keep])
    
    return selected_features

def split_data(df_features_train, df_features_test=None, test_size=0.2, random_state=42):
    """
    1) Separate features (X) and target (y) from df_features.
    2) Split into train/test sets (80/20 by default).
    3) Standardize the feature data.
    
    Parameters
    ----------
    df_features_train : pd.DataFrame
        DataFrame with rows = recordings, columns = features + 'class'.
        'class' holds the label (0 or 1).
    df_features_test : pd.DataFrame, optional
        DataFrame with rows = recordings, columns = features + 'class'.
        'class' holds the label (0 or 1).
        If None, then the test set will be extracted from the train set using the test_size.
    test_size : float, optional
        Proportion of the dataset to include in the test split (default=0.2).
    random_state : int, optional
        Seed used by the random number generator (default=42).

    Returns
    -------
    X_train : np.ndarray
        Training feature matrix after standardization.
    X_test : np.ndarray
        Test feature matrix after standardization.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    """

    # 1) Separate features (X) and target (y)
    df_copy = df_features_train.copy()
    y_train = df_copy['class'].values
    X_train = df_copy.drop(columns=['class']).values  # all columns except 'class'
    
    # 2) Split into train and test sets
    if df_features_test is not None:
        df_copy = df_features_test.copy()
        y_test = df_copy['class'].values
        X_test = df_copy.drop(columns=['class']).values  # all columns except 'class'
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train,
            test_size=test_size,
            random_state=random_state,
            stratify=y_train  # Keep class distribution consistent between train/test
        )
    
    # 3) Standardize (scale) feature data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def get_start_stop(eeg_path, annotation):
    """
    Get the start and stop times of the the segment of interest.
    """

    with warnings.catch_warnings(): #some edf files are anotated, prevent warnings while loading
        warnings.filterwarnings("ignore", message="Limited .* annotation")
        raw = mne.io.read_raw_edf(eeg_path, preload=False, verbose=False)
    df_annot = raw.annotations.to_data_frame()
    df_annot['t_start'] = (df_annot['onset'] - df_annot['onset'].min()).dt.total_seconds()
    df_annot['t_stop'] = (df_annot['t_start'] + df_annot['duration'])

    t_start = df_annot[df_annot['description'] == annotation]['t_start'].values[0]
    t_stop = df_annot[df_annot['description'] == annotation]['t_stop'].values[0]

    return t_start, t_stop