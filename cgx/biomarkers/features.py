"""
Pipeline for EEG PSD Analysis Using Welch’s Method
--------------------------------------------------
1. Load .edf files (assumes a folder containing ~10 files).
2. (Optional) Preprocess signals (filtering, re-referencing).
3. Compute PSD via raw.compute_psd(method='welch').
4. Extract power in standard EEG bands (delta, theta, alpha, beta, gamma) and some ratios of interest.
5. Store results in a pandas DataFrame with each file's features as one row.
"""

import os
import mne
import numpy as np
import pandas as pd 
from .visualization import plot_biomarkers
from .dataset import read_TUH_edf
from joblib import Parallel, delayed
import tqdm
from scipy.integrate import simpson
from fooof import FOOOFGroup, FOOOF
from .dataset import EEG_BANDS, RATIOS_OF_INTEREST, get_start_stop
from antropy import detrended_fluctuation, lziv_complexity, higuchi_fd, sample_entropy, perm_entropy
from time import sleep


FOOOFGROUP_MODEL = FOOOFGroup(aperiodic_mode='fixed', peak_width_limits=[1, 12], peak_threshold=1, max_n_peaks=7, verbose=False)
FOOOF_MODEL = FOOOF(aperiodic_mode='fixed', peak_width_limits=[1, 12], peak_threshold=1, max_n_peaks=7, verbose=False)


def compute_power_ratios(band_powers, roi=RATIOS_OF_INTEREST):
    """
    Given a dictionary of band powers for an EEG recording, compute
    specific ratios of interest (numerator bands / denominator bands).

    Parameters
    ----------
    band_powers : dict
        Keys are band names ('delta', 'theta', 'alpha', ...).
        Values are np.array of shape (n_channels,) giving the (relative) power
        for each channel in that band.
    roi : dict
        A dictionary specifying which ratio to compute. Each entry has:
            key (str) -> ratio name
            value -> list of two lists, e.g. [['delta'], ['alpha']]
                     meaning ratio = delta / alpha (per channel).
                     Multiple bands can be summed in numerator/denominator.

    Returns
    -------
    ratio_dict : dict
        Keys are the ratio names (e.g. 'delta/alpha').
        Values are np.array of shape (n_channels,) with the ratio for each channel.
    """
    ratios = {}

    for ratio_name, (numerator_bands, denominator_bands) in roi.items():
        # Sum the band powers across all bands in numerator
        numerator_power = 1e-25
        for band in numerator_bands:
            # if numerator_power is None:
            #     numerator_power = band_powers[band + '_abs'].copy()
            # else:
            numerator_power += band_powers[band + '_abs'].copy()

        # Sum the band powers across all bands in denominator
        denominator_power = 1e-25
        for band in denominator_bands:
            # if denominator_power is None:
            #     denominator_power = band_powers[band + '_abs'].copy()
            # else:
            denominator_power += band_powers[band + '_abs'].copy()

        # Compute ratio: (numerator / denominator) per channel
        # Watch out for zero or near-zero denominator to avoid divide-by-zero issues.
        ratio_vals = numerator_power / denominator_power
        ratios[ratio_name] = ratio_vals

    return ratios

def extract_band_powers(psd, ap_fit, freqs, bands=EEG_BANDS):
    """
    Given PSD (n_channels x n_freqs) and corresponding freqs (n_freqs,),
    integrate power within each defined band.
    Returns a dict mapping band_name -> np.array of shape (n_channels,).
    """

    psd_corrected = np.maximum(psd - 10**ap_fit, 0.0) # remove the aperiodic elements to only consider oscillatory activity
    band_powers = {}
    total_power = simpson(psd_corrected, dx=freqs[1] - freqs[0], axis=1)  # Integrate total power across all bands

    for band_name, (f_low, f_high) in bands.items():
        # Indices of freqs in the desired band
        idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        band_power = simpson(psd_corrected[:, idx], dx=freqs[1] - freqs[0], axis=1)  # Integrate over the band
        band_powers[band_name + '_abs'] = np.array(band_power)
        band_power /= (total_power + 1e-24) # Normalize by total power
        band_powers[band_name + '_rel'] = np.array(band_power)
    return band_powers

def compute_fooof(psd, freqs, single_file):
    # Extract aperiodic parameters for each channel
    aperiodic_params = {'offset': [], 'exponent': [], 'aperiodic_fit': [], 'full_fit': [],
                        'has_alpha_peak': [], 'alpha_peak_freq': [], 'alpha_peak_bw': []}
    for ch in range(psd.shape[0]):
        # Fit the model to each channel's PSD
        fm = FOOOF_MODEL
        fm.fit(freqs, psd[ch, :])
        offset, exponent = fm.aperiodic_params_
        aperiodic_params['offset'].append(offset)
        aperiodic_params['exponent'].append(exponent)
        # aperiodic_params['aperiodic_fit'].append(fm.get_model('aperiodic'))
        aperiodic_params['aperiodic_fit'].append(fm._ap_fit)
        aperiodic_params['full_fit'].append(fm.get_model('full'))

        alpha_peaks = [peak for peak in fm.peak_params_ if 7 <= peak[0] <= 13]
        if alpha_peaks:
            # Choose the most prominent (highest power)
            alpha_peak = max(alpha_peaks, key=lambda x: x[1])
            aperiodic_params['has_alpha_peak'].append(1)
            aperiodic_params['alpha_peak_freq'].append(alpha_peak[0])
            aperiodic_params['alpha_peak_bw'].append(alpha_peak[2])
        else:
            # No alpha peak found
            aperiodic_params['has_alpha_peak'].append(0)
            aperiodic_params['alpha_peak_freq'].append(0)
            aperiodic_params['alpha_peak_bw'].append(0)
    
    # Convert lists to numpy arrays
    for key in aperiodic_params:
        aperiodic_params[key] = np.array(aperiodic_params[key])

    
    
    # for visualization purposes
    if single_file:
        fg = FOOOFGROUP_MODEL
        fg.fit(freqs, psd)
        aperiodic_params['fg'] = fg
    return aperiodic_params

def compute_complexity(
    raw,
    chunk_sec=5,            # length (in seconds) of each chunk
    pe_orders=(5,),    # permutation entropy embedding dims to compute
    ):
    """
    Compute complexity metrics on an MNE Raw object using chunking.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed MNE Raw object (21 channels, 256 Hz, ~30min).
    chunk_sec : int
        Length of each chunk in seconds. Default is 5.
    pe_orders : tuple
        Embedding dimensions for Permutation Entropy. Default (5,).
    apen_m : int
        Embedding dimension 'm' for sample_entropy (SampEn).
    apen_r : float
        Tolerance 'r' for sample_entropy, typically fraction of std dev of the data.

    Returns
    -------
    pd.DataFrame
        A DataFrame (rows=channels) containing:
          [dfa, lzc, hfd, sampen, <perm_entropy_for_each_order>]
        Each value is the average over all chunks.
    """

    # -- 1) Extract the EEG data as np.array [n_channels, n_times]
    data = np.array(raw.get_data())  # shape => (n_channels, n_samples)
    sfreq = raw.info['sfreq']
    
    n_channels, n_samples = data.shape

    data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    
    # -- 2) Determine chunking indices
    chunk_size = int(sfreq * chunk_sec)
    n_chunks = n_samples // chunk_size  # integer division
    
    # Preallocate arrays to accumulate results
    # We'll store a running sum of each measure for each channel across chunks
    dfa_vals   = np.zeros(n_channels)
    lzc_vals   = np.zeros(n_channels)
    hfd_vals   = np.zeros(n_channels)
    sampen_vals= np.zeros(n_channels)
    # For permutation entropy, we store one average per embedding dimension
    pe_vals_dict = {order: np.zeros(n_channels) for order in pe_orders}
    
    # -- 3) Loop over chunks
    for i_chunk in range(n_chunks):
        start_ix = i_chunk * chunk_size
        end_ix   = start_ix + chunk_size
        
        # Extract chunk data => shape (n_channels, chunk_size)
        chunk_data = data[:, start_ix:end_ix]
        
        # For each channel, compute the measures & accumulate
        for ch_idx in range(n_channels):
            sig = chunk_data[ch_idx, :]
            
            # Detrended Fluctuation Analysis
            dfa_vals[ch_idx] += detrended_fluctuation(sig)

            # Lempel-Ziv Complexity
            # Because LZC can be sensitive to range, we can optionally
            # normalize or scale the signal. But for simplicity, we do raw:
            lzc_vals[ch_idx] += lziv_complexity(sig > 0, normalize=True)
            
            # Higuchi Fractal Dimension
            hfd_vals[ch_idx] += higuchi_fd(sig)
            
            # Sample Entropy (Note: by default AntroPy’s sample_entropy uses
            # m=2 and r=0.2 * std(signal). We can pass them as arguments too.)
            # We'll do a quick standardization so r=0.2 means "0.2 of std".
            sampen_vals[ch_idx] += sample_entropy(sig)

            # Permutation Entropy for multiple embedding orders
            # perm_entropy in AntroPy has parameters: signal, order=3, delay=1, normalize=False
            for order in pe_orders:
                # If you want normalized= True => range is [0..1], else absolute
                pe_vals_dict[order][ch_idx] += perm_entropy(
                    sig, order=order, delay=1, normalize=True
                )
    
    complexities = {
        "dfa":     dfa_vals / n_chunks,
        "lzc":     lzc_vals / n_chunks,
        "hfd":     hfd_vals / n_chunks,
        "SampEn":  sampen_vals / n_chunks,
        **{f"PermEn_{order}": pe_vals_dict[order] / n_chunks for order in pe_orders}
    }

    return complexities

def get_features(eeg_path, electrodes, single_file=False, t_start=300, t_stop=600, notch=60, start_stop_annotation=None):

    if (t_start is None) & (t_stop is None) & (isinstance(start_stop_annotation, str)):
        try:
            t_start, t_stop = get_start_stop(eeg_path, start_stop_annotation)
        except Exception as e:
            return None
    
    raw = read_TUH_edf(eeg_path, electrodes, t_start=t_start, t_stop=t_stop, high_pass=0.1, low_pass=70, notch=notch, proj = False)

    spectrum = raw.compute_psd(
            method='welch',
            fmin=1.0,
            fmax=70.0,
            n_fft=1024,
            n_overlap=512,
            window='hamming',
            verbose=False
    )
    
    psd, freqs = spectrum.get_data(return_freqs=True)
    aperiodic_params = compute_fooof(psd, freqs, single_file)
    band_powers = extract_band_powers(psd, aperiodic_params['aperiodic_fit'], freqs, EEG_BANDS)
    ratios = compute_power_ratios(band_powers, RATIOS_OF_INTEREST)
    complexities = compute_complexity(raw, chunk_sec=min(30, t_stop-t_start), pe_orders=(5,))
    # Sleep for 5 seconds to avoid memory issues
    sleep(5)
    return eeg_path, psd.shape[0], raw.info['ch_names'], freqs, psd, band_powers, ratios, aperiodic_params, complexities

def biomarker_analysis(eeg_path, electrodes, save_as=None, t_start=300, t_stop=600, notch=60, plot=True, batch_size=500, start_stop_annotation=None):

    if os.path.isfile(eeg_path):
        single_file = True
        features = [get_features(eeg_path=eeg_path, electrodes=electrodes, single_file=single_file, t_start=t_start, t_stop=t_stop, notch=notch, start_stop_annotation=start_stop_annotation)]
        df_results = format_dataframe(features)

    else:
        single_file = False
        # edf_files = [f for f in os.listdir(eeg_path) if f.lower().endswith('.edf')]
        edf_files = []
        for root, dirs, files in os.walk(eeg_path):
            for file in files:
                if file.lower().endswith('.edf'):
                    # Get the full path of the file
                    full_path = os.path.join(root, file)
                    edf_files.append(full_path)
        if not edf_files:
            print(f"No EDF files found in {eeg_path}.")
            return None
        
        batches = [edf_files[i:i + batch_size] for i in range(0, len(edf_files), batch_size)]
        for i, batch in enumerate(batches):
            if os.path.exists(f'./df_temp_{i}.pkl'):
                print(f'batch {i+1} already processed. Skipping...')
            else: 
                print(f"Processing batch {i+1}/{len(batches)}")
                features = Parallel(n_jobs=-1)(
                    delayed(get_features)(eeg_path=os.path.join(eeg_path, f), electrodes=electrodes, single_file=single_file, t_start=t_start, t_stop=t_stop, notch=notch, start_stop_annotation=start_stop_annotation) for f in tqdm.tqdm(batch, desc="Processing files...")
                )
                df_temp = format_dataframe(features)
                df_temp.to_pickle(f'./df_temp_{i}.pkl') ### temporary files that will be deleted afterwards
                del df_temp
        
        print(f"Combining and deleting {i+1} temporary files...")
        df_results = []
        for i in range(len(batches)):
            df_temp = pd.read_pickle(f'./df_temp_{i}.pkl')
            df_results.append(df_temp)
            os.remove(f'./df_temp_{i}.pkl')
        df_results = pd.concat(df_results)
    
    if save_as is not None:
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        df_results.to_pickle(save_as)
        print('Results saved at:', os.path.abspath(save_as))
    
    # ploting
    if plot: plot_biomarkers(df_results)

    return df_results

def format_dataframe(features):
    data_rows = []
    for feat in features:
        if feat is None: continue
        row_dict = {'filename': feat[0]}
        row_dict['n_channels'] = feat[1]
        row_dict['ch_names'] = feat[2]
        row_dict['freqs'] = feat[3]
        row_dict['psd'] = feat[4]
        for band_name, band_values in feat[5].items():
            row_dict[band_name] = band_values
        for ratio_name, ratio_values in feat[6].items():
            row_dict[ratio_name] = ratio_values
        for aperiodic_name, aperiodic_values in feat[7].items():
            row_dict[aperiodic_name] = aperiodic_values
        for complexity_name, complexity_values in feat[8].items():
            row_dict[complexity_name] = complexity_values
        data_rows.append(row_dict)

    # 5. Create a pandas DataFrame from the accumulated rows
    if not data_rows:
        return pd.DataFrame(index=pd.Index([], name='filename'))
    df_results = pd.DataFrame(data_rows).set_index('filename')
    return df_results