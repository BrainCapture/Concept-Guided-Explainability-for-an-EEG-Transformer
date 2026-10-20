"""Format TUAB / TUABEXB recordings for the 60-s pipeline (Sec. III-A).

Each recording is re-saved as `<out>/<subset>/S####/S####R##.edf` with annotations marking consecutive 60-s windows.
Scanning stops at the first window that passes the quality screen, which is annotated `<label>_60` (e.g. `abnormal_60`);
the windows before it are annotated `bad_60`. Downstream preprocessing and biomarker extraction use only the `<label>_60`
window.
"""
import argparse
import mne
import os

from tqdm import tqdm
from pathlib import Path

import shelve
from pyprep import NoisyChannels
import numpy as np
import contextlib
from joblib import Parallel, delayed

import sys
from cgx.biomarkers.dataset import read_TUH_edf, ELECTRODES_TUH



def create_output_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def evaluate_quality(raw, min_nchans=10, line_freqs=[60], hp_freq=0.1, lp_freq=80,       ### inspired by eegprep/src/preprocessing/methods.py
                        oha_threshold=40e-6, thv_threshold=40e-6, chv_threshold=80e-6,
                        min_unique=100):
    """
    Evaluates raw EEG quality using various thresholds and channel checks.

    Returns:
        Boolean indicating whether the data passes quality control.
    """
    n_chans = raw.info['nchan']
    if n_chans < min_nchans:
        return False

    
    # Basic filtering
    # raw.notch_filter(line_freqs, verbose=False)
    # raw.filter(hp_freq, lp_freq, verbose=False)

    # Check for discrete-value channels
    unique_counts = np.array([len(np.unique(ch)) for ch in raw._data])
    discrete_channels = [raw.ch_names[i] for i in np.where(unique_counts < min_unique)[0]]
    # Identify bad channels, suppressing output
    with suppress_stdout_stderr():
        noisych = NoisyChannels(raw)
        noisych.find_bad_by_SNR()   ### same as correlation + hfnoise methodes
        # noisych.find_bad_by_correlation()
        noisych.find_bad_by_deviation()
        # noisych.find_bad_by_hfnoise()
        noisych.find_bad_by_nan_flat()
    # Threshold checks
    oha = np.mean(np.abs(raw._data) > oha_threshold)
    thv = np.mean(np.std(raw._data, axis=0) > thv_threshold)
    chv = np.mean(np.std(raw._data, axis=1) > chv_threshold)

    bad_channels = discrete_channels + noisych.get_bads()
    bcr = len(bad_channels) / n_chans

    return (oha < 0.1) and (thv < 0.1) and (chv < 0.1) and (bcr < 0.1)

def add_annotations_to_raw(raw, label, window_lengths):
    
    onsets = []
    durations = []
    descriptions = []

    raw_processed = read_TUH_edf(raw.copy(), ELECTRODES_TUH, t_start=0, t_stop=None, high_pass=0.1, low_pass=80, notch=60, proj = False)       ### from cgx/biomarkers/dataset.py

    for window_length in window_lengths:
        for i in range(0, raw.n_times, window_length * int(raw.info['sfreq'])):
            if i + (window_length+1) * int(raw.info['sfreq']) > raw.n_times:
                break

            onset = i / raw.info['sfreq']
            onsets.append(onset)
            durations.append(window_length)

            quality_check = evaluate_quality(raw_processed.copy().crop(tmin=onset, tmax=onset+window_length))
            if quality_check:
                descriptions.append(label + '_' + str(window_length))
                break   ### here we only find the first good window and stop annotating
            else:
                descriptions.append('bad_' + str(window_length))

    annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    raw.set_annotations(annotations)





def add_annotations_and_save_file(subject, subject_dict, output_subset_path, folder_name, edf_file, label):
    
    if subject in subject_dict:
        subject_id = subject_dict[subject]

    if subject not in subject_dict:
        subject_id = max(list(subject_dict.values()) + [0]) + 1
        subject_dict[subject] = subject_id

    subject_identifier = 'S' + str(subject_id).zfill(4)
    directory = output_subset_path + subject_identifier + '/'
    create_output_dir(directory)

    session_id = len(os.listdir(directory)) + 1 # there is no overlap between subjects in the evaluation and training sets

    session_identifier = subject_identifier + 'R' + str(session_id).zfill(2)
    save_path = directory + session_identifier + '.edf'

    edf_path = folder_name + edf_file

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='error')

    # add_annotations_to_raw(raw, label, window_lengths=[6, 60, 600, 1200])
    add_annotations_to_raw(raw, label, window_lengths=[60])

    try:
        mne.export.export_raw(save_path, raw, fmt='edf', overwrite=True, physical_range=(-32767, 32767), verbose='error')
    except Exception as e:
        print(f'Error while saving recording {save_path}. Skipping...')
    
    subject_dict.sync()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True,
                        help='TUH edf root containing {train,eval}/{abnormal,normal}/ (e.g. tuh_eeg_abnormal/v3.0.1/edf/)')
    parser.add_argument('--output', required=True, help='Output root for formatted EDFs')
    parser.add_argument('--temp', required=True, help='Folder for resumable bookkeeping (shelve files)')
    parser.add_argument('--montage-subfolder', default='01_tcp_ar',
                        help="Subfolder below each label folder ('01_tcp_ar' for TUAB, '' for TUABEXB)")
    args = parser.parse_args()

    for subset in ['train', 'eval']:

        output_subset_path = os.path.join(args.output, subset) + '/'

        os.makedirs(args.temp, exist_ok=True)
        subject_dict = shelve.open(os.path.join(args.temp, f'subject_dict_{subset}.db'), writeback=False)
        count_dict = shelve.open(os.path.join(args.temp, f'count_dict_{subset}.db'), writeback=False)

        max_count = max(len(list(dict(count_dict).keys())), 0) + 1
        print(max_count)
        count = 1

        for label in ['abnormal', 'normal']:

            folder_name = os.path.join(args.input, subset, label, args.montage_subfolder) + '/'

            for edf_file in tqdm(os.listdir(folder_name), desc=f'Processing {subset}, {label}'):

                if count < max_count:
                    print(f'Skip file n°{count}.')
                else:
                    subject = edf_file.split('_')[0]
                    add_annotations_and_save_file(subject, subject_dict, output_subset_path, folder_name, edf_file, label)

                    count_dict[str(count)] = subject
                    count_dict.sync()
                count += 1

        count_dict.close()
        subject_dict.close()


if __name__ == "__main__":
    main()
