"""Extract the 25 channel-wise qEEG biomarkers (Table I) from the clean 60-s window of each formatted recording.

Run on the output of `format_tuh_60s_clean.py`. Recordings are split by their window annotation, so this writes
`<out-dir>/<subset>/normal.pkl` and `<out-dir>/<subset>/abnormal.pkl`.
"""
import argparse
import os

from cgx.biomarkers.dataset import ELECTRODES_TUH
from cgx.biomarkers.features import biomarker_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formatted', required=True, help='Root written by format_tuh_60s_clean.py')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--subsets', nargs='+', default=['train', 'eval'])
    parser.add_argument('--batch-size', type=int, default=500)
    args = parser.parse_args()

    for subset in args.subsets:
        for label in ['normal', 'abnormal']:
            print(f'### {subset} / {label}')
            biomarker_analysis(
                eeg_path=os.path.abspath(os.path.join(args.formatted, subset)),  # must be absolute
                electrodes=ELECTRODES_TUH,
                save_as=os.path.join(args.out_dir, subset, f'{label}.pkl'),
                t_start=None,
                t_stop=None,
                notch=60,
                plot=False,
                batch_size=args.batch_size,
                start_stop_annotation=f'{label}_60',
            )


if __name__ == '__main__':
    main()
