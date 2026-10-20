"""Combine `run_tcav.py` outputs across seeds and apply the relevance criteria (Sec. III-E/F).

For each seed directory, every `<concept>.pkl` is summarized per layer (mean TCAV score with 95% CI over the M
random counter-concepts, median linear-probe test accuracy, Mann-Whitney U p-value against the random concepts).
P-values are then Benjamini-Hochberg corrected jointly over all seeds, concepts and layers. Writes
`<out-dir>/all_concepts_<seed>_FDR.pkl` per seed and a stability table over the summarizer and extended classifier
layers.
"""
import argparse
import glob
import os
import pickle

import pandas as pd

from cgx.tcav.stats import apply_fdr, summarize_concept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-dirs', nargs='+', required=True,
                        help='One folder per model seed holding the <concept>.pkl files from run_tcav.py; '
                             'run biomarker and event concepts with the same --out-dir')
    parser.add_argument('--seed-names', nargs='+', default=None)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--q', type=float, default=0.05)
    parser.add_argument('--min-accuracy', type=float, default=0.6)
    parser.add_argument('--stability-layers', nargs='+', default=['summarizer', 'extended_classifier'])
    args = parser.parse_args()

    seed_names = args.seed_names or [os.path.basename(os.path.normpath(d)) for d in args.seed_dirs]
    tables = []
    for seed_dir in args.seed_dirs:
        rows = []
        for path in sorted(glob.glob(os.path.join(seed_dir, '*.pkl'))):  # cav/ holds Captum's cache
            with open(path, 'rb') as f:
                result = pickle.load(f)
            rows.append(summarize_concept(result, min_accuracy=args.min_accuracy))
        tables.append(pd.concat(rows, ignore_index=True))

    tables = apply_fdr(tables, q=args.q)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, table in zip(seed_names, tables):
        table.to_pickle(os.path.join(args.out_dir, f'all_concepts_{name}_FDR.pkl'))

    # Stability: share of (layer x seed) conditions meeting both criteria, and with TCAV > 0.5
    df = pd.concat([t.assign(seed=s) for s, t in zip(seed_names, tables)])
    df = df[df['layer'].isin(args.stability_layers)]
    stability = df.assign(relevant=df['significant'] & df['accurate'], positive=df['mean_score'] > 0.5) \
                  .groupby('concept')[['relevant', 'positive']].mean().sort_values('relevant', ascending=False)
    stability.to_csv(os.path.join(args.out_dir, 'stability.csv'))
    print(stability.to_string())


if __name__ == '__main__':
    main()
