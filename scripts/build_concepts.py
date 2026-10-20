"""Build biomarker-region, age, and random counter-concept sets from TUABEXB biomarkers (Sec. III-D).

Positive concepts `high_<biomarker>_<region>` sample N recordings (N/2 normal, N/2 abnormal) from the top quartile of
that feature on TUABEXB train. Random counter-concepts `random_1000 ... random_<1000+M-1>` sample N recordings each
from TUABEXB eval, which is disjoint from the positive-concept pool.
"""
import argparse
import os

from cgx.biomarkers.dataset import FEATURES, get_biomarker_data
from cgx.concepts.concept_generation import compute_positive_concepts, compute_random_concepts

REGIONS = ['pre-frontal', 'frontal', 'central', 'parietal', 'temporal', 'occipital']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--biomarker-dir', required=True,
                        help='TUABEXB biomarkers: train/{normal,abnormal}.pkl and eval/{normal,abnormal}.pkl')
    parser.add_argument('--ages', default=None,
                        help="Optional CSV with columns 'path_local' (formatted EDF path) and 'age'; enables the age concept")
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--N', type=int, default=100, help='Samples per concept')
    parser.add_argument('--M', type=int, default=100, help='Number of random counter-concepts')
    parser.add_argument('--seed', type=int, default=21)
    args = parser.parse_args()

    features = FEATURES['all']
    average_by = 'region'

    df_train = get_biomarker_data(df_0=os.path.join(args.biomarker_dir, 'train', 'normal.pkl'),
                                  df_1=os.path.join(args.biomarker_dir, 'train', 'abnormal.pkl'),
                                  df_age=args.ages, features=features, average_by=average_by)
    df_eval = get_biomarker_data(df_0=os.path.join(args.biomarker_dir, 'eval', 'normal.pkl'),
                                 df_1=os.path.join(args.biomarker_dir, 'eval', 'abnormal.pkl'),
                                 df_age=args.ages, features=features, average_by=average_by)

    compute_positive_concepts(df_train, features, REGIONS, comparison_method='high', N=args.N,
                              include_age=args.ages is not None,
                              save_as=os.path.join(args.out_dir, 'high_biomarker_region.csv'), seed=args.seed)
    compute_random_concepts(df_eval, N=args.N, M=args.M,
                            save_as=os.path.join(args.out_dir, 'random_biomarker_region.csv'))


if __name__ == '__main__':
    main()
