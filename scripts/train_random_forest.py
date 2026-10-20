"""Train the random forest on region-averaged qEEG biomarkers with iterative SHAP-based feature pruning (Sec. III-C).

Expects the biomarker tables produced by `scripts/extract_biomarkers.py` for TUAB train/eval (normal and abnormal).
"""
import argparse
import os

from cgx.biomarkers.random_forest import iterative_feature_pruning
from cgx.biomarkers.dataset import FEATURES, get_biomarker_data, select_nonredundant_features, split_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--biomarker-dir', required=True,
                        help='Folder with train/{normal,abnormal}.pkl and eval/{normal,abnormal}.pkl')
    parser.add_argument('--save-as', required=True, help='Output .pkl for the feature pruning log')
    parser.add_argument('--corr-thresh', type=float, default=0.9)
    parser.add_argument('--k-folds', type=int, default=3)
    parser.add_argument('--discard-fraction', type=float, default=0.1)
    parser.add_argument('--min-features', type=int, default=10)
    args = parser.parse_args()

    ################# GET DATA #################

    features = FEATURES['all']
    average_by = 'region'                   # None, 'region', 'lateralization', 'global'

    df_train = get_biomarker_data(df_0=os.path.join(args.biomarker_dir, 'train', 'normal.pkl'),
                                  df_1=os.path.join(args.biomarker_dir, 'train', 'abnormal.pkl'),
                                  features=features, average_by=average_by)
    df_eval = get_biomarker_data(df_0=os.path.join(args.biomarker_dir, 'eval', 'normal.pkl'),
                                 df_1=os.path.join(args.biomarker_dir, 'eval', 'abnormal.pkl'),
                                 features=features, average_by=average_by)

    print('Train shape:', df_train.shape)

    ################# REDUCE DATA #################

    selected_features = select_nonredundant_features(
        df_train,
        target_col='class',
        corr_method='pearson',
        corr_thresh=args.corr_thresh,
        n_keep=1
    )
    print(f"Selected features (N={len(selected_features)}):", selected_features)
    selected_features.append('class')
    df_train_reduced = df_train[selected_features]
    df_eval_reduced = df_eval[selected_features]

    ################# TRAIN MODEL #################

    param_grid = {
            'n_estimators': [i for i in range(10, 300, 10)],      # number of trees
            'max_depth': [i for i in range(5, 20, 1)] + [None,],     # tree depth
            'min_samples_leaf': [2, 3, 5, 7, 10] # number of samples per leaf to prevent overfitting
    }

    X_train, X_test, y_train, y_test = split_data(df_features_train=df_train_reduced, df_features_test=df_eval_reduced)
    iterative_feature_pruning(
        X_train, y_train,
        X_test, y_test,
        feature_names=list(df_train_reduced.columns[:-1]),
        k_folds=args.k_folds,
        param_grid=param_grid,
        discard_fraction=args.discard_fraction,
        min_features=args.min_features,
        save_as=args.save_as
    )


if __name__ == '__main__':
    main()
