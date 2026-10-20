import os
import numpy as np
import pandas as pd 
# from .dataset import read_TUH_edf, FEATURES, TOPOGRAPHICAL_MAP
# from joblib import Parallel, delayed
# import tqdm

# from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import shap
from sklearn.inspection import permutation_importance


from .visualization import plot_cf_matrix, plot_rf_cv, plot_feature_pruning


def train_random_forest(X_train, y_train, X_test, y_test, k_folds, param_grid, plot=False):
    """
    Train and tune a Random Forest classifier with basic hyperparameter ranges
    (n_estimators, max_depth, max_features) using GridSearchCV with k-folds.
    Evaluate the best model on the test set (accuracy, F1).

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test labels.
    k_folds: int
        Number of cross validation folds.
    param_grid: dict
        Parameter grid for grid search algorithm.
    plot: Bool
        Plot some plots.

    Returns
    -------
    best_model : RandomForestClassifier
        The best performing Random Forest model found by GridSearchCV.
    best_params : dict
        The best hyperparameter combination.
    accuracy : float
        Accuracy on the test set of the best_model.
    f1 : float
        F1 score on the test set of the best_model.
    """

    rf = RandomForestClassifier(random_state=42)

    # GridSearchCV with k-fold cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=k_folds,
        scoring='f1',   # or 'f1', 'roc_auc', etc., depending on your needs
        n_jobs=-1,             # use all available CPU cores
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    f1_val = grid_search.best_score_

    # check for overfitting
    y_pred_train = best_model.predict(X_train)
    f1_train = f1_score(y_train, y_pred_train, average='binary')

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    f1_test = f1_score(y_test, y_pred, average='binary')  # for binary classification

    results = grid_search.cv_results_  # a dict with keys like 'params', 'mean_test_score', etc.
    results = pd.DataFrame(results)#.sort_values('mean_test_score', ascending=False)
    results['param_max_depth'] = results['param_max_depth'].infer_objects(copy=False).fillna(np.inf)
    
    if plot:
        # plot_rf_cv(results, param_grid)
        plot_cf_matrix(y_test, y_pred)

    print('Best model:')
    print(f'### Parameters: {best_params}\n### F1 Train: {f1_train:.3f}\n### F1 Cross Validation: {f1_val:.3f}\n### F1 Test: {f1_test:.3f}')

    return best_model, best_params, f1_val, f1_test, y_pred

def compute_mdi_importance(model, feature_names):
    """
    Retrieve Mean Decrease in Impurity (MDI) feature importances from a trained RandomForest model.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained Random Forest model (e.g., best_model from GridSearchCV).
    feature_names : list of str
        Names of the features in the same order as the training data.
    
    Returns
    -------
    mdi_importances : np.ndarray
        Array of importances for each feature.
    """
    mdi_importances = model.feature_importances_
    return mdi_importances

def compute_permutation_importance(model, X_test, y_test, feature_names, scoring='f1'):
    """
    Compute permutation importance for each feature using the test set.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained Random Forest model.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    feature_names : list of str
        Names of the features in the same order as X_test.
    scoring : str, optional
        Metric for evaluating performance drop (e.g., 'accuracy', 'f1', etc.).

    Returns
    -------
    perm_importances : np.ndarray
        The mean decrease in performance for each feature.
    """
    result = permutation_importance(model, X_test, y_test, scoring=scoring, n_repeats=5, random_state=42)
    return result.importances_mean  # average importance across repeats

def compute_shap_values(model, X_sample):
    """
    Compute SHAP values using a TreeExplainer on a given sample of data.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained Random Forest model.
    X_sample : np.ndarray
        Sample of features for which to compute SHAP values.
        Typically, this could be your X_test or a subset of X_test.

    Returns
    -------
    explainer : shap.TreeExplainer
        The SHAP explainer object (useful for further plotting).
    shap_values : np.ndarray
        SHAP values for each sample/feature.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime

def iterative_feature_pruning(
    X_train, y_train,
    X_test, y_test,
    feature_names,
    k_folds,
    param_grid,
    discard_fraction=0.1,
    min_features=10,
    save_as=None
):
    """
    Iteratively prunes features based on SHAP values. Logs all outputs to a file and console.
    """

    # ----------------------------- Logging Setup -----------------------------
    log_file_path = os.path.splitext(save_as)[0] + "_log.txt" if save_as else "feature_pruning.log"
    
    logger = logging.getLogger("FeaturePruningLogger")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # reset in case of reuse
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    # ------------------------------------------------------------------------

    logger.info("Starting Iterative Feature Pruning")
    logger.info("Total initial features: %d", len(feature_names))
    logger.info("Minimum features to retain: %d", min_features)

    # Initialize
    current_features = feature_names.copy()
    best_f1 = -1.0
    best_subset = current_features.copy()
    best_params = None

    # Estimate total iterations
    tot_iter = 0
    num_features = len(feature_names)
    while num_features >= min_features:
        num_features = num_features - max(5, int(num_features * discard_fraction))
        tot_iter += 1

    iter = 0
    num_features, f1_vals, f1_tests = [], [], []
    log_data = {'k_folds': [], 'params': [], 'features': []}

    while len(current_features) > min_features:
        iter += 1
        logger.info("\n#########################################   Iteration %d/%d", iter, tot_iter)
        logger.info("Number of features: %d", len(current_features))

        indices = [feature_names.index(f) for f in current_features]
        X_train_sub = X_train[:, indices]
        X_test_sub = X_test[:, indices]

        best_model, best_model_params, f1_val, f1_test, y_pred = train_random_forest(
            X_train_sub, y_train, X_test_sub, y_test, k_folds, param_grid
        )

        num_features.append(len(current_features))
        f1_vals.append(f1_val)
        f1_tests.append(f1_test)

        log_data['k_folds'].append(k_folds)
        log_data['params'].append({key: [value] for key, value in best_model_params.items()})
        log_data['features'].append(current_features.copy())

        logger.info("Validation F1: %.4f", f1_val)
        logger.info("Test F1: %.4f", f1_test)
        logger.info("Best Params: %s", best_model_params)

        if f1_val > best_f1:
            best_f1 = f1_val
            best_subset = current_features.copy()
            best_params = best_model_params

        # SHAP-based importance
        _, shap_values = compute_shap_values(best_model, pd.DataFrame(data=X_test_sub, columns=current_features))
        shap_values = np.mean(np.abs(shap_values[..., 0]), axis=0)

        order = np.argsort(shap_values)[::-1]
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(shap_values) + 1)

        feat_order = np.argsort(rank)
        n_to_drop = max(5, int(discard_fraction * len(current_features)))
        n_keep = len(feat_order) - n_to_drop

        if n_keep <= min_features:
            logger.info("Stopping to avoid dropping below minimum feature threshold (%d)", min_features)
            break

        discard_indices = feat_order[n_keep:]
        discard_names = [current_features[i] for i in discard_indices]
        logger.info("Discarding features: %s", discard_names)

        keep_indices = feat_order[:n_keep]
        current_features = [current_features[i] for i in keep_indices]

    log_data['f1_val'] = f1_vals
    log_data['f1_test'] = f1_tests
    log_df = pd.DataFrame(log_data)

    plot_feature_pruning(log_df)

    if save_as is not None:
        if os.path.exists(save_as):
            logger.info("Appending to existing log at: %s", os.path.abspath(save_as))
            existing_log = pd.read_pickle(save_as)
            combined_log = pd.concat([existing_log, log_df], ignore_index=True)
            combined_log.to_pickle(save_as)
            logger.info("Results appended.")
        else:
            log_df.to_pickle(save_as)
            logger.info("Results saved at: %s", os.path.abspath(save_as))

    logger.info('\n-------------------------------------------------------------------------')
    logger.info("Best Test F1 found: %.4f with %d features.", best_f1, len(best_subset))
    logger.info("Best hyperparameters: %s", best_params)
    logger.info("Features used in best subset: %s", best_subset)

    return best_subset, best_params, best_f1, log_df
