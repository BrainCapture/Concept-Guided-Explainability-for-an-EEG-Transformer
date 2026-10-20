import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

def sample_examples(df, metric, region, comparison_method, N, seed=42):
    """
    Sample positive and negative examples based on a given metric and region.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns 'metric_region', 'age', 'class', and index 'filename'.
    metric : str
        The metric to consider (e.g., 'alpha').
    region : str
        The region to consider (e.g., 'frontal').
    comparison_method : str
        Either 'high' or 'low', indicating whether to sample values higher or lower than the mean.
    N : int
        Number of samples to select for positive and negative examples (must be even).

    Returns
    -------
    tuple
        Two lists: (positive_filenames, negative_filenames), or (np.nan, np.nan) if sampling is not possible.
    """
    # Check if N is even
    if N % 2 != 0:
        return np.nan

    # Form the column name
    if region is None:
        column_name = metric
    else:
        column_name = f"{metric}_{region}"

    # Check if column exists
    if column_name not in df.columns:
        return np.nan

    # Exclude outliers (5% on each side)
    lower_bound = df[column_name].quantile(0.025)
    upper_bound = df[column_name].quantile(0.975)
    filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    # Compute Q1 and Q3 of the filtered values
    q1 = filtered_df[column_name].quantile(0.25)
    q3 = filtered_df[column_name].quantile(0.75)

    # Determine positive candidates based on comparison_method
    if comparison_method == 'high':
        # Use Q3 as threshold and select values higher than Q3
        positive_candidates = filtered_df[filtered_df[column_name] >= q3]
    elif comparison_method == 'low':
        # Use Q1 as threshold and select values lower than Q1
        positive_candidates = filtered_df[filtered_df[column_name] <= q1]
    else:
        return np.nan  # invalid comparison_method

    # Split positive candidates by class
    positive_class0 = positive_candidates[positive_candidates['class'] == 0]
    positive_class1 = positive_candidates[positive_candidates['class'] == 1]

    n_per_class = N // 2

    # Check if enough samples in each class for positive examples
    if len(positive_class0) < n_per_class or len(positive_class1) < n_per_class:
        return np.nan

    # Sample positive examples
    positive_sample0 = positive_class0.sample(n=n_per_class, random_state=seed)
    positive_sample1 = positive_class1.sample(n=n_per_class, random_state=seed)
    positive_samples = pd.concat([positive_sample0, positive_sample1])

    # Prepare to select negative examples: exclude positive samples' indices
    # positive_indices = positive_samples.index
    # remaining_df = filtered_df.drop(positive_indices)

    # # Split remaining by class
    # negative_class0 = remaining_df[remaining_df['class'] == 0]
    # negative_class1 = remaining_df[remaining_df['class'] == 1]

    # # Check if enough samples in each class for negative examples
    # if len(negative_class0) < n_per_class or len(negative_class1) < n_per_class:
    #     return (np.nan, np.nan)

    # # Sample negative examples
    # negative_sample0 = negative_class0.sample(n=n_per_class, random_state=seed)
    # negative_sample1 = negative_class1.sample(n=n_per_class, random_state=seed)
    # negative_samples = pd.concat([negative_sample0, negative_sample1])

    # Return the filenames (index values)
    positive_filenames = positive_samples.index.tolist()
    # negative_filenames = negative_samples.index.tolist()

    # return (positive_filenames, negative_filenames)
    return positive_filenames

def compute_positive_concepts(df_biomarkers, metrics, regions, comparison_method, N, include_age = False, save_as=None, seed=42):
    """
    Compute positive and negative examples for each metric-region combination and compile them into a single dataframe.

    Parameters
    ----------
    df_biomarkers : pd.DataFrame
        Original dataframe with biomarker data.
    metrics : list of str
        List of metrics to consider (e.g., ['alpha', 'beta']).
    regions : list of str
        List of regions to consider (e.g., ['frontal', 'parietal']).
    comparison_method : str
        Either 'high' or 'low', indicating how to sample positive examples.
    N : int
        Number of positive and negative examples to sample per metric-region pair.
    save_as : str
        Path to save the resulting dataframe as a pickle file.

    Returns
    -------
    pd.DataFrame
        A dataframe containing all positive and negative samples, with a column indicating the sample type (1 for positive, -1 for negative)
        and another column indicating the associated concept (comparisonMethod_metric_region).
    """
    all_samples = []
    concept_count = 0
    for metric in metrics:
        for region in regions:
            # Sample positive and negative examples for this metric-region pair
            # positive_filenames, negative_filenames = sample_examples(df_biomarkers, metric, region, comparison_method, N, seed)
            positive_filenames = sample_examples(df_biomarkers, metric, region, comparison_method, N, seed)

            # Skip if sampling failed for this pair (returns np.nan)
            if isinstance(positive_filenames, float) and np.isnan(positive_filenames):
                continue

            # Get positive samples and add metadata
            pos_df = df_biomarkers.loc[positive_filenames].copy()
            # pos_df['example_type'] = 1  # positive example
            pos_df['concept'] = f"{comparison_method}_{metric}_{region}"  # track which feature this is for
            pos_df['id'] = concept_count
            concept_count += 1
            all_samples.append(pos_df)

            # # Get negative samples and add metadata
            # neg_df = df_biomarkers.loc[negative_filenames].copy()
            # neg_df['example_type'] = -1  # negative example
            # neg_df['concept'] = f"{comparison_method}_{metric}_{region}"
            # all_samples.append(neg_df)


    for i in range(3): # generate a few random to use as a baseline
        # add a random concept as sanity check
        random.seed(seed+i)
        a = list(range(len(df_biomarkers)))
        b = random.choices(a, k=N)

        pos_df = df_biomarkers.iloc[b].copy()
        pos_df['concept'] = f"random_concept_{i}"  # track which feature this is for
        pos_df['id'] = concept_count
        concept_count += 1
        all_samples.append(pos_df)

    if include_age:
        positive_filenames = sample_examples(df_biomarkers, metric='age', region=None, comparison_method=comparison_method, N=N, seed=seed)
        # Get positive samples and add metadata
        pos_df = df_biomarkers.loc[positive_filenames].copy()
        # pos_df['example_type'] = 1  # positive example
        pos_df['concept'] = f"{comparison_method}_age"  # track which feature this is for
        pos_df['id'] = concept_count
        concept_count += 1
        all_samples.append(pos_df)

    # Combine all samples into a single dataframe
    if not all_samples:
        df_concepts = pd.DataFrame()
    else:
        df_concepts = pd.concat(all_samples)

    df_concepts['files'] = df_concepts.index.map(lambda p: os.path.splitext(os.path.basename(p))[0])

    if save_as:
        # Save the dataframe to a pickle file
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        df_concepts.to_csv(save_as)
        print('Results saved at:', os.path.abspath(save_as))

    return df_concepts

def compute_random_concepts(df_biomarkers, N, M, save_as=None):
    """
    Compute positive and negative examples for each metric-region combination and compile them into a single dataframe.

    Parameters
    ----------
    df_biomarkers : pd.DataFrame
        Original dataframe with biomarker data.
    N : int
        Number of negative examples to sample.
    M : int
        Number of negative concepts to create
    save_as : str
        Path to save the resulting dataframe as a pickle file.

    Returns
    -------
    pd.DataFrame
        A dataframe containing all positive and negative samples, with a column indicating the sample type (1 for positive, -1 for negative)
        and another column indicating the associated concept (comparisonMethod_metric_region).
    """
    all_samples = []
    concept_count = 1000
    for i in range(M):

        random.seed(i)
        a = list(range(len(df_biomarkers)))
        b = random.choices(a, k=N)

        # Get positive samples and add metadata
        pos_df = df_biomarkers.iloc[b].copy()
        # pos_df['example_type'] = 1  # positive example
        pos_df['concept'] = f"random_{concept_count}"  # track which feature this is for
        all_samples.append(pos_df)
        pos_df['id'] = concept_count
        concept_count += 1
        # # Get negative samples and add metadata
        # neg_df = df_biomarkers.loc[negative_filenames].copy()
        # neg_df['example_type'] = -1  # negative example
        # neg_df['concept'] = f"{comparison_method}_{metric}_{region}"
        # all_samples.append(neg_df)

    # Combine all samples into a single dataframe
    if not all_samples:
        df_concepts = pd.DataFrame()
    else:
        df_concepts = pd.concat(all_samples)

    df_concepts['files'] = df_concepts.index.map(lambda p: os.path.splitext(os.path.basename(p))[0])

    if save_as:
        # Save the dataframe to a csv file
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        df_concepts.to_csv(save_as)
        print('Results saved at:', os.path.abspath(save_as))

    return df_concepts

# def get_concept_files(df_concepts, df_random, concept, return_type='name'):
#     if isinstance(df_concepts, str): df_concepts = pd.read_csv(df_concepts)
#     if isinstance(df_random, str): df_random = pd.read_csv(df_random)
#     if return_type == 'name':
#         positive_examples = (df_concepts[df_concepts['concept'] == concept]['files'].values, df_concepts[df_concepts['concept'] == concept]['id'].values[0])  # (names (list[str]), id (int))
#         random_examples = [(df_random[df_random['concept'] == random_id]['files'].values, df_random[df_random['concept'] == random_id]['id'].values[0]) for random_id in df_random['concept'].unique()] # [(names (list[str]), id (int)), ...]

#     elif return_type == 'path':
#         positive_examples = (df_concepts[df_concepts['concept'] == concept].index, df_concepts[df_concepts['concept'] == concept]['id'].values[0])  # (paths (list[str]), id (int))
#         random_examples = [(df_random[df_random['concept'] == random_id].index, df_random[df_random['concept'] == random_id]['id'].values[0]) for random_id in df_random['concept'].unique()] # [(paths (list[str]), id (int)), ...]
#     else: positive_examples, random_examples =  None, None

#     return positive_examples, random_examples

# def get_concept_files(concepts, concept_name, return_type='name'):
#     if isinstance(concepts, str): concepts = pd.read_csv(concepts)
#     if return_type == 'name': examples = concepts[concepts['concept'] == concept_name]['files'].values
#     elif return_type == 'path': examples = concepts[concepts['concept'] == concept_name].index
#     else: examples = None
#     id = concepts[concepts['concept'] == concept_name]['id'].values[0]
#     return examples, id

def get_concept_files(concepts, concept_name, return_type='name', events=False):
    if isinstance(concepts, str):
        concepts = pd.read_csv(concepts)
        
    concept_df = concepts[concepts['concept'] == concept_name]

    if return_type == 'name':
        examples = concept_df['files'].values
    elif return_type == 'path':
        examples = concept_df.index
    else:
        examples = None

    id = concept_df['id'].values[0]
    t_starts = concept_df['t_starts'].values if events else None

    return examples, id, t_starts