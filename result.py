import argparse
import os
import shutil
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import matthews_corrcoef
from collections import defaultdict
import matplotlib.colors as mcolors


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing train.txt, test.txt, and valid.txt files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to store the output files')
    parser.add_argument('--output_file_dir', type=str, required=True, help='Directory to store the output file')
    
    return parser.parse_args()

def get_df_res_label(df_ifacea):
    df_label = pd.DataFrame()
    df_label['naming'] = list(df_ifacea['residue_a']) + list(df_ifacea['residue_b'])
    df_label['label'] = list(df_ifacea['res_label_a']) + list(df_ifacea['res_label_b'])
    df_label = df_label.drop_duplicates().reset_index(drop=True)
    return df_label

def get_label_dict(df_naming):
    zip_iterator = zip(df_naming['label'], df_naming['naming'])
    dict_labels = dict(zip_iterator)
    return dict_labels

def set_names_df(df_out, df_ifacea):
    subject_name, obj_name = [], []
    df_outputs_type = df_out.copy()
    dict_labels = get_label_dict(get_df_res_label(df_ifacea))

    for i in range(len(df_outputs_type)):
        subject_name.append(dict_labels.get(df_outputs_type.iloc[i]['subject']))
        obj_name.append(dict_labels.get(df_outputs_type.iloc[i]['object']))

    df_outputs_type['subject_name'] = subject_name
    df_outputs_type['obj_name'] = obj_name
    df_outputs_type['pair'] = df_outputs_type['subject_name'] + '_' + df_outputs_type['obj_name']
    df_outputs_type['pair_relation'] = df_outputs_type['pair'] + '_' + df_out['relation'].astype(str)
    df_outputs_type['pair_time'] = df_outputs_type['pair'] + ' ' + df_out['time_stamp'].astype(str)
    return df_outputs_type

def set_names_df_output(df_out, df_ifacea):
    subject_name, obj_name = [], []
    df_outputs_type = df_out.copy()
    dict_labels = get_label_dict(get_df_res_label(df_ifacea))

    for i in range(len(df_outputs_type)):
        subject_name.append(dict_labels.get(df_outputs_type.iloc[i]['subject']))
        obj_name.append(dict_labels.get(df_outputs_type.iloc[i]['object']))

    df_outputs_type['subject_name'] = subject_name
    df_outputs_type['obj_name'] = obj_name
    df_outputs_type['pair'] = df_outputs_type['subject_name'] + '_' + df_outputs_type['obj_name']
    df_outputs_type['pair_time'] = df_outputs_type['pair'] + ' ' + df_out['time_stamp'].astype(str)
    return df_outputs_type

def get_res_heatmap_df(df_out):
    df_heat = df_out.pair_relation.value_counts()
    residue_a, residue_b, values, relation = [], [], [], []
    for pair in list(set(df_out['pair_relation'])):
        residue_a.append(pair.split('_')[0])
        residue_b.append(pair.split('_')[1])
        relation.append(pair.split('_')[2])
        values.append(df_heat[pair])
    df_heatvals = pd.DataFrame()
    df_heatvals['residue_a'] = residue_a
    df_heatvals['residue_b'] = residue_b
    df_heatvals['relation'] = relation
    df_heatvals['values'] = values
    return df_heatvals

def get_res_heatmap_df_output(df_out):
    df_heat = df_out.pair.value_counts()
    residue_a, residue_b, values = [], [], []
    for pair in list(set(df_out['pair'])):
        residue_a.append(pair.split('_')[0])
        residue_b.append(pair.split('_')[1])
        values.append(df_heat[pair])
    df_heatvals = pd.DataFrame()
    df_heatvals['residue_a'] = residue_a
    df_heatvals['residue_b'] = residue_b
    df_heatvals['values'] = values
    return df_heatvals

def heatmap_similarity_score(test_pivot, predicted_pivot, output_file):
    # Create a matrix from pivot tables
    matrix_test = test_pivot.values
    matrix_predicted = predicted_pivot.values
    
    # Matrix subtraction and calculating absolute values
    subs_matrix = np.abs(matrix_test - matrix_predicted)
    
    # Sum of all numbers in the matrix
    sum_of_matrix = np.sum(subs_matrix)
    
    # Find the number of elements in a matrix that are greater than zero
    len_positive_elements = len(subs_matrix[subs_matrix > 0])
    
    score = sum_of_matrix / len_positive_elements
    print("Sum of matrix:", sum_of_matrix)
    print("Length of positive numbers:", len_positive_elements)
    print("Score:", score)
    
    # Write the score to a file
    with open(output_file, "w") as file:
        file.write(f"Sum of matrix: {sum_of_matrix}\n")
        file.write(f"Length of positive numbers: {len_positive_elements}\n")
        file.write(f"Score: {score}\n")

def custom_sort(value):
    # Assuming format like "Number-String" e.g., "10-ARG"
    try:
        parts = value.split('-')
        numeric_part = int(parts[0])
        string_part = '-'.join(parts[1:]) # Handle cases like "10-ARG-A"
        return numeric_part, string_part
    except:
        # Fallback for unexpected formats
        return float('inf'), value

# Moved calculate_metrics to global scope
def calculate_metrics(gt_series, pred_series, total_possible):
    """Calculates TP, FP, FN, TN and standard metrics from 0/1 series."""
    TP = ((pred_series == 1) & (gt_series == 1)).sum()
    FP = ((pred_series == 1) & (gt_series == 0)).sum()
    FN = ((pred_series == 0) & (gt_series == 1)).sum()
    # TN calculation needs care. It's the total possible minus the observed states.
    TN = total_possible - (TP + FP + FN)
    TN = max(0, TN) # Ensure TN is not negative

    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    TPR = Recall
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0 # Specificity = TN / (TN+FP) = 1 - FPR
    F1 = 2 * ((Precision * Recall) / (Precision + Recall)) if (Precision + Recall) > 0 else 0
    mcc_denom = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))**(1/2)
    MCC = (TP * TN - FP * FN) / mcc_denom if mcc_denom > 0 else 0

    return {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'Recall': Recall, 'Precision': Precision, 'TPR': TPR, 'FPR': FPR,
            'F1': F1, 'MCC': MCC}

def main():
    args = parse_arguments()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_file_dir = args.output_file_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load labels.txt from input directory
    labels_path = os.path.join(input_dir, 'labels.txt')
    df_labels = pd.read_table(labels_path, sep=" ", header=None)
    df_labels.columns = [
        'itype', 'chain_a', 'chain_b', 'resname_a', 'resname_b',
        'resid_a', 'resid_b', 'atom_a', 'atom_b', 'time', 'itype_int',
        'chain_res_a', 'chain_res_b', 'chain_atom_res_a', 'chain_atom_res_b',
        'res_label_a', 'res_label_b', 'atom_label_a', 'atom_label_b'
    ]
    df_labels['chain_a_b_time'] = df_labels['chain_res_a'] + df_labels['chain_res_b'] + df_labels['time'].astype(str)
    df_labels['chain_a_b'] = df_labels['chain_res_a'] + df_labels['chain_res_b']
    df_labels['residue_a'] = df_labels['resid_a'].astype(str) + "-" + df_labels['resname_a']
    df_labels['residue_b'] = df_labels['resid_b'].astype(str) + "-" + df_labels['resname_b']
    
    # Load test.txt from input directory
    test_path = os.path.join(input_dir, 'test.txt')
    test_set = pd.read_table(test_path, delim_whitespace=True, header=None)
    test_set.columns = ['subject', 'relation', 'object', 'time_stamp']

    # Load train.txt from input directory
    train_path = os.path.join(input_dir, 'train.txt')
    train_set = pd.read_table(train_path, delim_whitespace=True, header=None)
    train_set.columns = ['subject', 'relation', 'object', 'time_stamp']   
    
    # Load valid.txt from input directory
    valid_path = os.path.join(input_dir, 'valid.txt')
    valid_set = pd.read_table(valid_path, delim_whitespace=True, header=None)
    valid_set.columns = ['subject', 'relation', 'object', 'time_stamp']   

    # Load output file from output file directory
    output_path = os.path.join(output_file_dir)
    output = pd.read_table(output_path, delim_whitespace=True, header=None)
    output.columns = ['subject', 'object', 'time_stamp']
    
    # Process test_set
    test_set_post = set_names_df(test_set, df_labels).drop_duplicates().reset_index(drop=True)
    test_set_post_process = pd.DataFrame()
    test_set_post_process['subject'] = test_set_post['subject_name']
    test_set_post_process['object'] = test_set_post['obj_name']
    test_set_post_process['time_stamp'] = test_set_post['time_stamp']
    test_set_post_process['pair_relation'] = test_set_post['pair_relation']
    test_set_post_process['pair'] = test_set_post['pair']

    # Generate heatmap data
    heatmap_df = get_res_heatmap_df(test_set_post_process)
    heatmap_df['pair'] = heatmap_df['residue_a'] + '-' + heatmap_df['residue_b']
    heatmap_df['time_stamp'] = test_set_post_process['time_stamp']
    del heatmap_df["pair"]
    min_value = heatmap_df['values'].min()
    max_value = heatmap_df['values'].max()
    heatmap_df['freq'] = 1 + (heatmap_df['values'] - min_value) / (max_value - min_value) * (100 - 1)
    heatmap_df['freq'] = heatmap_df['freq'].apply(lambda x: round(x, 2))

    # Save the heatmap data to a CSV file
    output_csv_path = os.path.join(output_dir, "ground_truth.csv")
    heatmap_df.to_csv(output_csv_path, index=False)

    # Process output file
    output_post = set_names_df_output(output, df_labels).drop_duplicates().reset_index(drop=True)
    output_post_process = pd.DataFrame()
    output_post_process['subject'] = output_post['subject_name']
    output_post_process['object'] = output_post['obj_name']
    output_post_process['time_stamp'] = output_post['time_stamp']
    output_post_process['pair'] = output_post['pair']

    # Generate heatmap data for output
    heatmap_output_df = get_res_heatmap_df_output(output_post_process)
    heatmap_output_df['pair'] = heatmap_output_df['residue_a'] + '-' + heatmap_output_df['residue_b']
    min_value_output = heatmap_output_df['values'].min()
    max_value_output = heatmap_output_df['values'].max()
    heatmap_output_df['freq'] = 1 + (heatmap_output_df['values'] - min_value_output) / (max_value_output - min_value_output) * (100 - 1)
    heatmap_output_df['freq'] = heatmap_output_df['freq'].apply(lambda x: round(x, 2))

    # Save the heatmap data to a CSV file
    prediction_csv_path = os.path.join(output_dir, "prediction.csv")
    heatmap_output_df.to_csv(prediction_csv_path, index=False)

    # TRAIN SET

    # --- Define Uncommon and Common based on TRAIN SET ---
    train_set_post = set_names_df(train_set, df_labels).drop_duplicates().reset_index(drop=True)
    train_set_post['freq_count'] = train_set_post.groupby('pair')['pair'].transform('count')
    last_time = np.max(train_set_post['time_stamp'])
    train_time_thrashold = last_time / 2

    # Process valid_set
    valid_set_post = set_names_df(valid_set, df_labels).drop_duplicates().reset_index(drop=True)

    # Define UNCOMMON pairs from train set
    a = train_set_post.loc[train_set_post['freq_count'] <= train_time_thrashold ]
    uncommon_pairs_train = a['pair'].unique() # Get unique uncommon pairs

    # Define COMMON pairs from train set (frequency > threshold)
    b = train_set_post.loc[train_set_post['freq_count'] > train_time_thrashold ]
    common_pairs_train = b['pair'].unique() # Get unique common pairs

    # Filter test/prediction data based on training set classifications
    test_filtered_uncommon = test_set_post[test_set_post["pair"].isin(uncommon_pairs_train)]
    predicted_filtered_uncommon = output_post[output_post["pair"].isin(uncommon_pairs_train)]

    test_filtered_common = test_set_post[test_set_post["pair"].isin(common_pairs_train)]
    predicted_filtered_common = output_post[output_post["pair"].isin(common_pairs_train)]
    
    # --- Generate BASELINE Predictions ---
    # 1. Get unique interacting pairs from the entire training set
    all_train_interaction_pairs = set(train_set_post['pair'].unique())
    print(f"Baseline: Found {len(all_train_interaction_pairs)} unique interaction pairs in the training set.")

    # 2. Get unique timestamps from the test set
    test_timestamps = sorted(list(test_set_post['time_stamp'].unique()))
    print(f"Baseline: Found {len(test_timestamps)} unique timestamps in the test set.")

    # 3. Generate baseline predictions: all train pairs at all test times
    baseline_preds_list = []
    if not all_train_interaction_pairs or not test_timestamps:
        print("Baseline: Cannot generate baseline predictions (no training pairs or no test timestamps).")
        baseline_predictions_df = pd.DataFrame(columns=['subject_name', 'obj_name', 'pair', 'time_stamp']) # Empty DF
    else:
        for pair in all_train_interaction_pairs:
            # Ensure pair splitting works even if residue names contain '_'
            parts = pair.split('_')
            subject_name = parts[0]
            obj_name = '_'.join(parts[1:]) # Re-join if object name had '_'

            # Basic check if splitting resulted in expected parts
            if not subject_name or not obj_name:
                 print(f"Warning: Skipping pair '{pair}' due to unexpected format.")
                 continue

            for ts in test_timestamps:
                baseline_preds_list.append({
                    'subject_name': subject_name,
                    'obj_name': obj_name,
                    'pair': pair,
                    'time_stamp': ts
                })
        baseline_predictions_df = pd.DataFrame(baseline_preds_list)
        # Ensure correct dtypes
        baseline_predictions_df['time_stamp'] = baseline_predictions_df['time_stamp'].astype(int)
        print(f"Baseline: Generated {len(baseline_predictions_df)} predictions.")

    # --- Define Time and Pair Universes ---
    # Initial ground truth and prediction DFs (just the observed interactions)
    ground_truth_df = test_set_post[['subject_name', 'obj_name', 'pair', 'time_stamp']].drop_duplicates()
    ground_truth_df['present'] = 1 # Mark ground truth interactions
    predictions_df = output_post[['subject_name', 'obj_name', 'pair', 'time_stamp']].drop_duplicates()
    predictions_df['present'] = 1 # Mark predicted interactions

    all_test_pairs = set(ground_truth_df['pair'].unique()) | set(predictions_df['pair'].unique())
    all_possible_pairs = all_test_pairs # Focus on pairs observed in test/predictions

    # Reuse test_timestamps from baseline section
    if not test_timestamps:
        # Handle case where test set might be empty or time column missing
        test_timestamps = sorted(list(set(ground_truth_df['time_stamp'].unique()) | set(predictions_df['time_stamp'].unique())))
        if not test_timestamps:
            raise ValueError("Could not determine timestamps from test or prediction data.")
    min_time, max_time = min(test_timestamps), max(test_timestamps)
    time_points = len(test_timestamps)

    # Create a complete grid of all possible pairs at all test times
    all_pairs_time_grid = pd.MultiIndex.from_product(
        [all_possible_pairs, test_timestamps], names=['pair', 'time_stamp']
    ).to_frame(index=False)

    # Merge ground truth and predictions onto the full grid
    pair_map = ground_truth_df[['pair', 'subject_name', 'obj_name']].drop_duplicates().set_index('pair')

    ground_truth_full = pd.merge(
        all_pairs_time_grid, ground_truth_df[['pair', 'time_stamp', 'present']],
        on=['pair', 'time_stamp'], how='left'
    ).fillna({'present': 0})
    ground_truth_full['present'] = ground_truth_full['present'].astype(int)
    ground_truth_full = ground_truth_full.join(pair_map, on='pair')

    predictions_full = pd.merge(
        all_pairs_time_grid, predictions_df[['pair', 'time_stamp', 'present']],
        on=['pair', 'time_stamp'], how='left'
    ).fillna({'present': 0})
    predictions_full['present'] = predictions_full['present'].astype(int)
    predictions_full = predictions_full.join(pair_map, on='pair') # Use same map

    # Create baseline_full similar to ground_truth_full/predictions_full
    if not baseline_predictions_df.empty:
        baseline_full = pd.merge(
            all_pairs_time_grid, baseline_predictions_df[['pair', 'time_stamp']],
            on=['pair', 'time_stamp'], how='left', indicator=True
        )
        baseline_full['present'] = np.where(baseline_full['_merge'] == 'both', 1, 0)
        baseline_full = baseline_full.drop(columns=['_merge'])
        baseline_full = baseline_full.join(pair_map, on='pair') # Add names
    else:
        # Create an empty dataframe with the right columns if baseline is empty
        baseline_full = pd.DataFrame(columns=['pair', 'time_stamp', 'present', 'subject_name', 'obj_name'])
        baseline_full['present'] = baseline_full['present'].astype(int)

    # --- Define Evaluation Series and Output Path --- #
    # Use the 'full' dataframes which include 0s for non-interactions
    ground_truth_eval = ground_truth_full.set_index(['pair', 'time_stamp'])['present']
    predictions_eval = predictions_full.set_index(['pair', 'time_stamp'])['present']
    if not baseline_predictions_df.empty:
        baseline_eval = baseline_full.set_index(['pair', 'time_stamp'])['present']
    else:
        # Ensure baseline_eval exists even if empty
        baseline_eval = pd.Series(0, index=ground_truth_eval.index, name='present')
    # Define scores file path early
    scores_file_path = os.path.join(output_dir, "PerformanceMetrics.txt")

    # --- Load raw data for total calculation (needed for TN) --- #
    with open(os.path.join(input_dir, "train.txt"), 'r') as fr: train_data = [[int(x) for x in line.split()] for line in fr]
    with open(os.path.join(input_dir, "valid.txt"), 'r') as fr: valid_data = [[int(x) for x in line.split()] for line in fr]
    with open(os.path.join(input_dir, "test.txt"), 'r') as fr: test_data = [[int(x) for x in line.split()] for line in fr]
    total_data = train_data + valid_data + test_data
    s = [item[0] for item in total_data]
    o = [item[2] for item in total_data]
    unique_s = len(np.unique(s))
    unique_o = len(np.unique(o))
    total_possible_s_o_pairs = unique_s * unique_o
    total_possible_interactions_over_time = total_possible_s_o_pairs * len(test_timestamps)

    # --- Calculate Stability Bins and Training Frequencies EARLY --- #
    print("--- Calculating Training Set Frequencies and Stability Bins ---")
    # Calculate bins and frequencies based on training data
    stability_bins, pair_freq_train = bin_edges_by_frequency(train_set_post)
    if stability_bins is None or pair_freq_train is None:
        print("Warning: Could not calculate stability bins or training frequencies. Dependent analyses will be skipped.")
        # Ensure variables are None if calculation failed
        stability_bins = None
        pair_freq_train = None

    # --- Performance Calculation (Write to File) --- #
    print("--- Calculating Overall Performance Metrics ---")
    with open(scores_file_path, "w") as scores: # Open file for writing metrics
        # -- BASELINE Performance --
        print("--- Calculating Performance for BASELINE (Predict all train pairs at all test times) ---", file=scores)
        if not baseline_predictions_df.empty:
            # Align indices before calculation
            common_index = ground_truth_eval.index.intersection(baseline_eval.index)
            baseline_metrics = calculate_metrics(ground_truth_eval[common_index], baseline_eval[common_index], total_possible_interactions_over_time)
            print("Performance metrics for BASELINE interactions:\nRecall: {:.4f}, Precision: {:.4f}, TPR: {:.4f}, FPR: {:.4f}, F1: {:.4f}, MCC: {:.4f}\n".format(
                baseline_metrics['Recall'], baseline_metrics['Precision'], baseline_metrics['TPR'], baseline_metrics['FPR'], baseline_metrics['F1'], baseline_metrics['MCC']), file=scores)
        else:
             print("Performance metrics for BASELINE interactions:\nCannot calculate metrics - no baseline predictions generated.\n", file=scores)
    
        # -- MODEL Performance on ALL interactions --
        print("--- Calculating Performance for MODEL (ALL Interactions) ---", file=scores)
        # Align indices
        common_index_model = ground_truth_eval.index.intersection(predictions_eval.index)
        all_metrics = calculate_metrics(ground_truth_eval[common_index_model], predictions_eval[common_index_model], total_possible_interactions_over_time)
        print("Performance metrics for MODEL (ALL interactions):\nRecall: {:.4f}, Precision: {:.4f}, TPR: {:.4f}, FPR: {:.4f}, F1: {:.4f}, MCC: {:.4f}\n".format(
            all_metrics['Recall'], all_metrics['Precision'], all_metrics['TPR'], all_metrics['FPR'], all_metrics['F1'], all_metrics['MCC']), file=scores)
    # Scores file is closed here implicitly by 'with' statement

    # --- Analysis 1: True vs. Predicted Interactions Over Time --- #
    print("--- Generating Analysis 1: Time Dynamics Plots ---")
    # Pass valid_set_post for including validation data in heatmap
    plot_time_vs_pair_heatmaps(ground_truth_full, predictions_full, valid_set_post, output_dir, num_pairs_to_show=50, valid_steps_to_show=20)
    representative_pairs_test = select_representative_pairs(ground_truth_full, n=7)
    plot_sample_trajectories(ground_truth_full, predictions_full, valid_set_post, representative_pairs_test, output_dir)
    plot_flip_rate_comparison(ground_truth_full, predictions_full, test_timestamps, output_dir, window_size=10)
    print("Analysis 1 plots saved.")

    # --- Analysis 1b: Trajectories based on Training Frequency Selection --- #
    print("--- Generating Analysis 1b: Sample Trajectories (Selected by Train Freq) ---")
    # Check if training frequency data is available
    if pair_freq_train is not None:
        representative_pairs_train = select_representative_pairs_train_freq(train_set_post, n=7)
        # Pass pair_freq_train for titles
        plot_sample_trajectories_train_selection(ground_truth_full, predictions_full, valid_set_post, representative_pairs_train, pair_freq_train, output_dir)
        print("Analysis 1b plot saved.")
    else:
        print("Skipping Analysis 1b: Training frequency data not available.")

    # --- Analysis 2: Model Predictions Over Time --- #
    print("--- Generating Analysis 2: Performance Over Time Plots ---")
    plot_metrics_vs_time(ground_truth_eval, predictions_eval, test_timestamps, total_possible_s_o_pairs, output_dir)
    plot_cumulative_error(ground_truth_eval, predictions_eval, test_timestamps, output_dir)
    print("Analysis 2 plots saved.")

    # --- Analysis 3: Performance by Interaction Stability --- #
    print("--- Generating Analysis 3: Stability-Based Performance ---")
    # Check if stability bins were calculated successfully
    if stability_bins is not None:
        # Append to the metrics file (already created)
        plot_metrics_by_stability(ground_truth_eval, predictions_eval, stability_bins, total_possible_s_o_pairs, output_dir, scores_file_path)
        # Calculate per-edge F1 scores on the test set
        f1_df = calculate_per_edge_f1(ground_truth_full, predictions_full)
        if f1_df is not None:
            # Plot F1 distribution by stability bin
            plot_f1_distribution_by_stability(f1_df, stability_bins, output_dir)
            # Plot Train Frequency vs Test F1 Scatter plot (requires pair_freq_train)
            if pair_freq_train is not None:
                 plot_train_freq_vs_test_f1(pair_freq_train, f1_df, stability_bins, output_dir)
            else:
                 print("Skipping Train Freq vs Test F1 scatter plot: Training frequency data not available.")
        print("Analysis 3 plots and metrics saved.")
    else:
        print("Skipping stability analysis: Stability bins not calculated.")

    # --- Final Cleanup and Summary ---
    print("--- Processing Summary ---")
    # Re-print locations of all generated files
    print("Performance metrics saved to:", scores_file_path)
    print("Heatmap similarity score saved to:", os.path.join(output_dir, "heatmap_similarity_score.txt"))
    # List all generated plots
    plot_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    print("Generated plots saved in:", output_dir)
    for plot_file in sorted(plot_files):
        print(f"- {plot_file}")


# --- Analysis 1 Functions ---

def plot_time_vs_pair_heatmaps(gt_full, pred_full, valid_set_post, output_dir, num_pairs_to_show=50, valid_steps_to_show=20):
    """Plots vertically stacked heatmaps of GT, Preds, and Overlay, including last valid steps."""
    # --- 1. Process Test Data --- 
    gt_pivot_test = gt_full.pivot(index='pair', columns='time_stamp', values='present')
    pred_pivot_test = pred_full.pivot(index='pair', columns='time_stamp', values='present')

    common_pairs = gt_pivot_test.index.intersection(pred_pivot_test.index)
    gt_pivot_test = gt_pivot_test.loc[common_pairs].fillna(0).astype(int)
    pred_pivot_test = pred_pivot_test.loc[common_pairs].fillna(0).astype(int)

    if len(common_pairs) > num_pairs_to_show:
        pair_freq = gt_pivot_test.sum(axis=1).sort_values(ascending=False)
        selected_pairs = pair_freq.head(num_pairs_to_show).index
        gt_pivot_test = gt_pivot_test.loc[selected_pairs]
        pred_pivot_test = pred_pivot_test.loc[selected_pairs]
        plot_title_suffix = f' (Top {num_pairs_to_show} Pairs)'
    else:
        selected_pairs = common_pairs # Use all common pairs
        plot_title_suffix = ' (All Common Pairs)'

    # Sort pairs for better visualization
    try:
        sorted_index = gt_pivot_test.index.map(lambda x: tuple(custom_sort(p) for p in x.split('_')))
        gt_pivot_test = gt_pivot_test.loc[sorted_index.sort_values().index]
        gt_pivot_test.index = gt_pivot_test.index.map(lambda x: f"{x[0][0]}-{x[0][1]}_{x[1][0]}-{x[1][1]}" if isinstance(x, tuple) and len(x)==2 and isinstance(x[0], tuple) and isinstance(x[1], tuple) else x)
        pred_pivot_test = pred_pivot_test.loc[gt_pivot_test.index]
    except Exception as e:
         print(f"Warning: Custom pair sorting failed ({e}). Falling back to simple string sort.")
         gt_pivot_test = gt_pivot_test.sort_index()
         pred_pivot_test = pred_pivot_test.loc[gt_pivot_test.index]

    overlay_matrix_test = pd.DataFrame(0, index=gt_pivot_test.index, columns=gt_pivot_test.columns)
    overlay_matrix_test[(gt_pivot_test == 1) & (pred_pivot_test == 0)] = 1 # FN
    overlay_matrix_test[(gt_pivot_test == 0) & (pred_pivot_test == 1)] = 2 # FP
    overlay_matrix_test[(gt_pivot_test == 1) & (pred_pivot_test == 1)] = 3 # TP

    # --- 2. Process Validation Data --- 
    valid_pivot = pd.DataFrame() # Default empty
    last_valid_timestamps = []
    if not valid_set_post.empty and valid_steps_to_show > 0:
        all_valid_timestamps = sorted(valid_set_post['time_stamp'].unique())
        if len(all_valid_timestamps) >= valid_steps_to_show:
            last_valid_timestamps = all_valid_timestamps[-valid_steps_to_show:]
            valid_data_filtered = valid_set_post[
                (valid_set_post['time_stamp'].isin(last_valid_timestamps)) &
                (valid_set_post['pair'].isin(selected_pairs)) # Use the same pairs as selected for test
            ]
            # Add 'present' column for validation data
            # Use assign to avoid SettingWithCopyWarning
            valid_data_filtered = valid_data_filtered.assign(present=1)
            # Pivot validation data
            valid_pivot = valid_data_filtered.pivot_table(
                index='pair', columns='time_stamp', values='present', fill_value=0
            )
            # Ensure all selected pairs are present, fill missing with 0
            valid_pivot = valid_pivot.reindex(gt_pivot_test.index, fill_value=0)
            # Ensure columns are sorted numerically
            valid_pivot = valid_pivot.reindex(sorted(valid_pivot.columns), axis=1)
        else:
            print(f"Warning: Not enough validation timestamps ({len(all_valid_timestamps)}) to show {valid_steps_to_show}.")

    # --- 3. Combine Data --- 
    if not valid_pivot.empty:
        # Concatenate horizontally: Validation | Test
        gt_combined = pd.concat([valid_pivot, gt_pivot_test], axis=1)
        
        # --- Fill prediction and overlay for validation period using validation ground truth --- 
        # Prediction plot: Show validation GT values (0 or 1)
        pred_combined = pd.concat([valid_pivot, pred_pivot_test], axis=1)
        
        # Overlay plot: Show TN (0) if valid GT is 0, TP (3) if valid GT is 1
        overlay_valid_part = valid_pivot.replace({0: 0, 1: 3})
        overlay_combined = pd.concat([overlay_valid_part, overlay_matrix_test], axis=1)
        
        valid_data_offset = len(last_valid_timestamps)
    else:
        # No validation data to show
        gt_combined = gt_pivot_test
        pred_combined = pred_pivot_test
        overlay_combined = overlay_matrix_test
        valid_data_offset = 0

    # --- 4. Plotting --- 
    cmap_gt = sns.color_palette(["#f0f0f0", "#0074D9"]) # Grey/Blue
    cmap_pred = sns.color_palette(["#f0f0f0", "#FFA500"]) # Grey/Orange (Now used for assumed validation GT as well)
    cmap_overlay = sns.color_palette(["#f0f0f0", "#0074D9", "#FFA500", "#2ecc71"]) # Grey(TN), Blue(FN), Orange(FP), Green(TP)
    # Add short descriptions back to overlay labels
    overlay_labels = ['TN (Correct Negative)', 'FN (Missed)', 'FP (False Positive)', 'TP (Correct Positive)']
    
    # Define colormaps for combined plots (No need for separate NaN color anymore)
    cmap_pred_viz = mcolors.ListedColormap(cmap_pred) # Use Pred colors (Grey/Orange) for Validation GT + Test Pred
    cmap_overlay_viz = mcolors.ListedColormap(cmap_overlay) # Use Overlay colors for Validation TN/TP + Test Overlay

    # Setup vertically stacked figure - remove sharex
    # Slightly increased height per label and width per timestamp
    fig_height = max(12, len(gt_combined.index) * 0.6) 
    fig_width = max(15, gt_combined.shape[1] * 0.25)
    fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height), sharex=False, sharey=True)

    common_heatmap_kws = {"linewidths": 0.1, "linecolor": 'lightgray'} # Base kws, cbar and yticklabels handled per plot

    current_yticklabels = gt_combined.index

    # Plot Ground Truth (Validation + Test)
    sns.heatmap(gt_combined, ax=axes[0], cmap=cmap_gt, cbar=False, yticklabels=current_yticklabels, **common_heatmap_kws)
    axes[0].set_title('Ground Truth (Last Validation + Test)')
    axes[0].set_ylabel('Residue Pair')
    axes[0].tick_params(axis='x', labelbottom=True) # Ensure x-tick labels are visible
    axes[0].set_xlabel('') # Remove x-axis title for top plot

    # Plot Predictions (Validation GT + Test Predictions)
    sns.heatmap(pred_combined.fillna(0).astype(int), ax=axes[1], cmap=cmap_pred_viz, vmin=0, vmax=1, cbar=False, yticklabels=current_yticklabels, **common_heatmap_kws)
    axes[1].set_title('Predictions (Validation GT + Test Predictions)')
    axes[1].set_ylabel('Residue Pair')
    axes[1].tick_params(axis='x', labelbottom=True) # Ensure x-tick labels are visible
    axes[1].set_xlabel('') # Remove x-axis title for middle plot

    # Plot Overlay (Validation TN/TP + Test Overlay)
    bounds_overlay = [0, 1, 2, 3, 4] # TN, FN, FP, TP
    norm_overlay = mcolors.BoundaryNorm(bounds_overlay, cmap_overlay_viz.N)

    cax = sns.heatmap(overlay_combined.fillna(0).astype(int), ax=axes[2], cmap=cmap_overlay_viz, norm=norm_overlay,
                      cbar=True, yticklabels=current_yticklabels, **common_heatmap_kws, # Use common_heatmap_kws here too
                      cbar_kws={"ticks": [0.5, 1.5, 2.5, 3.5], "label": "Result Type"}) # Place ticks in middle
    axes[2].set_title('Overlay (Validation TN/TP + Test Result)')
    axes[2].set_xlabel('Time Stamp') # Keep x-axis title only on bottom plot
    axes[2].set_ylabel('Residue Pair')
    axes[2].tick_params(axis='x', labelbottom=True) # Ensure x-tick labels are visible

    # Set overlay colorbar labels (now simplified)
    colorbar = cax.collections[0].colorbar
    colorbar.set_ticklabels(overlay_labels)

    # Dynamically adjust y-tick label font size
    num_labels = len(current_yticklabels)
    font_size = 10 # Default font size
    if fig_height > 0 and num_labels > 0: # Avoid division by zero
        space_per_label_pt = (fig_height / num_labels) * 72  # Available points per label
        # Aim for font size to be a fraction of available space, capped
        font_size = max(4, min(10, int(space_per_label_pt * 0.35)))
    else: # Fallback if calculation is not possible (e.g., no labels)
        if num_labels > 20: font_size = 8
        if num_labels > 35: font_size = 6
        if num_labels > 50: font_size = 5
        if num_labels > 70: font_size = 4
        
    for ax in axes:
        ax.tick_params(axis='y', labelsize=font_size)

    # Add vertical line separator if validation data was included
    if valid_data_offset > 0:
        for ax in axes:
            ax.axvline(x=valid_data_offset, color='red', linestyle='--', linewidth=2)
            # Optionally add text annotation
            ax.text(valid_data_offset / 2., ax.get_ylim()[0] * 1.02, 'Validation', 
                    ha='center', va='bottom', color='red', fontsize=10, weight='bold')
            ax.text(valid_data_offset + (gt_combined.shape[1] - valid_data_offset) / 2., ax.get_ylim()[0] * 1.02, 'Test', 
                    ha='center', va='bottom', color='black', fontsize=10, weight='bold')

    fig.suptitle(f'Interaction Dynamics: Validation History vs Test Prediction{plot_title_suffix}', fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    # Update filename
    new_filename = 'heatmap_time_vs_pairs_VERTICAL_with_valid.png'
    plt.savefig(os.path.join(output_dir, new_filename), dpi=300)
    print(f"Saved stacked heatmap with validation data to: {new_filename}") 
    plt.close(fig)


def select_representative_pairs(gt_full, n=7):
    """Selects n pairs representing a range of persistence levels based on TEST SET quantiles."""
    # Calculate persistence (frequency) for pairs present at least once in the TEST ground truth
    total_test_timestamps = gt_full['time_stamp'].nunique()
    if total_test_timestamps == 0:
        print("Warning: No timestamps found in test set ground truth.")
        return []
    pair_persistence = gt_full[gt_full['present'] == 1].groupby('pair')['present'].count() / total_test_timestamps
    pair_persistence = pair_persistence.sort_values()

    if pair_persistence.empty:
        print("Warning: No persistent pairs found in ground truth.")
        return []

    num_pairs_available = len(pair_persistence)
    print(f"Found {num_pairs_available} unique pairs with interactions in ground truth.")

    if num_pairs_available <= n:
        print(f"Selecting all {num_pairs_available} available pairs.")
        return pair_persistence.index.tolist()

    # Define quantiles to target (including min and max)
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    if n != 7: # Adjust quantiles if n is different (simple linear spacing for now)
        quantiles = np.linspace(0, 1, n)

    selected_pairs = set()
    # Use .quantile() which handles potential duplicates in persistence values better
    quantile_values = pair_persistence.quantile(quantiles, interpolation='nearest') # Find nearest actual value

    # Find indices corresponding to these quantile values
    selected_indices = set()
    for q_val in quantile_values:
        # Find the index (pair name) closest to this persistence value
        # Use idxmin() on the absolute difference to find the closest index
        closest_idx = (pair_persistence - q_val).abs().idxmin()
        selected_indices.add(closest_idx)

    # Ensure we have exactly n pairs, adding more if duplicates were picked
    additional_needed = n - len(selected_indices)
    if additional_needed > 0:
        print(f"Quantile selection yielded duplicates based on test freq. Adding {additional_needed} more pairs.")
        available_indices = pair_persistence.index.difference(list(selected_indices))
        if len(available_indices) >= additional_needed:
            additional_pairs = np.random.choice(available_indices, additional_needed, replace=False)
            selected_indices.update(additional_pairs)
        else:
            selected_indices.update(available_indices)

    final_selection = list(selected_indices)[:n]
    print(f"Selected {len(final_selection)} representative pairs based on TEST SET persistence quantiles: {final_selection}")
    return final_selection

def select_representative_pairs_train_freq(train_set_post, n=7):
    """Selects n pairs representing a range of persistence levels based on TRAINING SET quantiles."""
    if train_set_post is None or train_set_post.empty:
        print("Warning: Training set data is empty or None. Cannot select pairs by train frequency.")
        return []
        
    # Calculate persistence (frequency) for pairs present at least once in the TRAINING set
    total_train_timestamps = train_set_post['time_stamp'].nunique()
    if total_train_timestamps == 0:
        print("Warning: No timestamps found in training set.")
        return []
        
    pair_counts_train = train_set_post.groupby('pair').size()
    pair_persistence_train = pair_counts_train / total_train_timestamps
    pair_persistence_train = pair_persistence_train.sort_values()
    
    if pair_persistence_train.empty:
        print("Warning: No pairs found in training set after grouping.")
        return []

    num_pairs_available = len(pair_persistence_train)
    print(f"Found {num_pairs_available} unique pairs with interactions in training set.")

    if num_pairs_available <= n:
        print(f"Selecting all {num_pairs_available} available training pairs.")
        return pair_persistence_train.index.tolist()

    # Define quantiles
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    if n != 7:
        quantiles = np.linspace(0, 1, n)

    selected_indices = set()
    quantile_values = pair_persistence_train.quantile(quantiles, interpolation='nearest')

    for q_val in quantile_values:
        closest_idx = (pair_persistence_train - q_val).abs().idxmin()
        selected_indices.add(closest_idx)

    additional_needed = n - len(selected_indices)
    if additional_needed > 0:
        print(f"Quantile selection yielded duplicates based on train freq. Adding {additional_needed} more pairs.")
        available_indices = pair_persistence_train.index.difference(list(selected_indices))
        if len(available_indices) >= additional_needed:
            additional_pairs = np.random.choice(available_indices, additional_needed, replace=False)
            selected_indices.update(additional_pairs)
        else:
            selected_indices.update(available_indices)

    final_selection = list(selected_indices)[:n]
    print(f"Selected {len(final_selection)} representative pairs based on TRAINING SET persistence quantiles: {final_selection}")
    return final_selection


def plot_sample_trajectories(gt_full, pred_full, valid_set_post, pairs, output_dir):
    """Plots TEST SET ground truth vs predicted presence over time for selected pairs (selection based on TEST freq),
    prefixed by VALIDATION SET ground truth."""
    if not pairs:
        print("No representative pairs selected (based on test freq), skipping trajectory plot.")
        return

    num_pairs = len(pairs)
    fig_height = max(6, 2.5 * num_pairs)
    fig, axes = plt.subplots(num_pairs, 1, figsize=(14, fig_height), sharex=True, squeeze=False) # Increased width for time
    axes = axes.flatten()

    # Determine the last timestamp of the validation set to draw a separator
    last_valid_time = -1
    if not valid_set_post.empty:
        last_valid_time = valid_set_post['time_stamp'].max()

    for i, pair in enumerate(pairs):
        # Validation data for the pair
        valid_pair_gt = pd.DataFrame()
        if not valid_set_post.empty:
            valid_pair_gt = valid_set_post[valid_set_post['pair'] == pair].sort_values('time_stamp')
            # Ensure 'present' column for validation (assuming interaction means present=1)
            if not valid_pair_gt.empty and 'present' not in valid_pair_gt.columns:
                 valid_pair_gt = valid_pair_gt.assign(present=1)


        # Test data for the pair
        gt_pair_test = gt_full[gt_full['pair'] == pair].sort_values('time_stamp')
        pred_pair_test = pred_full[pred_full['pair'] == pair].sort_values('time_stamp')

        # Combine validation and test ground truth
        combined_gt_list = []
        if not valid_pair_gt.empty:
            combined_gt_list.append(valid_pair_gt[['time_stamp', 'present']])
        if not gt_pair_test.empty:
            combined_gt_list.append(gt_pair_test[['time_stamp', 'present']])
        
        combined_gt_df = pd.DataFrame()
        if combined_gt_list:
            combined_gt_df = pd.concat(combined_gt_list).drop_duplicates(subset=['time_stamp'], keep='first').sort_values('time_stamp')

        # Prepare prediction series: validation GT for validation period, actual predictions for test period
        combined_pred_list = []
        if not valid_pair_gt.empty: # Use validation GT as "prediction" for validation part
             combined_pred_list.append(valid_pair_gt[['time_stamp', 'present']])
        if not pred_pair_test.empty:
            combined_pred_list.append(pred_pair_test[['time_stamp', 'present']])

        combined_pred_df = pd.DataFrame()
        if combined_pred_list:
            combined_pred_df = pd.concat(combined_pred_list).drop_duplicates(subset=['time_stamp'], keep='first').sort_values('time_stamp')


        # Determine the full time range for plotting
        all_times = set()
        if not combined_gt_df.empty: all_times.update(combined_gt_df['time_stamp'])
        if not combined_pred_df.empty: all_times.update(combined_pred_df['time_stamp'])
        
        if not all_times:
            axes[i].set_title(f'Pair: {pair} (No data available)')
            axes[i].axis('off')
            continue
            
        time_range_sorted = sorted(list(all_times))

        # Reindex to full time range
        gt_plot = pd.Series(index=time_range_sorted, dtype='float64')
        if not combined_gt_df.empty:
            gt_plot = combined_gt_df.set_index('time_stamp')['present'].reindex(time_range_sorted, fill_value=0)

        pred_plot = pd.Series(index=time_range_sorted, dtype='float64')
        if not combined_pred_df.empty:
            pred_plot = combined_pred_df.set_index('time_stamp')['present'].reindex(time_range_sorted, fill_value=0)


        axes[i].step(gt_plot.index, gt_plot.values, where='post', label='Ground Truth (Valid+Test)', color='#0074D9', linewidth=1.5)
        axes[i].step(pred_plot.index, pred_plot.values + 0.05, where='post', label='Prediction (Valid GT+Test Pred)', color='#FFA500', linestyle='--', linewidth=1.5)

        # Add vertical line separator
        if last_valid_time != -1 and last_valid_time < time_range_sorted[-1] : # Only if valid data exists and is before end of test
             axes[i].axvline(x=last_valid_time + 0.5, color='red', linestyle='--', linewidth=1.2, label='Valid/Test Cutoff')
             # Add text annotations for Valid and Test periods
             min_plot_time, max_plot_time = time_range_sorted[0], time_range_sorted[-1]
             if last_valid_time >= min_plot_time: # Check if valid period is visible
                 axes[i].text((min_plot_time + last_valid_time) / 2, 1.08, 'Validation', ha='center', va='bottom', color='red', fontsize=9)
             if last_valid_time < max_plot_time: # Check if test period is visible
                 axes[i].text((last_valid_time + 1 + max_plot_time) / 2, 1.08, 'Test', ha='center', va='bottom', color='black', fontsize=9)


        gt_persistence_test = gt_pair_test['present'].mean() if not gt_pair_test.empty else 0 # Persistence calculated on TEST data
        axes[i].set_title(f'Pair: {pair} (Test Persistence: {gt_persistence_test:.2f})')
        axes[i].set_yticks([0, 1])
        axes[i].set_yticklabels(['Off', 'On'])
        axes[i].set_ylim(-0.1, 1.15) # Adjusted ylim slightly for text
        axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small') # Move legend outside
        axes[i].grid(True, axis='y', linestyle=':', alpha=0.7)

        if i == num_pairs - 1:
            axes[i].set_xlabel('Time Stamp (Validation + Test)')
        else:
            axes[i].tick_params(axis='x', labelbottom=False)

    fig.suptitle('Sample Pair Trajectories: Validation Ground Truth + Test Performance (Pairs Selected by Test Freq.)', fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.97]) # Adjust for legend
    plt.savefig(os.path.join(output_dir, 'sample_pair_trajectories_with_validation.png'), dpi=300) # New filename
    plt.close(fig)

def plot_sample_trajectories_train_selection(gt_full, pred_full, valid_set_post, pairs_selected_by_train, pair_freq_train, output_dir):
    """Plots TEST SET ground truth vs predicted presence over time for selected pairs (selection based on TRAIN freq),
    prefixed by VALIDATION SET ground truth."""
    if not pairs_selected_by_train:
        print("No representative pairs selected based on train freq, skipping trajectory plot.")
        return
    if pair_freq_train is None:
         print("Warning: Training frequency data not available for titles. Skipping trajectory plot.")
         return

    num_pairs = len(pairs_selected_by_train)
    fig_height = max(6, 2.5 * num_pairs)
    fig, axes = plt.subplots(num_pairs, 1, figsize=(14, fig_height), sharex=True, squeeze=False) # Increased width
    axes = axes.flatten()

    last_valid_time = -1
    if not valid_set_post.empty:
        last_valid_time = valid_set_post['time_stamp'].max()

    for i, pair in enumerate(pairs_selected_by_train):
        # Validation data
        valid_pair_gt = pd.DataFrame()
        if not valid_set_post.empty:
            valid_pair_gt = valid_set_post[valid_set_post['pair'] == pair].sort_values('time_stamp')
            if not valid_pair_gt.empty and 'present' not in valid_pair_gt.columns:
                valid_pair_gt = valid_pair_gt.assign(present=1)
        
        # Test data
        gt_pair_test = gt_full[gt_full['pair'] == pair].sort_values('time_stamp')
        pred_pair_test = pred_full[pred_full['pair'] == pair].sort_values('time_stamp')

        # Combine validation and test ground truth
        combined_gt_list = []
        if not valid_pair_gt.empty: combined_gt_list.append(valid_pair_gt[['time_stamp', 'present']])
        if not gt_pair_test.empty: combined_gt_list.append(gt_pair_test[['time_stamp', 'present']])
        combined_gt_df = pd.DataFrame()
        if combined_gt_list:
            combined_gt_df = pd.concat(combined_gt_list).drop_duplicates(subset=['time_stamp'], keep='first').sort_values('time_stamp')

        # Prepare prediction series
        combined_pred_list = []
        if not valid_pair_gt.empty: combined_pred_list.append(valid_pair_gt[['time_stamp', 'present']])
        if not pred_pair_test.empty: combined_pred_list.append(pred_pair_test[['time_stamp', 'present']])
        combined_pred_df = pd.DataFrame()
        if combined_pred_list:
            combined_pred_df = pd.concat(combined_pred_list).drop_duplicates(subset=['time_stamp'], keep='first').sort_values('time_stamp')
            
        all_times = set()
        if not combined_gt_df.empty: all_times.update(combined_gt_df['time_stamp'])
        if not combined_pred_df.empty: all_times.update(combined_pred_df['time_stamp'])

        if not all_times:
            train_persistence = pair_freq_train.get(pair, 0)
            axes[i].set_title(f'Pair: {pair} (Train Persistence: {train_persistence:.2f}) (No Valid/Test Data)')
            axes[i].axis('off')
            continue
        
        time_range_sorted = sorted(list(all_times))

        gt_plot = pd.Series(index=time_range_sorted, dtype='float64')
        if not combined_gt_df.empty:
            gt_plot = combined_gt_df.set_index('time_stamp')['present'].reindex(time_range_sorted, fill_value=0)

        pred_plot = pd.Series(index=time_range_sorted, dtype='float64')
        if not combined_pred_df.empty:
            pred_plot = combined_pred_df.set_index('time_stamp')['present'].reindex(time_range_sorted, fill_value=0)

        axes[i].step(gt_plot.index, gt_plot.values, where='post', label='Ground Truth (Valid+Test)', color='#0074D9', linewidth=1.5)
        axes[i].step(pred_plot.index, pred_plot.values + 0.05, where='post', label='Prediction (Valid GT+Test Pred)', color='#FFA500', linestyle='--', linewidth=1.5)

        if last_valid_time != -1 and last_valid_time < time_range_sorted[-1]:
            axes[i].axvline(x=last_valid_time + 0.5, color='red', linestyle='--', linewidth=1.2, label='Valid/Test Cutoff')
            min_plot_time, max_plot_time = time_range_sorted[0], time_range_sorted[-1]
            if last_valid_time >= min_plot_time:
                 axes[i].text((min_plot_time + last_valid_time) / 2, 1.08, 'Validation', ha='center', va='bottom', color='red', fontsize=9)
            if last_valid_time < max_plot_time:
                 axes[i].text((last_valid_time + 1 + max_plot_time) / 2, 1.08, 'Test', ha='center', va='bottom', color='black', fontsize=9)

        train_persistence = pair_freq_train.get(pair, 0) # Get TRAIN persistence for the title
        axes[i].set_title(f'Pair: {pair} (Train Persistence: {train_persistence:.2f})')
        axes[i].set_yticks([0, 1])
        axes[i].set_yticklabels(['Off', 'On'])
        axes[i].set_ylim(-0.1, 1.15)
        axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        axes[i].grid(True, axis='y', linestyle=':', alpha=0.7)

        if i == num_pairs - 1:
            axes[i].set_xlabel('Time Stamp (Validation + Test)')
        else:
            axes[i].tick_params(axis='x', labelbottom=False)

    fig.suptitle('Sample Pair Trajectories: Validation Ground Truth + Test Performance (Pairs Selected by Train Freq.)', fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.97]) # Adjust for legend
    plt.savefig(os.path.join(output_dir, 'sample_pair_trajectories_train_selection_with_validation.png'), dpi=300) # New filename
    plt.close(fig)


def plot_flip_rate_comparison(gt_full, pred_full, timestamps, output_dir, window_size=10):
    """Calculates and plots the number of state changes (on/off flips) in time windows."""

    def count_flips(df, pair_col='pair', time_col='time_stamp', present_col='present'):
        # Calculate differences between consecutive states for each pair
        df_sorted = df.sort_values([pair_col, time_col])
        df_sorted['prev_state'] = df_sorted.groupby(pair_col)[present_col].shift(1)
        # A flip occurs if the state is different from the previous state (and prev state exists)
        df_sorted['flip'] = (df_sorted[present_col] != df_sorted['prev_state']) & (df_sorted['prev_state'].notna())
        return df_sorted[df_sorted['flip']]

    gt_flips = count_flips(gt_full)
    pred_flips = count_flips(pred_full)

    # Bin flips into time windows
    bins = np.arange(min(timestamps), max(timestamps) + window_size, window_size)
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]

    if not labels: # Handle case with very few timestamps
        print("Not enough timestamps to create windows for flip rate analysis.")
        return

    gt_flips['time_window'] = pd.cut(gt_flips['time_stamp'], bins=bins, labels=labels, right=False)
    pred_flips['time_window'] = pd.cut(pred_flips['time_stamp'], bins=bins, labels=labels, right=False)

    gt_flip_counts = gt_flips.groupby('time_window').size()
    pred_flip_counts = pred_flips.groupby('time_window').size()

    # Combine counts for plotting
    flip_counts_df = pd.DataFrame({'Ground Truth': gt_flip_counts, 'Prediction': pred_flip_counts}).fillna(0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    flip_counts_df.plot(kind='bar', ax=ax, color=['#0074D9', '#FFA500'])
    ax.set_title(f'Interaction State Flips per Time Window (Size={window_size})')
    ax.set_xlabel('Time Window')
    ax.set_ylabel('Number of Flips (On<->Off)')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flip_rate_comparison.png'), dpi=300)
    plt.close(fig)


# --- Analysis 2 Functions ---

def plot_metrics_vs_time(gt_eval_series, pred_eval_series, timestamps, total_possible_pairs_per_ts, output_dir):
    """Calculates metrics cumulatively up to each timestamp and plots them."""
    metrics_over_time = defaultdict(list)
    timestamps_sorted = sorted(timestamps)

    # Align indices once
    common_index = gt_eval_series.index.intersection(pred_eval_series.index)
    gt_aligned = gt_eval_series[common_index]
    pred_aligned = pred_eval_series[common_index]

    # Get the MultiIndex levels for filtering by time
    pairs = gt_aligned.index.get_level_values(0)
    times = gt_aligned.index.get_level_values(1)


    for t_idx, t in enumerate(timestamps_sorted):
        # Filter data up to current time t
        mask = times <= t
        gt_cumulative = gt_aligned[mask]
        pred_cumulative = pred_aligned[mask]

        # Calculate total possible interactions up to this time
        # Assuming total_possible_pairs_per_ts is the number of unique s*o pairs
        total_possible_cumulative = total_possible_pairs_per_ts * (t_idx + 1)

        # Calculate metrics for cumulative data
        metrics = calculate_metrics(gt_cumulative, pred_cumulative, total_possible_cumulative)

        # Store metrics
        metrics_over_time['Time'].append(t)
        for key in ['Recall', 'Precision', 'F1', 'MCC', 'TPR', 'FPR']:
            metrics_over_time[key].append(metrics[key])

    metrics_df = pd.DataFrame(metrics_over_time)

    # Plotting
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    metrics_to_plot = ['Recall', 'Precision', 'F1', 'MCC', 'TPR', 'FPR']

    for i, metric in enumerate(metrics_to_plot):
        axes[i].plot(metrics_df['Time'], metrics_df[metric], marker='.', linestyle='-', label=metric)
        axes[i].set_title(f'Cumulative {metric} vs. Time')
        axes[i].set_ylabel(metric)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        if i >= 4: # Bottom row
             axes[i].set_xlabel('Time Stamp')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_time.png'), dpi=300)
    plt.close(fig)

def plot_cumulative_error(gt_eval_series, pred_eval_series, timestamps, output_dir):
    """Calculates and plots the cumulative sum of errors (FP + FN) over time."""
    errors_over_time = defaultdict(list)
    timestamps_sorted = sorted(timestamps)

    # Align indices once
    common_index = gt_eval_series.index.intersection(pred_eval_series.index)
    gt_aligned = gt_eval_series[common_index]
    pred_aligned = pred_eval_series[common_index]

    # Get the MultiIndex levels for filtering by time
    times = gt_aligned.index.get_level_values(1)

    cumulative_fp = 0
    cumulative_fn = 0

    for t in timestamps_sorted:
         # Filter data AT current time t
        mask_t = times == t
        gt_t = gt_aligned[mask_t]
        pred_t = pred_aligned[mask_t]

        fp_t = ((pred_t == 1) & (gt_t == 0)).sum()
        fn_t = ((pred_t == 0) & (gt_t == 1)).sum()

        cumulative_fp += fp_t
        cumulative_fn += fn_t

        errors_over_time['Time'].append(t)
        errors_over_time['Cumulative FP'].append(cumulative_fp)
        errors_over_time['Cumulative FN'].append(cumulative_fn)
        errors_over_time['Cumulative Errors (FP+FN)'].append(cumulative_fp + cumulative_fn)

    errors_df = pd.DataFrame(errors_over_time)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(errors_df['Time'], errors_df['Cumulative Errors (FP+FN)'], marker='.', linestyle='-', label='Cumulative Errors (FP+FN)')
    # Optional: Plot FP and FN separately
    # ax.plot(errors_df['Time'], errors_df['Cumulative FP'], marker='.', linestyle='--', label='Cumulative FP', alpha=0.7)
    # ax.plot(errors_df['Time'], errors_df['Cumulative FN'], marker='.', linestyle='--', label='Cumulative FN', alpha=0.7)

    ax.set_title('Cumulative Errors (FP + FN) vs. Time')
    ax.set_xlabel('Time Stamp')
    ax.set_ylabel('Cumulative Count')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_error_vs_time.png'), dpi=300)
    plt.close(fig)


# --- Analysis 3 Functions ---

def bin_edges_by_frequency(train_set_post):
    """Bins edges based on their frequency in the TRAINING set.
    Returns: 
        tuple: (stability_bins (Series), pair_freq_train (Series)) or (None, None) if error.
    """
    if train_set_post.empty:
        print("Warning: Training set data is empty. Cannot bin edges by training frequency.")
        return None, None

    # Calculate frequency based on presence in the training set
    total_train_timestamps = train_set_post['time_stamp'].nunique()
    if total_train_timestamps == 0:
         print("Warning: No timestamps found in training set. Cannot calculate frequency.")
         return None, None
         
    pair_counts_train = train_set_post.groupby('pair').size()
    pair_freq_train = pair_counts_train / total_train_timestamps

    bins = [-0.01, 0.05, 0.5, 1.01] # Bins: [0, 0.05), [0.05, 0.5), [0.5, 1.0]
    labels = ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)']

    stability_bins = pd.cut(pair_freq_train, bins=bins, labels=labels, right=False)
    # stability_bins = stability_bins.cat.add_categories('Undefined').fillna('Undefined') # Alignment handled later

    print("\nInteraction Stability Binning (based on TRAINING Set Frequency):")
    print(stability_bins.value_counts())

    return stability_bins, pair_freq_train # Return both Series

def plot_metrics_by_stability(gt_eval_series, pred_eval_series, stability_bins, total_possible_pairs_per_ts, output_dir, scores_file_path):
    """Calculates metrics for each stability bin (based on train freq), plots, and writes counts/metrics."""

    metrics_by_bin = {}
    total_possible_interactions_over_time = total_possible_pairs_per_ts * len(gt_eval_series.index.get_level_values(1).unique())

    # Align indices for evaluation data
    common_index = gt_eval_series.index.intersection(pred_eval_series.index)
    gt_aligned = gt_eval_series[common_index]
    pred_aligned = pred_eval_series[common_index]
    # Get all unique pairs present in the evaluation data (test set + predictions)
    eval_pairs = gt_aligned.index.get_level_values(0).unique()

    # Align stability bins (from training) with evaluation pairs
    # Assign 'Undefined' to pairs in eval but not in train stability bins
    stability_bins_aligned = stability_bins.reindex(eval_pairs).cat.add_categories('Undefined').fillna('Undefined')
    bin_counts = stability_bins_aligned.value_counts() # Counts based on eval pairs

    # Append performance to the scores file
    with open(scores_file_path, "a") as scores:
        print("\n--- Performance by Interaction Stability (TRAINING Set Frequency) ---", file=scores)

        # Use the categories from the aligned bins
        for bin_label in stability_bins_aligned.cat.categories:
            # Skip Undefined bin here, or handle it if desired
            # if bin_label == 'Undefined': continue 

            pairs_in_bin = stability_bins_aligned[stability_bins_aligned == bin_label].index
            pair_count = bin_counts.get(bin_label, 0) # Get count for this bin

            # Rename Moderate -> Uncommon for file output
            output_label = "Uncommon (5-50%)" if bin_label == "Moderate (5-50%)" else bin_label
            output_label = "Not in Train" if bin_label == "Undefined" else output_label # Rename Undefined
            print(f"\nMetrics for {output_label} interactions ({pair_count} pairs):", file=scores)

            if pairs_in_bin.empty or pair_count == 0:
                print(f"No pairs found for bin: {output_label}")
                # Ensure bin exists in metrics dict even if empty
                metrics_by_bin[bin_label] = {k: 0 for k in ['Recall', 'Precision', 'F1', 'MCC']}
                print("No interactions found in this bin for metric calculation.", file=scores)
                continue

            # Filter evaluation series to include only pairs in the current bin
            gt_bin = gt_aligned[gt_aligned.index.get_level_values(0).isin(pairs_in_bin)]
            pred_bin = pred_aligned[pred_aligned.index.get_level_values(0).isin(pairs_in_bin)]

            # Check if filtered data is empty (can happen if pair_count > 0 but no interactions in test)
            if gt_bin.empty and pred_bin.empty:
                 metrics = {'Recall': 0, 'Precision': 0, 'F1': 0, 'MCC': 0} # Or NaN?
                 print("No interactions present in test/predictions for this bin.", file=scores)
            else:
                # Calculate metrics using local TP/FP/FN/TN calculation within the loop
                TP_bin = ((pred_bin == 1) & (gt_bin == 1)).sum()
                FP_bin = ((pred_bin == 1) & (gt_bin == 0)).sum()
                FN_bin = ((pred_bin == 0) & (gt_bin == 1)).sum()
                TN_bin = ((pred_bin == 0) & (gt_bin == 0)).sum() # TNs within the observed bin slice

                Recall_bin = TP_bin / (TP_bin + FN_bin) if (TP_bin + FN_bin) > 0 else 0
                Precision_bin = TP_bin / (TP_bin + FP_bin) if (TP_bin + FP_bin) > 0 else 0
                F1_bin = 2 * ((Precision_bin * Recall_bin) / (Precision_bin + Recall_bin)) if (Precision_bin + Recall_bin) > 0 else 0
                mcc_denom_bin = ((TP_bin + FP_bin) * (TP_bin + FN_bin) * (TN_bin + FP_bin) * (TN_bin + FN_bin))**(1/2)
                MCC_bin = (TP_bin * TN_bin - FP_bin * FN_bin) / mcc_denom_bin if mcc_denom_bin > 0 else 0
                metrics = {'Recall': Recall_bin, 'Precision': Precision_bin, 'F1': F1_bin, 'MCC': MCC_bin}
                print(f"Recall: {Recall_bin:.4f}, Precision: {Precision_bin:.4f}, F1: {F1_bin:.4f}, MCC: {MCC_bin:.4f}", file=scores)

            metrics_by_bin[bin_label] = metrics

    # Prepare DataFrame for plotting, potentially removing 'Undefined' or renaming
    metrics_df = pd.DataFrame(metrics_by_bin).T
    # Optionally drop 'Undefined' row if you don't want to plot it
    metrics_df_plot = metrics_df.drop('Undefined', errors='ignore') 
    # Ensure desired plot order if needed
    plot_order = [l for l in ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)'] if l in metrics_df_plot.index]
    metrics_df_plot = metrics_df_plot.reindex(plot_order)

    # Plotting
    if not metrics_df_plot.empty:
        fig, ax = plt.subplots(figsize=(12, 7)) # Adjusted size slightly
        metrics_df_plot.plot(kind='bar', ax=ax)

        # Add text labels to the bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3, fontsize=9)

        ax.set_title('Performance Metrics by Interaction Stability (based on Training Freq.)')
        ax.set_xlabel('Stability Bin (Training Set Frequency)') # Label reflects training freq
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=0)
        ax.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left') # Move legend outside
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(bottom=0, top=max(1.05, ax.get_ylim()[1] * 1.05))

        plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout for external legend
        plt.savefig(os.path.join(output_dir, 'metrics_by_stability_bar_trainfreq.png'), dpi=300) # New filename
        plt.close(fig)
    else:
        print("No data to plot for metrics by stability.")


def calculate_per_edge_f1(ground_truth_full, pred_full):
    """Calculates F1 score for each edge based on test set performance.
    Returns:
        DataFrame: Index=pair, Columns=['F1'] or None if calculation fails.
    """
    # Align full dataframes on pair and time
    merged = pd.merge(
        ground_truth_full.add_suffix('_gt'),
        pred_full.add_suffix('_pred'),
        left_on=['pair_gt', 'time_stamp_gt'],
        right_on=['pair_pred', 'time_stamp_pred'],
        how='inner'
    )

    if merged.empty:
        print("Warning: Merging ground truth and predictions for per-edge F1 resulted in an empty dataframe.")
        return None

    per_edge_stats = []
    for pair, group in merged.groupby('pair_gt'):
        gt = group['present_gt']
        pred = group['present_pred']

        TP = ((pred == 1) & (gt == 1)).sum()
        FP = ((pred == 1) & (gt == 0)).sum()
        FN = ((pred == 0) & (gt == 1)).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        per_edge_stats.append({'pair': pair, 'F1': f1})

    if not per_edge_stats:
        print("Could not calculate per-edge F1 stats.")
        return None

    f1_df = pd.DataFrame(per_edge_stats).set_index('pair')
    return f1_df

def plot_f1_distribution_by_stability(f1_df, stability_bins, output_dir):
    """Plots the distribution of per-edge F1 scores grouped by stability bins."""
    # Add stability bin information (derived from TRAINING set frequency)
    eval_pairs_f1 = f1_df.index
    stability_bins_aligned_f1 = stability_bins.reindex(eval_pairs_f1).cat.add_categories('Undefined').fillna('Undefined')
    
    f1_with_bins = f1_df.join(stability_bins_aligned_f1.rename('Stability Bin'))
    # Exclude pairs that were not present in the training set (Undefined bin)
    f1_df_plot = f1_with_bins[f1_with_bins['Stability Bin'] != 'Undefined']
    
    if f1_df_plot.empty:
        print("No data to plot for F1 distribution by stability (after excluding pairs not in train).")
        return

    # Plotting (Boxplot)
    plt.figure(figsize=(10, 7))
    bin_order = [b for b in ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)'] if b in f1_df_plot['Stability Bin'].unique()]
    if bin_order:
         sns.boxplot(data=f1_df_plot, x='Stability Bin', y='F1', order=bin_order, palette='viridis')
         plt.title('Distribution of Per-Pair F1 Scores by Stability Bin (based on Training Freq.)')
         plt.xlabel('Stability Bin (Training Set Frequency)')
         plt.ylabel('F1 Score (calculated on Test Set)')
         plt.grid(True, axis='y', linestyle='--', alpha=0.6)
         plt.tight_layout()
         plt.savefig(os.path.join(output_dir, 'per_edge_f1_distribution_trainfreq.png'))
         plt.close()
    else:
         print("No valid stability bins found for plotting F1 distribution.")

def plot_train_freq_vs_test_f1(pair_freq_train, f1_df, stability_bins, output_dir):
    """Generates a scatter plot of Training Frequency vs Test F1 Score."""
    if pair_freq_train is None or f1_df is None or stability_bins is None:
        print("Skipping train freq vs test f1 plot due to missing input data.")
        return

    # Combine the data: Need Training Freq, Test F1, and Stability Bin (from Train Freq)
    # Ensure stability_bins index matches pair_freq_train index initially
    combined_df = pd.DataFrame({
        'Train Frequency': pair_freq_train,
        'Stability Bin': stability_bins
    })

    # Join with Test F1 scores (indexed by pair)
    combined_df = combined_df.join(f1_df, how='inner') # Inner join keeps only pairs present in both train and test results

    if combined_df.empty:
        print("No common pairs found between training frequency data and test F1 results. Cannot generate scatter plot.")
        return

    # Drop rows where stability bin might be NaN if any slipped through (though join should handle)
    combined_df = combined_df.dropna(subset=['Stability Bin', 'Train Frequency', 'F1'])

    plt.figure(figsize=(12, 8))
    bin_order = [b for b in ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)'] if b in combined_df['Stability Bin'].unique()]
    
    sns.scatterplot(
        data=combined_df, 
        x='Train Frequency', 
        y='F1', 
        hue='Stability Bin', 
        hue_order=bin_order, 
        palette='viridis', # Using viridis palette for potentially better contrast
        alpha=0.8,       # Slightly less transparent
        s=60             # Slightly larger markers
    )

    plt.title('Training Set Frequency vs. Test Set F1 Score per Pair')
    plt.xlabel('Pair Frequency in Training Set')
    plt.ylabel('F1 Score on Test Set')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Stability Bin (Train Freq.)')
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_train_freq_vs_test_f1.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

