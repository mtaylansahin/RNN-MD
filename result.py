import argparse
import os
import shutil
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


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
    numeric_part, string_part = value.split('-')
    return int(numeric_part), string_part

def find_matching_rows(df1, df2, col1, col2):
    return df2[df2[col2].isin(df1[col1])]

def main():
    args = parse_arguments()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_file_dir = args.output_file_dir

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
    # --------------------------------------------------------


    # Load raw data for total calculation (needed for TN)
    with open(os.path.join(input_dir, "train.txt"), 'r') as fr:
        train_data = []
        for line in fr: train_data.append([int(x) for x in line.split()])
    with open(os.path.join(input_dir, "valid.txt"), 'r') as fr:
        valid_data = []
        for line in fr: valid_data.append([int(x) for x in line.split()])
    with open(os.path.join(input_dir, "test.txt"), 'r') as fr:
        test_data = []
        for line in fr: test_data.append([int(x) for x in line.split()])

    total_data = train_data + valid_data + test_data
    s = [item[0] for item in total_data]
    o = [item[2] for item in total_data]
    unique_s = len(np.unique(s))
    unique_o = len(np.unique(o))
    total_possible_pairs_all = unique_o * unique_s
    time_points = (test_set["time_stamp"].max()) - (test_set["time_stamp"].min()) + 1

    # --- Performance Calculation ---
    scores_file_path = os.path.join(output_dir, "PerformanceMetrics.txt")
    scores = open(scores_file_path,"w")

    # Prediction performances of ALL interactions
    print("--- Calculating Performance for ALL Interactions ---", file=scores)
    drop_duplicate_output_all = output_post.drop_duplicates().reset_index(drop=True) # Use output_post directly
    drop_duplicate_test_all = test_set_post.drop_duplicates().reset_index(drop=True) # Use test_set_post directly

    # Use named columns for clarity
    so_all = drop_duplicate_output_all["subject_name"].astype(str).to_list()
    oo_all = drop_duplicate_output_all["obj_name"].astype(str).to_list()
    to_all = drop_duplicate_output_all["time_stamp"].astype(str).to_list()

    st_all = drop_duplicate_test_all["subject_name"].astype(str).to_list()
    ot_all = drop_duplicate_test_all["obj_name"].astype(str).to_list()
    tt_all = drop_duplicate_test_all["time_stamp"].astype(str).to_list()

    triplet_pred_all = set([so_all[i] + '_' + oo_all[i] + '_' + to_all[i] for i in range(len(so_all))])
    triplet_test_all = set([st_all[i] + '_' + ot_all[i] + '_' + tt_all[i] for i in range(len(st_all))])

    TP_all = len(triplet_pred_all & triplet_test_all)
    FP_all = len(triplet_pred_all - triplet_test_all)
    FN_all = len(triplet_test_all - triplet_pred_all)
    # TN_all calculation needs total possible pairs relevant *to the test set* if different from train/valid
    # Using total_possible_pairs_all * time_points assumes test entities are representative of all entities
    TN_all = total_possible_pairs_all * time_points - (TP_all + FP_all + FN_all)

    Recall_all = TP_all / (TP_all + FN_all) if (TP_all + FN_all) > 0 else 0
    Precision_all = TP_all / (TP_all + FP_all) if (TP_all + FP_all) > 0 else 0
    TPR_all = Recall_all
    FPR_all = FP_all / (FP_all + TN_all) if (FP_all + TN_all) > 0 else 0
    F1_all = 2 * ((Precision_all * Recall_all) / (Precision_all + Recall_all)) if (Precision_all + Recall_all) > 0 else 0
    mcc_denom_all = ((TP_all+FP_all)*(TP_all+FN_all)*(TN_all+FP_all)*(TN_all+FN_all))**(1/2)
    MCC_all = (TP_all*TN_all - FP_all*FN_all) / mcc_denom_all if mcc_denom_all > 0 else 0
    print("Performance metrics for ALL interactions:\nRecall: {:.4f}, Precision: {:.4f}, TPR: {:.4f}, FPR: {:.4f}, F1: {:.4f}, MCC: {:.4f}\n".format(Recall_all,Precision_all,TPR_all,FPR_all,F1_all,MCC_all),file=scores)


    # Prediction performances of UNCOMMON interactions
    print("--- Calculating Performance for UNCOMMON Interactions (Freq <= Threshold in Train Set) ---", file=scores)
    drop_duplicate_output_un = predicted_filtered_uncommon.drop_duplicates().reset_index(drop=True)
    drop_duplicate_test_un = test_filtered_uncommon.drop_duplicates().reset_index(drop=True)

    # Check if data exists before proceeding
    if not drop_duplicate_output_un.empty or not drop_duplicate_test_un.empty:
        so_un = drop_duplicate_output_un["subject_name"].astype(str).to_list()
        oo_un = drop_duplicate_output_un["obj_name"].astype(str).to_list()
        to_un = drop_duplicate_output_un["time_stamp"].astype(str).to_list()

        st_un = drop_duplicate_test_un["subject_name"].astype(str).to_list()
        ot_un = drop_duplicate_test_un["obj_name"].astype(str).to_list()
        tt_un = drop_duplicate_test_un["time_stamp"].astype(str).to_list()

        triplet_pred_un = set([so_un[i] + '_' + oo_un[i] + '_' + to_un[i] for i in range(len(so_un))])
        triplet_test_un = set([st_un[i] + '_' + ot_un[i] + '_' + tt_un[i] for i in range(len(st_un))])

        TP_un = len(triplet_pred_un & triplet_test_un)
        FP_un = len(triplet_pred_un - triplet_test_un)
        FN_un = len(triplet_test_un - triplet_pred_un)
        # Calculate total possible uncommon interactions based on unique subjects/objects in 'a'
        sol_un = len(a["subject"].unique())
        ool_un = len(a["object"].unique())
        total_threshold_un = sol_un * ool_un
        TN_un = total_threshold_un * time_points - (TP_un + FP_un + FN_un)

        Recall_un = TP_un / (TP_un + FN_un) if (TP_un + FN_un) > 0 else 0
        Precision_un = TP_un / (TP_un + FP_un) if (TP_un + FP_un) > 0 else 0
        TPR_un = Recall_un
        FPR_un = FP_un / (FP_un + TN_un) if (FP_un + TN_un) > 0 else 0
        F1_un = 2 * ((Precision_un * Recall_un) / (Precision_un + Recall_un)) if (Precision_un + Recall_un) > 0 else 0
        mcc_denom_un = ((TP_un+FP_un)*(TP_un+FN_un)*(TN_un+FP_un)*(TN_un+FN_un))**(1/2)
        MCC_un = (TP_un*TN_un - FP_un*FN_un) / mcc_denom_un if mcc_denom_un > 0 else 0
        print("Performance Metrics for UNCOMMON interactions:\nRecall: {:.4f}, Precision: {:.4f}, TPR: {:.4f}, FPR: {:.4f}, F1: {:.4f}, MCC: {:.4f}\n".format(Recall_un,Precision_un,TPR_un,FPR_un,F1_un,MCC_un),file=scores)
    else:
        print("No uncommon interactions found in test/predictions based on training threshold.\n", file=scores)


    # Prediction performances of COMMON interactions (frequency > threshold)
    print("--- Calculating Performance for COMMON Interactions (Freq > Threshold in Train Set) ---", file=scores)
    drop_duplicate_output_common = predicted_filtered_common.drop_duplicates().reset_index(drop=True)
    drop_duplicate_test_common = test_filtered_common.drop_duplicates().reset_index(drop=True)

    # Check if data exists before proceeding
    if not drop_duplicate_output_common.empty or not drop_duplicate_test_common.empty:
        so_common = drop_duplicate_output_common["subject_name"].astype(str).to_list()
        oo_common = drop_duplicate_output_common["obj_name"].astype(str).to_list()
        to_common = drop_duplicate_output_common["time_stamp"].astype(str).to_list()

        st_common = drop_duplicate_test_common["subject_name"].astype(str).to_list()
        ot_common = drop_duplicate_test_common["obj_name"].astype(str).to_list()
        tt_common = drop_duplicate_test_common["time_stamp"].astype(str).to_list()

        triplet_pred_common = set([so_common[i] + '_' + oo_common[i] + '_' + to_common[i] for i in range(len(so_common))])
        triplet_test_common = set([st_common[i] + '_' + ot_common[i] + '_' + tt_common[i] for i in range(len(st_common))])

        TP_common = len(triplet_pred_common & triplet_test_common)
        FP_common = len(triplet_pred_common - triplet_test_common)
        FN_common = len(triplet_test_common - triplet_pred_common)
        # Calculate total possible common interactions based on unique subjects/objects in 'b'
        sol_common = len(b["subject"].unique())
        ool_common = len(b["object"].unique())
        total_threshold_common = sol_common * ool_common
        TN_common = total_threshold_common * time_points - (TP_common + FP_common + FN_common)

        Recall_common = TP_common / (TP_common + FN_common) if (TP_common + FN_common) > 0 else 0
        Precision_common = TP_common / (TP_common + FP_common) if (TP_common + FP_common) > 0 else 0
        TPR_common = Recall_common
        FPR_common = FP_common / (FP_common + TN_common) if (FP_common + TN_common) > 0 else 0
        F1_common = 2 * ((Precision_common * Recall_common) / (Precision_common + Recall_common)) if (Precision_common + Recall_common) > 0 else 0
        mcc_denom_common = ((TP_common+FP_common)*(TP_common+FN_common)*(TN_common+FP_common)*(TN_common+FN_common))**(1/2)
        MCC_common = (TP_common*TN_common - FP_common*FN_common) / mcc_denom_common if mcc_denom_common > 0 else 0
        print("Performance Metrics for COMMON interactions (Freq > Threshold):\nRecall: {:.4f}, Precision: {:.4f}, TPR: {:.4f}, FPR: {:.4f}, F1: {:.4f}, MCC: {:.4f}\n".format(Recall_common,Precision_common,TPR_common,FPR_common,F1_common,MCC_common),file=scores)
    else:
        print("No common interactions found in test/predictions based on training threshold.\n", file=scores)

    scores.close()
"""
    df = pd.DataFrame(total_data, columns=['subject', 'relation', 'object', 'time_stamp'])

    df_set_post = set_names_df(df, df_labels).drop_duplicates().reset_index(drop=True)
    df_set_post_process = pd.DataFrame()
    df_set_post_process['subject'] = df_set_post['subject_name']
    df_set_post_process['object'] = df_set_post['obj_name']
    df_set_post_process['time_stamp'] = df_set_post['time_stamp']
    df_set_post_process['relation'] = df_set_post['relation']
    df_set_post_process['pair_relation'] = df_set_post['pair_relation']
    df_set_post_process['pair'] = df_set_post['pair']
    df_set_post_process['relation'].replace({0: 'H-bond'}, inplace=True)
    df_set_post_process['relation'].replace({1: 'Hydrophobic'}, inplace=True)
    df_set_post_process['relation'].replace({2: 'Ionic'}, inplace=True)
    df_set_post_process['Type'] = "Ground_truth"
    all_def_filtered = df_set_post_process.groupby(['time_stamp', 'pair']).head(1)

    df_set_post_process_merged = pd.concat([df_set_post_process, output_post_process], axis=0)
    df_set_post_process_merged.fillna("Predicted_Set", inplace=True)
    df_sorted = df_set_post_process_merged.sort_values(by="subject")

    # Generate heatmap for all pairwise interactions

    vmin = 0  # Minimum value
    vmax = 100  # Maximum value

    all_interactions = get_res_heatmap_df_output(all_def_filtered)
    min_value_all = all_interactions['values'].min()
    max_value_all = all_interactions['values'].max()
    all_interactions['freq'] = 1 + (all_interactions['values'] - min_value_all) / (max_value_all - min_value_all) * (100 - 1)
    all_interactions['freq'] = all_interactions['freq'].apply(lambda x: round(x, 2))
    all_interactions['numeric_order'] = all_interactions['residue_a'].apply(lambda x: int(x.split('-')[0]) if x.split('-')[0].isdigit() else float('inf'))
    all_interactions_sorted = all_interactions.sort_values(by='numeric_order', ascending=True)
    all_interactions_sorted['freq'] = all_interactions_sorted['freq'].round().astype(int)
    test_pivot_all = pd.pivot_table(all_interactions_sorted, index='residue_a', columns='residue_b', values='freq', aggfunc='sum', fill_value=0)
    test_pivot_all = test_pivot_all.reindex(all_interactions_sorted['residue_a'].unique())

    fig, ax = plt.subplots(figsize=(15, 12))

    ax = sns.heatmap(test_pivot_all, cmap='Blues', linewidths=0.5, linecolor='gray', annot_kws={"weight": "bold", "size": 16}, vmin=vmin, vmax=vmax, annot=True, fmt=".0f")
    ax.set_title('All Pairwise Interactions for MD Simulation', size=25, weight="bold")
    ax.set_xlabel('TF', size=25, weight="bold")
    ax.set_ylabel("")
    for t in ax.texts:
        if float(t.get_text()) > 0:
            t.set_text(t.get_text())  # if the value is greater than 0 then I set the text 
        else:
            t.set_text("")  # if not it sets an empty text
    ax.set_yticklabels(test_pivot_all.index, size=20, weight="bold")
    ax.set_xticklabels(test_pivot_all.columns, size=20, weight="bold")

    cbar1 = ax.collections[0].colorbar
    cbar1.ax.tick_params(labelsize=25)
    cbar1.set_label('Frequency', size=25, weight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_all.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Generate plot for interactions over time

    fig = go.Figure()

    sun = df_sorted[df_sorted["Type"] == "Ground_truth"]
    sun_2 = df_sorted[df_sorted["Type"] == "Predicted_Set"]

    # Add traces
    fig.add_trace(go.Scatter(x=sun["time_stamp"], y=sun["pair"], mode='markers',
                             marker_color="#0074D9", marker_symbol="arrow", name="Ground Truth"))
    fig.add_trace(go.Scatter(x=sun_2["time_stamp"], y=sun_2["pair"], mode='markers', opacity=0.4,
                             marker_color="#FFA500", marker_symbol="diamond", name="Predicted"))
    fig.update_layout(height=1700, width=800, template="plotly_white",
                      title="<b>Chart illustrating how interactions vary with time</b>", title_x=0.5,
                      xaxis_title="<b>Time</b>",
                      yaxis_title="<b>Pairs</b>")

    fig.write_image(os.path.join(output_dir, "interactions_during_time.png"), scale=6)

    print("Processing complete. Data saved to:", output_csv_path, prediction_csv_path, "heatmap_all.png", "interactions_during_time.png")


    # Load ground truth and predicted data
    ground_truth = pd.read_csv(os.path.join(output_dir, "ground_truth.csv"))
    predicted = pd.read_csv(os.path.join(output_dir, "prediction.csv"))
    ground_truth["freq"] = ground_truth["freq"].astype(int)
    predicted["freq"] = predicted["freq"].astype(int)

    ground_truth['pair'] = ground_truth['residue_a'] + '-' + ground_truth['residue_b']
    ground_truth['label'] = "False Negative (FN)"
    predicted['label'] = "False Positive (FP)"
    ground_truth['relation'].replace({0: 'H-bond', 1: 'Hydrophobic', 2: 'Ionic'}, inplace=True)
    ground_truth_sorted = ground_truth['residue_a'].apply(custom_sort).sort_values().index
    ground_truth_sorted = ground_truth.loc[ground_truth_sorted]
    predicted_sorted = predicted['residue_a'].apply(custom_sort).sort_values().index
    predicted_sorted = predicted.loc[predicted_sorted]

    vmin = 0  # Minimum value
    vmax = 100  # Maximum value

    filtered_df = test_set_post_process.groupby(['time_stamp', 'pair']).head(1)
    filtered_df_split = get_res_heatmap_df_output(filtered_df)
    filtered_df_split['pair'] = filtered_df_split['residue_a'] + '-' + filtered_df_split['residue_b']
    min_value_filtered_df_split = filtered_df_split['values'].min()
    max_value_filtered_df_split = filtered_df_split['values'].max()
    filtered_df_split['freq'] = 1 + (filtered_df_split['values'] - min_value_filtered_df_split) / (max_value_filtered_df_split - min_value_filtered_df_split) * (100 - 1)
    filtered_df_split['freq'] = filtered_df_split['freq'].apply(lambda x: round(x, 2))
    filtered_df_split_sorted = filtered_df_split['residue_a'].apply(custom_sort).sort_values().index
    filtered_df_split_sorted = filtered_df_split.loc[filtered_df_split_sorted]
    filtered_df_split_sorted['freq'] = filtered_df_split_sorted['freq'].round().astype(int)
    filtered_df_split_sorted['label'] = "False Negative (FN)"

    unique_rows_filtered_df_split_sorted = filtered_df_split_sorted[~filtered_df_split_sorted["pair"].isin(predicted_sorted["pair"])]

    filtered_df_split_sorted_heatmap = filtered_df_split.sort_values(by="residue_a")
    filtered_df_split_sorted_heatmap['freq'] = filtered_df_split_sorted_heatmap['freq'].round().astype(int)

    res_merg_split = pd.merge(filtered_df_split_sorted, predicted_sorted, how='outer')
    res = res_merg_split[['residue_a', 'residue_b', 'pair']]
    res_drop = res.drop_duplicates()

    res_merg_split_heatmap = pd.merge(filtered_df_split_sorted_heatmap, predicted_sorted,how='outer')
    res_heatmap = res_merg_split_heatmap[['residue_a', 'residue_b','pair']]
    res_drop_heatmap = res_heatmap.drop_duplicates()

    res_merg_ground_truth_heatmap = pd.merge(res_drop_heatmap, filtered_df_split_sorted_heatmap,how='outer')
    res_merg_predicted_heatmap = pd.merge(res_drop_heatmap, predicted_sorted,how='outer')

    res_merg_ground_truth_heatmap['numeric_order'] = res_merg_ground_truth_heatmap['residue_a'].apply(lambda x: int(x.split('-')[0]) if x.split('-')[0].isdigit() else float('inf'))
    res_merg_ground_truth_sorted_heatmap = res_merg_ground_truth_heatmap.sort_values(by='numeric_order', ascending=True)

    res_merg_predicted_heatmap['numeric_order'] = res_merg_predicted_heatmap['residue_a'].apply(lambda x: int(x.split('-')[0]) if x.split('-')[0].isdigit() else float('inf'))
    res_merg_predicted_sorted_heatmap = res_merg_predicted_heatmap.sort_values(by='numeric_order', ascending=True)

    test_pivot_heatmap = pd.pivot_table(res_merg_ground_truth_sorted_heatmap,index='residue_a', columns='residue_b', values='freq', aggfunc='sum', fill_value=0)
    predicted_sorted_pivot_heatmap = pd.pivot_table(res_merg_predicted_sorted_heatmap,index='residue_a', columns='residue_b', values='freq', aggfunc='sum', fill_value=0)

    test_pivot_sorted_heatmap = test_pivot_heatmap.reindex(res_merg_ground_truth_sorted_heatmap['residue_a'].unique())
    predicted_pivot_sorted_heatmap = predicted_sorted_pivot_heatmap.reindex(res_merg_predicted_sorted_heatmap['residue_a'].unique())

    merged_df_filtered = pd.concat([unique_rows_filtered_df_split_sorted, predicted_sorted], axis=0)
    merged_df_filtered.fillna("Predicted_Set", inplace=True)

    res_merg_ground_truth = pd.merge(res_drop, filtered_df_split_sorted, how='outer')
    res_merg_predicted = pd.merge(res_drop, predicted_sorted, how='outer')

    res_merg_ground_truth['numeric_order'] = res_merg_ground_truth['residue_a'].apply(lambda x: int(x.split('-')[0]) if x.split('-')[0].isdigit() else float('inf'))
    res_merg_ground_truth_sorted = res_merg_ground_truth.sort_values(by='numeric_order', ascending=True)

    res_merg_predicted['numeric_order'] = res_merg_predicted['residue_a'].apply(lambda x: int(x.split('-')[0]) if x.split('-')[0].isdigit() else float('inf'))
    res_merg_predicted_sorted = res_merg_predicted.sort_values(by='numeric_order', ascending=True)

    test_pivot_values = pd.pivot_table(res_merg_ground_truth, index='residue_a', columns='residue_b', values='values', aggfunc='sum', fill_value=0)
    test_pivot_values_sorted = pd.pivot_table(res_merg_ground_truth_sorted, index='residue_a', columns='residue_b', values='values', aggfunc='sum', fill_value=0)
    predicted_sorted_pivot_values = pd.pivot_table(res_merg_predicted, index='residue_a', columns='residue_b', values='values', aggfunc='sum', fill_value=0)
    predicted_sorted_pivot_values_sorted = pd.pivot_table(res_merg_predicted_sorted, index='residue_a', columns='residue_b', values='values', aggfunc='sum', fill_value=0)

    test_pivot_sorted_pivot_new = test_pivot_values_sorted.reindex(res_merg_ground_truth_sorted['residue_a'].unique())
    predicted_sorted_pivot_pivot_new = predicted_sorted_pivot_values_sorted.reindex(res_merg_predicted_sorted['residue_a'].unique())

    heatmap_similarity_score(test_pivot_values, predicted_sorted_pivot_values, os.path.join(output_dir, "heatmap_similarity_score.txt"))

    print("Similarity score saved to heatmap_similarity_score.txt")

    unique_rows_ground_truth = ground_truth_sorted[~ground_truth_sorted['pair'].isin(predicted_sorted['pair'])]
    unique_rows_predicted = predicted_sorted[~predicted_sorted['pair'].isin(ground_truth_sorted['pair'])]
    merged_df = pd.concat([ground_truth_sorted, predicted_sorted], axis=0)
    merged_df.fillna("Predicted_Set", inplace=True)
    unique_rows = pd.concat([unique_rows_filtered_df_split_sorted, unique_rows_predicted], axis=0)

    unique_rows['numeric_order'] = unique_rows['pair'].apply(lambda x: int(x.split('-')[0]) if x.split('-')[0].isdigit() else float('inf'))
    unique_rows = unique_rows.sort_values(by='numeric_order')

    # False Positive (FP) ve False Negative (FN) olarak veriyi ayır ve sırala
    df_fp = unique_rows[unique_rows['label'] == 'False Positive (FP)'].sort_values(by='numeric_order')
    df_fn = unique_rows[unique_rows['label'] == 'False Negative (FN)'].sort_values(by='numeric_order')

    # False Positive (FP) ve False Negative (FN) verilerini birleştir
    df_sorted = pd.concat([df_fn, df_fp])

    custom_colors = {'False Negative (FN)': 'Blue', 'False Positive (FP)': 'Purple'}
    plt.figure(figsize=(24, 17))
    ax = sns.barplot(y='freq', x='pair', hue='label', data=df_sorted, dodge=False, palette=custom_colors)
    plt.xlabel('Pair Interactions', size=35, weight="bold")
    plt.ylabel('Percentage', size=35, weight="bold")
    plt.yticks(fontweight='bold', fontsize=30)
    plt.xticks(fontweight='bold', fontsize=30)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.axhline(y=50, color='black', linestyle='--', linewidth=2)
    plt.legend(fontsize='large')
    ax.set_ylim(0, 100)
    plt.legend(fontsize='25')
    plt.title('Prediction Accuracy Per Pairwise Interactions Over Test Set)', size=35, weight="bold")
    plt.legend(fontsize='25')
    plt.savefig(os.path.join(output_dir, 'Prediction_Accuracy.png'), dpi=300, bbox_inches='tight')


    print("Prediction accuracy plot saved to Prediction_Accuracy.png")

    # Create a 1x2 grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the first heatmap
    ax1 = sns.heatmap(test_pivot_sorted_heatmap, ax=axes[0], cmap='Blues',linewidths=0.5, linecolor='gray',annot_kws={"weight": "bold","size":14},vmin=vmin, vmax=vmax,annot=True,fmt=".0f")
    axes[0].set_title('Ground Truth Interactions', size=20, weight="bold")
    axes[0].set_xlabel('TF', size=20, weight="bold")
    axes[0].set_ylabel("")
    #axes[0].set_ylabel('A chain', size=20, weight="bold")
    for t in ax1.texts:
        if float(t.get_text())>0:
            t.set_text(t.get_text()) #if the value is greater than 0 then I set the text 
        else:
            t.set_text("") # if not it sets an empty text
    ax1.set_yticklabels(test_pivot_sorted_heatmap.index, size = 15, weight="bold")
    ax1.set_xticklabels(test_pivot_sorted_heatmap.columns, size = 15, weight="bold")

    # Plot the second heatmap
    ax2 = sns.heatmap(predicted_pivot_sorted_heatmap, ax=axes[1],yticklabels=False, cmap='Purples',linewidths=0.5, linecolor='gray',vmin=vmin, vmax=vmax,annot=True,fmt=".0f",annot_kws={"weight": "bold","size":14})
    axes[1].set_title('Predicted Interactions', size=20, weight="bold")
    axes[1].set_xlabel('TF', size=20, weight="bold")
    axes[1].set_ylabel("")
    #axes[1].set_ylabel('A chain', size=20, weight="bold")
    for t in ax2.texts:
        if float(t.get_text())>0:
            t.set_text(t.get_text()) #if the value is greater than 0 then I set the text 
        else:
            t.set_text("") # if not it sets an empty text
            
    #ax2.set_yticklabels(predicted_160_20_20_sorted_pivot.index, size = 15, weight="bold")
    ax2.set_xticklabels(predicted_pivot_sorted_heatmap.columns, size = 15, weight="bold")
    # Add colorbars to each heatmap
    cbar1 = axes[0].collections[0].colorbar
    cbar2 = axes[1].collections[0].colorbar
    cbar1.ax.tick_params(labelsize=20)
    cbar2.ax.tick_params(labelsize=20)
    cbar1.set_label('Frequency',size=20, weight="bold")
    cbar2.set_label('Frequency',size=20, weight="bold")

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.savefig(os.path.join(output_dir, 'GroundTruth-PredictedSet_Heatmaps.png'), dpi=300, bbox_inches='tight')

    print("Heatmaps of ground truths and predicted interactions saved to GroundTruth-PredictedSet_Heatmaps.png")

    # TEST DATA BUBBLE HEATMAP DF

    test_bubble = get_res_heatmap_df(test_set_post_process)
    test_bubble['type'] = 'ACTUAL TEST DATA'
    test_bubble['freq'] = test_bubble['values'] / (np.max(test_set_post_process['time_stamp']) - np.min(test_set_post_process['time_stamp']) +1)

    # OUTPUT BUBBLE HEATMAP DF

    output_bubble = get_res_heatmap_df_output(output_post_process)
    output_bubble['type'] = 'PREDICTED DATA'
    output_bubble['freq'] = output_bubble['values'] / (np.max(output_post_process['time_stamp']) - np.min(output_post_process['time_stamp']) +1)

    # CONCAT DF TO CREATE BUBBLE DF

    bubble_heatmap_output = pd.concat([test_bubble, output_bubble])

    fig = px.scatter(bubble_heatmap_output, x="residue_b", y="residue_a",
                size="freq", color="type", size_max=30, opacity=0.6, color_discrete_map={'ACTUAL TEST DATA':'red','PREDICTED DATA':'lightblue'})

    fig.update_layout(title="Buble heatmap of ground truth and predicted test set",
                    yaxis_nticks=len(bubble_heatmap_output["residue_a"]),xaxis_nticks=len(bubble_heatmap_output["residue_b"]),
                    height =len(bubble_heatmap_output["residue_a"])*15)
    fig.update_layout(barmode='stack')
    fig.update_xaxes(categoryorder='category ascending',title='First Chain')
    fig.update_yaxes(categoryorder='category ascending',title='Second Chain')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='black')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='black')
    fig.write_image(os.path.join(output_dir, "Bubble_Heatmap.png"))

    print("Bubble heatmap saved to Bubble_Heatmaps.png")
"""


if __name__ == "__main__":
    main()

