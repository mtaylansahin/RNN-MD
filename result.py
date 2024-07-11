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
    train_set = pd.read_table(test_path, delim_whitespace=True, header=None)
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

    train_set_post = set_names_df(train_set, df_labels).drop_duplicates().reset_index(drop=True)
    train_set_post_process = pd.DataFrame()
    train_set_post_process['subject'] = train_set_post['subject_name']
    train_set_post_process['object'] = train_set_post['obj_name']
    train_set_post_process['time_stamp'] = train_set_post['time_stamp']

    train_set_post_process['type'] = "TRAIN SET"
    train_set_post_process['pair'] = train_set_post_process[['subject', 'object']].agg('-'.join, axis=1)

    train_set_post['freq_count'] = train_set_post.groupby('pair')['pair'].transform('count')
    a = train_set_post.loc[train_set_post['freq_count'] <= 50 ]
    sol = len(a["subject"].unique())
    ool = len(a["object"].unique())
    total_threshold = sol * ool

    find_matching_rows(a,test_set_post,"pair","pair")

    test_filtered = test_set_post[test_set_post["pair"].isin(a["pair"])]
    predicted_filtered = output_post[output_post["pair"].isin(a["pair"])]
    
    # Load and process train.txt, valid.txt, and test.txt from input directory
    with open(os.path.join(input_dir, "train.txt"), 'r') as fr:
        train_data = []
        train_times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            train_data.append([head, rel, tail, time])
            train_times.add(time)

    with open(os.path.join(input_dir, "valid.txt"), 'r') as fr:
        valid_data = []
        valid_times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            valid_data.append([head, rel, tail, time])
            valid_times.add(time)

    with open(os.path.join(input_dir, "test.txt"), 'r') as fr:
        test_data = []
        test_times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            test_data.append([head, rel, tail, time])
            test_times.add(time)

    total_data = train_data + valid_data + test_data
    s = []
    o = []
    for i in range(len(total_data)):
        s.append(total_data[i][0])
        o.append(total_data[i][2])
    unique_s = len(np.unique(s))
    unique_o = len(np.unique(o))
    total = unique_o * unique_s
    time_points = (test_set["time_stamp"].iloc[-1]) - (test_set["time_stamp"].iloc[0] - 1) 

    # Prediction performances of common interaction
    drop_duplicate_output = output.drop_duplicates().reset_index(drop=True)
    drop_duplicate_test = test_set.drop_duplicates().reset_index(drop=True)

    so = drop_duplicate_output["subject"].to_list()
    oo = drop_duplicate_output["object"].to_list()
    to = drop_duplicate_output["time_stamp"].to_list()

    st = drop_duplicate_test["subject"].to_list()
    ot = drop_duplicate_test["object"].to_list()
    tt = drop_duplicate_test["time_stamp"].to_list() 

    triplet_o = []
    triplet_t = []

    for i in range(0,len(so)):
        triplet_o.append(str(so[i])+str(oo[i])+str(to[i]))
    
    for i in range(0,len(st)):
        triplet_t.append(str(st[i])+str(ot[i])+str(tt[i]))

    TP = len(set(triplet_o) & set(triplet_t))
    FP = len(set(triplet_o) - set(triplet_t))
    FN = len(set(triplet_t) - set(triplet_o))
    TN = total * time_points - (TP + FP + FN)

    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    F1 = 2 * ((Precision * Recall) / (Precision + Recall))
    MCC = (TP*TN - FP*FN) / ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**(1/2)
    scores_file = os.path.join(output_dir)
    scores = open("PerformanceMetrics.txt","w")
    print("Performance metrics for commons\nRecall: {:.2f}, Precision: {:.2f}, TPR: {:.2f}, FPR: {:.2f}, F1: {:.2f}, MCC: {:.2f}".format(Recall,Precision,TPR,FPR,F1,MCC),file=scores)


    # Prediction performances of uncommon interaction

    drop_duplicate_output = predicted_filtered.drop_duplicates().reset_index(drop=True)
    drop_duplicate_test = test_filtered.drop_duplicates().reset_index(drop=True)

    so = drop_duplicate_output["subject"].to_list()
    oo = drop_duplicate_output["object"].to_list()
    to = drop_duplicate_output["time_stamp"].to_list()

    st = drop_duplicate_test["subject"].to_list()
    ot = drop_duplicate_test["object"].to_list()
    tt = drop_duplicate_test["time_stamp"].to_list()

    triplet_ot = []
    triplet_tt = []

    for i in range(0,len(so)):
        triplet_ot.append(str(so[i])+str(oo[i])+str(to[i]))

    for i in range(0,len(st)):
        triplet_tt.append(str(st[i])+str(ot[i])+str(tt[i]))
        
    TP = len(set(triplet_ot) & set(triplet_tt))
    FP = len(set(triplet_ot) - set(triplet_tt))
    FN = len(set(triplet_tt) - set(triplet_ot))
    TN = total_threshold * time_points - (TP + FP + FN)

    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    F1 = 2 * ((Precision * Recall) / (Precision + Recall))
    MCC = (TP*TN - FP*FN) / ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**(1/2)
    print("Performance Metrics for uncommons: Recall: {:.2f}, Precision: {:.2f}, TPR: {:.2f}, FPR: {:.2f}, F1: {:.2f}, MCC: {:.2f}".format(Recall,Precision,TPR,FPR,F1,MCC),file=scores)
    scores.close()
    shutil.move("PerformanceMetrics.txt",output_dir)

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

    res_merg_split = pd.merge(filtered_df_split_sorted, predicted_sorted, how='outer')
    res = res_merg_split[['residue_a', 'residue_b', 'pair']]
    res_drop = res.drop_duplicates()

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
    ax1 = sns.heatmap(test_pivot_sorted_pivot_new, ax=axes[0], cmap='Blues',linewidths=0.5, linecolor='gray',annot_kws={"weight": "bold","size":14},vmin=vmin, vmax=vmax,annot=True,fmt=".0f")
    axes[0].set_title('Ground Truth Interactions', size=20, weight="bold")
    axes[0].set_xlabel('TF', size=20, weight="bold")
    axes[0].set_ylabel("")
    #axes[0].set_ylabel('A chain', size=20, weight="bold")
    for t in ax1.texts:
        if float(t.get_text())>0:
            t.set_text(t.get_text()) #if the value is greater than 0 then I set the text 
        else:
            t.set_text("") # if not it sets an empty text
    ax1.set_yticklabels(test_pivot_sorted_pivot_new.index, size = 15, weight="bold")
    ax1.set_xticklabels(test_pivot_sorted_pivot_new.columns, size = 15, weight="bold")

    # Plot the second heatmap
    ax2 = sns.heatmap(predicted_sorted_pivot_pivot_new, ax=axes[1],yticklabels=False, cmap='Purples',linewidths=0.5, linecolor='gray',vmin=vmin, vmax=vmax,annot=True,fmt=".0f",annot_kws={"weight": "bold","size":14})
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
    ax2.set_xticklabels(predicted_sorted_pivot_pivot_new.columns, size = 15, weight="bold")
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

    fig = px.scatter(bubble_heatmap_output, x="chain_b", y="chain_a",
                size="freq", color="type", size_max=30, opacity=0.6, color_discrete_map={'ACTUAL TEST DATA':'red','PREDICTED DATA':'lightblue'})

    fig.update_layout(title="Buble heatmap of ground truth and predicted test set",
                    yaxis_nticks=len(bubble_heatmap_output["chain_a"]),xaxis_nticks=len(bubble_heatmap_output["chain_b"]),
                    height =len(bubble_heatmap_output["chain_a"])*15)
    fig.update_layout(barmode='stack')
    fig.update_xaxes(categoryorder='category ascending',title='First Chain')
    fig.update_yaxes(categoryorder='category ascending',title='Second Chain')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='black')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='black')
    fig.write_image(os.path.join(output_dir, "Bubble_Heatmap.png"))

    print("Bubble heatmap saved to Bubble_Heatmaps.png")

if __name__ == "__main__":
    main()

