import pandas as pd
import numpy as np
from math import comb
import math
import os
import seaborn as sns
import sys

data = sys.argv[1]
split_mcc = int(sys.argv[2])

test = pd.read_table("test.txt",delim_whitespace=True, header=None)
test.columns = ['subject', 'relation','object', 'time_stamp']
valid = pd.read_table("valid.txt",delim_whitespace=True, header=None)
valid.columns = ['subject', 'relation','object', 'time_stamp']
del valid["relation"]
train_set = pd.read_table("train.txt",delim_whitespace=True, header=None)
train_set.columns = ['subject', 'relation','object', 'time_stamp']
del train_set["relation"]
output = pd.read_table("{}".format(data),delim_whitespace=True, header=None)
output.columns = ['subject','object', 'time_stamp']

df_labels = pd.read_table("labels.txt", sep=" ",header=None)
df_labels.columns = ['itype', 'chain_a', 'chain_b', 'resname_a', 'resname_b',
                             'resid_a',
                             'resid_b', 'atom_a', 'atom_b', 'time', 'itype_int',
                             'chain_res_a', 'chain_res_b', 'chain_atom_res_a', 'chain_atom_res_b', 'res_label_a',
                             'res_label_b', 'atom_label_a', 'atom_label_b']
df_labels['chain_a_b_time'] =    df_labels['chain_res_a']    +   df_labels['chain_res_b']    +  df_labels['time'].astype(str)
df_labels['chain_a_b'] =    df_labels['chain_res_a']    +   df_labels['chain_res_b'] 
df_labels

def get_df_res_label(df_ifacea):
    df_label = pd.DataFrame()
    df_label['naming'] = list(df_ifacea['chain_res_a']) + list(df_ifacea['chain_res_b'])
    df_label['label'] = list(df_ifacea['res_label_a']) + list(df_ifacea['res_label_b'])
    df_label = df_label.drop_duplicates().reset_index(drop=True)
    return df_label

def get_label_dict(df_naming):
    zip_iterator = zip(df_naming['label'],df_naming['naming'])
    dict_labels = dict(zip_iterator)
    return dict_labels

def set_names_df(df_out, df_ifacea):
    subject_name, obj_name = [], []
    df_outputs_type = df_out.copy()

    dict_labels = get_label_dict(get_df_res_label(df_ifacea))

    # print(dict_labels)
    for i in range(len(df_outputs_type)):
        subject_name.append(dict_labels.get(df_outputs_type.iloc[i]['subject']))
        obj_name.append(dict_labels.get(df_outputs_type.iloc[i]['object']))

    df_outputs_type['subject_name'] = subject_name
    df_outputs_type['obj_name'] = obj_name
    df_outputs_type['pair'] = df_outputs_type['subject_name'] + '-' + df_outputs_type['obj_name']
    df_outputs_type['pair_time'] = df_outputs_type['pair'] + ' ' + df_out['time_stamp'].astype(str)
    return df_outputs_type

def get_res_heatmap_df(df_out):
    df_heat = df_out.pair.value_counts()
    chain_a, chain_b, values = [], [], []
    for pair in list(set(df_out['pair'])):
        chain_a.append(pair.split('-')[0])
        chain_b.append(pair.split('-')[1])
        values.append(df_heat[pair])
    df_heatvals = pd.DataFrame()
    df_heatvals['chain_a'] = chain_a
    df_heatvals['chain_b'] = chain_b
    df_heatvals['values'] = values
    return df_heatvals

# VALID SET

valid_set_post = set_names_df(valid, df_labels).drop_duplicates().reset_index(drop=True)
valid_set_post_process = pd.DataFrame()
valid_set_post_process['subject'] = valid_set_post['subject_name']
valid_set_post_process['object'] = valid_set_post['obj_name']
valid_set_post_process['time_stamp'] = valid_set_post['time_stamp']

valid_set_post_process['type'] = "VALID SET"
valid_set_post_process['pair'] = valid_set_post_process[['subject', 'object']].agg('-'.join, axis=1)
valid_set_post_process

# TRAIN SET

train_set_post = set_names_df(train_set, df_labels).drop_duplicates().reset_index(drop=True)
train_set_post_process = pd.DataFrame()
train_set_post_process['subject'] = train_set_post['subject_name']
train_set_post_process['object'] = train_set_post['obj_name']
train_set_post_process['time_stamp'] = train_set_post['time_stamp']

train_set_post_process['type'] = "TRAIN SET"
train_set_post_process['pair'] = train_set_post_process[['subject', 'object']].agg('-'.join, axis=1)
train_set_post_process

# Output

output_post = set_names_df(output, df_labels).drop_duplicates().reset_index(drop=True)
output_post_process = pd.DataFrame()
output_post_process['subject'] = output_post['subject_name']
output_post_process['object'] = output_post['obj_name']
output_post_process['time_stamp'] = output_post['time_stamp']

output_post_process['type'] = "PREDICTED SET"
output_post_process['pair'] = output_post_process[['subject', 'object']].agg('-'.join, axis=1)

train_set_post['freq_count'] = train_set_post.groupby('pair')['pair'].transform('count')
a = train_set_post.loc[train_set_post['freq_count'] <= split_mcc ]
sol = len(a["subject"].unique())
ool = len(a["object"].unique())
total_threshold = sol * ool
total_threshold

def find_matching_rows(df1, df2, col1, col2):
    return df2[df2[col2].isin(df1[col1])]

find_matching_rows(a,valid_set_post,"pair","pair")
valid_filtered = valid_set_post[valid_set_post["pair"].isin(a["pair"])]
predicted_filtered = output_post[output_post["pair"].isin(a["pair"])]

with open("train.txt", 'r') as fr:
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
with open("valid.txt", 'r') as fr:
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
with open("test.txt", 'r') as fr:
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
for i in range(0,len(total_data)):
    s.append(total_data[i][0])
    o.append(total_data[i][2])
unique_s = len(np.unique(s))
unique_o = len(np.unique(o))
total = unique_o * unique_s
print(unique_s * unique_o)
print(unique_s,unique_o)
time_points = (test["time_stamp"].iloc[-1]) - (test["time_stamp"].iloc[0] - 1)

# Prediction performances of %100 dataset

drop_duplicate_output = output_post.drop_duplicates().reset_index(drop=True)
drop_duplicate_valid = valid_set_post.drop_duplicates().reset_index(drop=True)

so = drop_duplicate_output["subject"].to_list()
oo = drop_duplicate_output["object"].to_list()
to = drop_duplicate_output["time_stamp"].to_list()

st = drop_duplicate_valid["subject"].to_list()
ot = drop_duplicate_valid["object"].to_list()
tt = drop_duplicate_valid["time_stamp"].to_list()

triplet_o = []
triplet_t = []

for i in range(0,len(so)):
    triplet_o.append(str(so[i])+str(oo[i])+str(to[i]))

for i in range(0,len(st)):
    triplet_t.append(str(st[i])+str(ot[i])+str(tt[i]))
    
TP = len(set(triplet_o) & set(triplet_t))
FP = len(set(triplet_o) - set(triplet_t))
FN = len(set(triplet_t) - set(triplet_o))
TN =  total * time_points - (TP + FP + FN)

Recall = TP / (TP + FN)
Precision = TP / (TP + FP)
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
F1 = 2 * ((Precision * Recall) / (Precision + Recall)) 
MCC = (TP*TN - FP*FN) / ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**(1/2)
scores = open("Scores","w")
print("Recall: {:.2f}, Precision: {:.2f}, TPR: {:.2f}, FPR: {:.2f}, F1: {:.2f}, MCC: {:.2f}".format(Recall,Precision,TPR,FPR,F1,MCC),file=scores)

# Prediction performances of %50 dataset

drop_duplicate_output = predicted_filtered.drop_duplicates().reset_index(drop=True)
drop_duplicate_test = valid_filtered.drop_duplicates().reset_index(drop=True)

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
print("Recall: {:.2f}, Precision: {:.2f}, TPR: {:.2f}, FPR: {:.2f}, F1: {:.2f}, MCC: {:.2f}".format(Recall,Precision,TPR,FPR,F1,MCC),file=scores)
scores.close()
