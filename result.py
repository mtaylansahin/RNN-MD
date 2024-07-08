import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from math import comb
import math
import sys

data = sys.argv[1]

test_set = pd.read_table("test.txt",delim_whitespace=True, header=None)
test_set.columns = ['subject', 'relation','object', 'time_stamp']
del test_set["relation"]
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

# TEST SET

test_set_post = set_names_df(test_set, df_labels).drop_duplicates().reset_index(drop=True)
test_set_post_process = pd.DataFrame()
test_set_post_process['subject'] = test_set_post['subject_name']
test_set_post_process['object'] = test_set_post['obj_name']
test_set_post_process['time_stamp'] = test_set_post['time_stamp']

test_set_post_process['type'] = "ACTUAL TEST SET"
test_set_post_process['pair'] = test_set_post_process[['subject', 'object']].agg('-'.join, axis=1)
test_set_post_process

# TRAIN SET

train_set_post = set_names_df(train_set, df_labels).drop_duplicates().reset_index(drop=True)
train_set_post_process = pd.DataFrame()
train_set_post_process['subject'] = train_set_post['subject_name']
train_set_post_process['object'] = train_set_post['obj_name']
train_set_post_process['time_stamp'] = train_set_post['time_stamp']

train_set_post_process['type'] = "TRAIN SET"
train_set_post_process['pair'] = train_set_post_process[['subject', 'object']].agg('-'.join, axis=1)


#OUTPUT SET

output_post = set_names_df(output, df_labels).drop_duplicates().reset_index(drop=True)
output_post_process = pd.DataFrame()
output_post_process['subject'] = output_post['subject_name']
output_post_process['object'] = output_post['obj_name']
output_post_process['time_stamp'] = output_post['time_stamp']

output_post_process['type'] = "PREDICTED SET"
output_post_process['pair'] = output_post_process[['subject', 'object']].agg('-'.join, axis=1)


# TEST DATA BUBBLE HEATMAP DF

test = get_res_heatmap_df(test_set_post_process)
test['type'] = 'ACTUAL TEST DATA'
test['freq'] = test['values'] / (np.max(test_set_post_process['time_stamp']) - np.min(test_set_post_process['time_stamp']) +1)

# OUTPUT BUBBLE HEATMAP DF

output_bh = get_res_heatmap_df(output_post_process)
output_bh['type'] = 'PREDICTED DATA'
output_bh['freq'] = output_bh['values'] / (np.max(output_post_process['time_stamp']) - np.min(output_post_process['time_stamp']) +1)

# CONCAT DF TO CREATE BUBBLE DF

bubble_heatmap_output_01 = pd.concat([test, output_bh])

# CONCAT DF TO CREATE SCATTER DF

scatter_output = pd.concat([test_set_post_process, train_set_post_process, output_post_process])

fig = px.scatter(bubble_heatmap_output_01, x="chain_b", y="chain_a",
                size="freq", color="type", size_max=30, opacity=0.6, color_discrete_map={'ACTUAL TEST DATA':'red','PREDICTED DATA':'lightblue'})

fig.update_layout(title="Buble heatmap of ground truth and predicted test set",
                  yaxis_nticks=len(bubble_heatmap_output_01["chain_a"]),xaxis_nticks=len(bubble_heatmap_output_01["chain_b"]),
                 height =len(bubble_heatmap_output_01["chain_a"])*15)
fig.update_layout(barmode='stack')
fig.update_xaxes(categoryorder='category ascending')
fig.update_yaxes(categoryorder='category ascending')
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='black')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='black')
fig.write_image("Bubble_Heatmap.png")

fig = px.scatter(scatter_output, y="pair", x="time_stamp", color="type")
fig.update_layout(title="Distribution the Time Stamps of Interactions in Train Set, Test Set and Predicted Set",
                  xaxis_title="Interaction Pairs",
                  yaxis_title="Time Stamp",yaxis={"dtick":1},height = 1500)
fig.update_yaxes(categoryorder='category ascending')
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='black')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='black')
fig.write_image("Scatter.png")

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
time_points = (test_set["time_stamp"].iloc[-1]) - (test_set["time_stamp"].iloc[0] - 1)  



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
scores = open("Scores","w")
print("Recall: {:.2f}, Precision: {:.2f}, TPR: {:.2f}, FPR: {:.2f}, F1: {:.2f}, MCC: {:.2f}".format(Recall,Precision,TPR,FPR,F1,MCC),file=scores)
scores.close()
