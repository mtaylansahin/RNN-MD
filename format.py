from matplotlib import pyplot as plt
import numpy as np # type: ignore
import os
import re
import pandas as pd
import sys
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')


def save_dataset(dataset, df):
    """
    This method saves the given dataset in RE-Net input format

    Parameters:
    dataset: the dataset name to be saved.

    Returns:
    train: dataset of training pairs
    test: dataset of test pairs
    valid: dataset of validation pairs
    """
    #train = df_for_introduction(df)
    #train = train.append(dataset[0:math.ceil(len(dataset) * 0.8)])
    #valid = dataset[len(train): len(train) + math.floor(len(dataset) * 0.1)]
    #test = dataset[len(train) + len(valid): len(train) + len(valid) + math.ceil(len(dataset) * 0.1)]
    last_time = np.max(dataset['time'])
    time_split_train = last_time * float(sys.argv[6])
    valid_ratio = 1 - float(sys.argv[7])
    time_split_valid = last_time * valid_ratio
    train = dataset[dataset['time'] <= time_split_train]
    valid = dataset[(dataset['time'] > time_split_train) & (dataset['time'] <= time_split_valid)]
    test = dataset[dataset['time'] > time_split_valid]
    print(train[0:])
    first_stat_entity = len(set(dataset['subject'])) + len(set(dataset['object']))
    second_stat_relations = len(set(dataset['relation'])) + 1 #new relation for introduction
    third_stat_time = len(set(train['time'])) + len(set(test['time'])) + len(set(valid['time']))
    stat = np.array([first_stat_entity, second_stat_relations, third_stat_time]).T

    #TODO "/"  could be a problem for Microsoft users
    np.savetxt(os.getcwd() + '/' + sys.argv[1] +'/stat.txt', stat, fmt='%d', newline=' ')
    np.savetxt(os.getcwd() + '/' + sys.argv[1] +'/test.txt', test.values, fmt='%d')
    np.savetxt(os.getcwd() + '/' + sys.argv[1] +'/valid.txt', valid.values, fmt='%d')
    np.savetxt(os.getcwd() + '/' + sys.argv[1] +'/train.txt', train.values, fmt='%d')

def df_for_introduction(df):
    df_introduce_objects = pd.DataFrame()
    label_sub = 'res_label_a'
    label_obj = 'res_label_b'

    if sys.argv[2] == 'atomic':
        label_sub = 'atom_label_a'
        label_obj = 'atom_label_b'

    len_sub = len(set(df[label_sub]))
    len_obj = len(set(df[label_obj]))

    intro_to_subjects = list(set(df[label_obj]))
    intro_to_objects = list(set(df[label_sub]))

    if (len_obj > len_sub):
        intro_to_objects = list(set(df[label_sub])) + list(np.repeat(np.max(df[label_sub]), len_obj - len_sub))

    if (len_sub > len_obj):
        intro_to_subjects = list(set(df[label_obj])) + list(np.repeat(np.max(df[label_obj]), len_sub - len_obj))

    df_introduce_objects['subject'] = intro_to_subjects
    df_introduce_objects['relation'] = np.repeat(3, max([len_obj, len_sub]))
    df_introduce_objects['object'] = intro_to_objects
    df_introduce_objects['time'] = np.repeat(0, max([len_obj, len_sub]))
    return df_introduce_objects

def df_to_dataset(df):
    dataset = pd.DataFrame()


    if sys.argv[2] == 'atomic':           
        dataset['subject'] = list(df['atom_label_a'])
        dataset['relation'] = list(df['itype_int'])
        dataset['object'] = list(df['atom_label_b'])
        dataset['time'] = list(df['time_stamp'])

    elif sys.argv[2] == 'residue':
        dataset['subject'] = list(df['res_label_a'])
        dataset['relation'] = list(df['itype_int'])
        dataset['object'] = list(df['res_label_b'])
        dataset['time'] = list(df['time_stamp'])
    return dataset.sort_values('time').drop_duplicates().reset_index(drop=True)

class Format:
    try:
        complex = sys.argv[1]
    except:
        print("""
    **********************************
    Please specify a valid complex 
    **********************************
    """)

    def __init__(self):
        self.current_replica = None
        self.input_directory = None
        self.input_files = None
        self.files_list = None

    def update_path(self, rep_no):
        """This function is for easily navigating through the interfacea files for complexes.
            Changes global variables: current_replica, input_directory, input_files, out_files, files_list

            Parameters:
            rep_no(int): the replica number

        """
        print(os.getcwd())
        self.current_replica = 'replica' + str(rep_no)
        self.input_directory = os.getcwd() + '/' + sys.argv[1] + '/' + self.current_replica
        self.input_files = self.input_directory + '/rep' + str(rep_no) + '-interfacea'  # for taking all input files (.interfacea)
        self.files_list = os.listdir(self.input_files)  # list of all interfacea files for given replica
        self.out_files = self.input_directory + '/merged_outputs' 
        print("current replica: " + self.current_replica)
        print("input directory: " + self.input_directory)

    ##WARNING: this is for the current file naming convention for storing .interfacea files
    def interfacea_to_df(self):
        df = pd.DataFrame()

        times = []

        self.update_path(sys.argv[3])

        for ifacea_file in self.files_list:
            time_stamp = int(np.array(re.findall('[0-9]+', ifacea_file)))-1

            interfacea_file = pd.read_table(self.input_files + '/' + ifacea_file, header=0,
                                                names=['itype', 'chain_a', 'chain_b', 'resname_a', 'resname_b',
                                                       'resid_a',
                                                       'resid_b', 'atom_a', 'atom_b'], sep="\s+")
            df = df.append(interfacea_file)
            times = np.hstack([times, np.repeat(time_stamp, len(interfacea_file))])

        df['time_stamp'] = times
        df_inter = df.loc[
            ((df["chain_a"] == sys.argv[4]) & (df["chain_b"] == sys.argv[5])) | ((df["chain_a"] == sys.argv[5]) & (df["chain_b"] == sys.argv[4]))]
        df_inter['itype_int'] = pd.Categorical(df_inter.itype).codes

        df_categorical = df_inter
        df_categorical['chain_res_a'] = df_inter['chain_a'] + df_inter['resid_a'].astype(str)
        df_categorical['chain_res_b'] = df_inter['chain_b'] + df_inter['resid_b'].astype(str)
        df_categorical['chain_atom_res_a'] = df_inter['chain_a'] + df_inter['atom_a'] + df_inter['resid_a'].astype(str)
        df_categorical['chain_atom_res_b'] = df_inter['chain_b'] + df_inter['atom_b'] + df_inter['resid_b'].astype(str)
        df_categorical['res_label_a'] = pd.Categorical(df_categorical.chain_res_a).codes
        df_categorical['res_label_b'] = pd.Categorical(df_categorical.chain_res_b).codes + np.max(df_categorical['res_label_a']) + 1
        df_categorical['atom_label_a'] = pd.Categorical(df_categorical.chain_atom_res_a).codes
        df_categorical['atom_label_b'] = pd.Categorical(df_categorical.chain_atom_res_b).codes
        ###NEW###
        np.savetxt(os.getcwd() + '/' + sys.argv[1] +'/labels.txt', df_categorical, fmt = "%s")
        return df_categorical
    
    def merged_interactions(self):
        os.makedirs(self.out_files, exist_ok=True)

        for i in range(0, len(self.files_list)):
            if self.files_list[i].endswith(".interfacea"):
                # Read the input file
                df = pd.read_table(self.input_files + "/" + self.files_list[i], delim_whitespace=True)
                
                # Drop the last two columns
                df_colDrop = df.drop(columns=[df.columns[-2], df.columns[-1]])
                
                # Remove duplicate rows
                df_dropped = df_colDrop.drop_duplicates()
                
                # Filter and write to respective files for hydrophobics, hbonds, and ionics
                filtered_df_hydrophobic = df_dropped[df_dropped["itype"] == 'hydrophobic']
                with open(self.out_files + "/" + "hydrophobics", 'a') as file:
                    filtered_df_hydrophobic.to_csv(file, header=False, index=False, sep="\t")

                filtered_df_hbond = df_dropped[df_dropped["itype"] == 'hbond']
                with open(self.out_files + "/" + "hbonds", 'a') as file:
                    filtered_df_hbond.to_csv(file, header=False, index=False, sep="\t")

                filtered_df_ionic = df_dropped[df_dropped["itype"] == 'ionic']
                with open(self.out_files + "/" + "ionics", 'a') as file:
                    filtered_df_ionic.to_csv(file, header=False, index=False, sep="\t")
    
    def frequency_graph(self, interact_type): #valid interact types: hydrophobic, ionics, hbonds

        b=pd.read_csv(self.out_files+"/"+ interact_type, sep="\t", header=None) #
        b.columns = ["itype", "chain_a", "chain_b", "resname_a", "resname_b", "resid_a", "resid_b" ]
        b["resname_a"] = b["resname_a"].astype(str) + b["chain_a"].astype(str) + b["resid_a"].astype(str)
        b["resname_b"] = b["resname_b"].astype(str) + b["chain_b"].astype(str) + b["resid_b"].astype(str)


        #Intermonomer interactions
        b_inter = b.loc[((b["chain_a"] == "A") & (b["chain_b"] == "C")) | ((b["chain_a"] == "C") & (b["chain_b"] == "A")) ]

        #Pairwise interactions
        b_inter['pairwise'] = b_inter['resname_a'].str.cat(b_inter['resname_b'],sep="-")

        #interaction freqs
        b_inter['aa_freq'] = b_inter.groupby('resname_a')['resname_a'].transform('count')
        b_inter['nuc_freq'] = b_inter.groupby('resname_b')['resname_b'].transform('count')
        b_inter['all_freq'] = b_inter.groupby('itype')['itype'].transform('count')
        b_inter['pair_freq'] = b_inter.groupby('pairwise')['pairwise'].transform('count') / 4 #over 4 reps
        b_inter['pair'] = b_inter.groupby('pairwise')['pairwise'].transform('count')
        b_inter['freq'] = (b_inter['pair'] / 200) * 100

        #print(b_inter)

        b_inter_sorted=b_inter.sort_values(('freq') , ascending=False)

        #setting colors
        colors = "Blues_r";
        if interact_type == "ionics":
            colors = "Reds_r"

        if interact_type == "hbonds":
            colors = "Greens_r"

        plt.figure(figsize=(13,8))
        ax = sns.barplot(x="pairwise", y="freq", palette=colors, data=b_inter_sorted)
        plt.xticks(rotation='vertical')


        plt.xticks(fontsize=13, fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')
        plt.ylim(ymax = 100)
        plt.xlabel('Interactions',  fontweight='bold')
        plt.ylabel('Percentage of ' + interact_type[:-1] + ' interactions', fontweight='bold')
        plt.title('General '  + interact_type[:-1] + ' interaction profile of 1JPS for Chain A-C Interaction',  fontweight='bold')


        plt.savefig(os.getcwd() + '/' + sys.argv[1] +"/general-" + interact_type +"-pairwise.png",dpi=300, bbox_inches="tight", format="png")

        return b_inter



def main():
    s = Format()
    df = s.interfacea_to_df()
    dataset = df_to_dataset(df)
    save_dataset(dataset, df)
    s.merged_interactions()
    b_inter = s.frequency_graph("ionics")
    b_inter = s.frequency_graph("hydrophobics")
    b_inter = s.frequency_graph("hbonds")


if __name__ == "__main__":
    main()
