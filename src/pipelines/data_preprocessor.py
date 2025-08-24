"""Data preprocessing utilities to generate RE-Net input files without subprocess.

This module ports the logic from format.py into callable functions that accept
explicit parameters and write outputs to a specified directory.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np  # type: ignore
import pandas as pd

from core.utils import get_logger, FileManager


logger = get_logger(__name__)


@dataclass
class PreprocessResult:
    """Container describing preprocessed dataset outputs."""
    output_directory: str
    labels_path: str
    train_path: str
    valid_path: str
    test_path: str
    stat_path: str


def _discover_interfacea_folder(data_directory: str, replica: str) -> Path:
    """Return the directory containing .interfacea files for a given replica.

    The expected structure is:
      <data_directory>/<replica>/rep<NUM>-interfacea/
    where <replica> is like 'replica1' and the folder becomes 'rep1-interfacea'.
    """
    replica_num = replica.replace('replica', '')
    input_directory = Path(data_directory) / replica
    interface_folder = input_directory / f"rep{replica_num}-interfacea"
    if not interface_folder.exists():
        raise FileNotFoundError(f"Interfacea folder not found: {interface_folder}")
    return interface_folder


def _read_interfacea_to_df(interface_folder: Path, chain1: str, chain2: str) -> pd.DataFrame:
    """Read all .interfacea files and construct labels dataframe similar to format.py."""
    df = pd.DataFrame()
    times: list[int] = []

    files_list = sorted(os.listdir(interface_folder))
    for ifacea_file in files_list:
        # Extract numeric timestamp portion and shift by -1 to match original behavior
        time_stamp = int(np.array(re.findall('[0-9]+', ifacea_file))) - 1
        interfacea_path = interface_folder / ifacea_file
        interfacea_file = pd.read_table(
            interfacea_path,
            header=0,
            names=[
                'itype', 'chain_a', 'chain_b', 'resname_a', 'resname_b',
                'resid_a', 'resid_b', 'atom_a', 'atom_b'
            ],
            sep="\s+"
        )
        df = df._append(interfacea_file)
        times = np.hstack([times, np.repeat(time_stamp, len(interfacea_file))])  # type: ignore

    df['time_stamp'] = times
    df_inter = df.loc[
        ((df["chain_a"] == chain1) & (df["chain_b"] == chain2)) |
        ((df["chain_a"] == chain2) & (df["chain_b"] == chain1))
    ]
    df_inter['itype_int'] = pd.Categorical(df_inter.itype).codes

    df_categorical = df_inter.copy()
    df_categorical['chain_res_a'] = df_inter['chain_a'] + df_inter['resid_a'].astype(str)
    df_categorical['chain_res_b'] = df_inter['chain_b'] + df_inter['resid_b'].astype(str)
    df_categorical['chain_atom_res_a'] = df_inter['chain_a'] + df_inter['atom_a'] + df_inter['resid_a'].astype(str)
    df_categorical['chain_atom_res_b'] = df_inter['chain_b'] + df_inter['atom_b'] + df_inter['resid_b'].astype(str)
    df_categorical['res_label_a'] = pd.Categorical(df_categorical.chain_res_a).codes
    df_categorical['res_label_b'] = pd.Categorical(df_categorical.chain_res_b).codes + np.max(df_categorical['res_label_a']) + 1
    df_categorical['atom_label_a'] = pd.Categorical(df_categorical.chain_atom_res_a).codes
    df_categorical['atom_label_b'] = pd.Categorical(df_categorical.chain_atom_res_b).codes
    return df_categorical


def _df_to_dataset(df: pd.DataFrame, interaction_type: str) -> pd.DataFrame:
    """Map raw labels dataframe into RE-Net dataset columns."""
    dataset = pd.DataFrame()
    if interaction_type == 'atomic':
        dataset['subject'] = list(df['atom_label_a'])
        dataset['relation'] = list(df['itype_int'])
        dataset['object'] = list(df['atom_label_b'])
        dataset['time'] = list(df['time_stamp'])
    else:
        # default to 'residue'
        dataset['subject'] = list(df['res_label_a'])
        dataset['relation'] = list(df['itype_int'])
        dataset['object'] = list(df['res_label_b'])
        dataset['time'] = list(df['time_stamp'])
    return dataset.sort_values('time').drop_duplicates().reset_index(drop=True)


def _split_by_time(dataset: pd.DataFrame, train_ratio: float, valid_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Time-based split of dataset into train/valid/test consistent with format.py."""
    last_time = np.max(dataset['time'])
    time_split_train = last_time * float(train_ratio)
    valid_cut = 1 - float(valid_ratio)
    time_split_valid = last_time * valid_cut
    train = dataset[dataset['time'] <= time_split_train]
    valid = dataset[(dataset['time'] > time_split_train) & (dataset['time'] <= time_split_valid)]
    test = dataset[dataset['time'] > time_split_valid]

    first_stat_entity = len(set(dataset['subject'])) + len(set(dataset['object']))
    second_stat_relations = len(set(dataset['relation'])) + 1  # new relation for introduction
    third_stat_time = len(set(train['time'])) + len(set(test['time'])) + len(set(valid['time']))
    stat = np.array([first_stat_entity, second_stat_relations, third_stat_time]).T
    return train, valid, test, stat


def run_preprocessing(
    data_directory: str,
    interaction_type: str,
    replica: str,
    chain1: str,
    chain2: str,
    train_ratio: float,
    validation_ratio: float,
    output_directory: str
) -> PreprocessResult:
    """Run preprocessing and write outputs to output_directory.

    Returns paths to generated files encapsulated in PreprocessResult.
    """
    file_manager = FileManager()
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read and transform
    interface_folder = _discover_interfacea_folder(data_directory, replica)
    labels_df = _read_interfacea_to_df(interface_folder, chain1, chain2)
    dataset = _df_to_dataset(labels_df, interaction_type)
    train_df, valid_df, test_df, stat = _split_by_time(dataset, train_ratio, validation_ratio)

    # Write outputs
    labels_path = out_dir / 'labels.txt'
    train_path = out_dir / 'train.txt'
    valid_path = out_dir / 'valid.txt'
    test_path = out_dir / 'test.txt'
    stat_path = out_dir / 'stat.txt'

    np.savetxt(stat_path, stat, fmt='%d', newline=' ')
    np.savetxt(test_path, test_df.values, fmt='%d')
    np.savetxt(valid_path, valid_df.values, fmt='%d')
    np.savetxt(train_path, train_df.values, fmt='%d')

    # Save labels as in original (space-separated)
    np.savetxt(labels_path, labels_df, fmt='%s')

    logger.info(f"Preprocessing written to {out_dir}")

    return PreprocessResult(
        output_directory=str(out_dir),
        labels_path=str(labels_path),
        train_path=str(train_path),
        valid_path=str(valid_path),
        test_path=str(test_path),
        stat_path=str(stat_path)
    )

