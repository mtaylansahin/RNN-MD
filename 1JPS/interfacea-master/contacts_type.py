import interfacea as ia
import pandas as pd

# below given lines are there to print the whole dataframe content
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

mol = ia.read('complex.pdb')

analyzer = ia.InteractionAnalyzer(mol)
analyzer.get_hydrophobic(include_intra=True)
analyzer.get_hbonds(include_intra=True)
analyzer.get_ionic(include_intra=True)

print(analyzer.itable._table) 
