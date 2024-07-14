import pandas as pd
import numpy as np

clone_cn_profiles = pd.read_csv("syns_data/clone_cn_profiles.tsv", sep="\t")
ctdna = pd.read_csv("syns_data/ctdna.tsv", sep="\t")

ctdna_cleaned = ctdna.dropna(subset=['copy'])

merged_data = pd.merge(clone_cn_profiles, ctdna_cleaned, on=["chr", "start", "end"])

clone_cn_profiles_array = merged_data[['A', 'C', 'E', 'normal']].to_numpy()
ctdna_array = merged_data['copy'].to_numpy()

print(clone_cn_profiles_array)
print(ctdna_array)
