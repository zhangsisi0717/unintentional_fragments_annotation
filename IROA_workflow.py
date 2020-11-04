from IROA_IDX import *

# dire ='/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA.db'

dire = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_final.pkl'
iroa = IROADataBase(dir=dire)
iroa.read_file()
for c in tqdm(iroa.compounds_list, desc="reading IROA compounds"):
    c.generate_mz_collection(mode='Positive') ##'Negative or Positive#


