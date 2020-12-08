from IROA_IDX import *
def read_iroa_data(mode):
    if mode not in ('Positive','Negative'):
        raise ValueError('mode has to be Positive or Negative!')
    # dire ='/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA.db'
    dire = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_final.pkl'
    iroa = IROADataBase(dir=dire)
    iroa.read_file()
    for c in tqdm(iroa.compounds_list, desc="reading IROA compounds"):
        c.generate_mz_collection(mode=mode) ##'Negative or Positive#

    return iroa
