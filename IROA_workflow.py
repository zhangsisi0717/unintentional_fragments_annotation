from IROA_IDX import *

# dire ='/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA.db'

dire = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_2.pkl'
iroa = IROA_db(dir=dire)
iroa.read_file()
for c in tqdm(iroa.compounds_list, desc="processing compounds"):
    c.generate_mz_collection(mode='Negative')
###################################################################################################
# file = dire + '/' + 'test.pkl'
# with open(dire,'rb') as f:
#     db = pkl.load(f)
# value = db['Positive']['11']
# pos_spec = IROA_Spectrum(spectrum_list=[(i,j) for i,j in value['spectrum'].items()],**value)
#
#
# for _,spec in tqdm(db['Positive'].items()):
#     if spec.get('m_z',None):
#         spec['precursor'] = spec.pop('m_z')
#
# for _,spec in tqdm(db['Negative'].items()):
#     if spec.get('m_z',None):
#         spec['precursor'] = spec.pop('m_z')
#
# with open(file,'wb') as f:
#     pkl.dump(db,f)
#
#
# a = {i:i for i in range(10)}
# with open(file,'wb') as f:
#     pkl.dump(a,f,protocol=pkl.HIGHEST_PROTOCOL)

# re = iroa.find_match(target=glc,save_matched_mz=True)
