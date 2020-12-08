from mona import *
def read_mona_data(mode):
    if mode not in ('Positive','Negative'):
        raise ValueError('mode has to be Positive or Negative!')
    mona_dire_neg ='../../../MoNA/mona_neg_cleaned.pkl'
    mona_dire_pos ='../../../MoNA/mona_pos_cleaned_new.pkl'
    mona = MonaDatabase(neg_dir=mona_dire_neg,pos_dir=mona_dire_pos)
    mona.read_file(cur_mode=mode) #Negative or Positive
    for c in tqdm(mona.compounds_list, desc="processing compounds"):
        c.generate_mz_collection(mode=mode)
    return mona
##########################################################################################################################

#def filter_invalid_spectra(rela_inte=0.01,mode='Negative'):
# import pickle as pkl
# with open('../../../MoNA/mona_lcms_pos.pkl', 'rb') as f:
#     mona_spec_pos = pkl.load(f)
#
# invalid_spec = []  #invalid==13761
# for index, value in tqdm(mona_spec_pos.items()):
#     length = len(value['spectrum_list'])
#     number = 0
#     for i in range(length):
#         if value['spectrum_list'][i][1] >= 1.:
#             number += 1
#     if number >= 100:
#         invalid_spec.append(index)
#
# mona_spec_pos_cleaned = mona_spec_pos
# for index in invalid_spec:
#     del mona_spec_pos_cleaned[index]
# with open('../../../MoNA/mona_pos_cleaned.pkl', 'wb') as f:
#     pkl.dump(mona_spec_pos_cleaned,f)
#
# ##37397
# s = (i for i in mona_spec_neg.values() if len([0 for j in i['spectrum_list'] if j[1] >= 1.]) >= 1000)
#
# r = next(s)
# # r['spectrum_list']
#
# plt.plot([i[0] for i in r['spectrum_list']], '.')
# plt.show()
# plt.plot([i[0] for i in r['spectrum_list']], [i[1] for i in r['spectrum_list']], '.')
# # plt.yscale('log')
# plt.show()
#
#
# imax = None
# nmax = 0
# for c in j:
#     n = sum(1 for i in c['spectrum'] if i == ' ') + 1
#     if n > nmax:
#         imax = c
#         nmax = n
#
# ########################################################################################################
# # mona_dire ='/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/MoNa'
# # with open('mona_lcms_neg.pkl','rb') as f:
# #     mona_spec_neg_2 = pkl.load(f)
# #
# # with open('mona_lcms_pos.pkl','rb') as f:
# #     mona_spec_pos = pkl.load(f)
# #
# # with open('MoNA-export-LC-MS-MS_Negative_Mode.json','r') as f:
# #     mona_spec_neg_raw = json.load(f)
# #
# # mona_spec_neg[0].keys()
# #dict_keys(['spectrum_id', 'name', 'spectrum_list', 'InChI', 'InChIKey', 'molecular_formula', 'total_exact_mass',
# #\'ms_level', 'collision_energy', 'precursor', 'precursor_type'])
#
# for key,value in tqdm(mona_spec_pos.items()):
#     value['molecular_formula'] = value.pop('molecular formula')
#     value['total_exact_mass'] = value.pop('total exact mass')
#     value['ms_level'] = value.pop('ms level')
#     value['collision_energy'] = value.pop('collision energy')
#     value['precursor'] = value.pop('precursor m/z')
#     value['precursor_type'] = value.pop('precursor type')
#
#
# with open('mona_lcms_neg.pkl','wb') as f:
#     pkl.dump(mona_spec_neg,f)
#
# with open('mona_lcms_pos.pkl','wb') as f:
#     pkl.dump(mona_spec_pos,f)
#
#
# mona_spec_comb = dict()
# mona_spec_comb['Negative'] = mona_spec_neg
# mona_spec_comb['Positive'] = mona_spec_pos
#
# with open('mona_spec_comb.pkl','wb') as f:
#     pkl.dump(mona_spec_comb,f)
#
# import pickle as pkl
# with open('../../../MoNA/mona_spec_comb.pkl', 'rb') as f:
#     mona_spec_comb = pkl.load(f)
#
# for key, value in tqdm(mona_spec_pos.items()):
#     for i in range(len(value['spectrum_list'])):
#         value['spectrum_list'][i] = list(value['spectrum_list'][i])
#         for j in range(len(value['spectrum_list'][i])):
#             value['spectrum_list'][i][j] = np.float(value['spectrum_list'][i][j])
#
# for i in range(len(mona_spec_neg_raw)):
#     if mona_spec_neg_raw[i]['id'] == 'UT001748':
#         print(i)
#         break
#
#
# with open(mona_dire + '/'+'comb.json') as f:
#     mona_combo = json.load(f) ##type:dic, len(mona_combo)==29
#
# with open(mona_dire+'/'+'MoNA-export-LC-MS-MS_Negative_Mode.json') as f:
#     mona_neg_raw = json.load(f) ##list: len==38480
# with open(mona_dire+'/'+'MoNA-export-LC-MS-MS_Positive_Mode.json') as f:
#     mona_neg_raw = json.load(f) ##len(86580)###
#
# # with open(mona_dire2+'/'+'lib_red_comb.json') as f:
# #     lib_red_comb = json.load(f) ##type:dic, len(mona_combo)==29
#
# ##lib_red_comb.keys()(['CID', 'InChIKey_x', 'spectrum', 'mode', 'ExactMass', 'InChIKey_y', 'InChI', \
# # 'MolecularWeight', 'IUPACName', 'MolecularFormula', 'Fingerprint2D'])
# #lib_red_comb['spectrum'].keys(): type:dict, len(spectrums) =='0'-'83317'
#
# ##lib_red_comb['mode']: type:
# #len(lib_red_comb['ExactMass']) #82548
# # mona_index_pos = []
# # mona_index_neg = []
# # for key,mode in lib_red_comb['mode'].items():
# #     if mode == 'positive':
# #         mona_index_pos.append(key)
# #     else: mona_index_neg.append(key)
# #
# # mona_keys = ['CID', 'spectrum', 'ExactMass','IUPACName', 'MolecularFormula']
# # mona_pos_spec = dict()
# # for i in tqdm(mona_index_pos, desc='process the pos spec'):
# #     mona_pos_spec[i] = dict()
# #     for j in mona_keys:
# #         mona_pos_spec[i][j] = lib_red_comb[j][i]
# #
# # with open(mona_dire2+'/'+'mona_pos_spec.pkl','wb') as f:
# #     pkl.dump(mona_pos_spec,f)
#
#
#
# #mona_neg_raw[0]['compound'][0]['inchi']
# #mona_neg_raw[0]['compound'][0]['inchiKey']
# #mona_neg_raw[0]['compound'][0]['metaData'][number]['name'] ->'total exact mass'
# #mona_neg_raw[0]['compound'][0]['metaData'][number]['value'] ->253.052112212
# #dict_keys(['compound', 'id', 'dateCreated', 'lastUpdated', 'lastCurated', 'metaData',
# # 'annotations', 'score', 'spectrum', 'splash', 'submitter', 'tags', 'library'])
# # success_index=[]
# failed_index =[]
# mona_neg_spectra = dict()
# temp =[]
# # for i in tqdm(range(len(mona_neg_raw)),leave=True):
# for i in tqdm(range(57983,len(mona_neg_raw)),leave=True):
#     spec_id = mona_neg_raw[i]['id']
#     # inchi = mona_neg_raw[i]['compound'][0]['inchi']
#     # inchiKey = mona_neg_raw[i]['compound'][0]['inchiKey']
#     if mona_neg_raw[i]['compound'][0]['names']:
#         name = mona_neg_raw[i]['compound'][0]['names'][0]['name']
#     else:
#         failed_index.append(i)
#         continue
#     spec = mona_neg_raw[i]['spectrum']  ##parse_spectrum
#     mz = []
#     intensity = []
#     index = 0
#     for jd in range(index + 1, len(spec)):
#         if spec[jd] == ':':
#             mz.append(spec[index:jd])
#             index = jd + 1
#             continue
#         elif spec[jd] == ' ':
#             intensity.append(spec[index:jd])
#             index = jd + 1
#             continue
#         elif jd == len(spec) - 1:
#             intensity.append(spec[index:])
#     spectrum_list = list(zip(mz, intensity))
#     temp +=[('spectrum_id', spec_id), ('name',name),('spectrum_list', spectrum_list)]
#     # ('inchi', inchi), ('inchiKey', inchiKey)
#
#     for j in mona_neg_raw[i]['compound'][0]['metaData']: ##cmp_info##
#         if j['name'] =='molecular formula':
#             formula = j['value']
#             temp.append(('molecular formula',formula))
#         elif j['name'] == 'total exact mass':
#             exact_mass = j['value']
#             temp.append(('total exact mass',exact_mass))
#         elif j['name'] == 'InChI':
#             temp.append(('InChI',j['value']))
#         elif j['name'] == 'InChIKey':
#             temp.append(('InChIKey',j['value']))
#     for k in mona_neg_raw[i]['metaData']: ##meta_data
#         if k['name'] == 'ms level':
#             ms_level = k['value']
#             temp.append(('ms level',ms_level))
#         elif k['name'] =='collision energy':
#             collision_energy = k['value']
#             temp.append(('collision energy',collision_energy))
#         elif k['name'] == 'precursor m/z':
#             precursor = k['value']
#             temp.append(('precursor m/z',precursor))
#         elif k['name'] == 'precursor type':
#             pre_type = k['value']
#             temp.append(('precursor type',pre_type))
#     mona_neg_spectra[i] = dict(temp)
#
# #######################################################################################







