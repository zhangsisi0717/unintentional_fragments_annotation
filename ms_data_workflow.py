from IROA_workflow import *
from MoNA_workflow import *
from mz_cloud_workflow import *
from spec_matching_result import *
from check_annotation_accurary import *
from msdata import *
from result_generate import *
import copy

# m = MSData.from_files('3T3_pro','/../../../Data/3T3_pro')
# m = MSData.from_files('3T3_pro','/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/sisi_codes_test/Data/3T3_pro')
m = MSData.from_files('IROA_MS1_pos','/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_09162020/IROA_MS1_pos')
m.apply_smooth_conv(m.gaussian_kernel(2, 12), rt_range=(0., np.inf))

#peak_detection_options = {'height': 1E-2, 'range_threshold': 1E-4, 'prominence': 1E-2}
#h_max = np.max(ints) - np.min(ints)
#prominence = h_max * prominence, height = h_max * height,
#delta = (rts[1:] - rts[:-1]).mean(), width = width / delta

#afer peak_detection:range_threshold = peak_heights * range_threshold, beyong range,set to 0
peak_detection_options = {'height': 0.1, 'range_threshold': 0.10, 'prominence': 0.1}
m.perform_feature_peak_detection(**peak_detection_options)
m.remove_duplicates_detection()
# m.perform_feature_peak_detection()
by = np.max(m.ints_raw, axis=1)

m.sort_feature(by=by)

gen_base_opt = dict(
    l1=2.,
    min_sin=5E-3,  # min_sin=5E-2,
    min_base_err=5E-4,  # min_base_err=5E-2,
    min_rt_diff=1.,
    max_cos_sim=0.90
)
m.generate_base(reset=True, allowed_n_overlap=(1,2), **gen_base_opt)
m.generate_base(reset=False, allowed_n_overlap=2, **gen_base_opt)
m.generate_base(reset=False, allowed_n_overlap=3, **gen_base_opt)
m.generate_base(reset=False, allowed_n_overlap=(4, None), **gen_base_opt)

# m.base_cos_similarity(plot=True)
# plt.show()
#
# m.base_rt_diff(plot=True)
# plt.show()


m.perform_peak_decomposition(l1=1.)

spec_params = dict(
    threshold=1E-5,
    # threshold=0.,
    max_rt_diff=4.,
    # max_rt_diff=np.inf,
    # max_mse=2E-2,
    max_mse=np.inf, ###np.inf
    # mz_upperbound=5.
    mz_upperbound=np.inf,
    # min_cos=.99
    min_cos=0.85
)
m.gen_spectrum(0, plot=False, load=False, **spec_params)
for i in tqdm(range(m.n_base), desc='generating spectrum'):
    m.gen_spectrum(i, plot=False, load=False, **spec_params)

# ###check base group and find_match within each group and plot###
test_path = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result_pos/pos_ms1_matching_1'
final_matching_results = []
# save_index = [i for i in range(len(m.base_index))if i % 30 == 0]
# save_index.append(len(m.base_index)-1)
idx_set=[]
import datetime
start = datetime.datetime.now()
for i in range(len(m.base_index)):
    print(f'current running basis {i}')
    spec = m.base_info[m.base_index[i]].spectrum
    spec_2 = copy.deepcopy(spec)
    if i in idx_set:
        result = GroupMatchingResult(recons_spec=spec_2,
                                 base_index_relative=i,
                                 base_index_abs=m.base_index[i],
                                     mode='Positive')
        result.gen_mzc_matching_result(total_layer_matching=1,n_candidates_further_matched=5,database=mzc,transform=None)
        result.gen_mona_matching_result(total_layer_matching=1,n_candidates_further_matched=5,database=mona,transform=None) ##start from 0th match##
        result.gen_iroa_matching_result(total_layer_matching=1,n_candidates_further_matched=5,database=iroa,transform=None)

    else:
        result = GroupMatchingResult(recons_spec=spec_2,
                                     base_index_relative=i,
                                     base_index_abs=m.base_index[i],
                                     mode='Positive')
        # result.gen_mzc_matching_result(total_layer_matching=1,n_candidates_further_matched=5,database=mzc,transform=None)
        result.gen_mona_matching_result(total_layer_matching=1,n_candidates_further_matched=5,database=mona,transform=None) ##start from 0th match##
        result.gen_iroa_matching_result(total_layer_matching=1,n_candidates_further_matched=5,database=iroa,transform=None)
    # result.gen_recur_matched_peaks()
    # result.count_total_matched_peaks()
    result.summarize_matching_re_all_db(mzc=False, mona=True, iroa=True)
    result.remove_db()
    final_matching_results.append(result)
    # if i in save_index:
    #     index = save_index.index(i)
    #     if index > 0:
    #         fi_re = [[j.sum_matched_results_iroa, j.sum_matched_results_mona, j.sum_matched_results_mzc] for j in
    #                  final_matching_results[save_index[int(index - 1)]:int(i + 1)]]
    #         prev = save_index[int(index - 1)]
    #         name = 'neg_102320_matching_1' + '_' + str(prev) + '-' + str(i) + '.pkl'
    #         with open(test_path + '/' + name, 'wb') as f:
    #             pkl.dump(fi_re, f)
end = datetime.datetime.now()
print(end - start)

######################generate matching results file#####################################################################
goundtruth_file_path = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result_pos/iroa_name_final_pos.csv'
path_to_store='/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result_pos/pos_ms1_matching_1'
grountruth_xlx = path_to_store+'/'+'df_truth_0.00.xlsx'
grountruth_csv = path_to_store+'/'+'df_truth_0.00.csv'

gen_result_groundtruth(m=m,goundtruth_file_path=goundtruth_file_path,path_to_store_xlx=grountruth_xlx,path_to_store_csv=grountruth_csv)


correct_match_xlx = path_to_store+'/'+ 'mona' +'_correct_match_0.00.xlsx'
correct_match_csv = path_to_store+'/'+ 'mona'+'_correct_match_0.00.csv'
gen_correct_matched_results(m,'mona',final_matching_results,goundtruth_file_path,correct_match_xlx,correct_match_csv)

# correct_match_path = cur_path + '/' + 'mzc_correct_match_0.00.csv'
not_matched_xlx = path_to_store+'/'+'not_matched_mona_new2.xlsx'
df_truth_path = path_to_store + '/' + 'df_truth_0.00.csv'
gen_not_matched_results(database=mona,df_truth_path=df_truth_path,correct_match_path=correct_match_csv,path_to_store=not_matched_xlx)
################################generate matched fragments all db_based on groundtruth#####################################################
import pandas as pd
cur_path2='/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result_pos/pos_ms1_matching_1'
iroa_path = cur_path2 + '/' + 'iroa_correct_match_0.00.csv'
mona_path = cur_path2 + '/' + 'mona_correct_match_0.00.csv'
mzc_path = cur_path2 + '/' + 'mzc_correct_match_0.00.csv'
save_path_xls = cur_path2+'/'+'matched_all_db_update.xlsx'
save_path_csv = cur_path2+'/'+'matched_all_db_update.csv'
gen_all_db_results(m,mona_path,iroa_path,mzc_path,save_path_xls,save_path_csv,iroa,mzc,mona)
###################################generate_correctly_matched_compounds##########################
# goundtruth_file_path = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result_pos/iroa_name_final_pos.csv'
# path_to_store='/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result_pos/pos_ms1_matching_1'
# df_re_truth=gen_file_matched_bases(m=m, file=goundtruth_file_path)
# df_re_truth.to_excel(path_to_store+'/'+'df_truth_0.00.xlsx', index=False)
# df_re_truth.to_csv(path_to_store+'/'+'df_truth_0.00.csv',index=False)
#
#
# re_df,mismatch_idx = gen_df_for_correct_annotation('mzc', final_matching_results, m, goundtruth_file_path)
# re_df.to_excel(path_to_store+'/'+'mzc_correct_match_0.00.xlsx', index=False)
# re_df.to_csv(path_to_store+'/'+'mzc_correct_match_0.00.csv',index=False)
# with open(path_to_store+'/'+'mzc_mismatch_idx_0.00.pkl','wb') as f:
#     pkl.dump(mismatch_idx,f)
#
# with open(path_to_store+'/'+'m.pkl','wb') as f:
#     pkl.dump(m,f)
# ################################################################
# # test_path = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result/negative_09302020'
# # with open(test_path+'/'+'5E3_peak_detection_results_raw.pkl','wb') as f:
# #     pkl.dump(m._detection_results_raw,f)
#
# # with open(path_to_store+'/'+'m.neg_mse5E4_rangethre0.10_Ker2_12_mincos9590_3.pkl','wb') as f:
# #     pkl.dump(m.base_info,f)
# num_f=[]
# for idx,con in m.base_info.items():
#     num_f.append(len(con.spectrum.mz))
#
# #########################################generate_not_matched_compounds##########################################
# cur_path = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result_pos/pos_ms1_matching_1'
# df_truth_0 = pd.read_csv(cur_path + '/' + 'df_truth_0.00.csv')
# df_db = pd.read_csv(cur_path + '/' + 'mzc_correct_match_0.00.csv') ###########modify here###########
#
# db_have_key=[]
# db_not_have_key=[]
#
# for idx in range(len(df_truth_0['InChIKey'].values)):
#     not_found = True
#     for cmp in mzc.compounds:  ########################need to modify ##
#         if cmp.InChIKey == df_truth_0['InChIKey'].values[idx]:
#             db_have_key.append(idx)
#             not_found = False
#             break
#     if not_found:
#         db_not_have_key.append(idx)
# # mzc_have_key_df = df_truth_0.reindex(mzc_have_key)
# db_not_matched=[]
# for idx in db_have_key:
#     if df_truth_0.iloc[idx].InChIKey not in df_db['InChIKey'].values:
#         db_not_matched.append(idx)
#
# not_matched_df = df_truth_0.reindex(db_not_matched)
# not_matched_df.to_excel(cur_path+'/'+'mzc_not_matched_df_new.xlsx', index = False)
# not_matched_df.to_csv(cur_path+'/'+'mzc_not_matched_df_new.csv',index=False)
# # not_matched_df.to_csv(cur_path+'/'+'iroa_not_matched_df_new.csv')
#
# # db_have_key_df = df_truth_0.reindex(db_have_key)
# # db_have_key_df.to_excel(cur_path+'/'+'mzc_have_key_df.xlsx', index = True)
# ##########################generate not-matched compounds###############################################################
# cur_path2 = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result_pos/pos_ms1_matching_1'
# df_mzc_sum = pd.read_csv(cur_path2 + '/' + 'mzc_not_matched_df_new.csv')
# df_mzc_sum['have_inchi']='NA'
# df_mzc_sum['Neg_spec']='NA'
# df_mzc_sum['Pos_spec']='NA'
#
# def gener_non_matched(database='mzc'):
#         if database == 'mzc':
#             for index in range(len(df_mzc_sum['InChIKey'].values)):
#                 target_cmp=[]
#                 for cmp in mzc.compounds:  ##############change here###########
#                     if cmp.InChIKey == df_mzc_sum['InChIKey'].values[index]:
#                         target_cmp.append(cmp)
#                         break
#                 if target_cmp:
#                     df_mzc_sum.iloc[[index],[5]] = 'Y'
#                     polarity=set()
#                     for s in target_cmp[0].spectra_1 + target_cmp[0].spectra_2:
#                         # for s in t:
#                         polarity.add(s.Polarity)
#                     # print('polarity={}'.format(polarity))
#                     if len(polarity)==1:
#                         # print('polarity.pop={}'.format(polarity.pop))
#                         if 'Negative' in polarity:
#                             df_mzc_sum.iloc[[index],[6]] = 'Y'
#                             df_mzc_sum.iloc[[index],[7]] = 'N'
#                         elif 'Positive' in polarity:
#                             df_mzc_sum.iloc[[index],[6]] = 'N'
#                             df_mzc_sum.iloc[[index],[7]] = 'Y'
#
#                     elif len(polarity) == 2:
#                         df_mzc_sum.iloc[[index],[6]] = 'Y'
#                         df_mzc_sum.iloc[[index],[7]] = 'Y'
#                 else:
#                     df_mzc_sum.iloc[[index],[5]] = 'N'
#
#     df_mzc_sum.to_excel(cur_path2+'/'+'not_matched_mzc_new2.xlsx', index = False)
##############COMBINE_MATCHED_RESULTS_ALL_DB###############################
cur_path2='/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result_pos/pos_ms1_matching_1'
df_iroa = pd.read_csv(cur_path2 + '/' + 'iroa_correct_match_0.00.csv')
df_mona = pd.read_csv(cur_path2 + '/' + 'mona_correct_match_0.00.csv')
df_mzc = pd.read_csv(cur_path2 + '/' + 'mzc_correct_match_0.00.csv')
df_iroa['database']='iroa'
df_mona['database']='mona'
df_mzc['database']='mzc'

matched_all_db=pd.concat([df_iroa, df_mona, df_mzc])
matched_all_db.drop_duplicates()
matched_all_db['spec_matched_peaks'] = 'NA'
matched_all_db['non_matched'] = 'NA'
matched_all_db['num_matched'] = 'NA'
matched_all_db['isotope'] = 'NA'
matched_all_db['num_isotope'] = 'NA'
matched_all_db['adduction'] = 'NA'
matched_all_db['num_adduction'] = 'NA'
matched_all_db['multimer'] = 'NA'
matched_all_db['num_multimer'] = 'NA'
# matched_all_db.to_excel(cur_path2+'/'+'matched_all_db.xlsx', index = False)
matched_all_db.to_csv(cur_path2+'/'+'matched_all_db.csv',index=False)
##############return_matched_peaks#########################################
cur_path2 = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result/neg_102320_matching_3_02'
matched_all_db2 = pd.read_csv(cur_path2 + '/' + 'matched_all_db.csv')
# matched_all_db.rename(columns={'other_matched':'non_matched'},inplace=True)
matched_all_db2['spec_matched_peaks'] = matched_all_db2['spec_matched_peaks'].apply(lambda x: list())
matched_all_db2['non_matched'] = matched_all_db2['non_matched'].apply(lambda x: list())
matched_all_db2['isotope'] = matched_all_db2['isotope'].apply(lambda x: dict())
matched_all_db2['adduction'] = matched_all_db2['adduction'].apply(lambda x: dict())
matched_all_db2['multimer'] = matched_all_db2['multimer'].apply(lambda x: dict())

for idx in tqdm(range(len(matched_all_db.index))):
    print(f'idx={idx}')
    base_idx = matched_all_db.loc[idx, 'base_idx'] #base_idx#
    print(f'base_idx={base_idx}')
    temp_spec = copy.deepcopy(m.base_info[m.base_index[base_idx]].spectrum)
    inchikey = matched_all_db.loc[idx, 'InChIKey']
    if matched_all_db.loc[idx, 'database'] == 'iroa':
        mat, non_mat,iso,add,mul = temp_spec.gen_matched_peaks_compound(inchIkey=inchikey, mode='Negative', database=iroa)
    elif matched_all_db.loc[idx, 'database'] == 'mona':
        mat, non_mat,iso,add,mul = temp_spec.gen_matched_peaks_compound(inchIkey=inchikey, mode='Negative', database=mona)
    elif matched_all_db.loc[idx, 'database'] == 'mzc':
        mat, non_mat,iso,add,mul = temp_spec.gen_matched_peaks_compound(inchIkey=inchikey, mode='Negative', database=mzc)

    matched_all_db.at[idx, 'spec_matched_peaks'] = mat
    matched_all_db.at[idx, 'non_matched'] = non_mat
    matched_all_db.at[idx, 'isotope'],matched_all_db.at[idx, 'adduction'], matched_all_db.at[idx, 'multimer']= iso,add,mul

    matched_all_db.iloc[idx,10] = len(mat)
    matched_all_db.iloc[idx,12],matched_all_db.iloc[idx,14], matched_all_db.iloc[idx,16] = len(iso),len(add),len(mul)

matched_all_db.to_excel(cur_path2+'/'+'matched_all_db_update.xlsx',index=False)
matched_all_db.to_csv(cur_path2+'/'+'matched_all_db_update.csv',index=False)

######################################################################################

