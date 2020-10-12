from IROA_workflow import *
from MoNA_workflow import *
from mz_cloud_workflow import *
from spec_matching_result import *
from check_annotation_accurary import *
from msdata import *
import copy

# m = MSData.from_files('3T3_pro','/../../../Data/3T3_pro')
# m = MSData.from_files('3T3_pro','/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/sisi_codes_test/Data/3T3_pro')
m = MSData.from_files('IROA_MS1_neg','/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_09162020/IROA_MS1_neg')
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
    max_cos_sim=0.95
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
    max_mse=np.inf,
    # mz_upperbound=5.
    mz_upperbound=np.inf,
    # min_cos=.99
    min_cos=0.90
)
m.gen_spectrum(0, plot=False, load=False, **spec_params)
for i in tqdm(range(m.n_base), desc='generating spectrum'):
    m.gen_spectrum(i, plot=False, load=False, **spec_params)

# for v in m.base_info.values():
#     print(v)

# ###check base group and find_match within each group and plot###
test_path = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result/neg_mse5E4_rangethre0.10_Ker2_12_mincos9590_3'
final_matching_results = []
save_index = [i for i in range(len(m.base_index))if i% 200==0]
save_index.append(len(m.base_index)-1)
import datetime
start=datetime.datetime.now()
for i in range(len(m.base_index)):
    spec = m.base_info[m.base_index[i]].spectrum
    spec_2 = copy.deepcopy(spec)
    result = GroupMatchingResult(recons_spec=spec_2,
                                 base_index_relative=i,
                                 base_index_abs=m.base_index[i])
    result.gen_mzc_matching_result(total_layer_matching=1,n_candidates_further_matched=10,database=mzc,transform=None)
    result.gen_mona_matching_result(total_layer_matching=1,n_candidates_further_matched=10,database=mona,transform=None) ##start from 0th match##
    result.gen_iroa_matching_result(total_layer_matching=1,n_candidates_further_matched=10,database=iroa,transform=None)
    # result.gen_recur_matched_peaks()
    # result.count_total_matched_peaks()
    result.summarize_matching_re_all_db(mzc=True,mona=True,iroa=True)
    result.remove_db()
    final_matching_results.append(result)
    if i in save_index:
        fi_re=[[j.sum_matched_results_iroa, j.sum_matched_results_mona,j.sum_matched_results_mzc] for j in final_matching_results]
        name = 'neg_mse5E4_rangethre0.10_Ker2_12_mincos9590_3' + '_'+ str(i) +'.pkl'
        with open(test_path+'/' + name,'wb') as f:
            pkl.dump(fi_re,f)
end=datetime.datetime.now()
print(end-start)
#############################################################
goundtruth_file_path = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result/iroa_name_final.csv'
path_to_store='/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result/neg_mse5E4_rangethre0.10_Ker2_12_mincos9590_3'
df_re_truth=gen_file_matched_bases(m=m, file=goundtruth_file_path)
df_re_truth.to_excel(path_to_store+'/'+'df_truth_0.05.xlsx', index = False)
re_df,mismatch_idx = gen_df_for_correct_annotation('iroa', final_matching_results, m, goundtruth_file_path)
re_df.to_excel(path_to_store+'/'+'iroa_correct_match.xlsx', index = False)
with open(path_to_store+'/'+'iroa_mismatch_idx.pkl','wb') as f:
    pkl.dump(mismatch_idx,f)
################################################################
# test_path = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result/negative_09302020'
# with open(test_path+'/'+'5E3_peak_detection_results_raw.pkl','wb') as f:
#     pkl.dump(m._detection_results_raw,f)

# with open(path_to_store+'/'+'m.neg_mse5E4_rangethre0.10_Ker2_12_mincos9590_3.pkl','wb') as f:
#     pkl.dump(m.base_info,f)
with open(path_to_store+'/'+'m.neg_mse5E4_rangethre0.10_Ker2_12_mincos9590_3.pkl','wb') as f:
    pkl.dump(m,f)


# with open(test_path +'/'+'neg_stds_non_trans_mse5E3_30.pkl','rb') as f:
#     neg_stds_non_trans_mse5E3_30 = pkl.load(f)
#
num_f=[]
for idx,con in m.base_info.items():
    num_f.append(len(con.spectrum.mz))

double_peak=[]
for _,re in m._detection_results.items():
    if re.n_overlaps == 2:
        double_peak.append(re)
