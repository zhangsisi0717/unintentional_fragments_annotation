from IROA_workflow import *
from MoNA_workflow import *
from mz_cloud_workflow import *
from spec_matching_result import *
from msdata import *
import copy

# m = MSData.from_files('3T3_pro','/../../../Data/3T3_pro')
# m = MSData.from_files('3T3_pro','/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/sisi_codes_test/Data/3T3_pro')
m = MSData.from_files('IROA_MS1_neg','/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_09162020/IROA_MS1_neg')
m.apply_smooth_conv(m.gaussian_kernel(4, 12), rt_range=(0., np.inf))

#peak_detection_options = {'height': 1E-2, 'range_threshold': 1E-4, 'prominence': 1E-2}
#h_max = np.max(ints) - np.min(ints)
#prominence = h_max * prominence, height = h_max * height,
#delta = (rts[1:] - rts[:-1]).mean(), width = width / delta

#afer peak_detection:range_threshold = peak_heights * range_threshold, beyong range,set to 0
peak_detection_options = {'height': 0.1, 'range_threshold': 0.01, 'prominence': 0.1}
m.perform_feature_peak_detection(**peak_detection_options)
m.remove_duplicates_detection()
# m.perform_feature_peak_detection()
by = np.max(m.ints_raw, axis=1)

m.sort_feature(by=by)

gen_base_opt = dict(
    l1=2.,
    min_sin=5E-2,  # min_sin=5E-2,
    min_base_err=5E-2,  # min_base_err=5E-2,
    min_rt_diff=1.,
    max_cos_sim=0.8
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
    min_cos=0.
)
m.gen_spectrum(0, plot=False, load=False, **spec_params)
for i in tqdm(range(m.n_base), desc='generating spectrum'):
    m.gen_spectrum(i, plot=False, load=False, **spec_params)

# for v in m.base_info.values():
#     print(v)

# ###check base group and find_match within each group and plot###
test_path = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result/negative'
final_matching_results = []
save_index = [30,60,90,100,130,153]
import datetime
start=datetime.datetime.now()
for i in range(0,154):
    spec = m.base_info[m.base_index[i]].spectrum
    spec_2 = copy.deepcopy(spec)
    result = GroupMatchingResult(recons_spec=spec_2,
                                 base_index_relative=i,
                                 base_index_abs=m.base_index[i])
    result.gen_mzc_matching_result(total_layer_matching=1,n_candidates_further_matched=5,database=mzc,transform=None)
    result.gen_mona_matching_result(total_layer_matching=1,n_candidates_further_matched=5,database=mona,transform=None) ##start from 0th match##
    result.gen_iroa_matching_result(total_layer_matching=1,n_candidates_further_matched=5,database=iroa,transform=None)
    # result.gen_recur_matched_peaks()
    # result.count_total_matched_peaks()
    result.summarize_matching_re_all_db(mzc=True,mona=True,iroa=True)
    final_matching_results.append(result)
    if i in save_index:
        fi_re=[[j.sum_matched_results_iroa, j.sum_matched_results_mona,j.sum_matched_results_mzc] for j in final_matching_results]
        name = 'matchre_neg_stds_non_trans' + '_'+ str(i) +'.pkl'
        with open(test_path+'/' + name,'wb') as f:
            pkl.dump(fi_re,f)
end=datetime.datetime.now()
print(end-start)
#############################################################
with open('3T3_pro_4ul_ms_data.pkl', "wb") as write_file:
    pkl.dump(m, write_file)

with open('3T3_pro_re_mona.pkl', "wb") as write_file:
    pkl.dump(result.mona_result, write_file)
# ##################################search IROA_database###
# iroa_re = iroa.find_match(target=spec, save_matched_mz=True, transform=math.sqrt) ##compare with iroa database
# iroa_re[0][1].bin_vec.matched_idx_mz #check matched mz for best match in iroa
# spec.gen_matched_mz(other=iroa_re[0][1]) #generate matched_mz and mis_matched_mz to spec.matched_mz and spec.mis_matched_mz
# spec.check_isotope()
# spec.check_adduction_list(exact_mass=iroa_re[0][0].MolecularWeight)
#
# #####################search mzcloud database#######
# mzc_re = mzc.find_match(target=spec, save_matched_mz=True, transform=math.sqrt)
# # mzc_re[0][1].bin_vec.matched_idx_mz
# # spec.compare_spectrum_plot(mzc_re[0][1])
# spec.gen_matched_mz(other=mzc_re[1][1])
# spec.check_isotope()
# spec.check_adduction_list(molecular_weight=mzc_re[1][0].MolecularWeight)
# spec.check_multimer(molecular_weight=mzc_re[1][0].MolecularWeight)
# sub_spec = spec.generate_sub_recon_spec()
# ####2nd round of matching for sub_spec in mzc
# mzc_re_2 = mzc.find_match(target=sub_spec, save_matched_mz=True, transform=math.sqrt)
# # mzc_re_2[0][1].bin_vec.matched_idx_mz
# # sub_spec.compare_spectrum_plot(mzc_re_2[1][1])
# sub_spec.gen_matched_mz(other=mzc_re_2[0][1])
# sub_spec.check_isotope()
# sub_spec.check_adduction_list(molecular_weight=mzc_re_2[0][0].MolecularWeight)
# sub_spec.check_multimer(molecular_weight=mzc_re_2[0][0].MolecularWeight)
# sub_sub_spec = sub_spec.generate_sub_recon_spec()
# ###############3rd round of matching for sub_sub_spec in mzc
# mzc_re_3 = mzc.find_match(target=sub_sub_spec, save_matched_mz=True, transform=math.sqrt)
# # mzc_re_3[1][1].bin_vec.matched_idx_mz
# # sub_sub_spec.compare_spectrum_plot(mzc_re_3[1][1])
# sub_sub_spec.gen_matched_mz(other=mzc_re_3[0][1])
# sub_sub_spec.check_isotope()
# sub_sub_spec.check_adduction_list(molecular_weight=mzc_re_3[0][0].MolecularWeight)
# sub_sub_spec.check_multimer(molecular_weight=mzc_re_3[0][0].MolecularWeight)
# sub_spec_4 = sub_sub_spec.generate_sub_recon_spec()
# #################4th round of matching for sub_spec_4 in mzc
# mzc_re_4 = mzc.find_match(target=sub_spec_4, save_matched_mz=True, transform=math.sqrt)
# # mzc_re_4[1][1].bin_vec.matched_idx_mz
# # sub_spec_4.compare_spectrum_plot(mzc_re_4[1][1])
# sub_spec_4.gen_matched_mz(other=mzc_re_4[0][1])
# sub_spec_4.check_isotope()
# sub_spec_4.check_adduction_list(molecular_weight=mzc_re_4[0][0].MolecularWeight)
# sub_spec_4.check_multimer(molecular_weight=mzc_re_4[0][0].MolecularWeight)
# sub_spec_5 = sub_spec_4.generate_sub_recon_spec()
# ######################search mona_database########
# ######
# mona_re = mona.find_match(target=spec, save_matched_mz=True, transform=math.sqrt)
# mona_re[0][1].bin_vec.matched_idx_mz
# spec.gen_matched_mz(other=mona_re[0][1]) ###generate matched_mz to
# spec.check_isotope()  ##check if there are other isotopope for matched_mz
# spec.check_adduction_list(molecular_weight=mona_re[0][0].total_exact_mass)
# spec.check_multimer(molecular_weight=mona_re[0][0].total_exact_mass)
# sub_spec = spec.generate_sub_recon_spec()

with open('iroa_ms1_matching_result_01.pkl','wb') as f:
    pkl.dump(iroa_ms1_matching_result_1,f)

test_path = '/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/IROA/IROA_MS1_matching_result/negative'
with open(test_path+'/'+'base_index.pkl','wb') as f:
    pkl.dump(m.base_index,f)


with open(test_path +'/'+'iroa_ms1_matching_result_01.pkl','rb') as f:
    temp = pkl.load(f)

########


