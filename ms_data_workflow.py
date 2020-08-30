from msdata import *
# from IROA_IDX import *

##dire:/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/sisi_codes_test/Data/3T3 #
m = MSData.from_files('IROA_ms1_neg','/Users/sisizhang/Dropbox/Share_Yuchen/Projects/in_source_fragments_annotation/sisi_codes_test/Data/IROA_ms1_neg')
m.apply_smooth_conv(m.gaussian_kernel(4, 12), rt_range=(0., np.inf))

m.perform_feature_peak_detection()
# peak_detection_options = {}
# by = np.zeros(m.n_feature)
# res = m.get_peak_detection_results(index=np.arange(m.n_feature), ordered=False)
# for i in range(m.n_feature):
#     r = res[i]
#     if r is not None:
#         if r.matching_idx is not None and r.peak_heights is not None:
#             by[i] = r.peak_heights[r.matching_idx]

by = np.max(m.ints_raw, axis=1)

m.sort_feature(by=by)

gen_base_opt = dict(
    l1=2.,
    min_sin=5E-2,  # min_sin=5E-2,
    min_base_err=5E-2,  # min_base_err=5E-2,
    min_rt_diff=2.,
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
    max_rt_diff=1.,
    # max_rt_diff=np.inf,
    # max_mse=2E-2,
    max_mse=np.inf,
    # mz_upperbound=5.
    mz_upperbound=np.inf,
    # min_cos=.99
    min_cos=0.
)
# m.gen_spectrum(0, plot=False, load=False, **spec_params)
for i in tqdm(range(m.n_base), desc='generating spectrum'):
    m.gen_spectrum(i, plot=False, load=False, **spec_params)

for v in m.base_info.values():
    print(v)

###check base group and find_match within each group and plot###
spec = m.base_info[m.base_index[6]].spectrum
##################################search IROA_database###
iroa_re = iroa.find_match(target=spec, save_matched_mz=True, transform=math.sqrt) ##compare with iroa database
iroa_re[0][1].bin_vec.matched_idx_mz #check matched mz for best match in iroa
spec.gen_matched_mz(other=iroa_re[0][1]) #generate matched_mz and mis_matched_mz to spec.matched_mz and spec.mis_matched_mz
spec.check_isotope()
spec.check_adduction_list(exact_mass=iroa_re[0][0].MolecularWeight)
#####################search mzcloud database#######
mzc_re = mzc.find_match(target=spec, save_matched_mz=True, transform=math.sqrt)
mzc_re[0][1].bin_vec.matched_idx_mz
spec.gen_matched_mz(other=mzc_re[0][1])
spec.check_isotope()
spec.check_adduction_list(exact_mass=mzc_re[0][0].MolecularWeight)

######################search mona_database########
mona_re = mona.find_match(target=spec, save_matched_mz=True, transform=math.sqrt)
mona_re[0][1].bin_vec.matched_idx_mz
spec.gen_matched_mz(other=mona_re[0][1]) ###generate matched_mz to
spec.check_isotope()  ##check if there are other isotopope for matched_mz