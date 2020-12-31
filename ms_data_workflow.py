from mona.py import *
from spec_matching_result import *
from msdata import *
import copy

####################################read_database##########################
mona_dire_neg ='../../../MoNA/mona_neg_cleaned.pkl'  #directory to store pickle file of MoNA database (negative mode)
mona_dire_pos ='../../../MoNA/mona_pos_cleaned_new.pkl' #directory to store pickle file of MoNA database (positive mode)
mona = MonaDatabase(neg_dir=mona_dire_neg,pos_dir=mona_dire_pos)
mona.read_file(cur_mode='Negative')  #cur_mode: 'Negative' or 'Positive'
for c in tqdm(mona.compounds_list, desc="processing compounds"):
    c.generate_mz_collection(mode='Negative') #mode: 'Negative' or 'Positive'

####################################read file####################################
m = MSData.from_files('data','/.../.../data')
# MSData.from_files(1st argument: data name, 2nd argument:directory_to_store_data), data name for the two arguments should be consistent#
# required data files including :
# data_ints.txt: .txt file that contains corresponding intensity information of features, shape=(number_of_features, number_of_time_points)
# data_mzs.txt: .txt file that contains mass-to-charge ratio values ranges for all features, shape = (number_of_features, 2)
# data_rts.txt: .txt file that contains retention time points, shape = (number_of_time_points, )
# data_peak_info.csv : .csv file contains peak information for all features

m.apply_smooth_conv(m.gaussian_kernel(2, 12), rt_range=(0., np.inf)) #smooth the raw data of peaks
peak_detection_options = {'height': 0.1, 'range_threshold': 0.1, 'prominence': 0.1}
m.perform_feature_peak_detection(**peak_detection_options) #detect peaks
m.remove_duplicates_detection() #remove duplicates
by = np.max(m.ints_raw, axis=1)
m.sort_feature(by=by) #sort features by intensity order(high to low)

####################################generate basis step####################################
gen_base_opt = dict(
    l1=2.,
    min_sin=5E-3,  # min_sin=5E-2,
    min_base_err=5E-4,  # min_base_err=5E-2,
    min_rt_diff=1.,
    max_cos_sim=0.9
)
m.generate_base(reset=True, allowed_n_overlap=(1,2), **gen_base_opt)
m.generate_base(reset=False, allowed_n_overlap=2, **gen_base_opt)
m.generate_base(reset=False, allowed_n_overlap=3, **gen_base_opt)
m.generate_base(reset=False, allowed_n_overlap=(4, None), **gen_base_opt)

####################################perform peak decompositon for other features####################################
m.perform_peak_decomposition(l1=1.) #decompose other features using generated basis group
spec_params = dict(
    threshold=1E-5,
    max_rt_diff=4.,
    max_mse=5E-2,
    mz_upperbound=np.inf,
    min_cos=0.85)
for i in tqdm(range(m.n_base), desc='generating spectrum'):
    m.gen_spectrum(i, plot=False, load=False, **spec_params) #generate reconstructed spectrum for each group

####################################database matching for each basis group####################################
final_matching_results = []  #create an empty list to contain the following matching result
import datetime
start = datetime.datetime.now()
for i in range(len(m.base_index)):
    print(f'current running basis {i}')
    spec = m.base_info[m.base_index[i]].spectrum
    spec_2 = copy.deepcopy(spec)

    result = GroupMatchingResult(recons_spec=spec_2,base_index_relative=i,base_index_abs=m.base_index[i],mode='Negative')
    # mode: 'Negative' or 'Positive'

    result.gen_mona_matching_result(total_layer_matching=1,n_candidates_further_matched=3,database=mona,transform=None)
    # total_layer_matching: number of layers for recursive matching(any positive integer), 0 :no recursive mathing, 1: one further recursive mathching(recommended)
    # n_candidates_further_matched: number of top-candidates considered for further recursive matching
    # transform: None or math.sqrt

    result.summarize_matching_re_all_db()
    result.remove_db()
    final_matching_results.append(result)
end = datetime.datetime.now()
print(end - start)

####################################check the matching results####################################
final_matching_results[0].sum_matched_results_mona # A dictionary that contains matching result for basis group 0
len(final_matching_results[0].sum_matched_results_mona) # if total_layer_matchin=1, n_candidates_further_matched=3, then there are 3^2 paths
final_matching_results[0].sum_matched_results_mona[0] # A dictionary for first path matching result of basis group 0
final_matching_results[0].sum_matched_results_mona[0]['candi_list_each_matching_layer'] # matching results (candidates and scores) for all matching layer)

