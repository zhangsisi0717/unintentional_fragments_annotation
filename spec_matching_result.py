from dataclasses import dataclass, field
from typing import Optional
from mona import *
from msdata import ReconstructedSpectrum
import copy


@dataclass()
class MoNAMatchingResult:
    database: Optional[MonaDatabase] = field(default=None, repr=False)
    mode: Optional[str] = "Negative"
    transform: Optional[math.sqrt] = field(default=None, repr=False)
    nth_candidate: Optional[int] = field(default=None, repr=True)  # if 0th_match, then nth_candidate is None
    nth_match: Optional[int] = field(default=None, repr=True)  # nth recursive match#

    base_index_relative: Optional[int] = field(default=None, repr=False)
    base_index_abs: Optional[int] = field(default=None, repr=False)

    cur_candidate_info: MonaCompounds = field(default=None, repr=False)  # if it is 1st match, then it is None
    parent_recons_spec: Optional[ReconstructedSpectrum] = field(default=None, repr=False)  # if 1st match,no parent
    parent_matched_mz: Optional[List] = field(default_factory=list, repr=False)

    current_recons_spec: Optional[ReconstructedSpectrum] = field(default=None, repr=False)
    current_raw_matching_result: Optional[List] = field(default_factory=list, repr=False)  # raw matching for cur_spec

    n_candidates_further_matched: Optional[int] = field(default=3, repr=False)  # n of candidates need further matching
    n_candidates_further_match_r: Optional[dict] = field(default_factory=dict,
                                                         repr=False)  # result of matching for candidates

    n_peaks_matched: Optional[int] = field(default=None, repr=False)  # n of peaks matched based on this candidate

    total_layer_matching: Optional[int] = field(default=3, repr=False)  # num of layer for recursive matching

    def __post_init__(self):
        if not self.nth_match:
            self.nth_match = 0
            self.parent_matched_mz = 'root'

        if self.total_layer_matching >= self.nth_match:
            if self.parent_recons_spec is not None:
                assert isinstance(self.parent_recons_spec, ReconstructedSpectrum)
                self.parent_matched_mz = self.parent_recons_spec.matched_mz
                self.n_peaks_matched = len(self.parent_recons_spec.matched_mz)
            if self.parent_matched_mz:
                if self.current_recons_spec is not None:
                    assert isinstance(self.current_recons_spec, ReconstructedSpectrum)
                    if self.database:
                        self.current_raw_matching_result = self.database.find_match(target=self.current_recons_spec,
                                                                                    save_matched_mz=True,
                                                                                    cos_threshold=1E-4,
                                                                                    reset_matched_mzs=True,
                                                                                    transform=self.transform)

                if len(self.current_raw_matching_result) < self.n_candidates_further_matched:
                    self.n_candidates_further_matched = len(self.current_raw_matching_result)

                for i in range(int(self.n_candidates_further_matched)):
                    cur_candi_info = self.current_raw_matching_result[i]
                    cur_spec = copy.deepcopy(self.current_recons_spec)
                    cur_spec.gen_matched_mz(self.current_raw_matching_result[i][1])
                    cur_spec.check_isotope()
                    cur_spec.check_adduction_list(molecular_weight=
                                                  self.current_raw_matching_result[i][0].total_exact_mass,
                                                  mode=self.mode)
                    cur_spec.check_multimer(molecular_weight=self.current_raw_matching_result[i][0].total_exact_mass,
                                            mode=self.mode)
                    sub_spec = cur_spec.generate_sub_recon_spec()
                    self.n_candidates_further_match_r[i] = MoNAMatchingResult(nth_match=int(self.nth_match + 1),
                                                                              nth_candidate=i,
                                                                              database=self.database,
                                                                              base_index_relative=
                                                                              self.base_index_relative,
                                                                              base_index_abs=self.base_index_abs,
                                                                              cur_candidate_info=cur_candi_info,
                                                                              parent_recons_spec=cur_spec,
                                                                              current_recons_spec=sub_spec,
                                                                              total_layer_matching=
                                                                              self.total_layer_matching,
                                                                              n_candidates_further_matched=
                                                                              self.n_candidates_further_matched,
                                                                              mode=self.mode)


@dataclass
class GroupMatchingResult:
    """
    Container class for reconstructed spectrum matching results

    """
    mode: Optional[str] = 'Negative'
    base_index_abs: Optional[int] = field(default=None, repr=False)
    base_index_relative: Optional[int] = field(default=None, repr=True)
    recons_spec: Optional[ReconstructedSpectrum] = field(default=None, repr=True)
    matched_against_mona: Optional[bool] = field(default=True, repr=False)
    mona_result: Optional[MoNAMatchingResult] = field(default=None, repr=False)
    sum_matched_results_mona: Optional[Dict] = field(default=None, repr=False)
    total_matched_peaks_mona: Optional[List] = field(default=None, repr=False)

    def gen_mona_matching_result(self, total_layer_matching: Optional[int] = 3,
                                 n_candidates_further_matched: Optional[int] = 1,
                                 database: [MonaDatabase] = None,
                                 base_index_abs: Optional[int] = base_index_abs,
                                 base_index_relative: Optional[int] = base_index_relative,
                                 transform=None) -> MoNAMatchingResult:

        if self.matched_against_mona:
            result = MoNAMatchingResult(current_recons_spec=self.recons_spec,
                                        total_layer_matching=total_layer_matching,
                                        n_candidates_further_matched=n_candidates_further_matched,
                                        database=database,
                                        base_index_abs=base_index_abs,
                                        base_index_relative=base_index_relative,
                                        transform=transform, mode=self.mode)

            self.mona_result = result

    def remove_db(self):
        def recursive_remove_database(cur_result):
            if cur_result.n_candidates_further_match_r is None:
                cur_result.database = None
                return
            cur_result.database = None
            for _, sub_result in cur_result.n_candidates_further_match_r.items():
                recursive_remove_database(sub_result)
            return

        if self.mona_result:
            recursive_remove_database(self.mona_result)

    @staticmethod
    def summarize_all_matching_result(result: MoNAMatchingResult):
        if not result.n_candidates_further_match_r:
            nth_candidate = [[]]
            nth_matching = [[]]
            matched_candi_info = [[]]
            matched_peaks = [[]]
            matched_adducts = [[]]
            candi_list_each_layer = [[]]
            return nth_candidate, nth_matching, matched_candi_info, matched_peaks, matched_adducts, candi_list_each_layer

        total_nth_candidate, total_nth_matching, total_candi_info = [], [], []
        total_matched_peaks, total_matched_adducts = [], []
        all_candi_list_each_layer = []
        for _, candi in result.n_candidates_further_match_r.items():
            nth_candidate, nth_matching, matched_candi_info, matched_peaks, matched_adducts, candi_list_each_layer = \
                GroupMatchingResult.summarize_all_matching_result(candi)
            for i in nth_candidate:
                temp_candidate = [candi.nth_candidate] + i
                total_nth_candidate.append(temp_candidate)
            for i in nth_matching:
                temp_nth_match = [candi.nth_match] + i
                total_nth_matching.append(temp_nth_match)
            for i in matched_candi_info:
                temp_matched_candi_info = [candi.cur_candidate_info] + i
                total_candi_info.append(temp_matched_candi_info)
            for i in matched_peaks:
                temp_total_matched_peaks = [candi.parent_recons_spec.matched_mz] + i
                total_matched_peaks.append(temp_total_matched_peaks)
            for i in matched_adducts:
                temp_matched_adducts = [{'matched_multimer': candi.parent_recons_spec.matched_multimer,
                                         'matched_isotope_mz': candi.parent_recons_spec.matched_isotope_mz,
                                         'matched_adducts': candi.parent_recons_spec.matched_adduction}] + i
                total_matched_adducts.append(temp_matched_adducts)
            for i in candi_list_each_layer:
                temp_candi_list_each_layer = [candi.current_raw_matching_result] + i
                all_candi_list_each_layer.append(temp_candi_list_each_layer)

        if result.parent_matched_mz == 'root':
            final_layer_list = []
            for i in all_candi_list_each_layer:
                temp = [result.current_raw_matching_result] + i
                final_layer_list.append(temp)
            all_candi_list_each_layer = final_layer_list

        return total_nth_candidate, total_nth_matching, total_candi_info, \
               total_matched_peaks, total_matched_adducts, all_candi_list_each_layer

    @staticmethod
    def gen_dic_for_matching_result(a: List, b: List, c: List, d: List, e: List, f: List):
        t = np.array([len(a), len(b), len(c), len(d), len(e), len(f)])
        p = t[:-1] - t[1:]
        if np.all(p == 0):
            re = dict()
            for i in range(t[0]):
                re[i] = {'candidates_matched_path': a[i],
                         'num_matched_peaks': sum([len(j) for j in d[i]]),
                         'candi_info_all': c[i],
                         'all_matched_peaks': d[i],
                         'all_matched_adducts': e[i], 'all_nth_matching': b[i],
                         'candi_list_each_matching_layer': f[i]}
            return re
        else:
            print(f)
            raise ValueError('length of input must all be the same!')

    def summarize_matching_re_all_db(self, mona: Optional[bool] = True, reset=True):
        if reset:
            self.sum_matched_results_mona = None

        if mona:
            a, b, c, d, e, f = GroupMatchingResult.summarize_all_matching_result(self.mona_result)
            self.sum_matched_results_mona = GroupMatchingResult.gen_dic_for_matching_result(a, b, c, d, e, f)
            total_peaks = []
            for idx, result in self.sum_matched_results_mona.items():
                total_peaks.append(result['num_matched_peaks'])
            self.total_matched_peaks_mona = total_peaks
