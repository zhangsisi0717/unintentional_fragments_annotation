from dataclasses import dataclass, field
from typing import Optional
from mzcloud import *
from msdata import ReconstructedSpectrum
import copy

# def recursive_spec_matching(recons_spec:[ReconstructedSpectrum]=None,n_recursion:Optional[int]=None,database:Optional[str]='MZcloud'):
#     if database not in ('MZcloud','MoNA','IROA'):
#         raise TypeError('Available database: MZcloud,MoNA or IROA')
#     if database == 'MZcloud':
#         mzc_result = dict()
#         from mz_cloud_workflow import *
#         if not n_recursion:

# @dataclass()
# class MZCloud_CandidateMatchingResult():
#
#     base_index_abs:Optional[int] = field(default=None,repr=False)
#     base_index_relative:Optional[int] = field(default=None,repr=True)
#
#     parent_recons_spec:Optional[ReconstructedSpectrum] = field(default=None,repr=True)
#     parent_matched_mz: Optional[List] = field(default_factory=list,repr=False)
#
#     current_recons_spec:Optional[ReconstructedSpectrum] = field(default=None,repr=False)
#     current_raw_matching_result: Optional[List] = field(default_factory=list,repr=False)
#
#     n_candidates_further_matched: Optional[int] = field(default=0,repr=False)
#     n_candidates_further_match_r:Optional[dict] = field(default_factory=dict,repr=False)
#
#     n_peaks_matched: Optional[int] = field(default=None,repr=False)

@dataclass
class MZCloudMatchingResult:  ####contain first raw_matching result###
    mzc_db: Optional[MZCloud] =field(default=None, repr=False)
    nth_match: Optional[int] = field(default=None, repr=True, init=True)  ##nth recursive match#

    base_index_relative: Optional[int] = field(default=None, repr=True)
    base_index_abs: Optional[int] = field(default=None, repr=False)

    cur_candidate_info: MZCloudCompound = field(default=None, repr=True)  ##if it is 1st match, then it is None
    parent_recons_spec: Optional[ReconstructedSpectrum] = field(default=None, repr=True)  ##if 1st match,no parent
    parent_matched_mz: Optional[List] = field(default_factory=list, repr=False)

    current_recons_spec: Optional[ReconstructedSpectrum] = field(default=None, repr=False)
    current_raw_matching_result: Optional[List] = field(default_factory=list, repr=False)  ##raw matching for cur_spec

    n_candidates_further_matched: Optional[int] = field(default=3, repr=False)  # n of candidates need further matching
    n_candidates_further_match_r: Optional[dict] = field(default_factory=dict,
                                                         repr=False)  # result of matching for candidates

    n_peaks_matched: Optional[int] = field(default=None, repr=False)  # n of peaks matched based on this candidate

    total_layer_matching:Optional[int] = field(default=3,repr=False)

    # sub_recon_spec: Optional[ReconstructedSpectrum] = field(default=None,repr=False)
    # raw_matching_result: Optional[List] = field(default_factory=list,repr=False)
    # n_recursive_matching: Optional[int] = field(default=2,repr=False)

    def __post_init__(self):
        if not self.nth_match:
            self.nth_match = 1

        if self.total_layer_matching >= self.nth_match:
            if self.parent_recons_spec is not None:
                assert isinstance(self.parent_recons_spec, ReconstructedSpectrum)
                self.parent_matched_mz = self.parent_recons_spec.matched_mz
                self.n_peaks_matched = len(self.parent_recons_spec.matched_mz)
            if self.current_recons_spec is not None:
                assert isinstance(self.current_recons_spec, ReconstructedSpectrum)
                if self.mzc_db:
                    self.current_raw_matching_result =self.mzc_db.find_match(target=self.current_recons_spec, save_matched_mz=True,
                                                                      transform=math.sqrt)
            # if not self.current_raw_matching_result:
            #     self.n_candidates_further_matched = 0
            if len(self.current_raw_matching_result) < self.n_candidates_further_matched:
                self.n_candidates_further_matched = len(self.current_raw_matching_result)

            for i in range(int(self.n_candidates_further_matched)):
                cur_candi_info = self.current_raw_matching_result[i][0]
                cur_spec = copy.deepcopy(self.current_recons_spec)
                cur_spec.gen_matched_mz(self.current_raw_matching_result[i][1])
                cur_spec.check_isotope()
                cur_spec.check_adduction_list(molecular_weight=self.current_raw_matching_result[i][0].MolecularWeight)
                cur_spec.check_multimer(molecular_weight=self.current_raw_matching_result[i][0].MolecularWeight)
                sub_spec = cur_spec.generate_sub_recon_spec()
                self.n_candidates_further_match_r[i] = MZCloudMatchingResult(nth_match=int(self.nth_match+1),
                                                                             mzc_db=self.mzc_db,
                                                                             base_index_relative=self.base_index_relative,
                                                                             base_index_abs=self.base_index_abs,
                                                                             cur_candidate_info=cur_candi_info,
                                                                             parent_recons_spec=cur_spec,
                                                                             current_recons_spec=sub_spec,
                                                                             total_layer_matching=self.total_layer_matching,
                                                                             n_candidates_further_matched=self.n_candidates_further_matched)



    # def candidates_further_mathching(self,add_n_layer=2):
    #     for n in range(len(self.n_candidates_further_matched)):
    #     ##check if exist isotope,adduction and multimer,then generate sub_spectrum for further matching
    #         cur_spec = self.n_candidates_further_match_r[n].parent_recons_spec
    #         cur_spec.recons_spec.check_isotope()
    #     self.recons_spec.check_adduction_list(molecular_weight=candidate.MolecularWeight)
    #     self.recons_spec.check_multimer(molecular_weight=candidate.MolecularWeight)
    #     sub_spec = self.recons_spec.generate_sub_recon_spec(recon_spec=self.recons_spec)
    #     print('generate sub_spectrum for this round')
    #     return sub_spec



# @dataclass
# class IROA_MatchingResult():
#
# @dataclass
# class MoNA_MatchingResult():


# @dataclass
# class GroupMatchingResult:  ###contains-mzcloudresult,iroa_result and mona_result###
#     base_index_abs: Optional[int] = field(default=None, repr=False)
#     base_index_relative: Optional[int] = field(default=None, repr=True)
#     recons_spec: Optional[ReconstructedSpectrum] = field(default=None, repr=True)
#     matched_against_mzcloud: Optional[bool] = field(default=True, repr=False)
#     matched_against_mona: Optional[bool] = field(default=True, repr=False)
#     matched_against_iroa: Optional[bool] = field(default=True, repr=False)
#     mzcloud_result: Optional[MZCloudMatchingResult] = field(default=None, repr=False)
#
#     # iroa_result: Optional[IROA_MatchingResult] =  field(default=None,repr=False)
#     # mona_result: Optional[MoNA_MatchingResult] = field(default=None,repr=False)
#
#     def __post_init__(self):
#         self.mzcloud_result = MZCloudMatchingResult(nth_match=1)
#
#     # def mzc_gen_candi_matching_r(self):
#     #     for i in range(self.n_candidates_further_matched):
#     #         copied_spec = copy.deepcopy(self.current_recons_spec)
#     #         self.n_candidates_further_match_r[i] = MZCloudMatchingResult()
#
#
# def spec_rough_matching(recons_spec: [ReconstructedSpectrum] = None, database: Optional[str] = 'MZcloud'):
#     if database not in ('MZcloud', 'MoNA', 'IROA'):
#         raise TypeError('Available database: MZcloud,MoNA or IROA')
#     if database == 'MZcloud':
#         pass
#         # return mzc_re
#
#     elif database == 'MoNA':
#         mona_re = mona.find_match(target=recons_spec, save_matched_mz=True, transform=math.sqrt)
#         return mona_re
#
#     else:
#         iroa_re = iroa.find_match(target=recons_spec, save_matched_mz=True, transform=math.sqrt)
#         return iroa_re


# def spec_further_matching(recons_spec: [ReconstructedSpectrum] = None, compound:)
