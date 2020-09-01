from dataclasses import dataclass, field
from typing import Optional
from mzcloud import *
from mona import *
from IROA_IDX import *
from msdata import ReconstructedSpectrum
import copy


@dataclass
class MZCloudMatchingResult:
    database: Optional[MZCloud] = field(default=None, repr=False)
    nth_match: Optional[int] = field(default=None, repr=True)  # nth recursive match#

    base_index_relative: Optional[int] = field(default=None, repr=False)
    base_index_abs: Optional[int] = field(default=None, repr=False)

    cur_candidate_info: MZCloudCompound = field(default=None, repr=False)  # if it is 1st match, then it is None
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
            self.nth_match = 1

        if self.total_layer_matching >= self.nth_match:
            if self.parent_recons_spec is not None:
                assert isinstance(self.parent_recons_spec, ReconstructedSpectrum)
                self.parent_matched_mz = self.parent_recons_spec.matched_mz
                self.n_peaks_matched = len(self.parent_recons_spec.matched_mz)
            if self.current_recons_spec is not None:
                assert isinstance(self.current_recons_spec, ReconstructedSpectrum)
                if self.database:
                    self.current_raw_matching_result = self.database.find_match(target=self.current_recons_spec,
                                                                                save_matched_mz=True,
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
                self.n_candidates_further_match_r[i] = MZCloudMatchingResult(nth_match=int(self.nth_match + 1),
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
                                                                             self.n_candidates_further_matched)


@dataclass()
class MoNAMatchingResult(MZCloudMatchingResult):
    database: Optional[MonaDatabase] = field(default=None, repr=False)

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
                if self.database:
                    self.current_raw_matching_result = self.database.find_match(target=self.current_recons_spec,
                                                                                save_matched_mz=True,
                                                                                transform=math.sqrt)

            if len(self.current_raw_matching_result) < self.n_candidates_further_matched:
                self.n_candidates_further_matched = len(self.current_raw_matching_result)

            for i in range(int(self.n_candidates_further_matched)):
                cur_candi_info = self.current_raw_matching_result[i][0]
                cur_spec = copy.deepcopy(self.current_recons_spec)
                cur_spec.gen_matched_mz(self.current_raw_matching_result[i][1])
                cur_spec.check_isotope()
                cur_spec.check_adduction_list(molecular_weight=self.current_raw_matching_result[i][0].total_exact_mass)
                cur_spec.check_multimer(molecular_weight=self.current_raw_matching_result[i][0].total_exact_mass)
                sub_spec = cur_spec.generate_sub_recon_spec()
                self.n_candidates_further_match_r[i] = MoNAMatchingResult(nth_match=int(self.nth_match + 1),
                                                                          database=self.database,
                                                                          base_index_relative=self.base_index_relative,
                                                                          base_index_abs=self.base_index_abs,
                                                                          cur_candidate_info=cur_candi_info,
                                                                          parent_recons_spec=cur_spec,
                                                                          current_recons_spec=sub_spec,
                                                                          total_layer_matching=
                                                                          self.total_layer_matching,
                                                                          n_candidates_further_matched=
                                                                          self.n_candidates_further_matched)


@dataclass()
class IROAMatchingResult(MZCloudMatchingResult):
    database: Optional[IROA_db] = field(default=None, repr=False)

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
                if self.database:
                    self.current_raw_matching_result = self.database.find_match(target=self.current_recons_spec,
                                                                                save_matched_mz=True,
                                                                                transform=math.sqrt)

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
                self.n_candidates_further_match_r[i] = IROAMatchingResult(nth_match=int(self.nth_match + 1),
                                                                          database=self.database,
                                                                          base_index_relative=self.base_index_relative,
                                                                          base_index_abs=self.base_index_abs,
                                                                          cur_candidate_info=cur_candi_info,
                                                                          parent_recons_spec=cur_spec,
                                                                          current_recons_spec=sub_spec,
                                                                          total_layer_matching=
                                                                          self.total_layer_matching,
                                                                          n_candidates_further_matched=
                                                                          self.n_candidates_further_matched)


@dataclass
class GroupMatchingResult:
    """
    Container class for reconstructed spectrum matching results

    """
    base_index_abs: Optional[int] = field(default=None, repr=False)
    base_index_relative: Optional[int] = field(default=None, repr=True)
    recons_spec: Optional[ReconstructedSpectrum] = field(default=None, repr=True)
    matched_against_mzcloud: Optional[bool] = field(default=True, repr=False)
    matched_against_mona: Optional[bool] = field(default=True, repr=False)
    matched_against_iroa: Optional[bool] = field(default=True, repr=False)
    mzcloud_result: Optional[MZCloudMatchingResult] = field(default=None, repr=False)
    mona_result: Optional[MoNAMatchingResult] = field(default=None, repr=False)
    iroa_result: Optional[IROAMatchingResult] = field(default=None, repr=False)

    def gen_mzc_matching_result(self, total_layer_matching: Optional[int] = 3,
                                n_candidates_further_matched: Optional[int] = 1,
                                database: [MZCloud] = None,
                                base_index_abs: Optional[int] = base_index_abs,
                                base_index_relative: Optional[int] = base_index_relative) -> MZCloudMatchingResult:

        if self.matched_against_mzcloud:
            result = MZCloudMatchingResult(current_recons_spec=self.recons_spec,
                                           total_layer_matching=total_layer_matching,
                                           n_candidates_further_matched=n_candidates_further_matched,
                                           database=database,
                                           base_index_abs=base_index_abs,
                                           base_index_relative=base_index_relative)

            self.mzcloud_result = result

    def gen_mona_matching_result(self, total_layer_matching: Optional[int] = 3,
                                 n_candidates_further_matched: Optional[int] = 1,
                                 database: [MZCloud] = None,
                                 base_index_abs: Optional[int] = base_index_abs,
                                 base_index_relative: Optional[int] = base_index_relative) -> MoNAMatchingResult:

        if self.matched_against_mona:
            result = MoNAMatchingResult(current_recons_spec=self.recons_spec,
                                        total_layer_matching=total_layer_matching,
                                        n_candidates_further_matched=n_candidates_further_matched,
                                        database=database,
                                        base_index_abs=base_index_abs,
                                        base_index_relative=base_index_relative)

            self.mona_result = result

    def gen_iroa_matching_result(self, total_layer_matching: Optional[int] = 3,
                                 n_candidates_further_matched: Optional[int] = 1,
                                 database: [MZCloud] = None,
                                 base_index_abs: Optional[int] = base_index_abs,
                                 base_index_relative: Optional[int] = base_index_relative) -> IROAMatchingResult:

        if self.matched_against_iroa:
            result = IROAMatchingResult(current_recons_spec=self.recons_spec,
                                        total_layer_matching=total_layer_matching,
                                        n_candidates_further_matched=n_candidates_further_matched,
                                        database=database,
                                        base_index_abs=base_index_abs,
                                        base_index_relative=base_index_relative)

            self.iroa_result = result
