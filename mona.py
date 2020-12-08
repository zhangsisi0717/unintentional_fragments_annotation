import json
import pickle as pkl
from pathlib import Path
from type_checking import *
from dataclasses import dataclass, field, asdict
import re
from matplotlib import pyplot as plt
from matplotlib import rc_context
import warnings
from tqdm.auto import tqdm
import math
from msdata import *
from binned_vec import *
from mzcloud import *
from IROA_IDX import *

@dataclass
class MonaSpectrum(Spectrum):
    Polarity: Union[str] = field(default=None,repr=False)
    name: Union[int, str, None] = field(default=None,repr=True)
    spectrum_id: Optional[str] = field(default=None,repr=True)
    InChI: Optional[str] = field(default=None,repr=False)
    InChIKey: Optional[str] = field(default=None,repr=False)
    total_exact_mass: Optional[float] = field(default=None,repr=False)
    molecular_formula: Optional[str] = field(default=None,repr=False)
    ms_level: Optional[str] = field(default=None,repr=True)
    collision_energy: Optional[str] = field(default=None,repr=False)
    precursor: Optional[float] = field(default=None,repr=False)
    precursor_type: Optional[str] = field(default=None,repr=False)
    matched_mzs: Optional[List[Tuple]] = field(default=None,repr=False)
    # def __post_init__(self):
    #     super().__post_init__()

#dict_keys(['spectrum_id', 'name', 'spectrum_list', 'InChI', 'InChIKey', 'molecular_formula', 'total_exact_mass',
#\'ms_level', 'collision_energy', 'precursor', 'precursor_type'])

@dataclass
class MonaCompounds:
    id: Union[int, str, None] = field(default=None,repr=True)
    name: Union[int, str, None] = field(default=None,repr=True)
    InChI: Optional[str] = field(default=None,repr=False)
    InChIKey: Optional[str] = field(default=None,repr=False)
    total_exact_mass: Optional[float] = field(default=None,repr=False)
    molecular_formula: Optional[str] = field(default=None,repr=False)
    spectra_1: List[MonaSpectrum] = field(default_factory=list, repr=False)
    spectra_2: List[MonaSpectrum] = field(default_factory=list, repr=False)
    mzs_1: Optional[MZCollection] = field(default=None, repr=False, init=False)
    mzs_2: Optional[MZCollection] = field(default=None, repr=False, init=False)
    mzs_filtered: Optional[MZCollection] = field(default=None, repr=False, init=False)
    mzs_union: Optional[MZCollection] = field(default=None, repr=False, init=False)
    mz: Optional[float] = field(default=None, repr=False, init=False)

    def add_spectra_1(self, spectra_1: MonaSpectrum)->List[MonaSpectrum]:
        self.spectra_1.append(spectra_1)

    def add_spectra_2(self, spectra_2: MonaSpectrum)->List[MonaSpectrum]:
        self.spectra_2.append(spectra_2)

    def generate_mz_collection(self, mode: str = 'Negative', ppm: float = 30.,
                               rela_threshold: float = 1E-2,
                               save: bool = True,
                               filter_func: Optional[Callable[[MonaSpectrum], bool]] = None,
                               ) -> Tuple[MZCollection, MZCollection, MZCollection, MZCollection]:

        if mode not in ('Negative', 'Positive'):
            raise ValueError("mode must be either 'Negative' or 'Positive'")

        mzs_1 = MZCollection(ppm=ppm)
        mzs_2 = MZCollection(ppm=ppm)
        if self.spectra_1:
            spectra_1 = self.spectra_1
            for s in spectra_1:
                mz_array = s.mz[s.relative_intensity >= rela_threshold]
                for mz in mz_array:
                    mzs_1.add(mz)
        if self.spectra_2:
            spectra_2 = self.spectra_2
            for s in spectra_2:
                mz_array = s.mz[s.relative_intensity >= rela_threshold]
                for mz in mz_array:
                    mzs_2.add(mz)

        if filter_func is None:
            filter_func = lambda x: True

        if mzs_1 and mzs_2:
            in_mzs2 = [i in mzs_2 for i in mzs_1.center]
            mzs_filtered: MZCollection = mzs_1.filter(leave=in_mzs2, inplace=False)
            mzs_union: MZCollection = MZCollection.union(mzs_1, mzs_2)
            mzs_union.sort()

        elif (not mzs_1) and (mzs_2):
            mzs_union = mzs_2
            mzs_union.sort()
            mzs_filtered = mzs_2

        if save:
            self.mzs_1 = mzs_1
            self.mzs_2 = mzs_2
            self.mzs_filtered = mzs_filtered
            self.mzs_union = mzs_union


@dataclass()
class MonaDatabase:
    name: Optional[str] = 'mona'
    neg_dir: Optional[str] = field(default=None, repr=False)
    pos_dir: Optional[str] = field(default=None, repr=False)
    compounds_list: Optional[List[MonaCompounds]] = field(default_factory=list, repr=False)
    compounds_dic: Optional[Dict] = field(default_factory=dict, repr=False) ##key:compounds_name value:List[Spectrum]
    n_compounds: Optional[int] = field(default=None, repr=True)
    positive_spectra: Optional[List[MonaSpectrum]] = field(default_factory=list, repr=False)
    negative_spectra: Optional[List[MonaSpectrum]] = field(default_factory=list, repr=False)

    def read_file(self, cur_mode='Negative'):
        if cur_mode == 'Negative':
            if self.neg_dir:
                with open(self.neg_dir, 'rb') as f:
                    db = pkl.load(f)
                for key, value in tqdm(db.items(), desc='Reading MoNA negative spectra'):
                    neg_spec = MonaSpectrum(**value, mode='Negative')
                    self.negative_spectra.append(neg_spec)
                    if neg_spec.name not in self.compounds_dic:
                        self.compounds_dic[neg_spec.name] = [neg_spec]
                    else:
                        self.compounds_dic[neg_spec.name].append(neg_spec)
                i = 0
                for cmp_name,comp in tqdm(self.compounds_dic.items(), desc='Creating compound list'):
                    self.compounds_list.append(MonaCompounds(id=i, name=cmp_name, spectra_2=comp,\
                                                             InChI = comp[0].InChI, InChIKey=comp[0].InChIKey,\
                                                             total_exact_mass=comp[0].total_exact_mass,\
                                                             molecular_formula=comp[0].molecular_formula))
                    i += 1
            else:
                raise ValueError('must input the directory of MoNA_neg_spectra.pkl file!')

        elif cur_mode == 'Positive':
            if self.pos_dir:
                with open(self.pos_dir, 'rb') as f:
                    db = pkl.load(f)
                for key, value in tqdm(db.items(), desc='Reading MoNA positive spectra'):
                    pos_spec = MonaSpectrum(**value,mode='Positive')
                    self.positive_spectra.append(pos_spec)
                    if pos_spec.name not in self.compounds_dic:
                        self.compounds_dic[pos_spec.name] = [pos_spec]
                    else: self.compounds_dic[pos_spec.name].append(pos_spec)
                i=0
                for cmp_name,comp in tqdm(self.compounds_dic.items(), desc='Creating compound list'):
                    self.compounds_list.append(MonaCompounds(id=i, name=cmp_name,spectra_2=comp, \
                                                             InChI=comp[0].InChI, InChIKey=comp[0].InChIKey, \
                                                             total_exact_mass=comp[0].total_exact_mass, \
                                                             molecular_formula=comp[0].molecular_formula))
                    i+=1

            else:
                raise ValueError('must input the directory of MoNA_pos_spectra.pkl file!')

        else:
            raise ValueError('cur_mode must be Negative or Positive!')

        self.n_compounds = len(self.compounds_list)

    def add_to_compounds_list(self, compound: MonaCompounds) -> List[MonaCompounds]:
        self.compounds_list.append(compound)

    def add_positive_spectra(self, spectrum: MonaSpectrum) -> List[MonaSpectrum]:
        if spectrum.mode == 'Positive':
            self.positive_spectra.append(spectrum)
        else:
            raise ValueError('spectrum mode is not positive!')

    def add_negative_spectra(self,spectrum:MonaSpectrum)->List[MonaSpectrum]:
        if spectrum.mode == 'Negative':
            self.negative_spectra.append(spectrum)
        else:
            raise ValueError('spectrum mode is not negative!')

    @property
    def n_pos_spectra(self):
        return len(self.positive_spectra)

    @property
    def n_neg_spectra(self):
        return len(self.negative_spectra)

    def find_match(self, target: Union[ReconstructedSpectrum, IROASpectrum, MZCloudSpectrum, MonaSpectrum],
                   rela_threshold: float = 1E-2,
                   # mode: str = 'Negative',
                   cos_threshold: float = 1E-4,
                   transform: Optional[Callable[[float], float]] = None, save_matched_mz: bool = True,
                   reset_matched_mzs: bool = True,
                   precur_mass_diff_threshold: Optional[Numeric] = np.inf) -> List[Tuple[MonaCompounds, MonaSpectrum, float]]:

        # if mode not in ('Negative', 'Positive'):
        #     raise ValueError("mode can only be 'Negative' or 'Positive'.")

        candidates: List[MonaCompounds] = []

        if self.compounds_list is None:
            raise ValueError("Run read_file() first.")


        for c in tqdm(self.compounds_list, desc="finding candidates compounds", leave=True):
            for mz, intensity in zip(target.mz, target.relative_intensity):
                if intensity > rela_threshold and mz in c.mzs_union:
                    candidates.append(c)
                    break

        res = []
        for c in tqdm(candidates, desc="looping through candidates", leave=True):
            for s in c.spectra_1 + c.spectra_2:
                if_choose_s = False
                # cos = s.bin_vec.cos(other=target.bin_vec, transform=transform,
                #                     save_matched_mz=save_matched_mz,
                #                     reset_matched_idx_mz=reset_matched_idx_mz)
                if s.precursor:
                    for mz in target.mz:
                        if (abs(s.precursor-mz)/s.precursor) * 1E6 <= precur_mass_diff_threshold:
                            if_choose_s = True
                if not s.precursor:
                    if_choose_s = True

                if reset_matched_mzs:
                    s.matched_mzs = None

                if if_choose_s:
                    cos, matched_mzs = target.cos(other=s,func=transform)
                #matched_mzs: all the matched (mzs,abs_ints) in target
                    if cos > cos_threshold:
                        res.append((c, s, cos))
                        if save_matched_mz:
                            s.matched_mzs = matched_mzs

        res.sort(key=lambda x: x[2], reverse=True)
        cmp = set()
        cmp_list = []
        for c, s, cos in res:
            if c.id not in cmp:
                cmp.add(c.id)
                cmp_list.append((c, s, cos))

        return cmp_list









