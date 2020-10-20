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
from msdata import ReconstructedSpectrum,Spectrum
from binned_vec import *
from mzcloud import *


@dataclass
class IROASpectrum(Spectrum):
    name: Optional[str] = field(default=None,repr=True)
    ms_level: Optional[str] = field(default='MS2',repr=True)
    id: Optional[str] = field(default=None,repr=False)
    cpdID: Optional[str] = field(default=None,repr=False)
    Polarity: Optional[str] = field(default=None, repr=False)
    spectrum: Optional[dict] = field(default=None,repr=False)
    precursor: Optional[float] = field(default=None,repr=False)
    MolecularWeight: Optional[Numeric] = field(default=None,repr=False)
    InChIKey: Optional[str] = field(default=None, repr=False)
    matched_mzs: Optional[List[Tuple]] = field(default=None,repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.Polarity = self.mode
        self.spectrum_list_abs = self.spectrum_list


@dataclass
class IROACompounds:
    name: Optional[str] = field(default=None, repr=True)
    spectra_1: Optional[List[IROASpectrum]] = field(default_factory=list, repr=False)
    spectra_2: Optional[List[IROASpectrum]] = field(default_factory=list, repr=False)
    mzs_1: Optional[MZCollection] = field(default=None, repr=False, init=False)
    mzs_2: Optional[MZCollection] = field(default=None, repr=False, init=False)
    mzs_filtered: Optional[MZCollection] = field(default=None, repr=False, init=False)
    mzs_union: Optional[MZCollection] = field(default=None, repr=False, init=False)
    cpdID: Optional[str] = field(default=None,repr=False)
    MolecularWeight:Optional[Numeric] = field(default=None,repr=False)
    InChIKey: Optional[str] = field(default=None, repr=False)

    def add_spectra_1(self, spectra_1:IROASpectrum)->List[IROASpectrum]:
        self.spectra_1.add(spectra_1)

    def add_spectra_2(self, spectra_2:IROASpectrum)->List[IROASpectrum]:
        self.spectra_2.add(spectra_2)

    def generate_mz_collection(self, mode: str = 'Negative', ppm: float = 30.,
                               rela_threshold: float = 1E-2,
                               save: bool = True,
                               filter_func: Optional[Callable[[IROASpectrum], bool]] = None,
                               ) -> Tuple[MZCollection, MZCollection, MZCollection, MZCollection]:

        if mode not in ('Negative', 'Positive'):
            raise ValueError("mode must be either 'Negative' or 'Positive'")

        mzs_1 = MZCollection(ppm=ppm)
        mzs_2 = MZCollection(ppm=ppm)
        if self.spectra_1:
            spectra_1: List[IROASpectrum] = [i for i in self.spectra_1 if i.Polarity == mode]
            for s in spectra_1:
                mz_array = s.mz[s.relative_intensity >= rela_threshold]
                for mz in mz_array:
                    mzs_1.add(mz)
        if self.spectra_2:
            spectra_2: List[IROASpectrum] = [i for i in self.spectra_2 if i.Polarity == mode]
            if spectra_2:
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
class IROADataBase:
    dir: Optional[str] = None
    compounds_list: Optional[List[IROACompounds]] = field(default_factory=list, repr=None)
    compounds_dic: Optional[Dict] = field(default_factory=dict,repr=None) ##key:compounds_name value:List[Spectrum]
    n_compounds: Optional[int] = field(default=None,repr=True)
    positive_spectra: Optional[List[IROASpectrum]] = field(default_factory=list, repr=False)
    negative_spectra: Optional[List[IROASpectrum]] = field(default_factory=list, repr=False)

    def read_file(self):
        if dir is not None:
            with open(self.dir, 'rb') as f:
                db = pkl.load(f)

            for key,value in tqdm(db['Positive'].items(),desc='Reading positive spectra'):
                pos_spec = IROASpectrum(spectrum_list=[(i, j) for i, j in value['spectrum'].items()], **value)
                self.positive_spectra.append(pos_spec)
                if pos_spec.name not in self.compounds_dic:
                    self.compounds_dic[pos_spec.name] = [pos_spec]
                else:self.compounds_dic[pos_spec.name].append(pos_spec)

            for key,value in tqdm(db['Negative'].items(),desc='Reading negative spectra'):
                neg_spec = IROASpectrum(spectrum_list=[(i, j) for i, j in value['spectrum'].items()], **value)
                self.negative_spectra.append(neg_spec)
                if neg_spec.name not in self.compounds_dic:
                    self.compounds_dic[neg_spec.name] = [neg_spec]
                else: self.compounds_dic[neg_spec.name].append(neg_spec)

            for cmp_name, comp in tqdm(self.compounds_dic.items(),desc='Creating compound list'):
                self.compounds_list.append(IROACompounds(name=cmp_name, spectra_2=comp,
                                                         MolecularWeight=comp[0].MolecularWeight,
                                                         InChIKey=comp[0].InChIKey,cpdID=comp[0].cpdID))

        else:
            raise warnings.warn('must input the directory of the pickle file')

    def add_to_compounds_list(self, compound: IROACompounds)-> List[IROACompounds]:
        self.compounds_list.append(compound)

    def add_positive_spectra(self, spectrum:IROASpectrum)->List[IROASpectrum]:
        if spectrum.mode == 'Positive':
            self.positive_spectra.append(spectrum)
        else:
            raise ValueError('spectrum mode is not positive!')

    def add_negative_spectra(self, spectrum:IROASpectrum)->List[IROASpectrum]:
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

    def find_match(self, target: Union[ReconstructedSpectrum, IROASpectrum, MZCloudSpectrum],
                   rela_threshold: float = 1E-2,
                   mode: str = 'Negative',
                   cos_threshold: float = 1E-5,
                   transform: Optional[Callable[[float], float]] = None,
                   save_matched_mz: bool = True,
                   reset_matched_mzs: bool = True) -> List[Tuple[MZCloudCompound, MZCloudSpectrum, float]]:

        if mode not in ('Negative', 'Positive'):
            raise ValueError("mode can only be 'Negative' or 'Positive'.")

        candidates: List[MZCloudCompound] = []

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
                if s.Polarity == mode:
                    if_choose_s = False
                    # cos = s.bin_vec.cos(other=target.bin_vec, transform=transform,
                    #                     save_matched_mz=save_matched_mz,
                    #                     reset_matched_idx_mz=reset_matched_idx_mz)
                    if s.precursor:
                        for mz in target.mz:
                            if (abs(s.precursor-mz)/s.precursor) * 1E6 <= 70:
                                if_choose_s = True
                    if not s.precursor:
                        if_choose_s = True


                    if reset_matched_mzs:
                        s.matched_mzs = None

                    if if_choose_s:
                        cos, matched_mzs = target.cos(other=s,func=transform)
                        if cos > cos_threshold:
                            res.append((c, s, cos))
                            if save_matched_mz:
                                s.matched_mzs = matched_mzs

        res.sort(key=lambda x: x[2], reverse=True)

        cmp = set()
        cmp_list = []
        for c, s, cos in res:
            if c.cpdID not in cmp:
                cmp.add(c.cpdID)
                cmp_list.append((c, s, cos))

        return cmp_list










