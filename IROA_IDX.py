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
class IROA_Spectrum(Spectrum):
    name: Optional[str] = field(default=None,repr=True)
    ms_level: Optional[str] = field(default='2',repr=True)
    id: Optional[str] = field(default=None,repr=False)
    cpdID: Optional[str] = field(default=None,repr=False)
    Polarity: Optional[str] = field(default=None, repr=False)
    spectrum: Optional[dict] = field(default=None,repr=False)
    precursor : Optional[float] = field(default=None,repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.Polarity = self.mode


    # mz: Optional[np.ndarray] = field(default=np.array([i for i in spectrum.keys()]),repr=False)
    # intensity: Optional[np.ndarray] = field(default=np.array([i for i in spectrum.values()]),repr=False)
    # spectrum_list = field(default=self.spectrum,repr=False)
    # def __post_init__(self):
    #     if self.spectrum is not None:
    #         self.mz = np.array([i for i in self.spectrum.keys()])
    #         self.intensity = np.array([i for i in self.spectrum.values()])
    #         self.spectrum_list = [(i,j) for i,j in self.spectrum.items()]
    #         self.n_peaks = len(self.mz)
    #         self.max_intensity = 0. if len(self.mz)==0 else np.max(self.intensity)
    #
    #     if self.max_intensity > 0.:
    #         self.relative_intensity = self.intensity / self.max_intensity
    #     else:
    #         self.relative_intensity = self.intensity
    #
    #     if len(self.mz) > 0.:
    #         self.n_peaks = len(self.mz)
    #
    #     else:
    #         raise warnings.warn('number of peaks is 0')
@dataclass
class IROA_compounds:
    name: Optional[str] = field(default=None, repr=True)
    spectra_1: Optional[List[IROA_Spectrum]] = field(default_factory=list, repr=False)
    spectra_2: Optional[List[IROA_Spectrum]] = field(default_factory=list, repr=False)
    mzs_1: Optional[MZCollection] = field(default=None, repr=False, init=False)
    mzs_2: Optional[MZCollection] = field(default=None, repr=False, init=False)
    mzs_filtered: Optional[MZCollection] = field(default=None, repr=False, init=False)
    mzs_union: Optional[MZCollection] = field(default=None, repr=False, init=False)
    cpdID: Optional[str] = field(default=None,repr=False)


    def add_spectra_1(self,spectra_1:IROA_Spectrum)->List[IROA_Spectrum]:
        self.spectra_1.add(spectra_1)

    def add_spectra_2(self,spectra_2:IROA_Spectrum)->List[IROA_Spectrum]:
        self.spectra_2.add(spectra_2)

    def generate_mz_collection(self, mode: str = 'Negative', ppm: float = 30.,
                               rela_threshold: float = 1E-2,
                               save: bool = True,
                               filter_func: Optional[Callable[[IROA_Spectrum], bool]] = None,
                               ) -> Tuple[MZCollection, MZCollection, MZCollection, MZCollection]:

        if mode not in ('Negative', 'Positive'):
            raise ValueError("mode must be either 'Negative' or 'Positive'")

        mzs_1 = MZCollection(ppm=ppm)
        mzs_2 = MZCollection(ppm=ppm)
        if self.spectra_1:
            spectra_1: List[IROA_Spectrum] = [i for i in self.spectra_1 if i.Polarity == mode]
            for s in spectra_1:
                mz_array = s.mz[s.relative_intensity >= rela_threshold]
                for mz in mz_array:
                    mzs_1.add(mz)
        if self.spectra_2:
            spectra_2: List[IROA_Spectrum] = [i for i in self.spectra_2 if i.Polarity == mode]
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
class IROA_db:
    dir: Optional[str] = None
    compounds_list: Optional[List[IROA_compounds]] = field(default_factory=list,repr=None)
    compounds_dic: Optional[Dict] = field(default_factory=dict,repr=None) ##key:compounds_name value:List[Spectrum]
    n_compounds: Optional[int] = field(default=None,repr=True)
    positive_spectra: Optional[List[IROA_Spectrum]] = field(default_factory=list, repr=False)
    negative_spectra: Optional[List[IROA_Spectrum]] = field(default_factory=list, repr=False)

    def read_file(self):
        if dir is not None:
            with open(self.dir, 'rb') as f:
                db = pkl.load(f)

            for key,value in tqdm(db['Positive'].items(),desc='Reading positive spectra'):
                pos_spec = IROA_Spectrum(spectrum_list=[(i,j) for i,j in value['spectrum'].items()],**value)
                self.positive_spectra.append(pos_spec)
                if pos_spec.name not in self.compounds_dic:
                    self.compounds_dic[pos_spec.name] = [pos_spec]
                else:self.compounds_dic[pos_spec.name].append(pos_spec)

            for key,value in tqdm(db['Negative'].items(),desc='Reading negative spectra'):
                neg_spec = IROA_Spectrum(spectrum_list=[(i,j) for i,j in value['spectrum'].items()],**value)
                self.negative_spectra.append(neg_spec)
                if neg_spec.name not in self.compounds_dic:
                    self.compounds_dic[neg_spec.name] = [neg_spec]
                else: self.compounds_dic[neg_spec.name].append(neg_spec)

            for cmp_name, comp in tqdm(self.compounds_dic.items(),desc='Creating compound list'):
                self.compounds_list.append(IROA_compounds(name=cmp_name,spectra_2=comp))

        else:
            raise warnings.warn('must input the directory of the pickle file')


    def add_to_compounds_list(self, compound: IROA_compounds)-> List[IROA_compounds]:
        self.compounds_list.append(compound)

    def add_positive_spectra(self,spectrum:IROA_Spectrum)->List[IROA_Spectrum]:
        if spectrum.mode == 'Positive':
            self.positive_spectra.append(spectrum)
        else:
            raise ValueError('spectrum mode is not positive!')

    def add_negative_spectra(self,spectrum:IROA_Spectrum)->List[IROA_Spectrum]:
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

    def find_match(self, target: Union[ReconstructedSpectrum, IROA_Spectrum,MZCloudSpectrum],
                   rela_threshold: float = 1E-2,
                   mode: str = 'Negative',
                   cos_threshold: float = 1E-3,
                   transform: Optional[Callable[[float], float]] = None
                   ) -> List[Tuple[MZCloudCompound, MZCloudSpectrum, float]]:

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
            for s in c.Spectra_1 + c.Spectra_2:
                if s.Polarity == mode:
                    cos = s.bin_vec.cos(target.bin_vec, transform=transform)
                    if cos > cos_threshold:
                        res.append((c, s, cos))
        res.sort(key=lambda x: x[2], reverse=True)

        cmp = set()
        cmp_list = []
        for c, s, cos in res:
            if c.cpdID not in cmp:
                cmp.add(c.cpdID)
                cmp_list.append((c, s, cos))

        return cmp_list










