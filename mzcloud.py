from pathlib import Path
import pickle as pkl
import json
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


class MZCollection:
    """
    A collection of mz values, used for fast spectrum matching. When matching, each mz will be considered a match
    if it is within a tolerance of one mz value stored in the class. This tolerance is characterized by a ppm value,
    i.e., for a mz value p, if there exists a mz value q in MZCollection such that

        1./ (1 + ppm * 1E-6) q <= p <= (1 + ppm * 1E-6) q

    then we think of p as equal to q and belongs to this collection.
    """

    @staticmethod
    def union(mzc1: "MZCollection", mzc2: "MZCollection") -> "MZCollection":
        ppm = min(mzc1.ppm, mzc2.ppm)
        mzc_union: "MZCollection" = MZCollection(ppm=ppm)
        for mz in mzc1.center:
            mzc_union.add(mz)

        for mz in mzc2.center:
            mzc_union.add(mz)

        return mzc_union

    def __init__(self, ppm: float = 30.) -> None:
        self.ppm: float = ppm
        self.lc: float = 1. / (1. + self.ppm * 1E-6)
        self.uc: float = 1. + self.ppm * 1E-6

        self.lb: List[float] = []
        self.ub: List[float] = []
        self.center: List[float] = []
        self.min: List[float] = []
        self.max: List[float] = []

        self.n = 0

    def __repr__(self) -> str:
        return "MZCollection(n_mz={})".format(self.n)

    def add(self, mz: float) -> None:

        for i in range(self.n):
            if self.lb[i] <= mz <= self.ub[i]:  # identified to belong to a group
                change_flag: bool = True
                if mz < self.min[i]:
                    self.min[i] = mz
                elif mz > self.max[i]:
                    self.max[i] = mz
                else:
                    change_flag = False

                if change_flag:  # need to recompute center, lb, ub
                    self.center[i] = math.sqrt(self.min[i] * self.max[i])
                    self.lb[i], self.ub[i] = self.center[i] * self.lc, self.center[i] * self.uc

                # once identified, break
                break
        # if loop finishes without breaking
        else:  # create new group
            self.lb.append(mz * self.lc)
            self.ub.append(mz * self.uc)
            self.center.append(mz)
            self.min.append(mz)
            self.max.append(mz)

            self.n += 1

    def __contains__(self, item: Float) -> bool:
        for lower, upper in zip(self.lb, self.ub):
            if lower <= item <= upper:
                return True
        else:
            return False

    def filter(self, leave: Union[Iterable[bool]], inplace: bool = False) -> Optional["MZCollection"]:

        leave = [bool(i) for i in leave]
        if len(leave) != self.n:
            raise ValueError("expect leave of length {}, got {} instead.".format(self.n, len(leave)))

        if inplace:
            self.center = [v for v, i in zip(self.center, leave) if i]
            self.lb = [v for v, i in zip(self.lb, leave) if i]
            self.ub = [v for v, i in zip(self.ub, leave) if i]
            self.min = [v for v, i in zip(self.min, leave) if i]
            self.max = [v for v, i in zip(self.max, leave) if i]

            self.n = len(self.center)
        else:
            res = MZCollection(ppm=self.ppm)
            res.center = [v for v, i in zip(self.center, leave) if i]
            res.lb = [v for v, i in zip(self.lb, leave) if i]
            res.ub = [v for v, i in zip(self.ub, leave) if i]
            res.min = [v for v, i in zip(self.min, leave) if i]
            res.max = [v for v, i in zip(self.max, leave) if i]

            res.n = len(res.center)

            return res

    def sort(self, reverse=False) -> None:

        idx = [i for i, c in sorted(enumerate(self.center), key=lambda x: x[1], reverse=reverse)]

        self.center: List[float] = [self.center[i] for i in idx]
        self.lb: List[float] = [self.lb[i] for i in idx]
        self.ub: List[float] = [self.ub[i] for i in idx]
        self.min: List[float] = [self.min[i] for i in idx]
        self.max: List[float] = [self.max[i] for i in idx]

    # def match_score(self, recons_spectrum: Union[ReconstructedSpectrum, "MZCloudSpectrum"],
    #                 transform: Optional[Callable[[float], float]] = None,
    #                 threshold: float = 0.,
    #                 mz_upperbound: float = 5.
    #                 ) -> float:
    #     """
    #     Return the match score of a reconstructed spectrum
    #     :param recons_spectrum:
    #     :param transform:
    #     :param threshold:
    #     :param mz_upperbound:
    #     :return: -1 if matching failed, or a number between 0 and 1.
    #     """
    #
    #     if self.n == 0:  # self is empty
    #         return -1.  # failed
    #
    #     if recons_spectrum.mz is None or recons_spectrum.relative_intensity is None:
    #         raise ValueError("Invalid reconstructed spectrum")
    #
    #     matched_mass: float = 0.
    #     unmatched_mass: float = 0.
    #
    #     if recons_spectrum.arg_max_mz is not None:
    #         mz_ub = recons_spectrum.arg_max_mz + mz_upperbound
    #     else:
    #         mz_ub = np.inf
    #
    #     for mz, rel_int in zip(recons_spectrum.mz, recons_spectrum.relative_intensity):
    #
    #         if mz > mz_ub:
    #             continue  # skip
    #
    #         if rel_int < threshold:  # skip
    #             continue
    #         if transform is None:
    #             inc = rel_int
    #         else:
    #             inc = transform(rel_int)
    #         if mz in self:  # got a matched fragments
    #             matched_mass += inc
    #         else:
    #             unmatched_mass += inc
    #
    #     total_mass: float = matched_mass + unmatched_mass
    #     if total_mass == 0.:
    #         return -1.  # failed
    #     else:
    #         return matched_mass / total_mass


@dataclass()
class MZCloudOptions:
    # regular expression that matches valid file names for MS1 trees
    # by default, matches files with 7 digits and ends with ".json"
    ms1_file_pattern: Optional[Pattern] = re.compile(r"^[0-9]{7}\.json$")

    # regular expression that matches valid file names for MS1 trees
    # by default, matches files with 7 digits and ends with ".json"
    ms2_file_pattern: Optional[Pattern] = re.compile(r"^[0-9]{7}\.json$")

    ms1_file_str: str = "{:07d}.json"
    ms2_file_str: str = "{:07d}.json"
    dpi: Numeric = 200
    figsize: Tuple[Numeric, Numeric] = (8, 6)


@dataclass()
class MZCloudTree:
    Id: Optional[int] = field(default=None, repr=False)
    id: Optional[int] = field(default=None, repr=True, init=False)
    OrderId: Optional[int] = field(default=None, repr=False)
    Polarity: Optional[str] = field(default=None, repr=True)
    Contributor: Optional[str] = field(default=None, repr=False)
    InstrumentName: Optional[str] = field(default=None, repr=False)
    IonizationMethod: Optional[str] = field(default=None, repr=False)
    CompoundId: Optional[int] = field(default=None, repr=False)
    CreationDate: Optional[str] = field(default=None, repr=False)
    ModifyDate: Optional[str] = field(default=None, repr=False)
    MZC_env: Optional["MZCloud"] = field(default=None, repr=False)

    def __post_init__(self):
        self.id = self.Id

    @property
    def spectra_1(self):
        if self.MZC_env is None:
            raise RuntimeError("No MZCloud associated.")

        try:
            return self.MZC_env.tree_to_spectra(self, ms_level=1)
        except FileNotFoundError:
            return []

    @property
    def spectra_2(self):
        if self.MZC_env is None:
            raise RuntimeError("No MZCloud associated.")

        try:
            return self.MZC_env.tree_to_spectra(self, ms_level=1)
        except FileNotFoundError:
            return []


@dataclass()
class MZCloudCompound:
    Id: Optional[int] = field(default=None, repr=False)
    id: Optional[int] = field(default=None, repr=True, init=False)
    MolecularWeight: Optional[Numeric] = field(default=None, repr=False, init=None)
    n_trees: int = field(default=0, repr=False, init=False)
    CompoundName: Optional[str] = None
    SystematicName: Optional[str] = field(default=None, repr=False)
    SearchCompoundName: Optional[str] = field(default=None, repr=False)
    SearchSystematicName: Optional[str] = field(default=None, repr=False)
    CAS: Optional[int] = field(default=None, repr=False)
    SMILES: Optional[str] = field(default=None, repr=False)
    InChI: Optional[str] = field(default=None, repr=False)
    InChIKey: Optional[str] = field(default=None, repr=False)
    IdNumbers: Optional[List] = field(default=None, repr=False)
    Synonyms: Optional[List] = field(default=None, repr=False)
    CompoundClasses: Optional[List] = field(default=None, repr=False)
    CreationDate: Optional[str] = field(default=None, repr=False)
    ModifyDate: Optional[str] = field(default=None, repr=False)
    SpectralTrees: Optional[List] = field(default=None, repr=False)
    trees: List[MZCloudTree] = field(default_factory=list, repr=False, init=False)
    tree_id: List[int] = field(default_factory=list, repr=False, init=False)
    precursor_mz: Optional[MZCollection] = field(default=None, repr=False, init=False)
    mzs_1: Optional[MZCollection] = field(default=None, repr=False, init=False)
    mzs_2: Optional[MZCollection] = field(default=None, repr=False, init=False)
    mzs_filtered: Optional[MZCollection] = field(default=None, repr=False, init=False)
    mzs_union: Optional[MZCollection] = field(default=None, repr=False, init=False)
    mz: Optional[float] = field(default=None, repr=False, init=False)
    MZC_env: Optional["MZCloud"] = field(default=None, repr=False)

    def __post_init__(self):
        self.id = self.Id
        if self.SpectralTrees is not None:
            self.trees = [MZCloudTree(MZC_env=self.MZC_env, **i) for i in self.SpectralTrees]
            self.n_trees = len(self.trees)
            self.tree_id = [i.Id for i in self.trees]

    def tree_iter(self) -> Iterator[MZCloudTree]:
        return self.trees.__iter__()

    @property
    def spectra_1(self):
        if self.MZC_env is None:
            raise RuntimeError("No MZCloud associated.")

        return self.MZC_env.to_spectra(self.trees, ignore_exception=True, ms_level=1)

    @property
    def spectra_2(self):
        if self.MZC_env is None:
            raise RuntimeError("No MZCloud associated.")

        return self.MZC_env.to_spectra(self.trees, ignore_exception=True, ms_level=2)

    def generate_mz_collection(self, mode: str = 'Negative', ppm: float = 20.,
                               threshold: float = 1E-2,
                               save: bool = True,
                               filter_func: Optional[Callable[["MZCloudSpectrum"], bool]] = None
                               ) -> Tuple[MZCollection, MZCollection, MZCollection, MZCollection]:

        if self.MZC_env is None:
            raise RuntimeError("No MZCloud associated.")

        return self.MZC_env.generate_mz_collection(compound=self,
                                                   mode=mode,
                                                   ppm=ppm,
                                                   threshold=threshold,
                                                   save=save,
                                                   filter_func=filter_func)

    def get_precursor(self, mode: str = 'Negative',
                      ppm: float = 30.,
                      save: bool = True) -> MZCollection:

        if self.MZC_env is None:
            raise RuntimeError("No MZCloud associated.")

        return self.MZC_env.get_precursor(compound=self, mode=mode, ppm=ppm, save=save)

@dataclass()
class MZCloudSpectrum(Spectrum):
    Spectrum_label: str = field(default='MZCloud',repr=False)
    mode: str = field(default=None,repr=False)
    Id: Optional[int] = field(default=None, repr=False)
    id: Optional[int] = field(default=None, repr=True, init=False)
    MolecularWeight: Optional[Numeric] = field(default=None, repr=False, init=False)

    PrecursorPeaks: Optional[List] = field(default=None, repr=False)
    Postprocessing: Optional[str] = field(default=None, repr=False)
    SpectrumKind: Optional[str] = field(default=None, repr=False)
    Peaks: Optional[List] = field(default=None, repr=False)
    MassRange: Optional[Dict] = field(default=None, repr=False)
    IsolationWidth: Optional[Dict] = field(default=None, repr=False)
    Analyzer: Optional[str] = field(default=None, repr=False)
    IonActivations: Optional[List] = field(default=None, repr=False)
    Filter: Any = field(default=None, repr=False)
    mz_range: Tuple[Optional[float], Optional[float]] = field(default=(None, None), repr=True, init=False)
    accuracy: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    resolution: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    is_not_nan: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    spectrum_valid: bool = field(default=False, repr=False, init=False)

    lb: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    ub: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    ppm: Optional[float] = field(default=None, repr=False, init=False)

    transform: Optional[str] = field(default=None, repr=False, init=False)
    transformed_int: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    transformed_int_norm: Optional[Float] = field(default=None, repr=False, init=False)
    threshold: Optional[float] = field(default=None, repr=False, init=False)

    MZC_env: Optional["MZCloud"] = field(default=None, repr=False)
    matched_mzs: Optional[List[Tuple]] = field(default=None,repr=False)

    def __post_init__(self):
        self.id = self.Id

        if self.Peaks is None:
            self.n_peaks = 0
        else:
            self.n_peaks = len(self.Peaks)

        if self.MassRange is not None:
            self.mz_range = (self.MassRange.get('Min'), self.MassRange.get('Max'))

        if self.Peaks is not None:

            # sort by mz value
            self.Peaks.sort(key=lambda x: -np.inf if x.get('MZ') is None else x.get('MZ'))

            mz = np.array([i['MZ'] if 'MZ' in i else np.nan for i in self.Peaks])
            intensity = np.array([i['Abundance'] if 'Abundance' in i else np.nan for i in self.Peaks])
            accuracy = np.array([i['Accuracy'] if 'Accuracy' in i else np.nan for i in self.Peaks])
            resolution = np.array([i['Resolution'] if 'Resolution' in i else np.nan for i in self.Peaks])

            self.is_not_nan = ~((np.isnan(mz)) | (np.isnan(intensity)))

            if np.any(self.is_not_nan):
                # if any of the spectrum has a non NAN value
                self.mz = mz[self.is_not_nan]
                self.intensity = intensity[self.is_not_nan]
                self.spectrum_list = list(zip(self.mz, self.intensity))
                self.spectrum_list_abs = self.spectrum_list
                self.accuracy = accuracy[self.is_not_nan]
                self.resolution = resolution[self.is_not_nan]
                self.max_intensity = np.max(self.intensity)
                self.relative_intensity = self.intensity / self.max_intensity
                self.spectrum_valid = True
                self.n_valid_peaks = len(self.mz)

                self.reduced_spectrum_list = [(mz, ints) for mz, ints in self.spectrum_list if ints > self.max_intensity * self.rela_threshold_reduce]
                self.reduced_mz = np.array([mz for mz, intensity in self.reduced_spectrum_list])
                self.reduced_intensity = np.array([intensity for mz, intensity in self.reduced_spectrum_list])
                self.reduced_rela_intensity = self.reduced_intensity / self.max_intensity
                self.reduced_n_peaks = len(self.reduced_intensity)

                # binned sparse vector
                self.bin_vec = BinnedSparseVector(ppm=self.bin_ppm)
                self.bin_vec.add(x=self.mz, y=self.relative_intensity,y_abs=self.intensity)


class MZCloud:
    @staticmethod
    def file_to_tree_id(file: Union[str, Path]):
        if isinstance(file, Path):
            stem = file.stem
        else:
            stem = Path(file).stem
        return int(stem)

    def __init__(self, root_dir: str, **options) -> None:
        self.name: Optional[str] = 'mzc'
        self.root_path: Path = Path(root_dir)
        if not self.root_path.is_dir():
            raise FileNotFoundError("root_dir '{}' is not a directory".format(root_dir))

        self.options = MZCloudOptions(**options)

        self.ms1_path: Path = Path(root_dir) / "MS1"
        self.ms2_path: Path = Path(root_dir) / "MS2"

        self.comp_metadata_path: Path = Path(root_dir) / "metadata.json"
        self.mw_inchikey_map_path: Path = Path(root_dir) / "mw_inchikey_map.pkl"

        self.ms1_available: bool = self.ms1_path.is_dir()
        self.ms2_available: bool = self.ms2_path.is_dir()
        self.comp_metadata_available: bool = self.comp_metadata_path.is_file()
        self.mw_inchikey_map_available: bool = self.mw_inchikey_map_path.is_file()

        self.compounds: Optional[List[MZCloudCompound]] = None
        self.compounds_dict: Optional[Dict[int, MZCloudCompound]] = None

        self.tree_to_compounds_dict: Optional[Dict[int, MZCloudCompound]] = None

        self._n_ms1_files: Optional[int] = None
        self._n_ms2_files: Optional[int] = None

        self.ms1_spectra: Dict[int, List[MZCloudSpectrum]] = dict()
        self.ms2_spectra: Dict[int, List[MZCloudSpectrum]] = dict()

    def ms1_iter(self, tree_id: bool = True) -> Iterator:
        if self.options.ms1_file_pattern is not None:
            if tree_id:
                ms1_iter = ((f, int(f.stem)) for f in self.ms1_path.glob("*")
                            if self.options.ms1_file_pattern.search(f.name) and f.is_file())
            else:
                ms1_iter = (f for f in self.ms1_path.glob("*")
                            if self.options.ms1_file_pattern.search(f.name) and f.is_file())
        else:
            if tree_id:
                ms1_iter = ((f, int(f.stem)) for f in self.ms1_path.glob("*") if f.is_file())
            else:
                ms1_iter = (f for f in self.ms1_path.glob("*") if f.is_file())

        return ms1_iter

    def ms2_iter(self, tree_id: bool = True) -> Iterator:
        if self.options.ms2_file_pattern is not None:
            if tree_id:
                ms2_iter = ((f, int(f.stem)) for f in self.ms2_path.glob("*")
                            if self.options.ms2_file_pattern.search(f.name))
            else:
                ms2_iter = (f for f in self.ms2_path.glob("*") if self.options.ms2_file_pattern.search(f.name))
        else:
            if tree_id:
                ms2_iter = ((f, int(f.stem)) for f in self.ms2_path.glob("*"))
            else:
                ms2_iter = self.ms2_path.glob("*")

        return ms2_iter

    @property
    def n_ms1_trees(self) -> int:
        if self._n_ms1_files is None:
            self._n_ms1_files = len(list(self.ms1_iter()))
        return self._n_ms1_files

    @property
    def n_ms2_trees(self) -> int:
        if self._n_ms2_files is None:
            self._n_ms2_files = len(list(self.ms2_iter()))
        return self._n_ms2_files

    def tree_to_spectra(self, tree: Union[int, MZCloudTree], ms_level: int = 1, save=True,
                        load=True) -> List[MZCloudSpectrum]:

        if isinstance(tree, MZCloudTree):
            tree_id = tree.id
        else:
            tree_id = int(tree)

        if ms_level not in (1, 2):
            raise ValueError("ms_level must be 1 or 2.")

        if load:
            if ms_level == 1 and tree_id in self.ms1_spectra:
                return self.ms1_spectra[tree_id]
            elif tree_id in self.ms2_spectra:
                return self.ms2_spectra[tree_id]

        if ms_level == 1:
            parent = self.ms1_path
            name = self.options.ms1_file_str.format(tree_id)
        else:  # ms_level == 2:
            parent = self.ms2_path
            name = self.options.ms2_file_str.format(tree_id)

        file = parent / name

        if not file.is_file():
            raise FileNotFoundError("tree file '{}' not found".format(file.name))

        with open(file.as_posix(), 'r') as fp:
            res = json.load(fp)

        spectra = [MZCloudSpectrum(MZC_env=self, **spectrum) for spectrum in res]
        if save:
            if ms_level == 1:
                self.ms1_spectra[tree_id] = spectra
            else:  # ms_level == 2
                self.ms2_spectra[tree_id] = spectra

        return spectra

    def to_spectra(self, trees: Iterable[Union[int, MZCloudTree]],
                   ms_level: int = 1,
                   ignore_exception: bool = True
                   ) -> List[List[MZCloudSpectrum]]:
        res = []
        for t in trees:
            try:
                res.append(self.tree_to_spectra(t, ms_level=ms_level))
            except FileNotFoundError:
                if not ignore_exception:
                    raise

        return res

    def plot_spectrum(self, spectrum: MZCloudSpectrum, relative: bool = True, **kwargs) -> None:

        if spectrum.spectrum_valid:

            with rc_context(rc={'figure.dpi': self.options.dpi, 'figure.figsize': self.options.figsize}):
                if relative:  # relative intensity
                    _, _, baseline = \
                        plt.stem(spectrum.mz, spectrum.relative_intensity, use_line_collection=True,
                                 markerfmt=' ', **kwargs)
                    plt.ylabel("relative intensity")
                else:  # absolute intensity
                    _, _, baseline = \
                        plt.stem(spectrum.mz, spectrum.intensity, use_line_collection=True,
                                 markerfmt=' ', **kwargs)
                    plt.ylabel("intensity")

                baseline.set_xdata([0.01, .99])
                baseline.set_transform(plt.gca().get_yaxis_transform())
                plt.xlabel("mz")

        else:
            msg = "not a valid spectrum"
            warnings.warn(msg)

    def __str__(self):
        return "MZCloud(root_dir='{}', " \
               "n_MS1_trees={}, n_MS2_trees={})".format(self.root_path.as_posix(), self.n_ms1_trees, self.n_ms2_trees)

    def read_all_trees(self, enable_tqdm: bool = True, leave: bool = True) -> None:

        for _, tree_id in tqdm(self.ms1_iter(tree_id=True), desc="reading MS1 files", total=self.n_ms1_trees,
                               disable=not enable_tqdm, leave=leave):
            self.tree_to_spectra(tree=tree_id, ms_level=1, save=True)

        for _, tree_id in tqdm(self.ms2_iter(tree_id=True), desc="reading MS2 files", total=self.n_ms2_trees,
                               disable=not enable_tqdm, leave=leave):
            self.tree_to_spectra(tree=tree_id, ms_level=2, save=True)

    def read_comp_metadata(self) -> None:
        if not self.mw_inchikey_map_available:
            raise ValueError('mw_inchikey_map not available')

        with open(self.mw_inchikey_map_path.as_posix(), 'rb') as f:
            mw_inchi_map = pkl.load(f)

        if not self.comp_metadata_available:
            raise ValueError("metadata not available")

        with open(self.comp_metadata_path.as_posix(), 'r') as fp:
            res = json.load(fp)

        self.compounds = [MZCloudCompound(MZC_env=self, **v) for v in res]

        self.compounds_dict = dict()
        self.tree_to_compounds_dict = dict()

        for c in self.compounds:
            c.MolecularWeight = mw_inchi_map.get(c.InChIKey, None)
            self.compounds_dict[c.id] = c
            for t in c.tree_id:
                self.tree_to_compounds_dict[t] = c

    def get_precursor(self, compound: MZCloudCompound, mode: str = 'Negative',
                      ppm: float = 30.,
                      save: bool = True) -> MZCollection:
        if mode not in ('Negative', 'Positive'):
            raise ValueError("mode must be either 'Negative' or 'Positive'")

        precursor = MZCollection(ppm=ppm)

        for t in compound.trees:

            if t.Polarity == mode:
                try:
                    s = self.tree_to_spectra(tree=t, ms_level=2)
                    for i in s:
                        for j in i.PrecursorPeaks:
                            if j.get('MZ') is not None:
                                precursor.add(j.get('MZ'))

                except FileNotFoundError:
                    pass

        if save:
            compound.precursor_mz = precursor
            if precursor.n == 1:
                compound.mz = precursor.center[0]

        return precursor

    def generate_mz_collection(self, compound: MZCloudCompound, mode: str = 'Negative', ppm: float = 30.,
                               threshold: float = 1E-2,
                               save: bool = True,
                               filter_func: Optional[Callable[[MZCloudSpectrum], bool]] = None
                               ) -> Tuple[MZCollection, MZCollection, MZCollection, MZCollection]:

        if mode not in ('Negative', 'Positive'):
            raise ValueError("mode must be either 'Negative' or 'Positive'")

        spectra_1: List[MZCloudSpectrum] = []
        spectra_2: List[MZCloudSpectrum] = []

        if filter_func is None:
            filter_func = lambda x: True

        for t in compound.trees:
            if t.Polarity != mode:
                continue
            try:
                s = self.tree_to_spectra(tree=t, ms_level=1)
                spectra_1 += [i for i in s if i.Polarity == mode and filter_func(i)]
            except FileNotFoundError:
                pass

            try:
                s = self.tree_to_spectra(tree=t, ms_level=2)
                spectra_2 += [i for i in s if i.Polarity == mode and filter_func(i)]
            except FileNotFoundError:
                pass

        mzs_1 = MZCollection(ppm=ppm)
        mzs_2 = MZCollection(ppm=ppm)

        for s in spectra_1:
            s.MolecularWeight = compound.MolecularWeight
            s.ms_level = 'MS1'
            mz_array = s.mz[s.relative_intensity >= threshold]
            for mz in mz_array:
                mzs_1.add(mz)
        for s in spectra_2:
            s.MolecularWeight = compound.MolecularWeight
            s.ms_level = 'MS2'
            mz_array = s.mz[s.relative_intensity >= threshold]
            for mz in mz_array:
                mzs_2.add(mz)

        mzs_1.sort()
        mzs_2.sort()

        in_mzs2 = [i in mzs_2 for i in mzs_1.center]
        mzs_filtered: MZCollection = mzs_1.filter(leave=in_mzs2, inplace=False)

        mzs_union: MZCollection = MZCollection.union(mzs_1, mzs_2)
        mzs_union.sort()

        if save:
            compound.mzs_1 = mzs_1
            compound.mzs_2 = mzs_2
            compound.mzs_filtered = mzs_filtered
            compound.mzs_union = mzs_union

        return mzs_1, mzs_2, mzs_filtered, mzs_union

    def filter_compounds(self, func: Callable[[MZCloudCompound], bool]):

        return (i for i in self.compounds if func(i))

    def find_match(self, target: Union[ReconstructedSpectrum, MZCloudSpectrum],
                   threshold: float = 1E-3,
                   mode: str = 'Negative',
                   cos_threshold: float = 1E-4,
                   transform: Optional[Callable[[float], float]] = None,
                   save_matched_mz: bool = True,
                   reset_matched_mzs: bool = True
                   ) -> List[Tuple[MZCloudCompound, MZCloudSpectrum, float]]:

        if mode not in ('Negative', 'Positive'):
            raise ValueError("mode can only be 'Negative' or 'Positive'.")

        candidates: List[MZCloudCompound] = []

        if self.compounds is None:
            raise ValueError("Run read_comp_metadata() first.")

        for c in tqdm(self.compounds, desc="finding candidates compounds", leave=True):
            for mz, intensity in zip(target.mz, target.relative_intensity):
                if intensity > threshold and mz in c.mzs_union:
                    candidates.append(c)
                    break

        res = []
        for c in tqdm(candidates, desc="looping through candidates", leave=True):
            for t in c.spectra_1 + c.spectra_2:
                for s in t:
                    if s.Polarity == mode:
                        if_choose_s = False
                        # cos = s.bin_vec.cos(other=target.bin_vec, transform=transform,
                        #                     save_matched_mz=save_matched_mz,
                        #                     reset_matched_idx_mz=reset_matched_idx_mz)
                        if s.PrecursorPeaks:
                            for mz in target.mz:
                                if (abs(s.PrecursorPeaks[0]['MZ']-mz)/s.PrecursorPeaks[0]['MZ']) * 1E6 <= 70:
                                    if_choose_s = True
                        if not s.PrecursorPeaks:
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
            if c.id not in cmp:
                cmp.add(c.id)
                cmp_list.append((c, s, cos))

        return cmp_list

    __repr__ = __str__
