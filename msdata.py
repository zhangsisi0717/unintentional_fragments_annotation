import pandas as pd
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import matplotlib
import BQP
from scipy import signal
from tqdm.auto import tqdm
from dataclasses import dataclass, field, asdict
from type_checking import *
from binned_vec import *
import copy



class DimensionMismatchError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)


@dataclass()
class MSDataOptions:
    """
    Options for MSData
    """
    tqdm: bool = True
    dpi: int = 200
    figsize: Tuple[Numeric, Numeric] = (8, 6)
    bqp_solver: str = 'proj'
    slope_err_tol: float = 1E-2
    height: float = 1E-2
    height_type: str = 'rel'
    range_threshold: float = 1E-4
    range_threshold_type: str = 'rel'
    prominence: float = 1E-2
    prominence_type: str = 'rel'
    width: float = 3.
    width_type: str = 'rel'


@dataclass()
class PeakDetectionResults:
    """
    Container class for peak detection results
    """

    # metadata
    absolute_index: [int] = None
    n_peaks: Optional[int] = None  # number of peaks
    mz_range: Tuple[float, float] = None
    rt_range: Optional[Tuple[Numeric, Numeric]] = field(default=None,repr=False)

    # peak position
    peaks: Optional[np.ndarray] = field(default=None, repr=False)
    abs_peaks: Optional[np.ndarray] = field(default=None, repr=False)

    # peak information
    peak_heights: Optional[np.ndarray] = field(default=None, repr=False)
    rel_peak_heights: Optional[np.ndarray] = field(default=None, repr=False)

    prominences: Optional[np.ndarray] = field(default=None, repr=False)
    rel_prominences: Optional[np.ndarray] = field(default=None, repr=False)

    base_heights: Optional[np.ndarray] = field(default=None, repr=False)
    rel_base_heights: Optional[np.ndarray] = field(default=None, repr=False)

    left_bases: Optional[np.ndarray] = field(default=None, repr=False)
    abs_left_bases: Optional[np.ndarray] = field(default=None, repr=False)

    right_bases: Optional[np.ndarray] = field(default=None, repr=False)
    abs_right_bases: Optional[np.ndarray] = field(default=None, repr=False)

    widths: Optional[np.ndarray] = field(default=None, repr=False)
    abs_widths: Optional[np.ndarray] = field(default=None, repr=False)

    width_heights: Optional[np.ndarray] = field(default=None, repr=False)
    rel_width_heights: Optional[np.ndarray] = field(default=None, repr=False)

    left_ips: Optional[np.ndarray] = field(default=None, repr=False)
    abs_left_ips: Optional[np.ndarray] = field(default=None, repr=False)

    right_ips: Optional[np.ndarray] = field(default=None, repr=False)
    abs_right_ips: Optional[np.ndarray] = field(default=None, repr=False)

    left_ranges: Optional[np.ndarray] = field(default=None, repr=False)
    abs_left_ranges: Optional[np.ndarray] = field(default=None, repr=False)

    right_ranges: Optional[np.ndarray] = field(default=None, repr=False)
    abs_right_ranges: Optional[np.ndarray] = field(default=None, repr=False)

    # matched peak information
    matching_idx: Optional[int] = field(default=None, repr=False)
    matching_peak: Optional[int] = field(default=None, repr=False)
    matching_abs_peak: Optional[float] = field(default=None, repr=True)
    matching_range: Optional[Tuple[int, int]] = field(default=None, repr=False)
    matching_abs_range: Optional[Tuple[float, float]] = field(default=None, repr=False)

    overlaps: Optional[np.ndarray] = field(default=None, repr=False)
    n_overlaps: Optional[int] = field(default=None, repr=False)


@dataclass()
class PeakDecompositionResults:
    """
    Container class for peak decomposition results
    """
    B: Optional[np.ndarray] = field(default=None, repr=False)
    c: Optional[np.ndarray] = field(default=None, repr=False)
    const_term: Optional[bool] = field(default=None, repr=False)
    l1: Optional[Numeric] = field(default=None, repr=False)
    l2: Optional[Numeric] = field(default=None, repr=False)
    mse: Optional[Numeric] = field(default=None, repr=False)
    rt_range: Optional[Tuple[Numeric, Numeric]] = field(default=None, repr=False)
    status: Optional[int] = None
    v: Optional[np.ndarray] = field(default=None, repr=False)  # normalized_vector
    v_max: Optional[Numeric] = field(default=None, repr=False)  # max of absolute intensity of the vector
    v_recons: Optional[np.ndarray] = field(default=None, repr=False)  # reconstructed vector by using normalized basis


@dataclass
class Spectrum:
    Polarity: Optional[str] = field(default=None, repr=True)
    mode: Optional[str] = field(default=None, repr=True)
    ms_level: Optional[str] = field(default=None, repr=False)

    spectrum_list: Optional[List[Tuple[float, float]]] = field(default=None, repr=False)
    spectrum_list_abs: Optional[List[Tuple[float, float]]] = field(default=None, repr=False)
    mz: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    intensity: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    relative_intensity: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    n_peaks: int = field(default=0, repr=True, init=False)
    max_intensity: Optional[Numeric] = field(default=None, repr=False, init=False)
    arg_max_mz: Optional[Float] = field(default=None, repr=False, init=False)
    bin_ppm: float = field(default=60., repr=False, init=True)
    bin_vec: Optional[BinnedSparseVector] = field(default=None, repr=False, init=False)

    spectrum_list_reduced: Optional[List[Tuple[float, float]]] = field(default=None, repr=False)
    reduced_mz: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    reduced_intensity: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    reduced_rela_intensity: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    rela_threshold_reduce: Optional[Numeric] = field(default=0.01, repr=False, init=False)
    reduced_n_peaks:int = field(default=0, repr=True, init=False)

    def __post_init__(self):
        if self.spectrum_list is not None:
            self.n_peaks = len(self.spectrum_list)
            self.mz = np.array([mz for mz, intensity in self.spectrum_list])
            self.intensity = np.array([intensity for mz, intensity in self.spectrum_list])
            self.max_intensity = 0. if self.n_peaks == 0 else np.max(self.intensity)

            self.spectrum_list_reduced = [(mz, ints) for mz, ints in self.spectrum_list if
                                          ints > self.max_intensity * self.rela_threshold_reduce]
            self.reduced_mz = np.array([mz for mz, intensity in self.spectrum_list_reduced])
            self.reduced_intensity = np.array([intensity for mz, intensity in self.spectrum_list_reduced])
            self.reduced_n_peaks = len(self.reduced_intensity)

        elif self.mz and self.intensity:
            if len(self.mz) == len(self.intensity):
                self.n_peaks = len(self.mz)
                self.spectrum_list = list(zip(self.mz, self.intensity))
                self.max_intensity = 0. if self.n_peaks == 0 else np.max(self.intensity)
                self.spectrum_list_reduced = [(mz, ints) for mz, ints in self.spectrum_list if ints > self.max_intensity * self.rela_threshold_reduce]
                self.reduced_mz = np.array([mz for mz, intensity in self.spectrum_list_reduced])
                self.reduced_intensity = np.array([intensity for mz, intensity in self.spectrum_list_reduced])
                self.reduced_n_peaks = len(self.reduced_intensity)

        else:
                raise DimensionMismatchError('length of mz must be equal to length of intensity!')

        if self.max_intensity > 0.:
            self.relative_intensity = self.intensity / self.max_intensity
            self.reduced_rela_intensity = self.reduced_intensity/self.max_intensity
        else:
            self.relative_intensity = self.intensity
        if self.n_peaks > 0:
            arg_max = np.argmax(self.relative_intensity)
            self.arg_max_mz = self.mz[arg_max]

        self.bin_vec = BinnedSparseVector(ppm=self.bin_ppm)
        self.bin_vec.add(x=self.mz, y=self.relative_intensity, y_abs=self.intensity)
        if not self.spectrum_list_abs:
            self.spectrum_list_abs = self.spectrum_list

    def compare_spectrum_plot(self, other,
                              dpi: Numeric = 200, figsize: Tuple[Numeric, Numeric] = (8, 6),
                              threshold: float = 1E-2,
                              mz_delta=20.,
                              label=tuple(['self.spectrum', 'other.spectrum']),
                              **kwargs):

        if self.n_peaks is None or self.n_peaks == 0:
            raise ValueError("self is not a valid spectrum.")

        if other.n_peaks is None or other.n_peaks == 0:
            raise ValueError("other is not a valid spectrum.")

        self_valid = self.relative_intensity >= threshold
        other_valid = other.relative_intensity >= threshold

        mz_min = max(0., min(np.min(self.mz[self_valid]), np.min(other.mz[other_valid])) - mz_delta)
        mz_max = max(np.max(self.mz[self_valid]), np.max(other.mz[other_valid])) + mz_delta

        with matplotlib.rc_context(rc={'figure.dpi': dpi, 'figure.figsize': figsize}):
            markerline, stemline, baseline = \
                plt.stem(self.mz[self_valid], self.relative_intensity[self_valid], use_line_collection=True,
                         markerfmt=' ',
                         label=label[0],
                         **kwargs)
            plt.setp(baseline, 'color', 'grey')
            plt.setp(stemline, 'color', 'dodgerblue')

            markerline, stemline, baseline = \
                plt.stem(other.mz[other_valid], -other.relative_intensity[other_valid], use_line_collection=True,
                         markerfmt=' ',
                         label=label[1],
                         **kwargs)
            plt.setp(baseline, 'color', 'grey')
            plt.setp(stemline, 'color', 'salmon')

            baseline.set_xdata([0.01, .99])
            baseline.set_transform(plt.gca().get_yaxis_transform())

            plt.legend()

            plt.ylabel("relative intensity")

            plt.xlabel("mz")

            plt.xlim((mz_min, mz_max))

        return markerline, stemline, baseline

    def plot(self, func: Optional[Callable[[Numeric], Numeric]] = None, relative: bool = True, dpi: Numeric = 200,
             figsize: Tuple[Numeric, Numeric] = (8, 6),
             threshold: float = 1E-2, mz_delta=20., **kwargs) -> Tuple[Any, Any, Any]:

        """
        Use matplotlib.pyplot.stem() to plot the spectrum.


        :param func: optional function to apply to intensity (element-wise) before plotting
        :param relative: if set to True (default), will use relative intensity
        :param dpi: dpi of the plot; default is 200
        :param figsize: figure size, default is (8, 6)
        :param kwargs: additional settings will be passed to matplotlib.pyplot.stem()
        :return: if succeed, will return markerline, stemline, baseline just like matplotlib.pyplot.stem()
        """

        if self.n_peaks != 0 and self.n_peaks is not None:
            self_valid = self.relative_intensity >= threshold
            mz_min = max(0., np.min(self.mz[self_valid]) - mz_delta)
            mz_max = np.max(self.mz[self_valid]) + mz_delta

            if relative:  # relative intensity
                if func is None:
                    intensity = self.relative_intensity
                else:
                    intensity = np.array([func(i) for i in self.relative_intensity])
            else:  # absolute intensity
                if func is None:
                    intensity = self.intensity
                else:
                    intensity = np.array([func(i) for i in self.intensity])

            with matplotlib.rc_context(rc={'figure.dpi': dpi, 'figure.figsize': figsize}):

                markerline, stemline, baseline = \
                    plt.stem(self.mz, intensity, use_line_collection=True,
                             markerfmt=' ', **kwargs)
                plt.ylabel("relative intensity")

                baseline.set_xdata([0.01, .99])
                baseline.set_transform(plt.gca().get_yaxis_transform())
                plt.xlabel("mz")
                plt.xlim((mz_min, mz_max))

            return markerline, stemline, baseline

        else:
            msg = "not a valid spectrum"
            warnings.warn(msg)
            return None, None, None

    def match_matrix(self, other, ppm: Optional[float] = 20) -> Optional[np.matrix]:
        if np.all(self.mz) and np.all(other.mz):
            mza_log = np.log(self.mz)
            mzb_log = np.log(other.mz)

            sigma = 2 * np.log(1 + ppm * 1E-6)

            diff = np.subtract.outer(mza_log, mzb_log)

            s = np.exp(- (diff / sigma) ** 2)

            return s
        else:
            raise ValueError('mzs of spectrum is None !')

    def match_list(self, other, ppm=20, threshold=1E-8):

        s = self.match_matrix(other=other, ppm=ppm)

        assert len(s.shape) == 2, "mza, mzb must be 1-d ndarray"

        indices = np.argwhere(s > threshold)

        return [{'ia': i, 'ib': j, 'mza': self.mz[i], 'mzb': other.mz[j],'ppm': 1E6 * np.abs(self.mz[i] - other.mz[j]) / self.mz[i], 'coeff': s[i][j]} for i, j in indices]

    def match_df(self, other, ppm=20, threshold=1E-8):
        from pandas import DataFrame
        return DataFrame(self.match_list(other=other, ppm=ppm, threshold=threshold))

    def true_inner(self,other, ppm: Optional[float] = 30, func=None, vectorize=True):

        s = self.match_matrix(other=other, ppm=ppm)

        if func is not None:
            try:

                if not vectorize:
                    raise TypeError

                inta_t = np.array([func(i) for i in self.intensity])
                intb_t = np.array([func(i) for i in other.intensity])

            except TypeError:

                inta_t = np.array([func(i) for i in self.intensity])

                intb_t = np.array([func(i) for i in other.intensity])

        else:

            inta_t = self.intensity
            intb_t = other.intensity
        return inta_t.dot(s.dot(intb_t))

    def cos(self, other, ppm=40, matched_ppm=60, func=None, vectorize=True):
        df = self.match_df(other=other, ppm=20, threshold=1E-8)
        cosine = self.true_inner(other=other, ppm=ppm, func=func, vectorize=vectorize) / np.sqrt(
            self.true_inner(other=self, ppm=ppm, func=func, vectorize=vectorize)
            * other.true_inner(other=other, ppm=ppm, func=func, vectorize=vectorize))
        if not df.empty:
            matched_mzs = list()
            for i in df[df['ppm'] <= matched_ppm]['ia'].values:
                if self.spectrum_list_abs[i] not in matched_mzs:
                    matched_mzs.append(self.spectrum_list_abs[i])
            # matched_mzs = [self.spectrum_list[i] for i in df[df['ppm'] <= matched_ppm]['ia'].values]
            return cosine, matched_mzs
        else:
            return cosine, None


@dataclass
class ReconstructedSpectrum(Spectrum):
    spectrum_label: str = field(default=tuple(['Reconstructed', 'Reference']), repr=False)
    spectrum_list_abs: Optional[List[tuple]] = field(default_factory=list, repr=False)
    intensity_abs_recon: Optional[list] = field(default_factory=list, repr=False)
    matched_mz: Optional[List[Tuple]] = field(default_factory=list, repr=False)
    mis_matched_mz: Optional[List[Tuple]] = field(default_factory=list, repr=False)
    matched_isotope_mz: Optional[dict] = field(default_factory=dict, repr=False)
    matched_adduction: Optional[dict] = field(default_factory=dict, repr=False)
    matched_multimer: Optional[dict] = field(default_factory=dict, repr=False)
    sub_recon_spec: Optional[Spectrum] = field(default=None, repr=False)
    Polarity: Optional[str] = field(default=None, repr=False)
    mode: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.spectrum_list_abs:
            self.intensity_abs_recon = [j for _, j in self.spectrum_list_abs]
            self.bin_vec = BinnedSparseVector()
            self.bin_vec.add(x=self.mz, y=self.relative_intensity, y_abs=self.intensity_abs_recon)

    # def gen_matched_mz(self, other: [Spectrum], reset_matched_mz=True):
    #     if reset_matched_mz:
    #         self.matched_mz = []
    #         self.mis_matched_mz = []
    #     recon_matched_mz = []
    #     recon_mis_matched_mz = []
    #     if other.bin_vec.matched_idx_mz:
    #         for idx in other.bin_vec.matched_idx_mz['matched_idx']:
    #             for mz, idx_int in self.bin_vec.mz_idx_dic.items():
    #                 if idx_int[0] == idx:
    #                     recon_matched_mz.append((mz, idx_int[1]))
    #
    #         for mz, idx_int in self.bin_vec.mz_idx_dic.items():
    #             if (mz, idx_int[1]) not in recon_matched_mz:
    #                 recon_mis_matched_mz.append((mz, idx_int[1]))
    #
    #         self.matched_mz = recon_matched_mz
    #         self.mis_matched_mz = recon_mis_matched_mz
    #
    #     return recon_matched_mz

    def gen_matched_mz(self, other: [Spectrum], reset_matched_mz=True):
        if reset_matched_mz:
            self.matched_mz = list()
            self.mis_matched_mz = list()
            if other.matched_mzs:
                self.matched_mz = [tuple(i) for i in other.matched_mzs]
                for mz, ints in self.spectrum_list_abs:
                    if (mz, ints) not in self.matched_mz and (mz,ints) not in self.mis_matched_mz:
                        self.mis_matched_mz.append((mz, ints))
            else:
                self.mis_matched_mz = self.spectrum_list_abs

    def check_isotope(self, ppm=30, reset=True):
        if reset:
            self.matched_isotope_mz = dict()
        percen_thre = [0.20, 0.1, 0.05]
        isotope_mz = dict()
        if self.matched_mz:
            for mz, intensity in self.matched_mz:  ##matched_mz
                for mis_mz, mis_intensity in self.mis_matched_mz:  ##mis_matched_mz
                    mz_13C = [mz + 1.003354, mz + 2 * 1.003354, mz + 3 * 1.003354]  ###13C of matched mz
                    for j in range(len(mz_13C)):
                        if abs(mis_mz - mz_13C[j]) / mz_13C[j] * 1E6 <= ppm * 2:
                            if mis_intensity <= intensity * percen_thre[j]:
                                if not isotope_mz.get((mis_mz, mis_intensity)):
                                    isotope_mz[(mis_mz, mis_intensity)] = [(mz, intensity)]
                                    self.mis_matched_mz.remove((mis_mz, mis_intensity))
                                    if (mis_mz,mis_intensity) not in self.matched_mz:
                                        self.matched_mz.append((mis_mz, mis_intensity))

            self.matched_isotope_mz = isotope_mz
        return isotope_mz

    def check_adduction_list(self, molecular_weight: Numeric = None, ppm: Optional[Numeric] = 30,
                             reset: bool = True, mode: str = 'Negative'):
        ##exact_mass-adduction##
        if reset:
            self.matched_adduction = dict()
        if molecular_weight:
            adduction_list = {'Positive': {'M+H-2H2O': (35.012788115999996, 1),
                                           'M+H-H2O': (17.00278811000001, 1),
                                           'M-H2O+NH4': (-0.022711889999982304, 1),
                                           'M+H': (-1.0073118899999827, 1),
                                           'M+Li': (-7.016011889999987, 1),
                                           'M+NH4': (-18.03381188999998, 1),
                                           'M+Na': (-22.989211890000007, 1),
                                           'M+CH3OH+H': (-33.03351189, 1),
                                           'M+K': (-38.96311188999999, 1),
                                           'M+ACN+H': (-42.03381188999998, 1),
                                           'M+2Na-H': (-44.97111189, 1),
                                           'M+ACN+Na': (-64.01581188999998, 1),
                                           '[M+H+Na]2+': (molecular_weight-(molecular_weight+1.0078+22.9897)/2,2),
                                           '[M+2H]2+': (molecular_weight-(molecular_weight+2*1.0078)/2, 2),
                                           '[M+2Na]2+': (molecular_weight-(molecular_weight+2*22.9897)/2, 2),
                                           '[M+3H]3+': (molecular_weight-(molecular_weight+3*1.0078)/3, 3),
                                           '[M+2H+Na]3+': (molecular_weight-(molecular_weight+2*1.0078+22.9897)/3,3),
                                           '[M+H+2Na]3+': (molecular_weight-(molecular_weight+1.0078+2*22.9897)/3,3)},
                              'Negative': {'M-H': (1.0072881100000188, 1),
                                           'M+F': (-18.998411884000006, 1),
                                           'M-H2O-H': (-19.01838811600001, 1),
                                           'M+Na-2H': (-20.974711883999987, 1),
                                           'M+Cl': (-34.96941188400001, 1),
                                           'M+K-2H': (-36.948611884, 1),
                                           'M+FA-H': (-44.998211884, 1),
                                           'M+CH3COO': (-59.013811884000006, 1),
                                           '[2M-2H+Na]-': (molecular_weight-(2*molecular_weight-1.0078*2+22.9897),1),
                                           '[M-2H]2-': (molecular_weight-(molecular_weight-2*1.0078)/2, 2),
                                           '[M-3H]3-': (molecular_weight-(molecular_weight-3*1.0078)/3, 3)}}

            matched_adduction = dict()
            if mode not in ('Negative', 'Positive'):
                raise TypeError('mode must be Negative or Positive')
            else:
                for mis_mz in self.mis_matched_mz:
                    for ad, mz in adduction_list[mode].items():
                        if abs((molecular_weight - mis_mz[0] - mz[0]) / mz[0]) * 1E6 <= ppm * 2:
                            matched_adduction[mis_mz] = (ad, mis_mz[0], mz[1])
                            if mis_mz not in self.matched_mz:
                                self.matched_mz.append(mis_mz)
                            self.mis_matched_mz.remove(mis_mz)
                self.matched_adduction = matched_adduction

        return self.matched_adduction

    def check_multimer(self, molecular_weight: Numeric = None, ppm: Optional[Numeric] = 30,
                       reset: bool = True, mode: str = 'Negative'):
        if reset:
            self.matched_multimer = dict()
        if molecular_weight:
            if mode not in ('Negative', 'Positive'):
                raise TypeError('mode must be Negative or Positive')
            else:
                if mode == 'Negative':
                    for mz, intensity in self.matched_mz:
                        for mis_mz, mis_intensity in self.mis_matched_mz:
                            if (abs(mz * 2 + 1.0073118899999827 - mis_mz) / (
                                    mz * 2 + 1.0073118899999827)) * 1E6 <= ppm * 2:
                                self.matched_multimer[(mis_mz, mis_intensity)] =\
                                    ('dimer of frag/precur', (mz, intensity))
                                self.mis_matched_mz.remove((mis_mz, mis_intensity))
                                if (mis_mz,mis_intensity) not in self.matched_mz:
                                    self.matched_mz.append((mis_mz, mis_intensity))

                            if (abs(molecular_weight * 2 - 1.0073118899999827 - mis_mz) / (
                                    molecular_weight * 2 - 1.0073118899999827)) * 1E6 <= ppm * 2:
                                if (mis_mz, mis_intensity) not in self.matched_multimer.keys():
                                    if (mis_mz, mis_intensity) not in self.matched_mz:
                                        self.matched_multimer[(mis_mz, mis_intensity)] = ('dimer of molecule',
                                                                                          molecular_weight)
                                        self.mis_matched_mz.remove((mis_mz, mis_intensity))
                                        if (mis_mz,mis_intensity) not in self.matched_mz:
                                            self.matched_mz.append((mis_mz, mis_intensity))
                elif mode == 'Positive':
                    for mz, intensity in self.matched_mz:
                        for mis_mz, mis_intensity in self.mis_matched_mz:
                            if (abs(mz * 2 - 1.0073118899999827 - mis_mz) / (
                                    mz * 2 - 1.0073118899999827)) * 1E6 <= ppm * 2:
                                self.matched_multimer[(mis_mz, mis_intensity)] = (
                                'dimer of frag/precur', (mz, intensity))
                                self.mis_matched_mz.remove((mis_mz, mis_intensity))
                                if (mis_mz,mis_intensity) not in self.matched_mz:
                                    self.matched_mz.append((mis_mz, mis_intensity))

                            if (abs(molecular_weight * 2 + 1.0073118899999827 - mis_mz) / (
                                    molecular_weight * 2 + 1.0073118899999827)) * 1E6 <= ppm * 2:
                                if (mis_mz, mis_intensity) not in self.matched_multimer.keys():
                                    if (mis_mz, mis_intensity) not in self.matched_mz:
                                        self.matched_multimer[(mis_mz, mis_intensity)] = ('dimer of molecule',
                                                                                          molecular_weight)
                                        self.mis_matched_mz.remove((mis_mz, mis_intensity))
                                        if (mis_mz,mis_intensity) not in self.matched_mz:
                                            self.matched_mz.append((mis_mz, mis_intensity))

        return self.matched_multimer

    def generate_sub_recon_spec(self, reset: bool = True):
        if reset:
            self.sub_recon_spec = None
            new = self.mis_matched_mz.copy()
            new.sort(reverse=True, key=lambda x: x[1])
            mis_matched_rela = [list(i) for i in self.mis_matched_mz]
            if new:
                max_val = new[0][1]
                for i in range(len(mis_matched_rela)):
                    mis_matched_rela[i][1] = (mis_matched_rela[i][1]) / max_val
            self.sub_recon_spec = ReconstructedSpectrum(spectrum_list=mis_matched_rela,spectrum_list_abs=self.mis_matched_mz)

        return self.sub_recon_spec

    def gen_matched_peaks_compound(self,inchIkey: str, database, mode: Union['Negative','Positive'], reset=True):
        if reset:
            self.matched_mz = None
            self.mis_matched_mz = None
        res=[]
        mw = None
        if database.name == 'iroa':
            cor_candi=[]
            for can in database.compounds_list:
                if can.InChIKey == inchIkey:
                    cor_candi.append(can)
            for candi in cor_candi:
                for s in candi.spectra_1 + candi.spectra_2:
                    if s.mode == mode:
                        if_choose_s = False
                        if s.precursor:
                            for mz in self.mz:
                                if (abs(s.precursor-mz)/s.precursor) * 1E6 <= 70:
                                    if_choose_s = True
                        if not s.precursor:
                            if_choose_s = True

                        if if_choose_s:
                            cos, matched_mzs = self.cos(other=s,func=None)
                            s.matched_mzs = matched_mzs
                            if cos > 1E-4:
                                res.append((cor_candi[0],s, cos))
            res.sort(key=lambda x: x[2], reverse=True)
            if res:
                mw = res[0][0].MolecularWeight
        elif database.name == 'mona':
            cor_candi=[]
            for can in database.compounds_list:
                if can.InChIKey == inchIkey:
                    cor_candi.append(can)
            for candi in cor_candi:
                for s in candi.spectra_1 + candi.spectra_2:
                    if s.mode == mode:
                        if_choose_s = False
                        if s.precursor:
                            for mz in self.mz:
                                if (abs(s.precursor-mz)/s.precursor) * 1E6 <= 70:
                                    if_choose_s = True
                        if not s.precursor:
                            if_choose_s = True

                        if if_choose_s:
                            cos, matched_mzs = self.cos(other=s,func=None)
                            s.matched_mzs = matched_mzs
                            if cos > 1E-4:
                                res.append((cor_candi[0],s, cos))
            res.sort(key=lambda x: x[2], reverse=True)
            if res:
                mw = res[0][0].total_exact_mass

        elif database.name == 'mzc':
            cor_candi=[]
            for can in database.compounds:
                if can.InChIKey == inchIkey:
                    cor_candi.append(can)
            for candi in cor_candi:
                for t in candi.spectra_1 + candi.spectra_2:
                    for s in t:
                        if s.Polarity == mode:
                            if_choose_s = False
                            if s.PrecursorPeaks:
                                for mz in self.mz:
                                    if (abs(s.PrecursorPeaks[0]['MZ']-mz)/s.PrecursorPeaks[0]['MZ']) * 1E6 <= 70:
                                        if_choose_s = True
                            if not s.PrecursorPeaks:
                                if_choose_s = True

                            if if_choose_s:
                                cos, matched_mzs = self.cos(other=s,func=None)
                                s.matched_mzs = matched_mzs
                                if cos > 1E-4:
                                    res.append((cor_candi[0],s, cos))
            res.sort(key=lambda x: x[2], reverse=True)
            if res:
                mw = res[0][0].MolecularWeight

        if res:
            target_spec = res[0][1]
            self.gen_matched_mz(target_spec,reset_matched_mz=True)
            self.check_isotope()
            self.check_adduction_list(mode=mode,molecular_weight=mw)
            self.check_multimer(mode=mode,molecular_weight=mw)
            return self.matched_mz, self.mis_matched_mz,self.matched_isotope_mz,self.matched_adduction,self.matched_multimer
        else:
            return ['res_empty'],['res_empty'],['empty'],['empty'],['empty']


@dataclass
class BaseProperties:
    """
    Container class for base information
    """
    abs_index: Optional[Int] = None
    cond: Optional[Float] = field(default=None, repr=False)
    index: Optional[Int] = None
    mse: Optional[Float] = field(default=None, repr=False)
    n_overlaps: Optional[Int] = field(default=None, repr=False)
    rt: Optional[Float] = field(default=None, repr=False)
    sin: Optional[Float] = field(default=None, repr=False)
    v: Optional[np.ndarray] = field(default=None, repr=False)  # normalized basis
    v_max: Optional[Float] = field(default=None, repr=False)  # max ints of the orginal basis vector

    cos_sim: Optional[Float] = field(default=None, repr=False)

    coefficient: Optional[np.ndarray] = field(default=None, repr=False)
    coefficient_abs: Optional[np.ndarray] = field(default=None, repr=False)

    n_components: Optional[int] = None
    threshold: Optional[float] = field(default=None, repr=False)
    mz_upperbound: Optional[float] = field(default=None, repr=False)
    max_mse: Optional[float] = field(default=None, repr=False)
    max_rt_diff: Optional[float] = field(default=None, repr=False)
    min_cos: Optional[float] = field(default=None, repr=False)
    # spectrum_list: Optional[List[Tuple[float, float]]] = field(default=None, repr=False, init=None)
    spectrum: Optional[ReconstructedSpectrum] = field(default=None, repr=False, init=None)


class MSData:
    """
    Container for MS data
    """

    @staticmethod
    def gaussian_kernel(scale: Numeric, half_len: Numeric) -> np.ndarray:
        """
        Gaussian kernel, normalized
        :param scale: standard deviation
        :param half_len: half of the length of the kernel
        :return:
        """
        x = np.arange(-half_len, half_len + 1, 1.)
        r = np.exp(- (x / scale) ** 2 / 2.)
        r /= r.sum()
        return r

    def plt_context(self, dpi: Optional[Numeric] = None, figsize: Optional[Numeric] = None) -> matplotlib.rc_context:
        if dpi is None:
            dpi = self.options.dpi
        if figsize is None:
            figsize = self.options.figsize
        return matplotlib.rc_context(rc={'figure.dpi': dpi, 'figure.figsize': figsize})

    @staticmethod
    def from_files(prefix: str, directory: Optional[str] = None, sep: str = '_', **options) -> "MSData":
        """
        Create a MSData instance from files
        :param prefix: a string which specifies the name of the data set;
                       note that if prefix is None, then sep will NOT appear in the file name
        :param directory: directory of the files
        :param sep: separator between file name and prefix
        :return: an MSData instance
        """
        if directory is None:
            directory = '.'

        root = Path(directory)

        # required files: rt, mz, int, peak_info
        files = dict(rt_file=root / sep.join(i for i in (prefix, 'rts.txt') if i is not None),
                     # file for retention time, txt file

                     mz_file=root / sep.join(i for i in (prefix, 'mzs.txt') if i is not None),
                     # file for mz values, txt file

                     int_file=root / sep.join(i for i in (prefix, 'ints.txt') if i is not None),
                     # file for intensities, txt file

                     peak_info_file=root / sep.join(i for i in (prefix, 'peak_info.csv') if i is not None)
                     # file for peak information, csv file
                     )

        # check files' existence
        for file, path in files.items():
            if not path.exists():
                raise FileNotFoundError('{} \'{}\' not found. '.format(file, path.as_posix()))

        rts = pd.read_table(files['rt_file'].as_posix(), header=None, delimiter=",").values.flatten()

        mzs = pd.read_table(files['mz_file'].as_posix(), header=None, delimiter=",").values
        ints = pd.read_table(files['int_file'].as_posix(), header=None, delimiter=",").values

        df = pd.read_csv(files['peak_info_file'].as_posix())
        df.rename(columns={'Unnamed: 0': 'peak_id'}, inplace=True)

        extra_info = {}
        extra_info.update(dict((k, v.as_posix()) for k, v in files.items()))
        extra_info['prefix'] = prefix
        extra_info['dir'] = root.as_posix()

        ms_data = MSData(rts=rts, mzs=mzs, ints=ints, df=df, extra_info=extra_info, **options)

        return ms_data

    def __init__(self, rts: np.ndarray, mzs: np.ndarray, ints: np.ndarray,
                 df: pd.DataFrame, extra_info: Optional[dict] = None, **options) -> None:

        # handle options
        self.options: MSDataOptions = MSDataOptions(**options)
        self._ints_duplicates_raw = None
        self._ints_duplicates_sm_raw = None
        self._mzs_duplicates_raw = None
        self._mzs_duplicates = None
        self._df_duplicates = None
        self.extra_info: Dict[Any, Any] = {}
        if extra_info is not None:
            self.extra_info.update(extra_info)

        # check dimensions
        n_rt_raw = rts.shape[0]
        if rts.shape != (n_rt_raw,):
            raise DimensionMismatchError('incompatible dimension for rt_file')

        n_feature = mzs.shape[0]
        if mzs.shape != (n_feature, 2):
            raise DimensionMismatchError('incompatible dimension for mz_file')

        if ints.shape != (n_feature, n_rt_raw):
            raise DimensionMismatchError('incompatible dimension for int_file')

        if df.shape[0] != n_feature:
            raise DimensionMismatchError('incompatible dimension for peak_info_file')

        self.n_feature: int = n_feature
        self.n_rt_raw: int = n_rt_raw
        self.n_rt: int = n_rt_raw

        self._rts_raw: np.ndarray = np.copy(rts)  # raw retention time
        self._rts: np.ndarray = np.copy(rts)  # smoothed retention time

        self._rts_raw_intercept, self._rts_raw_slope, rel = self.line_model(self._rts_raw)
        if rel > self.options.slope_err_tol:
            msg = "Warning: non-uniform scanning interval: difference has a relative error of {:.2E}".format(rel)
            print(msg)
            uniform_rts, interp_ints = self.rts_interp_model(self._rts_raw, ints)
            self._non_uni_rts_raw = np.copy(rts)
            self._rts_raw = uniform_rts
            self._rts_raw_intercept, self._rts_raw_slope, rel = self.line_model(self._rts_raw)
            self._ints_raw = interp_ints
            self._non_uni_ints_raw = np.copy(ints)
        else:
            self._ints_raw: np.ndarray = np.copy(ints)  # raw intensity, not smoothed, uses abs index
            self._ints_sm_raw: np.ndarray = np.copy(
                ints)  # smoothed but not re-ordered, i.e., uses abs index for feature
            self._ints: np.ndarray = np.copy(ints)
        # re-ordered and smoothed intensity, i.e., use relative index for feature
        self._mzs_raw: np.ndarray = np.copy(mzs)  # raw mz
        self._mzs: np.ndarray = np.copy(mzs)  # re-ordered mz
        self._rts_intercept = self._rts_raw_intercept
        self._rts_slope = self._rts_raw_slope
        self._df: pd.DataFrame = df  # metadata data frame
        self._df['id'] = self._df.index.copy()

        self._view_order: np.ndarray = np.arange(self.n_feature)  # mapping from relative index to absolute index

        self._kernel: Optional[np.ndarray] = None  # kernel used to smooth the intensity

        # container for peak detection results, uses relative index
        self._detection_results: Dict[Int, PeakDetectionResults] = dict()
        self._detection_results_raw: Dict[Int, PeakDetectionResults] = dict()  # same as above, uses abs index
        self._detection_results_duplicates: Dict[Int, PeakDetectionResults] = dict()

        # container for peak decomposition results, uses relative index
        self._decomposition_results: Dict[Int, PeakDecompositionResults] = dict()

        self.base_info: Dict[Int, BaseProperties] = dict()
        self.base_index: List[Int] = []
        self.B: Optional[np.ndarray] = None
        self.M: Optional[np.ndarray] = None  # self.M = self.B.dot(self.B.T)
        self.cos_sim: Optional[np.ndarray] = None

    @staticmethod
    def line_model(x: np.ndarray) -> Tuple[Float, Float, Float]:
        x = np.asarray(x)
        delta = x[1:] - x[:-1]

        delta_mean = delta.mean()
        delta_range = delta.max() - delta.min()
        rel_err = delta_range / delta_mean

        return x[0], delta.mean(), rel_err

    @staticmethod
    def rts_interp_model(rts_raw: np.ndarray, ints_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        delta = rts_raw[1:] - rts_raw[:-1]
        delta_mean = delta.mean()
        uniform_rt = np.arange(start=rts_raw[0], stop=rts_raw[-1] + delta_mean, step=delta_mean)
        interp_ints = np.interp(uniform_rt, rts_raw, ints_raw[0])
        for i in tqdm(range(1, ints_raw.shape[0]), desc='Starting interpolating:'):
            temp_ints = np.interp(uniform_rt, rts_raw, ints_raw[i])
            interp_ints = np.vstack((interp_ints, temp_ints))
        return uniform_rt, interp_ints

    @property
    def order(self) -> np.ndarray:
        return self._view_order

    @property
    def rts_raw(self) -> np.ndarray:
        return self._rts_raw

    @property
    def rts(self) -> np.ndarray:
        return self._rts

    @property
    def mzs_raw(self) -> np.ndarray:
        return self._mzs_raw

    @property
    def mzs(self) -> np.ndarray:
        return self._mzs

    @property
    def ints_raw(self) -> np.ndarray:
        return self._ints_raw

    @property
    def ints_unordered(self) -> np.ndarray:
        return self._ints_sm_raw

    @property
    def ints(self) -> np.ndarray:
        return self._ints

    @property
    def df(self) -> pd.DataFrame:
        return self._df.iloc[self.order]

    @property
    def df_raw(self) -> pd.DataFrame:
        return self._df

    @property
    def non_uni_rts_raw(self) -> np.ndarray:
        return self._non_uni_rts_raw

    @property
    def non_uni_ints_raw(self) -> np.ndarray:
        return self._non_uni_ints_raw

    def absolute_index(self, index: Union[int, Iterable[int]], ordered: bool = True) -> np.ndarray:

        try:
            iter(index)
        except TypeError:  # not iterable
            index = np.array([int(index)])
        else:  # iterable
            index = np.array([int(i) for i in index])

        if ordered:
            index = self.order[index]

        return index

    def __str__(self) -> str:
        prefix = self.extra_info.get('prefix')
        root = self.extra_info.get('dir')
        if prefix is not None:
            prefix_str = "prefix='{}'".format(prefix)
        else:
            prefix_str = ""
        if root is not None:
            root_str = "dir='{}'".format(root)
        else:
            root_str = ""

        n_ft_str = "n_feature={}".format(self.n_feature)
        n_rt_str = "n_rt={}".format(self.n_rt)

        return "MSData({})".format(', '.join(i for i in [n_ft_str, n_rt_str, prefix_str, root_str] if i != ""))

    def rts_interp(self, i):
        return np.interp(i, np.arange(self.n_rt), self.rts)

    def sort_feature(self, by: Optional[np.ndarray] = None, descending: bool = True) -> None:
        """
        sort the features (i.e. chromatograms)
        only sort the order of self.mzs, self.ints, self._detection_results


        :param by: the value used to perform the sorting; array-like with shape (self.n_feature,); if set to None (
        default), will reset the order
        :param descending: if set to True (default), will sort by descending order
        :type descending: bool
        :return: None
        """
        if by is None:  # reset
            view_order = np.arange(self.n_feature)
        else:
            if not isinstance(by, np.ndarray):
                raise TypeError('by must be a numpy array.')
            if not by.shape == (self.n_feature,):
                raise DimensionMismatchError('by should have shape ({},) '
                                             '(got {} instead)'.format(self.n_feature, by.shape))

            if descending:
                view_order = np.argsort(-by)
            else:
                view_order = np.argsort(by)

        if not np.all(view_order == self._view_order):  # if the order has been changed
            self._view_order = view_order
            self._mzs = np.copy(self._mzs_raw[self._view_order])
            self._ints = np.copy(self._ints_sm_raw[self._view_order])

            # reset the base if the order has been changed
            self.reset_base()

            self._detection_results = dict()
            for i, j in enumerate(self._view_order):
                res = self._detection_results_raw.get(j)
                if res is not None:
                    self._detection_results[i] = res

            # self._detection_results = \
            #     dict((i, self._detection_results_raw.get(j)) for i, j in enumerate(self._view_order))

    def apply_smooth_conv(self, kernel: np.ndarray,
                          min_zero: bool = True, rt_range: Optional[Tuple[Numeric, Numeric]] = None) -> None:
        """
        Smoothing the chromatography by convolution with a kernel

        :param kernel: kernel to use; array-like

        :param min_zero: bool, if set to True, will subtract the minimal value from each chromatography
        :type min_zero: bool

        :param rt_range: 2-tuple, the first one is minimal rt (if no minimal is needed, set to 0. or -np.inf),
        the second one is the maximal rt (if no maximal is needed, set to np.inf)

        :return: None
        """

        self._kernel = kernel

        if np.abs(np.sum(kernel) - 1.) > 1E-4:
            warnings.warn("kernel not normalized!")
        if np.any(kernel < 0.):
            warnings.warn("kernel contains negative value")

        # apply smoothing to rts
        self._rts = np.convolve(self._rts_raw, kernel, 'valid')  ##raw_length - kernel_len + 1 #
        self.n_rt = self._rts.shape[0]

        self._rts_intercept, self._rts_slope, rel = self.line_model(self._rts)
        if rel > self.options.slope_err_tol:
            msg = "non-uniform retention time: difference has a relative error of {:.2E}".format(rel)
            warnings.warn(msg)

        # apply smoothing to ints
        _ints = np.copy(self._ints_raw)
        # if min_zero:
        #     _ints -= np.min(_ints, axis=1).reshape((-1, 1))
        self._ints_sm_raw = -np.ones((self.n_feature, self.n_rt))
        for i in range(self.n_feature):
            self._ints_sm_raw[i, :] = np.convolve(_ints[i, :], kernel, 'valid')

        self._ints = np.copy(self._ints_sm_raw[self._view_order])

        if rt_range is not None:  # need to filter rt
            rt_idx = np.array([i for (i, t) in enumerate(self._rts) if rt_range[0] <= t <= rt_range[1]])
            self._rts = self._rts[rt_idx]
            self.n_rt = self._rts.shape[0]
            self._ints_sm_raw = self._ints_sm_raw[:, rt_idx]

        if min_zero:
            self._ints_sm_raw -= np.min(self._ints_sm_raw, axis=1).reshape((-1, 1))

    def plot_peak(self, index, rt_range=None, smooth=True, ordered=True,
                  dpi=None, figsize=None):
        """
        plot a peak
        :param index: a single index indicating the feature, index = self.order[index]
        :type index: int
        :param rt_range:
        :param smooth:
        :param ordered:
        :param dpi:
        :param figsize:
        :return: None
        """

        if ordered:
            index = self.order[index]

        if smooth:  # plot smoothed
            rts = self.rts
            ints = self._ints_sm_raw
        else:
            rts = self.rts_raw
            ints = self._ints_raw

        if rt_range is None:  # all range
            rt_idx = np.arange(rts.shape[0])
        else:
            rt_idx = np.array([i for (i, t) in enumerate(rts) if rt_range[0] <= t <= rt_range[1]])
        rts = rts[rt_idx]

        with self.plt_context(dpi=dpi, figsize=figsize):
            plt.plot(rts, ints[index, rt_idx], linewidth=0.8)
            if smooth:
                detected = self._detection_results_raw.get(index)  # uses absolute index
                if detected is not None:
                    peaks = detected.abs_peaks
                    heights = detected.peak_heights
                    base_heights = detected.base_heights
                    plt.plot(peaks, heights, 'ro', markersize=2)
                    plt.vlines(x=peaks, ymin=base_heights, ymax=heights, linewidth=.5, color="C1")

                    plt.hlines(y=detected.width_heights, xmin=detected.abs_left_ips,
                               xmax=detected.abs_right_ips, linewidth=.5, color='C1')

                    plt.hlines(y=base_heights, xmin=detected.abs_left_bases,
                               xmax=detected.abs_right_bases, linewidth=.5,
                               color='C1')

                    matching_range = detected.matching_abs_range
                    if matching_range is not None:
                        plt.axvspan(matching_range[0], matching_range[1], alpha=0.2)
                        idx = detected.matching_idx
                        plt.vlines(x=peaks[idx], ymin=base_heights[idx], ymax=heights[idx], linewidth=.5)
                        plt.hlines(y=detected.width_heights[idx], xmin=detected.abs_left_ips[idx],
                                   xmax=detected.abs_right_ips[idx], linewidth=.5)

                        plt.hlines(y=base_heights[idx], xmin=detected.abs_left_bases[idx],
                                   xmax=detected.abs_right_bases[idx], linewidth=.5)

    def feature(self, index: Union[int, Iterable[int]],
                rt_range: Optional[Tuple[Numeric, Numeric]] = None,
                smooth: bool = True, ordered: bool = True, plot: bool = False,
                dpi: Optional[Numeric] = None,
                figsize: Optional[Tuple[Numeric, Numeric]] = None) -> Tuple[np.ndarray, np.ndarray]:

        abs_index = self.absolute_index(index, ordered=ordered)

        if smooth:  # plot smoothed
            rts = self.rts
            ints = self._ints_sm_raw

        else:
            rts = self.rts_raw
            ints = self._ints_raw


        if rt_range is None:  # all range
            rt_idx = np.arange(rts.shape[0])
        else:
            rt_idx = np.array([i for (i, t) in enumerate(rts) if rt_range[0] <= t <= rt_range[1]])

        rts = np.copy(rts[rt_idx])
        ints = np.copy(ints[abs_index, :][:, rt_idx])


        if plot:
            with self.plt_context(dpi=dpi, figsize=figsize):
                for i in range(len(abs_index)):
                    plt.plot(rts, ints[i, :])

        return rts, ints

    # def peak_detection(self, rts: np.ndarray, ints: np.ndarray,
    #                    range_threshold: float = 0.001, range_threshold_type: str = 'rel',
    #                    height: float = 0.01, height_type: str = 'rel',
    #                    prominence: float = 0.01, prominence_type: str = 'rel',
    #                    width: float = 3., width_type: str = 'rel',
    #                    plot: bool = False, dpi: Optional[Numeric] = None,
    #                    figsize: Optional[Tuple[Numeric, Numeric]] = None, **kwargs) -> PeakDetectionResults:
    #
    #     rts = np.asarray(rts)
    #     ints = np.asarray(ints)
    #
    #     n_rt = rts.shape[0]
    #     assert rts.shape == (n_rt,)
    #     assert ints.shape == (n_rt,)
    #
    #     h_max = np.max(ints) - np.min(ints)  # max diff
    #
    #     if range_threshold_type[0] in 'rR':  # relative
    #         range_threshold = h_max * range_threshold
    #
    #     if height_type[0] in 'rR':  # relative
    #         height = h_max * height
    #
    #     if prominence_type[0] in 'rR':  # relative
    #         prominence = h_max * prominence
    #
    #     if width_type[0] in 'rR':  # relative
    #         delta = (rts[1:] - rts[:-1]).mean()
    #         width = width / delta
    #
    #     peaks, properties = signal.find_peaks(ints, height=height, prominence=prominence, width=width, **kwargs)
    #
    #     properties['peaks'] = peaks
    #
    #     n_peaks = peaks.shape[0]
    #
    #     properties['n_peaks'] = n_peaks
    #
    #     rt_idx = np.arange(n_rt)
    #
    #     above_threshold = ints >= range_threshold
    #     left_range = np.copy(peaks)
    #     right_range = np.copy(peaks)
    #     for i in range(n_peaks):
    #         while above_threshold[left_range[i]] and left_range[i] > 0:
    #             left_range[i] -= 1
    #         while above_threshold[right_range[i]] and right_range[i] < n_rt - 1:
    #             right_range[i] += 1
    #
    #     properties['left_ranges'] = left_range
    #     properties['right_ranges'] = right_range
    #
    #     properties['base_heights'] = ints[peaks] - properties['prominences']
    #
    #     abs_properties = dict()
    #     for k, v in properties.items():
    #         if k in ('left_ranges', 'right_ranges', 'left_bases', 'right_bases', 'left_ips', 'right_ips', 'peaks'):
    #             abs_properties['abs_' + k] = np.interp(v, rt_idx, rts)
    #     properties.update(abs_properties)
    #
    #     properties['abs_widths'] = properties['abs_right_ips'] - properties['abs_left_ips']
    #
    #     rel_properties = dict()
    #     for k, v in properties.items():
    #         if k in ('peak_heights', 'prominences', 'width_heights', 'base_heights'):
    #             rel_properties['rel_' + k] = v / h_max
    #     properties.update(rel_properties)
    #
    #     results = PeakDetectionResults(**properties)
    #
    #     if plot:
    #         with self.plt_context(dpi=dpi, figsize=figsize):
    #             plt.plot(rts, ints, linewidth=.8)
    #             for i in peaks:
    #                 plt.plot(rts[i], ints[i], 'ro', markersize=2)
    #             plt.vlines(x=rts[peaks], ymin=ints[peaks] - properties["prominences"],
    #                        ymax=ints[peaks], linewidth=.8, color="C1")
    #
    #             plt.hlines(y=properties['width_heights'], xmin=properties['abs_left_ips'],
    #                        xmax=properties['abs_right_ips'], linewidth=.8, color='C1')
    #
    #             plt.hlines(y=ints[peaks] - properties["prominences"], xmin=properties['abs_left_bases'],
    #                        xmax=properties['abs_right_bases'], linewidth=.8,
    #                        color='C1')
    #
    #     return results

    def peak_detection(self, rts: np.ndarray, ints: np.ndarray,
                       range_threshold: float = 0.001, range_threshold_type: str = 'rel',
                       height: float = 0.01, height_type: str = 'rel',
                       prominence: float = 0.01, prominence_type: str = 'rel',
                       width: float = 3., width_type: str = 'rel',
                       plot: bool = False, dpi: Optional[Numeric] = None,
                       figsize: Optional[Tuple[Numeric, Numeric]] = None, **kwargs) -> PeakDetectionResults:

        rts = np.asarray(rts)
        ints = np.asarray(ints)

        n_rt = rts.shape[0]
        assert rts.shape == (n_rt,)
        assert ints.shape == (n_rt,)

        h_max = np.max(ints) - np.min(ints)  # max diff

        if height_type[0] in 'rR':  # relative
            height = h_max * height

        if prominence_type[0] in 'rR':  # relative
            prominence = h_max * prominence

        if width_type[0] in 'rR':  # relative
            delta = (rts[1:] - rts[:-1]).mean()
            width = width / delta

        peaks, properties = signal.find_peaks(ints, height=height, prominence=prominence, width=width, **kwargs)

        properties['peaks'] = peaks

        n_peaks = peaks.shape[0]

        properties['n_peaks'] = n_peaks

        rt_idx = np.arange(n_rt)

        left_range = np.copy(peaks)
        right_range = np.copy(peaks)

        peak_heights = ints[peaks]

        if range_threshold_type[0] in 'rR':  # relative
            range_threshold = peak_heights * range_threshold
        else:
            range_threshold = np.ones(shape=peak_heights.shape) * range_threshold

        for i in range(n_peaks):
            while ints[left_range[i]] > range_threshold[i] and left_range[i] > 0:
                left_range[i] -= 1
            while ints[right_range[i]] > range_threshold[i] and right_range[i] < n_rt - 1:
                right_range[i] += 1

        properties['left_ranges'] = left_range
        properties['right_ranges'] = right_range

        properties['base_heights'] = ints[peaks] - properties['prominences']
        # prominence:vertical distance between the peak and its lowest contour line#

        abs_properties = dict()
        for k, v in properties.items():
            if k in ('left_ranges', 'right_ranges', 'left_bases', 'right_bases', 'left_ips', 'right_ips', 'peaks'):
                abs_properties['abs_' + k] = np.interp(v, rt_idx, rts)
        properties.update(abs_properties)

        properties['abs_widths'] = properties['abs_right_ips'] - properties['abs_left_ips']

        rel_properties = dict()
        for k, v in properties.items():
            if k in ('peak_heights', 'prominences', 'width_heights', 'base_heights'):
                rel_properties['rel_' + k] = v / h_max
        properties.update(rel_properties)

        results = PeakDetectionResults(**properties)

        if plot:
            with self.plt_context(dpi=dpi, figsize=figsize):
                plt.plot(rts, ints, linewidth=.8)
                for i in peaks:
                    plt.plot(rts[i], ints[i], 'ro', markersize=2)
                plt.vlines(x=rts[peaks], ymin=ints[peaks] - properties["prominences"],
                           ymax=ints[peaks], linewidth=.8, color="C1")

                plt.hlines(y=properties['width_heights'], xmin=properties['abs_left_ips'],
                           xmax=properties['abs_right_ips'], linewidth=.8, color='C1')

                plt.hlines(y=ints[peaks] - properties["prominences"], xmin=properties['abs_left_bases'],
                           xmax=properties['abs_right_bases'], linewidth=.8,
                           color='C1')

        return results

    def perform_feature_peak_detection(self, index: Union[Iterable[int], None] = None,
                                       rt_range: Optional[Tuple[Numeric, Numeric]] = None,
                                       smooth: bool = True, ordered: bool = True,
                                       enable_tqdm: Optional[bool] = None,
                                       **peak_detection_options) -> None:

        if enable_tqdm is None:
            enable_tqdm = self.options.tqdm

        peak_detection_option_keys = ('height', 'height_type', 'prominence', 'prominence_type', 'width', 'width_type')

        options = {k: v for k, v in asdict(self.options).items() if k in peak_detection_option_keys}

        options.update(peak_detection_options)

        if index is None:
            index = np.arange(self.n_feature)  # relative index
            index_raw = np.copy(self._view_order)  # abs index
        else:
            index = np.array([int(i) for i in index])  # relative index
            index_raw = self.absolute_index(index, ordered=ordered)  # abs index

        rts, ints = self.feature(index=index, rt_range=rt_range, smooth=smooth,
                                 ordered=False)  # use absolute index

        for i, idx_raw in tqdm(enumerate(index_raw), total=len(index_raw),
                               disable=(not enable_tqdm), desc="peak detection"):

            idx = index[i]  # relative index
            properties = self.peak_detection(rts=rts, ints=ints[i, :], plot=False, **options)
            properties.rt_range = rt_range
            properties.absolute_index = idx_raw
            properties.mz_range = self.mzs_raw[idx_raw]

            if properties.n_peaks > 0:  # at least 1 peak detected
                matching_rt = self.df_raw.iloc[idx]['rt']
                matching_idx = np.argmin(np.abs(properties.abs_peaks - matching_rt))  # find the closest one
                abs_left_range = properties.abs_left_ranges[matching_idx]
                abs_right_range = properties.abs_right_ranges[matching_idx]

                if abs_left_range <= matching_rt <= abs_right_range:  # OK
                    properties.matching_idx = matching_idx
                    properties.matching_abs_range = (abs_left_range, abs_right_range)
                    left_range = properties.left_ranges[matching_idx]
                    right_range = properties.right_ranges[matching_idx]
                    properties.matching_range = (left_range, right_range)
                    properties.matching_peak = properties.peaks[matching_idx]
                    properties.matching_abs_peak = properties.abs_peaks[matching_idx]

                    overlaps = ((properties.left_ranges <= right_range) & (properties.right_ranges >= left_range))
                    properties.overlaps = overlaps
                    properties.n_overlaps = np.sum(overlaps)

            # save results
            self._detection_results[idx] = properties  # this uses relative index
            self._detection_results_raw[idx_raw] = properties  # this uses abs index

    def remove_duplicates_detection(self):
        if not self._detection_results_duplicates:
            self._detection_results_duplicates = self._detection_results_raw
            duplicates = dict()
            unique_detection_result = dict()
            unique_abs_index = list()
            for rela_index, re in self._detection_results_raw.items():
                mz_min, mz_max = np.round(re.mz_range[0],decimals=3), np.round(re.mz_range[1],decimals=3)
                if re.matching_abs_peak is not None:
                    matching_abs_peak = np.round(re.matching_abs_peak)
                    if not duplicates.get((mz_min, mz_max,matching_abs_peak)):
                        duplicates[(mz_min, mz_max, matching_abs_peak)] = [(rela_index,re.absolute_index)]
                    else:
                        duplicates[(mz_min, mz_max,matching_abs_peak)].append((rela_index,re.absolute_index))
            i = 0
            for mzrange, rela_index in duplicates.items():
                unique_detection_result[i] = self._detection_results_raw[rela_index[0][0]]
                unique_abs_index.append(rela_index[0][0])
                i += 1
            self._detection_results_raw = unique_detection_result
            self._detection_results = unique_detection_result
            self.n_feature = len(unique_detection_result)
            self._ints_duplicates_raw = self._ints_raw
            self._ints_duplicates_sm_raw = self._ints_sm_raw
            self._ints_raw = self._ints_raw[unique_abs_index]
            self._ints_sm_raw = self._ints_sm_raw[unique_abs_index]
            self._ints = np.copy(self._ints_raw)
            self._mzs_duplicates_raw = self._mzs_raw
            self._mzs_duplicates = self._mzs
            self._mzs_raw = self._mzs_raw[unique_abs_index]
            self._mzs = np.copy(self._mzs_raw)
            self._df_duplicates = self._df
            self._df = self._df.iloc[unique_abs_index]

    def get_peak_detection_results(self, index: Union[int, Sequence[int]],
                                   ordered: bool = True) -> Union[PeakDetectionResults, List[PeakDetectionResults]]:
        index = self.absolute_index(index, ordered=ordered)
        if len(index) == 1:
            return self._detection_results_raw.get(index[0])
        else:
            return [self._detection_results_raw.get(i) for i in index]

    def decompose(self, index: Int, B: np.ndarray,
                  const_term: bool, l1: Float = 1., l2: Float = 1E-5,
                  save: bool = True, M: Optional[np.ndarray] = None) -> PeakDecompositionResults:
        """
        Perform a peak decomposition for a single feature, using B as the base matrix
        :param index:
        :param B:
        :param const_term:
        :param l1:
        :param l2:
        :param save:
        :param M:
        :return:
        """

        info = PeakDecompositionResults(
            const_term=const_term,
            l1=l1,
            l2=l2
        )
        peak_info = self._detection_results.get(index)

        if peak_info is None:
            info.status = -1  # status -1: cannot process
            if save:
                self._decomposition_results[index] = info
            return info

        rt_range = peak_info.matching_range
        if rt_range is None:
            info.status = -1  # status -1: cannot process
            if save:
                self._decomposition_results[index] = info
            return info

        rt_idx = np.arange(self.n_rt)
        rt_range_idx = (rt_range[0] <= rt_idx) & (rt_idx <= rt_range[1])

        v = np.copy(self.ints[index, :])
        v[~rt_range_idx] = 0.
        v -= np.min(v)
        v_max = np.max(v)
        if v_max > 0.:  # avoid NAN
            v /= v_max
        else:
            warnings.warn("v_max is 0")

        c = BQP.decompose(v, B, l1=l1, l2=l2, const_term=const_term, M=M)
        v_recons = c.dot(B)

        mse = np.linalg.norm(v - v_recons, 2) / np.sqrt(self.n_rt)

        info.status = 0  # success
        info.rt_range = rt_range
        info.v = v
        info.B = B
        info.c = c
        info.v_recons = v_recons
        info.v_max = v_max
        info.mse = mse

        if save:
            self._decomposition_results[index] = info

        return info

    def reset_base(self) -> None:
        """
        Reset the base information and the base matrix
        :return:
        """
        self.base_info: Dict[Int, BaseProperties] = dict()
        self.base_index: List[Int] = []
        self.B: Optional[np.ndarray] = np.ones(shape=(1, self.n_rt))
        self.M: Optional[np.ndarray] = np.array([[self.n_rt]], dtype=np.float64)

    @property
    def n_base(self) -> int:
        return len(self.base_index)

    def plot_decomposition(self, index: int, threshold: float = 1E-2,
                           dpi: Optional[Numeric] = None,
                           figsize: Optional[Tuple[Numeric, Numeric]] = None) -> None:
        decomp_info: Optional[PeakDecompositionResults] = self._decomposition_results.get(index)
        if decomp_info is None:
            raise ValueError("No decomposition results.")

        v = decomp_info.v
        c = decomp_info.c

        base_idx = np.arange(len(c) - 1)[(c[1:] >= threshold)]

        with self.plt_context(dpi=dpi, figsize=figsize):

            plt.plot(self.rts, v)
            plt.plot(self.rts, decomp_info.v_recons, '--')

            for i in base_idx:
                base = self.base_info[self.base_index[i]]
                coeff = c[i + 1]  # coefficient
                plt.plot(self.rts, - base.v * coeff)

    def generate_base(self, index: Optional[Sequence[int]] = None, reset: bool = False,
                      l1: float = 1., l2: float = 1E-5,
                      min_base_err: float = 5E-2, min_rt_diff: float = 1.,
                      allowed_n_overlap: Union[Int, Tuple[Optional[Int], Optional[Int]]] = (1, 1),
                      min_sin: float = 1E-2,
                      max_cos_sim: float = 0.9,
                      leave: bool = True, enable_tqdm: bool = True) -> None:
        """
        :param index: always interpreted as relative index
        :param reset:
        :param l1:
        :param l2:
        :param min_base_err:
        :param min_rt_diff:
        :param allowed_n_overlap:
        :param min_sin:
        :param max_cos_sim:
        :param leave:
        :param enable_tqdm:
        :return:
        """

        if hasattr(allowed_n_overlap, "__len__") and hasattr(allowed_n_overlap, "__getitem__"):
            if allowed_n_overlap[0] is not None:
                min_n_overlap = int(allowed_n_overlap[0])
            else:
                min_n_overlap = 1

            if allowed_n_overlap[1] is not None:
                max_n_overlap = int(allowed_n_overlap[1])
            else:
                max_n_overlap = np.inf
        else:
            min_n_overlap = int(allowed_n_overlap)
            max_n_overlap = int(allowed_n_overlap)

        if index is None:
            index = np.arange(self.n_feature)
        else:
            index = np.array([int(i) for i in index])

        if reset:
            self.reset_base()

        n_base_before = len(self.base_index)

        # generate base
        index_iter = tqdm(enumerate(index), disable=(not enable_tqdm), leave=leave, total=len(index))
        for i, ft_idx in index_iter:

            index_iter.set_description("generated # basis: {} --> {}".format(n_base_before, len(self.base_index)))

            if ft_idx in self.base_index:  # if already in base, skip
                continue

            # fixed self._detection_results, which now uses __relative__ index
            peak_info = self._detection_results.get(ft_idx)

            if peak_info is None:  # no peak info, skip
                continue

            n_overlaps = peak_info.n_overlaps
            if n_overlaps is None:  # no peak detected, skip
                continue

            if peak_info.n_overlaps < min_n_overlap or peak_info.n_overlaps > max_n_overlap:
                # invalid n_overlap, skip
                continue

            rt = peak_info.matching_abs_peak  # absolute retention time
            base_rts = [d.rt for d in self.base_info.values()]

            if base_rts:
                rt_diff = np.min(np.abs(np.array(base_rts) - rt))
            else:
                rt_diff = np.inf

            if rt_diff < min_rt_diff:
                # rt_diff too small, skip
                continue

            # now do a decomposition, no saving when generating base
            decompose_info = self.decompose(ft_idx, self.B, const_term=True, l1=l1, l2=l2, save=False)

            mse = decompose_info.mse
            if mse < min_base_err:
                # mse too small, skip
                continue

            # compute sin(theta)
            v = decompose_info.v
            c_ols = np.linalg.solve(self.M, self.B.dot(v))  # OLS solution
            v_ols_recon = c_ols.dot(self.B)
            sin = np.linalg.norm(v - v_ols_recon, 2) / np.linalg.norm(v, 2)

            if sin < min_sin:  # sin(theta) too small, skip
                continue

            # compute cosine similarity:
            cos_vec = self.B[1:, :].dot(v)  # self.B[1:, :]: do not include the first
            cos_vec = cos_vec / np.linalg.norm(v, 2)
            cos_vec = cos_vec / np.linalg.norm(self.B[1:, :], 2, axis=1)

            if cos_vec.size > 0:
                cos_sim = np.max(cos_vec)
            else:
                cos_sim = 0.  # when bases is empty we define the cos sim to be 0.

            if cos_sim > max_cos_sim:  # cos similarity too large; skip
                continue

            # more tests can be added here; when failed, use continue to skip

            # finally, added as a basis

            self.B = np.vstack([self.B, decompose_info.v])
            self.M = self.B.dot(self.B.T)  # can be made more efficient

            base_info = BaseProperties(
                index=ft_idx,  # relative index
                abs_index=self._view_order[ft_idx],  # abs index
                rt=rt,  # abs retention time
                n_overlaps=peak_info.n_overlaps,
                mse=mse,
                v=decompose_info.v,
                v_max=decompose_info.v_max,
                sin=sin,
                cos_sim=cos_sim,
                cond=np.linalg.cond(self.M)  # condition number of M; just for inspection, can be commented out
            )

            self.base_index.append(ft_idx)
            self.base_info[ft_idx] = base_info

        # n_base_after = len(self._base_info)

        # if verbose:
        #     print("number of basis: {} --> {}".format(n_base_before, n_base_after))

    def base_cos_similarity(self, zero_diag: bool = True, include_const: bool = False,
                            plot: bool = False,
                            dpi: Optional[Numeric] = None,
                            figsize: Optional[Tuple[Numeric, Numeric]] = None
                            ) -> np.ndarray:
        """
        Compute the cos similarity matrix for the bases
        :param zero_diag:
        :param include_const:
        :param plot:
        :return:
        """

        if include_const:
            B = self.B
            if self.M is not None:
                M = self.M
            else:
                M = B.dot(B.T)
        else:  # do not include the constant term
            B = self.B[1:]
            if self.M is not None:
                M = self.M[1:, 1:]
            else:
                M = B.dot(B.T)
        B_norm = np.linalg.norm(B, 2, axis=1)
        if M.size > 0:
            cos = M / B_norm.reshape((1, -1)) / B_norm.reshape((-1, 1))
        else:
            raise ValueError("Empty base set.")

        zero_diag_cos = np.copy(cos)

        for i in range(zero_diag_cos.shape[0]):
            zero_diag_cos[i, i] = 0.

        if plot:
            arg_max = np.unravel_index(np.argmax(zero_diag_cos), shape=zero_diag_cos.shape)
            max_cos_sim = zero_diag_cos[arg_max[0], arg_max[1]]
            with self.plt_context(dpi=dpi, figsize=figsize):
                plt.imshow(cos)  # always show diagonal term as 1
                plt.title("cos sim, max = {:.4f} @ ({}, {})".format(max_cos_sim, arg_max[0], arg_max[1]))
                cbar = plt.colorbar()
                # cbar.ax.set_ylabel('cos similarity', rotation=270)

        if zero_diag:
            return zero_diag_cos
        else:
            return cos

    def base_rt_diff(self, plot: bool = False,
                     abs: bool = True,
                     dpi: Optional[Numeric] = None,
                     figsize: Optional[Tuple[Numeric, Numeric]] = None):

        if not self.base_info:
            raise ValueError("Empty base set.")

        rt_vec = np.array([i.rt for i in self.base_info.values()])

        rt_diff_mat = np.subtract.outer(rt_vec, rt_vec)

        abs_rt_diff_mat = np.abs(rt_diff_mat)

        if plot:

            off_diag = np.copy(abs_rt_diff_mat)
            for i in range(off_diag.shape[0]):
                off_diag[i, i] = + np.inf

            arg_min = np.unravel_index(np.argmin(off_diag), shape=off_diag.shape)

            min_abs_diff_rt = off_diag[arg_min[0], arg_min[1]]

            with self.plt_context(dpi=dpi, figsize=figsize):
                plt.imshow(abs_rt_diff_mat)
                plt.title("RT diff, min = {:.4f} s @ ({}, {})".format(min_abs_diff_rt, arg_min[0], arg_min[1]))
                plt.colorbar()

        if abs:
            return abs_rt_diff_mat
        else:
            return rt_diff_mat

    def base_compare_plot(self, index_a: int, index_b: int, dpi: Optional[Numeric] = None,
                          figsize: Optional[Tuple[Numeric, Numeric]] = None,
                          xlim: Union[str, Tuple[Numeric, Numeric]] = 'auto',
                          xlim_rt_delta: float = 20.
                          ) -> None:

        ft_idx_a: int = self.base_index[index_a]
        ft_idx_b: int = self.base_index[index_b]

        base_a: BaseProperties = self.base_info[ft_idx_a]
        base_b: BaseProperties = self.base_info[ft_idx_b]

        rt_a = base_a.rt
        rt_b = base_b.rt

        cos_sim = base_a.v.dot(base_b.v) / np.linalg.norm(base_a.v) / np.linalg.norm(base_b.v)
        rt_diff = np.abs(rt_a - rt_b)

        peak_a: PeakDetectionResults = self._detection_results[ft_idx_a]
        peak_b: PeakDetectionResults = self._detection_results[ft_idx_b]

        range_a = peak_a.matching_abs_range
        range_b = peak_b.matching_abs_range

        if xlim == 'auto':
            range_left = max(self.rts[0], min(range_a[0], range_b[0]) - xlim_rt_delta)
            range_right = min(self.rts[-1], max(range_a[1], range_b[1]) + xlim_rt_delta)

            xlim = (range_left, range_right)

        with self.plt_context(dpi=dpi, figsize=figsize):
            plt.plot(self.rts, base_a.v, color='#1f77b4', label=index_a)
            plt.axvline(x=rt_a, color='#1f77b4', alpha=0.7)
            plt.axvspan(xmin=range_a[0], xmax=range_a[1], ymax=1., ymin=0.5, alpha=.1, color='#1f77b4')

            plt.plot(self.rts, -base_b.v, color='#ff7f0e', label=index_b)
            plt.axvline(x=rt_b, color='#ff7f0e', alpha=0.7)
            plt.axvspan(xmin=range_b[0], xmax=range_b[1], ymax=.5, ymin=0., alpha=.1, color='#ff7f0e')

            plt.xlim(xlim)
            plt.legend()
            plt.title("cos sim: {:.4f},  rt diff: {:.4f}s".format(cos_sim, rt_diff))

            plt.xlabel("retention time / s")
            plt.ylabel("relative intensity")

    def perform_peak_decomposition(self, index: Optional[Union[int, Iterable[int]]] = None,
                                   l1: float = 1., l2: float = 1E-5, enable_tqdm: bool = True,
                                   leave: bool = True) -> None:

        if index is None:
            index = np.arange(self.n_feature)
        else:
            try:
                iter(index)
            except TypeError:  # not iterable
                index = [int(index)]
            else:
                index = [int(i) for i in index]

        for ft_idx in tqdm(index, desc="peak decomposition", leave=leave, disable=(not enable_tqdm)):

            decompose_info = self.decompose(ft_idx, self.B, const_term=True, l1=l1, l2=l2, save=True, M=self.M)

            # if decompose_info['status'] == -1:  # status -1: cannot process
            #     continue
            if ft_idx in self.base_index:
                decompose_info.status = 1  # status 1: is a basis

            # -- don't set extra status --
            # elif decompose_info['mse'] > mse_tol:
            #     decompose_info['status'] = -2  # status -2: failed
            # else:
            #     decompose_info['status'] = 0

            self._decomposition_results[ft_idx] = decompose_info  # this line should be unnecessary

    def spectrum_coefficient(self, base_index: int, threshold: float = 1E-2,  ##or threshold = 1E-2
                             max_mse: float = 1E-2, max_rt_diff: float = .5,
                             min_cos: float = 0.9,
                             save: bool = True, load: bool = True) -> np.ndarray:
        ##return:normalized coefficient of the basis
        if base_index < 0 or base_index >= self.n_base:  # index out of range
            raise ValueError

        base_ft_index = self.base_index[base_index]
        base_info = self.base_info[base_ft_index]

        if load and base_info.coefficient is not None and base_info.threshold == threshold and \
                base_info.max_mse == max_mse and base_info.max_rt_diff == max_rt_diff:
            return base_info.coefficient, base_info.coefficient_abs
        base_rt = base_info.rt

        c_idx = int(base_index) + 1  # index in c vector;
        # note that the first element in c vector corresponds to const term, hence +1

        coefficient = np.zeros(shape=self.n_feature)  # initialize as zero
        abs_coefficient = np.zeros(shape=self.n_feature)

        for idx in range(self.n_feature):
            # both are relative index
            decomp_info = self._decomposition_results.get(idx)
            peak_info = self._detection_results.get(idx)
            if decomp_info is None or peak_info is None:  # no relevant info available, skip
                continue
            c = decomp_info.c
            v_max = decomp_info.v_max
            mse = decomp_info.mse
            rts = peak_info.abs_peaks
            if c is None or v_max is None or mse is None or rts is None:  # decomposition info incomplete, skip
                continue
            if mse > max_mse:  # mse too large, skip
                continue
            rts = np.array(rts)
            if np.min(np.abs(rts - base_rt)) > max_rt_diff:  # rt difference is too large, skip
                continue
            if len(c) <= c_idx:  # c_idx out of range, skip
                continue

            cos = decomp_info.v.dot(base_info.v) / np.linalg.norm(decomp_info.v) / np.linalg.norm(base_info.v)

            if cos < min_cos:  # cos similarity too small, skip
                continue

            coefficient[idx] = v_max * c[c_idx]
            # v_max:the max intens of a certain feature,
            # c : decompose_info.c of a certain feature
            # c_idx:idx of coefficient of the self.basis

        if len(coefficient) > 0 and np.max(coefficient) > 0.:
            abs_coefficient = np.copy(coefficient)
            coefficient /= np.max(coefficient)

        coefficient[coefficient < threshold] = 0.
        abs_coefficient[coefficient < threshold] = 0.
        n_components = np.sum(coefficient > 0.)

        if save:  # save
            base_info.threshold = threshold
            base_info.max_mse = max_mse
            base_info.max_rt_diff = max_rt_diff
            base_info.coefficient = coefficient
            base_info.coefficient_abs = abs_coefficient
            base_info.n_components = n_components
            base_info.min_cos = min_cos

        return coefficient, abs_coefficient

    def plot_coelution(self, base_index: int, dpi: Optional[float] = None,
                       figsize: Optional[Tuple[Numeric, Numeric]] = None,
                       threshold: float = 1E-2, max_mse: float = 1E-2, max_rt_diff: float = .5,
                       min_cos: float = 0.9,
                       xlim: Optional[Union[str, Tuple[Numeric, Numeric]]] = 'auto',
                       xlim_delta_rt: float = 20.,
                       save: bool = True, load: bool = True,
                       mz_upperbound: float = np.inf, ##could set mz_upperbound to infinite
                       rescale: bool = True
                       ) -> None:
        """
        Plot the coelution graph for a certain base identified by base_index(relative_index)
        :param rescale:
        :param mz_upperbound:
        :param base_index: index of the base
        :param dpi:
        :param figsize:
        :param threshold:
        :param max_mse:
        :param max_rt_diff:
        :param xlim:
        :param xlim_delta_rt:
        :param save:
        :param load:
        :return:
        """

        base_ft_index = self.base_index[base_index]
        base_info: dict = asdict(self.base_info[base_ft_index])
        base_info.update(asdict(self._detection_results[base_ft_index]))

        base_v = base_info['v']

        base_range = base_info['matching_abs_range']
        base_rt = base_info['matching_abs_peak']

        if isinstance(xlim, str) and xlim[0] in 'aA':  # auto
            xlim = (max(self.rts[0], base_range[0] - xlim_delta_rt),
                    min(self.rts[-1], base_range[1] + xlim_delta_rt))

        coefficient, _ = self.spectrum_coefficient(base_index=base_index, threshold=threshold,
                                                                 max_mse=max_mse, max_rt_diff=max_rt_diff,
                                                                 min_cos=min_cos,
                                                                 save=save, load=load)

        arg_max = np.argmax(coefficient)
        arg_max_mz = self.mzs[arg_max, :].mean()

        range_idxs = (base_range[0] <= self.rts) & (self.rts <= base_range[1])

        mz_upperbound = arg_max_mz + mz_upperbound

        with self.plt_context(dpi=dpi, figsize=figsize):

            if xlim is not None:
                plt.xlim(xlim)

            for i in range(self.n_feature):
                if coefficient[i] >= threshold:

                    # mz = self.mzs[i, :].mean()

                    mz = self.df.iloc[i].mz

                    if mz > mz_upperbound:
                        continue  # skipped

                    decomp = self._decomposition_results.get(i)

                    if decomp is None:
                        continue

                    v = decomp.v

                    if v is None:
                        continue

                    max_signal = np.max(v[range_idxs])
                    if max_signal > 0.:
                        if rescale:
                            plt.plot(self.rts, v * coefficient[i] / max_signal)
                        else:
                            plt.plot(self.rts, v / max_signal)

            plt.axvspan(xmin=base_range[0], xmax=base_range[1], alpha=0.2)
            plt.axvline(x=base_rt, alpha=0.5)
            plt.plot(self.rts, -base_v, '--')
            plt.ylim((-1.05, 1.05))
            plt.xlabel("retention time / s")
            plt.ylabel("relative intensity")

    def gen_spectrum(self, base_index: int, plot: bool = False,
                     dpi: Optional[float] = None, figsize: Optional[Tuple[Numeric, Numeric]] = None,
                     threshold: float = 1E-2, max_mse: float = 1E-2,
                     max_rt_diff: float = 0.5,
                     min_cos: float = 0.9,
                     xlim: Optional[Union[str, Tuple[Numeric, Numeric]]] = 'auto',
                     xlim_delta_mz: float = 50.,
                     save: bool = True, load: bool = True,
                     mz_upperbound: float = np.inf,
                     ) -> ReconstructedSpectrum:

        base_info = self.base_info[self.base_index[base_index]]

        if load and base_info.spectrum is not None and base_info.threshold == threshold and \
                base_info.max_mse == max_mse and base_info.max_rt_diff == max_rt_diff \
                and base_info.mz_upperbound == mz_upperbound and base_info.min_cos == min_cos:

            spectrum = base_info.spectrum
            spectrum_list = spectrum.spectrum_list

        else:

            coefficient, abs_coefficient = self.spectrum_coefficient(base_index=base_index, threshold=threshold,
                                                                     max_mse=max_mse, max_rt_diff=max_rt_diff,
                                                                     min_cos=min_cos,
                                                                     save=save, load=load)
            # n_coe = len([i for i in coefficient if i > 0])
            # n_abs_coe = len([i for i in abs_coefficient if i > 0])

            base_info.mz_upperbound = mz_upperbound
            spectrum_list = [(self.df.iloc[i].mz, v) for i, v in enumerate(coefficient) if v > 0.]
            spectrum_list_abs = [(self.df.iloc[i].mz, v) for i, v in enumerate(abs_coefficient) if v > 0.]

            spectrum_list.sort(key=lambda x: x[0])
            spectrum_list_abs.sort(key=lambda x: x[0])

            arg_max = np.argmax(coefficient)
            arg_max_mz = self.mzs[arg_max, :].mean()

            spectrum_list = [(mz, intensity) for mz, intensity in spectrum_list if mz <= arg_max_mz + mz_upperbound]
            spectrum_list_abs = [(mz, intensity) for mz, intensity in spectrum_list_abs if
                                 mz <= arg_max_mz + mz_upperbound]

            spectrum = ReconstructedSpectrum(spectrum_list=spectrum_list, spectrum_list_abs=spectrum_list_abs)

        if plot:

            if isinstance(xlim, str) and xlim[0] in 'aA':  # auto
                xlim = (max(0., spectrum_list[0][0] - xlim_delta_mz),
                        spectrum_list[-1][0] + xlim_delta_mz)

            spectrum_array = np.array(spectrum_list).T
            with self.plt_context(dpi=dpi, figsize=figsize):

                if xlim is not None:
                    plt.xlim(xlim)

                _, _, baseline = \
                    plt.stem(spectrum_array[0], spectrum_array[1], use_line_collection=True, markerfmt=' ')

                baseline.set_xdata([0.01, .99])
                baseline.set_transform(plt.gca().get_yaxis_transform())
                plt.xlabel("mz")
                plt.ylabel("relative intensity")

        if save:
            base_info.spectrum = spectrum

        return spectrum

    __repr__ = __str__
