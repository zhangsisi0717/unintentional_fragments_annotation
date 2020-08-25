from type_checking import *
import numpy as np


class BinnedSparseVector:

    def __init__(self, ppm: float = 60.) -> None:

        if ppm >= 1E6:
            raise ValueError("ppm cannot be greater than 1E6, and is recommended to be less than 1E3. ")

        self.ppm: float = ppm
        self.log_base: float = np.log(1 + ppm * 1E-6)
        self.dict: dict = dict()
        self._norm: Optional[float] = None

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        #x:self.mz y:self.relative_intensity

        n = x.shape[0]
        assert x.shape == (n, ) and y.shape == (n, )
        assert np.all(x > 0.)

        index = np.floor(np.log(x) / self.log_base).astype(int)

        for i in range(n):
            self.dict[index[i]] = self.dict.get(index[i], 0.) + y[i]

        # reset self._norm
        self._norm = None

    def __repr__(self):
        return self.dict.__repr__()

    @property
    def norm(self) -> float:
        if self._norm is None:
            if self.dict:
                self._norm = np.linalg.norm(np.array([i for i in self.dict.values()]), 2)
            else:
                self._norm = 0.

        return self._norm

    def inner(self, other: "BinnedSparseVector", transform: Optional[Callable[[float], float]] = None):

        if self.ppm != other.ppm:
            raise ValueError("cannot compute inner product of two BinnedSparseVector of different ppm.")

        inner_product = 0.

        if transform is None:
            for i, v in self.dict.items():
                #
                # if i in other.dict:  # do not directly access other.dict[i]!
                #     inner_product += v * other.dict[i]

                for j in (i, i-1, i+1):
                    if j in other.dict:  # do not directly access other.dict[i]!
                        inner_product += v * other.dict[j]
                        break
        else:
            for i, v in self.dict.items():
                #
                # if i in other.dict:  # do not directly access other.dict[i]!
                #     inner_product += v * other.dict[i]

                for j in (i, i-1, i+1):
                    if j in other.dict:  # do not directly access other.dict[i]!
                        inner_product += transform(v) * transform(other.dict[j])
                        break

        return inner_product

    def cos(self, other: "BinnedSparseVector", transform: Optional[Callable[[float], float]] = None) -> float:
        if transform is None:
            if self.norm > 0. and other.norm > 0.:
                return self.inner(other) / self.norm / other.norm
            else:
                raise ValueError("cos is not defined when one of the BinnedSparseVector has norm 0.")
        else:
            self_norm = np.linalg.norm(np.array([transform(v) for v in self.dict.values()]), 2)
            other_norm = np.linalg.norm(np.array([transform(v) for v in other.dict.values()]), 2)

            if self_norm > 0. and other_norm > 0:
                return self.inner(other, transform=transform) / self_norm / other_norm
