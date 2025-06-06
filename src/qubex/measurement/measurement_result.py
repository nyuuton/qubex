from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, reduce
from typing import Collection

import numpy as np
from numpy.typing import NDArray

from ..analysis import visualization as viz
from ..backend import SAMPLING_PERIOD
from .state_classifier import StateClassifier

SAMPLING_PERIOD_SINGLE = SAMPLING_PERIOD
SAMPLING_PERIOD_AVG = SAMPLING_PERIOD * 4


class MeasureMode(Enum):
    SINGLE = "single"
    AVG = "avg"

    @cached_property
    def integral_mode(self) -> str:
        if self == MeasureMode.SINGLE:
            return "single"
        elif self == MeasureMode.AVG:
            return "integral"
        else:
            raise ValueError(f"Invalid mode: {self}")


@dataclass(frozen=True)
class MeasureData:
    target: str
    mode: MeasureMode
    raw: NDArray
    classifier: StateClassifier | None = None

    @cached_property
    def n_states(self) -> int:
        if self.classifier is None:
            raise ValueError("Classifier is not set")
        else:
            return self.classifier.n_states

    @cached_property
    def kerneled(
        self,
    ) -> NDArray:
        if self.mode == MeasureMode.SINGLE:
            return np.mean(self.raw, axis=1)
        elif self.mode == MeasureMode.AVG:
            return np.asarray(np.mean(self.raw))
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @cached_property
    def classified(self) -> NDArray:
        if self.mode == MeasureMode.SINGLE:
            if self.classifier is not None:
                return self.classifier.predict(self.kerneled)
            else:
                raise ValueError("Classifier is not set")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @cached_property
    def length(self) -> int:
        return len(self.raw)

    @cached_property
    def times(self) -> NDArray[np.float64]:
        if self.mode == MeasureMode.SINGLE:
            return np.arange(self.length) * SAMPLING_PERIOD_SINGLE
        elif self.mode == MeasureMode.AVG:
            return np.arange(self.length) * SAMPLING_PERIOD_AVG
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @cached_property
    def counts(self) -> dict[str, int]:
        if len(self.classified) == 0:
            raise ValueError("No classification data available")
        classified_labels = self.classified
        count = np.bincount(classified_labels, minlength=self.n_states)
        state = {str(label): count[label] for label in range(len(count))}
        return state

    @cached_property
    def probabilities(self) -> NDArray[np.float64]:
        if len(self.classified) == 0:
            raise ValueError("No classification data available")
        total = sum(self.counts.values())
        return np.array([count / total for count in self.counts.values()])

    @cached_property
    def standard_deviations(self) -> NDArray[np.float64]:
        if len(self.classified) == 0:
            raise ValueError("No classification data available")
        return np.sqrt(
            self.probabilities * (1 - self.probabilities) / sum(self.counts.values())
        )

    def plot(
        self,
        title: str | None = None,
        return_figure: bool = False,
        save_image: bool = False,
    ):
        if self.mode == MeasureMode.SINGLE:
            return viz.scatter_iq_data(
                data={self.target: np.asarray(self.kerneled)},
                title=title or f"Readout IQ data : {self.target}",
                return_figure=return_figure,
                save_image=save_image,
            )
        elif self.mode == MeasureMode.AVG:
            return viz.plot_waveform(
                data=self.raw,
                sampling_period=SAMPLING_PERIOD_AVG,
                title=title or f"Readout waveform : {self.target}",
                xlabel="Capture time (ns)",
                ylabel="Signal (arb. units)",
                return_figure=return_figure,
                save_image=save_image,
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def plot_fft(
        self,
        return_figure: bool = False,
        save_image: bool = False,
    ):
        return viz.plot_fft(
            x=self.times,
            y=self.raw,
            title=f"Fourier transform : {self.target}",
            xlabel="Frequency (GHz)",
            ylabel="Signal (arb. units)",
            return_figure=return_figure,
            save_image=save_image,
        )


@dataclass(frozen=True)
class MeasureResult:
    mode: MeasureMode
    data: dict[str, MeasureData]
    config: dict

    @cached_property
    def counts(self) -> dict[str, int]:
        return self.get_counts()

    @cached_property
    def probabilities(self) -> dict[str, float]:
        return self.get_probabilities()

    @cached_property
    def standard_deviations(self) -> dict[str, float]:
        return self.get_standard_deviations()

    @cached_property
    def mitigated_counts(self) -> dict[str, int]:
        return self.get_mitigated_counts()

    @cached_property
    def mitigated_probabilities(self) -> dict[str, float]:
        return self.get_mitigated_probabilities()

    def get_basis_indices(
        self,
        targets: Collection[str] | None = None,
    ) -> list[tuple[int, ...]]:
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        if targets is None:
            targets = self.data.keys()
        dimensions = [self.data[target].n_states for target in targets]
        return list(np.ndindex(*[dim for dim in dimensions]))

    def get_basis_labels(
        self,
        targets: Collection[str] | None = None,
    ) -> list[str]:
        basis_indices = self.get_basis_indices(targets)
        return ["".join(str(i) for i in basis) for basis in basis_indices]

    def get_classified_data(
        self,
        targets: Collection[str] | None = None,
    ) -> NDArray:
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        if targets is None:
            targets = self.data.keys()
        return np.column_stack([self.data[target].classified for target in targets])

    def get_counts(
        self,
        targets: Collection[str] | None = None,
    ) -> dict[str, int]:
        classified_data = self.get_classified_data(targets)
        classified_labels = np.array(
            ["".join(map(str, row)) for row in classified_data]
        )
        counts = dict(Counter(classified_labels))
        basis_labels = self.get_basis_labels(targets)
        counts = {
            basis_label: counts.get(basis_label, 0) for basis_label in basis_labels
        }
        return counts

    def get_probabilities(
        self,
        targets: Collection[str] | None = None,
    ) -> dict[str, float]:
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        total = sum(self.get_counts(targets).values())
        return {key: count / total for key, count in self.get_counts(targets).items()}

    def get_standard_deviations(
        self,
        targets: Collection[str] | None = None,
    ) -> dict[str, float]:
        if len(self.data) == 0:
            raise ValueError("No classification data available")
        return {
            key: np.sqrt(prob * (1 - prob) / total)
            for key, prob, total in zip(
                self.get_counts(targets).keys(),
                self.get_probabilities(targets).values(),
                self.get_counts(targets).values(),
            )
        }

    def get_classifier(self, target: str) -> StateClassifier:
        if target not in self.data:
            raise ValueError(f"Target {target} not found in data")
        classifier = self.data[target].classifier
        if classifier is None:
            raise ValueError(f"Classifier for target {target} is not set")
        return classifier

    def get_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> NDArray:
        targets = tuple(targets)
        confusion_matrices = []
        for target in targets:
            cm = self.get_classifier(target).confusion_matrix
            n_shots = cm[0].sum()
            confusion_matrices.append(cm / n_shots)
        return reduce(np.kron, confusion_matrices)

    def get_inverse_confusion_matrix(
        self,
        targets: Collection[str],
    ) -> NDArray:
        targets = tuple(targets)
        confusion_matrix = self.get_confusion_matrix(targets)
        return np.linalg.inv(confusion_matrix)

    def get_mitigated_counts(
        self,
        targets: Collection[str] | None = None,
    ) -> dict[str, int]:
        if targets is None:
            targets = self.data.keys()
        raw = self.get_counts(targets)
        cm_inv = self.get_inverse_confusion_matrix(targets)
        mitigated = np.array(list(raw.values())) @ cm_inv
        basis_labels = self.get_basis_labels(targets)
        mitigated_counts = {
            basis_label: int(mitigated[i]) for i, basis_label in enumerate(basis_labels)
        }
        return mitigated_counts

    def get_mitigated_probabilities(
        self,
        targets: Collection[str] | None = None,
    ) -> dict[str, float]:
        if targets is None:
            targets = self.data.keys()
        raw = self.get_probabilities(targets)
        cm_inv = self.get_inverse_confusion_matrix(targets)
        mitigated = np.array(list(raw.values())) @ cm_inv
        basis_labels = self.get_basis_labels(targets)
        mitigated_probabilities = {
            basis_label: mitigated[i] for i, basis_label in enumerate(basis_labels)
        }
        return mitigated_probabilities

    def plot(
        self,
        return_figure: bool = False,
        save_image: bool = False,
    ):
        if self.mode == MeasureMode.SINGLE:
            data = {
                qubit: np.asarray(data.kerneled) for qubit, data in self.data.items()
            }
            return viz.scatter_iq_data(
                data=data,
                return_figure=return_figure,
                save_image=save_image,
            )
        elif self.mode == MeasureMode.AVG:
            figures = []
            for data in self.data.values():
                fig = data.plot(
                    return_figure=return_figure,
                    save_image=save_image,
                )
                figures.append(fig)
            if return_figure:
                return figures
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def plot_fft(
        self,
        return_figure: bool = False,
        save_image: bool = False,
    ):
        figures = []
        for data in self.data.values():
            fig = data.plot_fft(
                return_figure=return_figure,
                save_image=save_image,
            )
            figures.append(fig)
        if return_figure:
            return figures


@dataclass(frozen=True)
class MultipleMeasureResult:
    mode: MeasureMode
    data: dict[str, list[MeasureData]]
    config: dict

    def plot(
        self,
        return_figure: bool = False,
        save_image: bool = False,
    ):
        for qubit, data_list in self.data.items():
            figures = []
            for capture_index, data in enumerate(data_list):
                fig = data.plot(
                    title=f"{qubit} : data[{capture_index}]",
                    return_figure=return_figure,
                    save_image=save_image,
                )
                figures.append(fig)
            if return_figure:
                return figures
        return None
