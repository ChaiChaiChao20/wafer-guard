from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEGACY_ERROR_CODES = (100.0, 200.0, 300.0)


@dataclass
class CausalPreprocessor:
    """Train-only robust imputation/scaling plus explicit PS-L semantics.

    sensor_0 == 0 identifies a PS-L wafer. sensor_0 is then a route sentinel,
    and sensor_1 is a firmware category. Those values never enter continuous
    statistics. Their continuous coordinates are neutralized only in the model
    matrix and are accompanied by explicit route/code features.
    """

    medians_: np.ndarray | None = None
    means_: np.ndarray | None = None
    scales_: np.ndarray | None = None
    feature_count_: int | None = None

    def fit(self, train_features: np.ndarray) -> "CausalPreprocessor":
        x = np.asarray(train_features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError("Expected a 2-D sensor matrix with sensor_0 and sensor_1")
        legacy = np.isfinite(x[:, 0]) & (x[:, 0] == 0.0)
        fit_values = x.copy()
        fit_values[legacy, 0:2] = np.nan

        count = x.shape[1]
        medians = np.zeros(count, dtype=np.float64)
        means = np.zeros(count, dtype=np.float64)
        scales = np.ones(count, dtype=np.float64)
        for col in range(count):
            observed = fit_values[:, col]
            observed = observed[np.isfinite(observed)]
            if observed.size == 0:
                continue
            median = float(np.median(observed))
            medians[col] = median
            means[col] = float(np.mean(observed))
            scale = float(np.std(observed))
            scales[col] = scale if scale > 1e-8 else 1.0

        self.medians_ = medians
        self.means_ = means
        self.scales_ = scales
        self.feature_count_ = count
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.medians_ is None or self.means_ is None or self.scales_ is None:
            raise RuntimeError("Call fit() on the training period before transform()")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.feature_count_:
            raise ValueError("Sensor matrix shape differs from the fitted training matrix")

        legacy = np.isfinite(x[:, 0]) & (x[:, 0] == 0.0)
        codes = x[:, 1]
        numeric = x.copy()
        numeric[legacy, 0:2] = np.nan
        missing = ~np.isfinite(numeric)
        missing[legacy, 0:2] = True

        # Do not scale PS-L firmware codes or the PS-L route sentinel. They are
        # masked before scaling. The normalized zeros below are neutral model
        # coordinates, not imputations of continuous measurements.
        imputed = np.where(np.isfinite(numeric), numeric, self.medians_[None, :])
        scaled = (imputed - self.means_[None, :]) / self.scales_[None, :]
        scaled[legacy, 0:2] = 0.0
        scaled = np.clip(scaled, -12.0, 12.0).astype(np.float32)

        code_features = np.zeros((x.shape[0], len(LEGACY_ERROR_CODES) + 1), dtype=np.float32)
        for index, code in enumerate(LEGACY_ERROR_CODES):
            code_features[:, index] = (legacy & np.isfinite(codes) & (codes == code)).astype(np.float32)
        known = np.zeros(x.shape[0], dtype=bool)
        for code in LEGACY_ERROR_CODES:
            known |= np.isfinite(codes) & (codes == code)
        code_features[:, -1] = (legacy & ~known).astype(np.float32)

        missing_features = missing.astype(np.float32)
        legacy_feature = legacy.astype(np.float32)[:, None]
        return np.concatenate((scaled, missing_features, legacy_feature, code_features), axis=1)

    def fit_transform(self, train_features: np.ndarray) -> np.ndarray:
        return self.fit(train_features).transform(train_features)

    def save(self, path: str) -> None:
        if self.medians_ is None or self.means_ is None or self.scales_ is None:
            raise RuntimeError("Preprocessor has not been fitted")
        np.savez_compressed(
            path,
            medians=self.medians_,
            means=self.means_,
            scales=self.scales_,
            feature_count=np.asarray([self.feature_count_], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str) -> "CausalPreprocessor":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                medians_=data["medians"].astype(np.float64),
                means_=data["means"].astype(np.float64),
                scales_=data["scales"].astype(np.float64),
                feature_count_=int(data["feature_count"][0]),
            )

