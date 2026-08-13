"""Engineered DNA sequence representations used by primitive source heads."""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Sequence

import numpy as np


BASES = "ACGT"
PAIR_INDEX = {
    left + right: index
    for index, (left, right) in enumerate(itertools.product(BASES, repeat=2))
}
EIIP = {"A": 0.1260, "C": 0.1340, "G": 0.0806, "T": 0.1335}
PROMOTER_MOTIFS = [
    re.compile(pattern)
    for pattern in ["TATAAA", "TATATA", "TATA", "CAAT", "CCAAT", "GGGCGG", "CGCG", "[CT][CT]A[ACGT][AT][CT][CT]"]
]


def normalize_sequence(sequence: str) -> str:
    return "".join(base if base in BASES else "N" for base in sequence.upper())


def _kmers(k: int) -> tuple[str, ...]:
    return tuple("".join(word) for word in itertools.product(BASES, repeat=k))


def kmer_frequencies(sequences: Sequence[str], k: int) -> np.ndarray:
    vocabulary = {word: index for index, word in enumerate(_kmers(k))}
    output = np.zeros((len(sequences), len(vocabulary)), dtype=np.float32)
    for row, raw in enumerate(sequences):
        sequence = normalize_sequence(raw)
        valid = 0
        for start in range(max(len(sequence) - k + 1, 0)):
            index = vocabulary.get(sequence[start : start + k])
            if index is not None:
                output[row, index] += 1.0
                valid += 1
        if valid:
            output[row] /= valid
    return output


def cksnap(sequences: Sequence[str], max_gap: int = 5) -> np.ndarray:
    output = np.zeros((len(sequences), 16 * (max_gap + 1)), dtype=np.float32)
    for row, raw in enumerate(sequences):
        sequence = normalize_sequence(raw)
        for gap in range(max_gap + 1):
            offset = gap + 1
            denominator = 0
            block = slice(16 * gap, 16 * (gap + 1))
            for start in range(max(len(sequence) - offset, 0)):
                index = PAIR_INDEX.get(sequence[start] + sequence[start + offset])
                if index is not None:
                    output[row, 16 * gap + index] += 1.0
                    denominator += 1
            if denominator:
                output[row, block] /= denominator
    return output


def zcurve(sequences: Sequence[str], bins: int = 12) -> np.ndarray:
    rows = []
    for raw in sequences:
        sequence = normalize_sequence(raw)
        length = max(len(sequence), 1)
        row = []
        for bin_index in range(bins):
            start = round(bin_index * length / bins)
            stop = round((bin_index + 1) * length / bins)
            segment = sequence[start:stop] or sequence
            denominator = max(len(segment), 1)
            a, c, g, t = (segment.count(base) / denominator for base in BASES)
            row.extend([a + g - c - t, a + c - g - t, a + t - c - g, g + c, a - t, g - c])
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def eiip_autocorrelation(
    sequences: Sequence[str],
    max_lag: int = 24,
    spectrum_bins: int = 32,
) -> np.ndarray:
    rows = []
    for raw in sequences:
        values = np.asarray([EIIP.get(base, 0.0) for base in normalize_sequence(raw)], dtype=np.float32)
        if values.size == 0:
            values = np.zeros(1, dtype=np.float32)
        centered = values - values.mean()
        denominator = float(np.dot(centered, centered)) + 1e-8
        row = [float(values.mean()), float(values.std()), float(values.min()), float(values.max())]
        row.extend(
            float(np.dot(centered[:-lag], centered[lag:]) / denominator)
            if lag < centered.size
            else 0.0
            for lag in range(1, max_lag + 1)
        )
        fft_length = max(64, 2 ** math.ceil(math.log2(max(centered.size, 2))))
        spectrum = np.abs(np.fft.rfft(centered, n=fft_length)).astype(np.float32)[1:]
        spectrum = np.pad(spectrum[:spectrum_bins], (0, max(0, spectrum_bins - spectrum.size)))
        row.extend((spectrum / (np.linalg.norm(spectrum) + 1e-8)).tolist())
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def dinucleotide_autocorrelation(sequences: Sequence[str], max_lag: int = 12) -> np.ndarray:
    rows = []
    for raw in sequences:
        sequence = normalize_sequence(raw)
        indices = [PAIR_INDEX.get(sequence[index : index + 2], -1) for index in range(max(len(sequence) - 1, 0))]
        one_hot = np.zeros((max(len(indices), 1), 16), dtype=np.float32)
        for position, index in enumerate(indices):
            if index >= 0:
                one_hot[position, index] = 1.0
        frequencies = one_hot.mean(axis=0)
        centered = one_hot - frequencies
        denominator = np.sum(centered * centered, axis=0) + 1e-8
        row = frequencies.tolist()
        for lag in range(1, max_lag + 1):
            correlation = (
                np.sum(centered[:-lag] * centered[lag:], axis=0) / denominator
                if lag < len(one_hot)
                else np.zeros(16)
            )
            row.extend(correlation.tolist())
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def center_context(sequences: Sequence[str], flank: int = 20) -> np.ndarray:
    windows = [(-flank, -10), (-10, -3), (-3, 4), (4, 11), (11, flank + 1)]
    rows = []
    for raw in sequences:
        sequence = normalize_sequence(raw)
        center = len(sequence) // 2
        row = [float(sequence[center] == base) if sequence else 0.0 for base in BASES]
        for left, right in windows:
            segment = sequence[max(0, center + left) : min(len(sequence), center + right)] or sequence
            denominator = max(len(segment), 1)
            row.extend(segment.count(base) / denominator for base in BASES)
            row.extend(
                [
                    (segment.count("G") + segment.count("C")) / denominator,
                    segment.count("CG") / max(denominator - 1, 1),
                ]
            )
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def promoter_motif_position(sequences: Sequence[str], bins: int = 10) -> np.ndarray:
    rows = []
    for raw in sequences:
        sequence = normalize_sequence(raw)
        length = max(len(sequence), 1)
        row = []
        for pattern in PROMOTER_MOTIFS:
            positions = np.asarray([match.start() / length for match in pattern.finditer(sequence)])
            row.append(len(positions) / length)
            row.extend(
                [float(positions.mean()), float(positions.std()), float(positions.min()), float(positions.max())]
                if positions.size
                else [0.0, 0.0, 0.0, 0.0]
            )
        for bin_index in range(bins):
            start = round(bin_index * length / bins)
            stop = round((bin_index + 1) * length / bins)
            segment = sequence[start:stop] or sequence
            denominator = max(len(segment), 1)
            row.extend(
                [
                    segment.count("TATA") / max(denominator - 3, 1),
                    segment.count("CAAT") / max(denominator - 3, 1),
                    segment.count("CG") / max(denominator - 1, 1),
                    (segment.count("G") + segment.count("C")) / denominator,
                ]
            )
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)

