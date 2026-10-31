import numpy as np

from cxrbench.data import split_augment_then_split, split_clean
from cxrbench.protocols import PAPER_ACC, PAPER_K, PAPER_N


def test_paper_numbers():
    assert PAPER_K == 978
    assert PAPER_N == 1043
    assert 0.92 < PAPER_ACC < 0.95


def test_clean_split_is_disjoint():
    labels = np.repeat([0, 1, 2], [1000, 1000, 1000])
    train, val, test = split_clean(labels, seed=42)
    assert len(train) + len(val) + len(test) == len(labels)
    assert not set(train[:, 0]) & set(test[:, 0])
    assert not set(val[:, 0]) & set(test[:, 0])


def test_augment_then_split_items_are_unique():
    labels = np.repeat([0, 1, 2], [1000, 1000, 1000])
    train, val, test = split_augment_then_split(labels, seed=42)
    total = len(train) + len(val) + len(test)
    assert total == len(labels) * 4
    items = {(int(i), int(v)) for i, v in np.vstack([train, val, test])}
    assert len(items) == total
