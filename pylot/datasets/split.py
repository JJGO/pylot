from typing import List, Tuple

from pydantic import validate_arguments
from sklearn.model_selection import train_test_split


@validate_arguments
def data_splits(
    values: List[str], splits: Tuple[float, float, float], seed: int
) -> Tuple[List[str], List[str], List[str]]:

    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate entries found in values")

    if (s := sum(splits)) != 1.0:
        raise ValueError(f"Splits must add up to 1.0, got {splits}->{s}")

    train_size, val_size, test_size = splits
    values = sorted(values)
    if test_size == 0.0:
        trainval, test = values, []
    else:
        trainval, test = train_test_split(
            values, test_size=test_size, random_state=seed
        )
    val_ratio = val_size / (train_size + val_size)
    if val_size == 0.0:
        train, val = trainval, []
    else:
        train, val = train_test_split(trainval, test_size=val_ratio, random_state=seed)

    assert sorted(train + val + test) == values, "Missing Values"

    return (train, val, test)
