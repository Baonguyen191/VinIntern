import pandas as pd


def generate_folds(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    initial_train_years: int = 2,
    test_years: int = 1,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    folds = []
    i = 0
    while True:
        train_end = data_start + pd.DateOffset(
            years=initial_train_years + i * test_years
        )
        test_end = train_end + pd.DateOffset(years=test_years)

        if train_end >= data_end:
            break

        if test_end > data_end:
            test_end = data_end + pd.Timedelta(days=1)

        folds.append((data_start, train_end, train_end, test_end))
        i += 1

    return folds
