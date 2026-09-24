from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
NORMALIZED_DIR = ROOT_DIR / "data_normalized"
if not NORMALIZED_DIR.exists():
    NORMALIZED_DIR = ROOT_DIR / "data" / "normalized"


def load_telemetry(
    module: str,
    data_dir: str | Path = NORMALIZED_DIR,
    columns: list[str] | None = None,
    entities: list[str] | None = None,
) -> pd.DataFrame:
    path = Path(data_dir) / f"telemetry_{module}.parquet"
    filters = [("entity_id", "in", list(entities))] if entities else None
    df = pd.read_parquet(path, columns=columns, filters=filters)
    return df.sort_values(["entity_id", "ts"]).reset_index(drop=True)


def load_equipment_params(data_dir: str | Path = NORMALIZED_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "equipment_params.csv", comment="#")
