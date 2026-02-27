"""API configuration and utility functions"""
from pathlib import Path
from typing import Any, Optional, Tuple
import hashlib


class FEMethod:
    """Fixed effects handling methods"""
    MUNDLAK = "mundlak"
    DEMEAN = "demean"
    AUTO = "auto"
    AUTO_FE = "auto_fe"


# ============================================================================
# File Format Configuration
# ============================================================================

# Glob extensions that map to DuckDB reader functions
_FILE_READERS = {
    ".parquet": "read_parquet",
    ".parquet.gz": "read_parquet",
    ".csv": "read_csv_auto",
    ".csv.gz": "read_csv_auto",
    ".tsv": "read_csv_auto",
    ".tsv.gz": "read_csv_auto",
    ".json": "read_json_auto",
    ".ndjson": "read_json_auto",
    ".jsonl": "read_json_auto",
    ".arrow": "read_arrow",
    ".feather": "read_arrow",
}

# Name used when an in-memory object is registered as a DuckDB view
_DUCKDB_VIEW_NAME = "_duckreg_data"


# ============================================================================
# Data Source Utilities
# ============================================================================


def _resolve_table_name(data_path: Path) -> str:
    """Create a DuckDB table/scan expression from a file-system path.

    Supports single files and directories (glob-scanned as parquet).
    """
    if data_path.is_file():
        # Match on the *full* suffix chain (e.g. '.csv.gz')
        name = data_path.name.lower()
        reader = None
        for ext, fn in _FILE_READERS.items():
            if name.endswith(ext):
                reader = fn
                break
        if reader is None:
            supported = list(_FILE_READERS.keys())
            raise ValueError(
                f"Unsupported file format: '{data_path.suffix}'. "
                f"Supported extensions: {supported}"
            )
        return f"{reader}('{data_path}')"
    elif data_path.is_dir():
        return f"read_parquet('{data_path}/**/*.parquet')"
    raise ValueError(f"Data path not found: {data_path}")


def _resolve_db_path(data: str, cache_dir: Optional[str], db_name: Optional[str]) -> str:
    """Resolve a persistent DuckDB database path from file-based inputs."""
    if db_name is not None:
        db_path = Path(db_name)
        db_path.parent.mkdir(exist_ok=True, parents=True)
        return str(db_path)

    data_path = Path(data).resolve()
    cache_dir = Path(cache_dir) if cache_dir else (
        data_path.parent if data_path.is_file() else data_path
    ) / ".duckreg"

    cache_dir.mkdir(exist_ok=True, parents=True)
    data_hash = hashlib.md5(str(data_path).encode()).hexdigest()[:8]
    return str(cache_dir / f"duckreg_{data_hash}.db")


def _resolve_data_source(
    data: Any,
    cache_dir: Optional[str] = None,
    db_name: Optional[str] = None,
) -> Tuple[str, str, Any]:
    """Unified data-source resolver.

    Accepts file paths (str/Path), in-memory DataFrames (pandas, Polars,
    PyArrow), and any other object that DuckDB can register as a view.

    Returns
    -------
    db_path : str
        DuckDB database path.  ``':memory:'`` for non-file sources unless
        *db_name* is provided.
    table_name : str
        SQL expression or view name used to reference the data inside DuckDB.
    obj_to_register : object or None
        When not ``None`` the caller must register this object in the DuckDB
        connection under *_DUCKDB_VIEW_NAME* before querying.
    """
    # ---- pandas DataFrame ---------------------------------------------------
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            db_path = db_name or ":memory:"
            return db_path, _DUCKDB_VIEW_NAME, data
    except ImportError:
        pass

    # ---- PyArrow Table / RecordBatch ----------------------------------------
    try:
        import pyarrow as pa
        if isinstance(data, (pa.Table, pa.RecordBatch)):
            db_path = db_name or ":memory:"
            return db_path, _DUCKDB_VIEW_NAME, data
    except ImportError:
        pass

    # ---- Polars DataFrame ---------------------------------------------------
    try:
        import polars as pl
        if isinstance(data, pl.DataFrame):
            db_path = db_name or ":memory:"
            return db_path, _DUCKDB_VIEW_NAME, data
    except ImportError:
        pass

    # ---- DuckDB relation (already tied to a connection) ---------------------
    try:
        import duckdb
        if isinstance(data, duckdb.DuckDBPyRelation):
            # Materialise into an in-memory view; the relation carries its own
            # connection so we cannot move it — the caller is responsible for
            # passing the same connection via duckdb_kwargs if needed.
            db_path = db_name or ":memory:"
            return db_path, _DUCKDB_VIEW_NAME, data
    except ImportError:
        pass

    # ---- File/directory path (str or pathlib.Path) --------------------------
    if isinstance(data, (str, Path)):
        data_str = str(data)
        resolved_db = _resolve_db_path(data_str, cache_dir, db_name)
        table_ref = _resolve_table_name(Path(data_str).resolve())
        return resolved_db, table_ref, None

    raise TypeError(
        f"Unsupported data type: {type(data)!r}.  "
        "Pass a file path (str/Path), pandas/Polars/PyArrow DataFrame, "
        "or a DuckDB relation."
    )
