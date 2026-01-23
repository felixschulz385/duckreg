"""API configuration and utility functions"""
from pathlib import Path
from typing import Optional
import hashlib


class FEMethod:
    """Fixed effects handling methods"""
    MUNDLAK = "mundlak"
    DEMEAN = "demean"
    AUTO = "auto"


# ============================================================================
# File Format Configuration
# ============================================================================

_FILE_READERS = {
    ".parquet": "read_parquet",
    ".csv": "read_csv",
    ".parquet.gz": "read_parquet",
}


# ============================================================================
# Data Source Utilities
# ============================================================================


def _resolve_table_name(data_path: Path) -> str:
    """Create DuckDB table reference from data path"""
    if data_path.is_file():
        suffix = data_path.suffix.lower()
        if suffix not in _FILE_READERS:
            raise ValueError(f"Unsupported file format: {suffix}. Supported: {list(_FILE_READERS.keys())}")
        return f"{_FILE_READERS[suffix]}('{data_path}')"
    elif data_path.is_dir():
        return f"read_parquet('{data_path}/**/*.parquet')"
    raise ValueError(f"Data path not found: {data_path}")


def _resolve_db_path(data: str, cache_dir: Optional[str], db_name: Optional[str]) -> str:
    """Resolve database path from inputs"""
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
