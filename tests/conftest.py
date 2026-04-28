import pytest
import pandas as pd


def pytest_addoption(parser):
    parser.addoption(
        "--force-regen",
        action="store_true",
        default=False,
        help="Force regeneration of test data",
    )


@pytest.fixture(scope="session")
def force_regen(request):
    return request.config.getoption("--force-regen")


@pytest.fixture
def parquet_path_factory(tmp_path):
    def _write(df: pd.DataFrame, name: str = "data.parquet") -> str:
        path = tmp_path / name
        df.to_parquet(path, index=False)
        return str(path)

    return _write
