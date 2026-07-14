# Migration notes for 0.4.6

The defaults are now `fitter="auto"`, `compression="auto"`,
`demean_backend="auto"`, and `retain_compressed=False`. Automatic compression
chooses only between two exact representations; rounded compression remains an
explicit opt-in.

`se_method="none"` now strictly skips covariance work. IV first-stage
coefficients remain available, while covariance-based F statistics and p-values
are `None`.

Models close their DuckDB connection after fitting. Use
`retain_compressed=True` to preserve `df_compressed`; otherwise accessing it
after fitting raises an error explaining how to opt in. `close()` is idempotent,
and models support `with` statements.

DuckDB resource settings are typed top-level arguments. In particular,
`threads=1` is now applied rather than treated as an implicit default.
