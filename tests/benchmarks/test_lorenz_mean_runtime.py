import pytest

from benchmarks import lorenz_mean_runtime


def test_parse_spill_diagnostics_selects_exact_entry():
    log = """
ptxas info    : Function properties for wanted
ptxas . 7 bytes stack frame, 10 bytes spill stores, 8 bytes spill loads
ptxas info    : Function properties for helper
ptxas . 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
"""

    assert lorenz_mean_runtime.parse_spill_diagnostics(
        log, "wanted"
    ) == (10, 8)

    with pytest.raises(SystemExit, match="found 0"):
        lorenz_mean_runtime.parse_spill_diagnostics(log, "missing")
    with pytest.raises(SystemExit, match="found 2"):
        lorenz_mean_runtime.parse_spill_diagnostics(
            log + log, "wanted"
        )


def test_chunk_defaults_scale_with_smoke_run_count():
    total_memory = 12 * 2**30
    default_runs, default_proportion = (
        lorenz_mean_runtime.resolve_chunk_settings(
            None, None, None, total_memory
        )
    )
    smoke_runs, smoke_proportion = (
        lorenz_mean_runtime.resolve_chunk_settings(
            1024, None, None, total_memory
        )
    )

    assert default_runs == 2**22
    assert smoke_runs == 1024
    assert default_proportion * total_memory == 24 * 2**20
    assert smoke_proportion * total_memory == 6 * 2**10


def test_explicit_chunk_settings_override_defaults():
    assert lorenz_mean_runtime.resolve_chunk_settings(
        1024, 2048, 0.01, 12 * 2**30
    ) == (2048, 0.01)


@pytest.mark.parametrize(
    "argv",
    (
        ["0"],
        ["--repeats", "0"],
        ["--chunked-runs", "-1"],
        ["--chunked-proportion", "nan"],
        ["--chunked-proportion", "1.1"],
        ["--repeats", "2", "--min-count", "3"],
    ),
)
def test_cli_rejects_invalid_numeric_domains(argv):
    with pytest.raises(SystemExit):
        lorenz_mean_runtime._parse_args(argv)
