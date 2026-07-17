import math
import queue
import subprocess
import sys
import threading

import pytest

from benchmarks import ab_gate


def test_worker_timeout_is_one_deadline_despite_chatter():
    script = (
        "import time\n"
        "while True:\n"
        " print('noise', flush=True)\n"
        " time.sleep(0.001)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE,
        text=True,
    )
    proc.output_queue = queue.Queue()

    def read_output():
        for line in proc.stdout:
            proc.output_queue.put(line)
        proc.output_queue.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    with pytest.raises(SystemExit, match="timed out"):
        ab_gate.read_reply(proc, "@READY", "test", 0.05)
    assert proc.poll() is not None


def meta_line(key="fixed", **overrides):
    values = {
        "regs": 37,
        "spill_store_bytes": 0,
        "spill_load_bytes": 0,
        "shared": 0,
        "dynshared": 32768,
        "const": 1112,
        "blocks_per_sm": 3,
        "sms": 48,
        "blocksize": 32,
        "runs_per_block": 32,
        "runs": 1024,
        "chunks": 5,
    }
    values.update(overrides)
    fields = " ".join(f"{name}={values[name]}" for name in values)
    return f"@META {key} {fields}"


def test_parse_meta_requires_exact_schema():
    key, meta = ab_gate.parse_meta(meta_line())

    assert key == "fixed"
    assert tuple(meta) == ab_gate.META_FIELDS
    assert meta["spill_store_bytes"] == 0

    with pytest.raises(ValueError, match="schema"):
        ab_gate.parse_meta(
            meta_line().replace("regs=37", "registers=37")
        )
    with pytest.raises(ValueError, match="positive"):
        ab_gate.parse_meta(meta_line(chunks=0))


def test_parse_times_requires_exact_positive_finite_samples():
    kernel, wall = ab_gate.parse_times(
        "@TIMES fixed kernel 1.25 1.5 wall 2.5 3.0",
        "fixed",
        2,
    )

    assert kernel == [1.25, 1.5]
    assert wall == [2.5, 3.0]

    invalid = (
        "@TIMES fixed kernel 1.0 wall 2.0",
        "@TIMES fixed kernel 0 1.0 wall 2.0 3.0",
        "@TIMES fixed kernel nan 1.0 wall 2.0 3.0",
        "@TIMES other kernel 1.0 2.0 wall 3.0 4.0",
    )
    for line in invalid:
        with pytest.raises(ValueError):
            ab_gate.parse_times(line, "fixed", 2)


def test_classify_deltas_marks_boundary_disagreement():
    delta, verdict, distrust = ab_gate.classify_deltas(
        [0.1, 0.2, 0.8, 0.9], 0.5
    )

    assert math.isclose(delta, 0.5)
    assert verdict == "ok"
    assert distrust

    delta, verdict, distrust = ab_gate.classify_deltas(
        [0.6, 0.7], 0.5
    )
    assert math.isclose(delta, 0.65)
    assert verdict == "REGRESSION"
    assert not distrust

    for invalid in ([], [math.nan], [math.inf], [-math.inf]):
        with pytest.raises(ValueError, match="finite and nonempty"):
            ab_gate.classify_deltas(invalid, 0.5)


@pytest.mark.parametrize(
    "argv",
    (
        ["--block-solves", "0"],
        ["--threshold", "nan"],
        ["--wall-threshold", "inf"],
        ["--chunked-proportion", "0"],
        ["--gap", "2", "1"],
        ["--worker-timeout", "0"],
    ),
)
def test_cli_rejects_invalid_numeric_domains(argv):
    with pytest.raises(SystemExit):
        ab_gate.parse_args(argv)
