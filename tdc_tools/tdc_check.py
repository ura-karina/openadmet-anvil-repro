from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    try:
        import tdc  # type: ignore
    except Exception as exc:
        print(f"ERROR: failed to import tdc: {exc}", file=sys.stderr)
        return 2

    report: dict[str, object] = {"tdc_file": getattr(tdc, "__file__", None)}

    # Try the modern BenchmarkGroup API (PyTDC)
    try:
        from tdc.benchmark_group import BenchmarkGroup  # type: ignore

        group = BenchmarkGroup(name="ADMET_Group")
        # This is intentionally defensive: different PyTDC versions expose different methods.
        dataset_names: list[str] = []
        for attr in ("dataset_names", "datasets", "benchmark_names", "name_list"):
            if hasattr(group, attr):
                val = getattr(group, attr)
                dataset_names = list(val() if callable(val) else val)
                if dataset_names:
                    break

        report["benchmark_group_available"] = True
        report["admet_group_dataset_names"] = dataset_names

    except Exception as exc:
        report["benchmark_group_available"] = False
        report["benchmark_group_error"] = repr(exc)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

