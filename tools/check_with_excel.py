#!/usr/bin/env python3
"""Check solver runs with the course-provided ``Solution checker.xlsx``.

The workbook has one sheet per instance (``TSPA``, ``TSPB``).  Its layout is:

    A3:D202   node table: id, x, y, cost
    F3:F102   the solution: M node ids
    F103      =F3, so the tour closes
    G/H/I     LOOKUP of x, y and cost for every solution node
    I2        SUM(I3:I102)   node cost
    J2        SUM(J3:J102)   tour length (J102 closes the cycle)
    K2        I2 + J2        objective

This script does not recompute any distances or objectives.  It copies the
workbook to a temporary file, pastes one tour at a time into ``F3:F102``, calls
``calculateAll()`` and reads I2, J2 and K2.  All arithmetic is therefore done by
the workbook's own formulas.  The only thing checked in Python is feasibility
(M distinct in-range ids), because a repeated or out-of-range id would make the
LOOKUP meaningless.

A headless LibreOffice listener is started automatically if none is reachable on
the chosen port.  The script needs the ``uno`` module that ships with
LibreOffice; on many systems the system ``python3`` provides it, otherwise use
the interpreter from ``/usr/lib/libreoffice/program/``.

Usage:

    python3 tools/check_with_excel.py results.json --out report/excel_check.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def start_listener(port: int) -> subprocess.Popen:
    """Start a private headless LibreOffice and wait for its UNO socket."""
    profile = tempfile.mkdtemp(prefix="lo_checker_profile_")
    proc = subprocess.Popen(
        [
            "soffice",
            "--headless",
            "--invisible",
            "--norestore",
            "--nologo",
            "--nodefault",
            f"--accept=socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext",
            f"-env:UserInstallation={Path(profile).as_uri()}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if port_is_open(port):
            return proc
        time.sleep(0.2)
    proc.terminate()
    raise RuntimeError(f"LibreOffice did not open port {port} within 30 s")


def connect(port: int):
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local
    )
    return resolver.resolve(
        f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext"
    )


def node_count(path: Path) -> int:
    """Number of parsable ``x;y;cost`` rows, matching the Go loader."""
    n = 0
    for line in path.read_text().splitlines():
        parts = line.strip().split(";")
        if len(parts) != 3:
            continue
        try:
            int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="?", default="results.json")
    ap.add_argument("--checker", default="Solution checker.xlsx")
    ap.add_argument("--out", default="report/excel_check.json")
    ap.add_argument("--port", type=int, default=2002)
    ap.add_argument("--instances", default="TSPA.csv,TSPB.csv")
    args = ap.parse_args()

    runs = json.load(open(args.results))
    checker = Path(args.checker).resolve()
    if not checker.exists():
        raise SystemExit(f"checker not found: {checker}")

    instance_paths = {
        Path(p.strip()).name: Path(p.strip())
        for p in args.instances.split(",")
        if p.strip()
    }
    sizes = {name: node_count(path) for name, path in instance_paths.items()}

    started = None
    if not port_is_open(args.port):
        started = start_listener(args.port)
    ctx = connect(args.port)
    desktop = ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.frame.Desktop", ctx
    )

    # Work on a copy: the provided workbook must stay untouched.
    tmpdir = tempfile.mkdtemp(prefix="excel_check_")
    book = Path(tmpdir) / checker.name
    shutil.copy(checker, book)

    hidden = PropertyValue()
    hidden.Name, hidden.Value = "Hidden", True
    doc = desktop.loadComponentFromURL(
        uno.systemPathToFileUrl(str(book)), "_blank", 0, (hidden,)
    )

    sheets = {}
    for name in sizes:
        sheet_name = Path(name).stem
        sheets[name] = doc.Sheets.getByName(sheet_name)

    records: list[dict] = []
    t0 = time.time()
    for i, run in enumerate(runs, 1):
        name = run["instance"]
        n = sizes[name]
        m = (n + 1) // 2
        tour = run["tour"]
        valid = (
            len(tour) == m
            and len(set(tour)) == m
            and all(0 <= v < n for v in tour)
        )

        sheet = sheets[name]
        sheet.getCellRangeByName("F3:F102").setDataArray(
            tuple((float(v),) for v in tour)
        )
        doc.calculateAll()

        excel_objective = int(sheet.getCellRangeByName("K2").getValue())
        excel_tour_len = int(sheet.getCellRangeByName("J2").getValue())
        excel_node_cost = int(sheet.getCellRangeByName("I2").getValue())

        records.append(
            {
                "method": run["strategy"],
                "instance": name,
                "iteration": run["iteration"],
                "valid": valid,
                "solver_objective": run["objective"],
                "excel_objective": excel_objective,
                "objective_matches": excel_objective == run["objective"],
                "solver_tour_len": run["tour_len"],
                "excel_tour_len": excel_tour_len,
                "solver_node_cost": run["node_cost"],
                "excel_node_cost": excel_node_cost,
                "components_match": (
                    excel_tour_len == run["tour_len"]
                    and excel_node_cost == run["node_cost"]
                ),
            }
        )
        if i % 400 == 0:
            print(f"  {i}/{len(runs)} checked ({time.time() - t0:.1f}s)", flush=True)

    doc.close(True)
    if started is not None:
        try:
            desktop.terminate()
        except Exception:
            started.terminate()
    shutil.rmtree(tmpdir, ignore_errors=True)

    bad = [
        r
        for r in records
        if not (r["valid"] and r["objective_matches"] and r["components_match"])
    ]
    payload = {
        "checker": checker.name,
        "runs": records,
        "totals": {"checked": len(records), "failing": len(bad)},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"runs checked: {len(records)}, failing: {len(bad)}")
    print(f"checker: {checker.name} (all objective arithmetic from the workbook)")
    print(f"wrote {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
