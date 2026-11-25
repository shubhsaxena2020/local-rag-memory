#!/usr/bin/env python3
"""
verify_all.py - Run all functional test packages sequentially with safety checks.

Usage:
    python work/r3-testmatrix-verifyall1/verify_all.py

Requires:
    - Project virtual environment at <project_root>/.venv/Scripts/python.exe
    - ops/local_benchmark.lock (acquired exclusively for the batch)
    - ops/watchdog.state (valid RUN timestamp <=120s, temp<85C, and actual free RAM >=8GiB before lock and before each package)

Test Suites:
    eval, judge, storage, lifecycle, retrieval, grounding, graph, embedding_bench, reranker, chaos, mcp_server
    (eval through chaos target their actual <package>/tests directory; mcp_server targets its top-level test_mcp_server.py)

Output:
    - Evidence files under <script_dir>/evidence/<run_id>/<suite>_output.txt
    - Concise summary table printed to stdout with test counts (tests run, passed, skipped, failures, errors, exit)
    - Returns 0 if all packages pass, nonzero if any fails

Note: Latency/soak/scale/migration/protocol/security matrix categories are NOT RUN.
      Fuzz, protocol, concurrency soak, crash/power-loss, corruption/recovery,
      dependency failure, 1k-100k scale, migration, time, security, mutation,
      latency/RAM benchmarks, and soak are NOT RUN unless corresponding automated suites exist.
"""
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

# -------------------------- Configuration --------------------------
def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for p in [current] + list(current.parents):
        if (p / "ops").is_dir() and (p / "mcp_server").is_dir():
            return p
    return Path(__file__).resolve().parents[2]

PROJECT_ROOT = find_project_root()
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
LOCK_FILE = PROJECT_ROOT / "ops" / "local_benchmark.lock"
WATCHDOG_FILE = PROJECT_ROOT / "ops" / "watchdog.state"
EVIDENCE_BASE = Path(__file__).resolve().parent / "evidence"

# Test suites: (suite_name, relative_path_to_target, is_directory)
# mcp_server targets its real top-level test_mcp_server.py; all other suites target each package's actual tests directory
SUITE_SPECS = [
    ("eval", "eval/tests", True),
    ("judge", "judge/tests", True),
    ("storage", "storage/tests", True),
    ("lifecycle", "lifecycle/tests", True),
    ("retrieval", "retrieval/tests", True),
    ("grounding", "grounding/tests", True),
    ("graph", "graph/tests", True),
    ("embedding_bench", "embedding_bench/tests", True),
    ("reranker", "reranker/tests", True),
    ("chaos", "chaos/tests", True),
    ("mcp_server", "mcp_server/test_mcp_server.py", False),
]
# -------------------------------------------------------------------

# Windows API for GlobalMemoryStatusEx
class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", wintypes.ULARGE_INTEGER),
    ]

def get_available_ram_gb() -> float:
    """Return available physical memory in GiB using GlobalMemoryStatusEx."""
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(stat)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        raise OSError("Failed to call GlobalMemoryStatusEx")
    return stat.ullAvailPhys / (1024 ** 3)  # bytes to GiB

def parse_watchdog(line: str):
    """Parse watchdog line: '2026-10-20T01:12:11 RUN ram=51.9 temp=61.9 cpu=30'
    Returns (timestamp_str, ram_percent, temp, cpu) or None if invalid."""
    parts = line.strip().split()
    if len(parts) < 4:
        return None
    timestamp_str = parts[0]
    if parts[1] != "RUN":
        return None
    ram_percent = None
    temp = None
    cpu = None
    for token in parts[2:]:
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        k = k.lower()
        if k == "ram":
            try:
                ram_percent = float(v)
            except ValueError:
                pass
        elif k == "temp":
            try:
                temp = float(v)
            except ValueError:
                pass
        elif k == "cpu":
            try:
                cpu = float(v)
            except ValueError:
                pass
    if timestamp_str is None or ram_percent is None or temp is None or cpu is None:
        return None
    return timestamp_str, ram_percent, temp, cpu

def check_watchdog() -> bool:
    """Read watchdog.state and validate RUN, timestamp <=120s, temp<85C.
    Returns True if valid, False otherwise."""
    if not WATCHDOG_FILE.is_file():
        print(f"ERROR: Watchdog file not found: {WATCHDOG_FILE}")
        return False
    try:
        content = WATCHDOG_FILE.read_text(encoding="utf-8").strip()
        if not content:
            print(f"ERROR: Watchdog file empty: {WATCHDOG_FILE}")
            return False
        parsed = parse_watchdog(content)
        if parsed is None:
            print(f"ERROR: Invalid watchdog format: {content}")
            return False
        timestamp_str, ram_percent, temp, cpu = parsed
        try:
            dt = datetime.fromisoformat(timestamp_str)
        except Exception:
            print(f"ERROR: Invalid timestamp in watchdog: {timestamp_str}")
            return False

        if dt.tzinfo is not None:
            now = datetime.now(timezone.utc).astimezone()
            delta_sec = (now - dt).total_seconds()
        else:
            now = datetime.now()
            delta_sec = (now - dt).total_seconds()

        if delta_sec < -5.0:
            print(f"ERROR: Watchdog timestamp in future: {timestamp_str}")
            return False
        if delta_sec > 120.0:
            print(f"ERROR: Watchdog timestamp too old: {delta_sec:.1f}s > 120s")
            return False
        if temp >= 85.0:
            print(f"ERROR: Temp too high: {temp}C (limit <85C)")
            return False
        return True
    except Exception as e:
        print(f"ERROR: Failed to parse watchdog state: {e}")
        return False

def acquire_lock(owner_token: str) -> bool:
    """Create lock file exclusively and write owner token.
    Returns True if acquired, False if already exists."""
    if LOCK_FILE.is_file():
        print(f"ERROR: Lock file already exists: {LOCK_FILE}")
        return False
    try:
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCK_FILE, "x", encoding="utf-8") as f:
            json.dump(
                {
                    "pid": os.getpid(),
                    "owner_token": owner_token,
                    "description": "verify_all functional test matrix",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
                f,
            )
        return True
    except FileExistsError:
        print(f"ERROR: Lock file already exists (race): {LOCK_FILE}")
        return False
    except Exception as e:
        print(f"ERROR: Failed to create lock file: {e}")
        return False

def release_lock(owner_token: str):
    """Remove lock file only if it still contains our token."""
    try:
        if LOCK_FILE.is_file():
            payload = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
            if payload.get("owner_token") == owner_token:
                LOCK_FILE.unlink()
            else:
                print(f"WARNING: Lock file owned by another process: {payload.get('owner_token', 'unknown')}")
    except Exception as e:
        print(f"WARNING: Failed to remove lock file: {e}")

def validate_suite_paths() -> bool:
    """Validate that all test suite target paths and python environment exist before starting the batch."""
    if not VENV_PYTHON.is_file():
        print(f"ERROR: Virtualenv python executable not found: {VENV_PYTHON}")
        return False
    all_ok = True
    for name, rel_path, is_dir in SUITE_SPECS:
        target = PROJECT_ROOT / rel_path
        if not target.exists():
            print(f"ERROR: Suite '{name}' target path does not exist: {target}")
            all_ok = False
        elif is_dir and not target.is_dir():
            print(f"ERROR: Suite '{name}' target path is not a directory: {target}")
            all_ok = False
        elif not is_dir and not target.is_file():
            print(f"ERROR: Suite '{name}' target path is not a file: {target}")
            all_ok = False
    return all_ok

def run_suite_test(name: str, rel_path: str, is_dir: bool) -> tuple[int, str, str]:
    """Run tests for a suite.
    For mcp_server, runs its real top-level test_mcp_server.py.
    For other suites, runs unittest discover targeting each package's actual test directory.
    Returns (exit_code, stdout, stderr)."""
    target = PROJECT_ROOT / rel_path
    if not target.exists():
        return 2, "", f"Suite target not found: {target}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    if is_dir:
        cmd = [
            str(VENV_PYTHON), "-m", "unittest", "discover",
            "-s", str(target),
            "-p", "test_*.py"
        ]
    else:
        # Run top-level test file directly relative to PROJECT_ROOT
        test_file_arg = target.relative_to(PROJECT_ROOT).as_posix()
        cmd = [
            str(VENV_PYTHON), "-m", "unittest",
            test_file_arg
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
            timeout=300  # 5 minute timeout per suite
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 3, "", "Test execution timed out (5 minutes)"
    except Exception as e:
        return 4, "", f"Failed to run tests: {e}"

def parse_unittest_output(stdout: str, stderr: str, exit_code: int) -> dict:
    """Parse unittest output to extract test counts.
    Returns dict with keys: tests_run, passed, skipped, failures, errors, discovery_failure, no_output."""
    combined = stdout + "\n" + stderr
    stripped = combined.strip()

    if not stripped:
        return {
            "tests_run": 0,
            "passed": 0,
            "skipped": 0,
            "failures": 0,
            "errors": 1,
            "discovery_failure": True,
            "no_output": True,
        }

    # Match 'Ran X tests' or 'Ran X test'
    m_run = re.search(r"Ran\s+(\d+)\s+tests?", combined)
    tests_run = int(m_run.group(1)) if m_run else 0

    m_fail = re.search(r"failures=(\d+)", combined)
    failures = int(m_fail.group(1)) if m_fail else 0

    m_err = re.search(r"errors=(\d+)", combined)
    errors = int(m_err.group(1)) if m_err else 0

    m_skip = re.search(r"skipped=(\d+)", combined)
    skipped = int(m_skip.group(1)) if m_skip else 0

    is_failed_test_loader = (
        "_FailedTest" in combined
        or "Failed to import test module" in combined
        or "discover failed" in combined.lower()
    )
    is_discovery_failure = tests_run == 0 or m_run is None or is_failed_test_loader
    if is_discovery_failure and failures == 0 and errors == 0:
        errors = 1

    # Defined strictly: passed = run - skipped - failures - errors
    passed = max(0, tests_run - skipped - failures - errors)

    return {
        "tests_run": tests_run,
        "passed": passed,
        "skipped": skipped,
        "failures": failures,
        "errors": errors,
        "discovery_failure": is_discovery_failure,
        "no_output": False,
    }

def main() -> int:
    # Validate all suite target paths before starting the batch
    if not validate_suite_paths():
        return 1

    # Step 1: Initial watchdog check
    if not check_watchdog():
        return 1

    # Step 2: Check actual free RAM >=8 GiB
    try:
        avail_gb = get_available_ram_gb()
        if avail_gb < 8.0:
            print(f"ERROR: Insufficient free RAM: {avail_gb:.2f} GiB (need >=8.00 GiB)")
            return 1
    except Exception as e:
        print(f"ERROR: Failed to query available RAM: {e}")
        return 1

    # Step 3: Acquire lock after preflight checks
    owner_token = f"{os.getpid()}-{int(time.time())}-{random.randint(10000, 99999)}"
    if not acquire_lock(owner_token):
        return 1

    try:
        # Generate unique run ID and create evidence directory under lock
        run_id = f"{int(time.time())}-{os.getpid()}-{random.randint(1000, 9999)}"
        evidence_dir = EVIDENCE_BASE / run_id
        if evidence_dir.exists():
            print(f"ERROR: Evidence directory already exists: {evidence_dir}")
            return 1
        evidence_dir.mkdir(parents=True, exist_ok=False)

        results = []
        any_failed = False

        for name, rel_path, is_dir in SUITE_SPECS:
            # Pre-suite watchdog check
            if not check_watchdog():
                print(f"ERROR: Watchdog check failed before suite '{name}'")
                return 1

            # Pre-suite actual RAM check
            try:
                avail_gb = get_available_ram_gb()
                if avail_gb < 8.0:
                    print(f"ERROR: Insufficient free RAM before suite '{name}': {avail_gb:.2f} GiB")
                    return 1
            except Exception as e:
                print(f"ERROR: Failed to query available RAM before suite '{name}': {e}")
                return 1

            # Verify lock ownership
            if LOCK_FILE.is_file():
                payload = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
                if payload.get("owner_token") != owner_token:
                    print(f"ERROR: Lock file owned by another process: {payload.get('owner_token', 'unknown')}")
                    return 1
            else:
                print(f"ERROR: Lock file missing before suite '{name}'")
                return 1

            evidence_file = evidence_dir / f"{name}_output.txt"
            if evidence_file.exists():
                print(f"ERROR: Evidence file already exists (refusing overwrite): {evidence_file}")
                return 1

            print(f"Running tests for suite: {name}")
            exit_code, stdout, stderr = run_suite_test(name, rel_path, is_dir)

            evidence_content = (
                f"SUITE: {name}\n"
                f"TARGET: {rel_path}\n"
                f"EXIT CODE: {exit_code}\n\n"
                f"--- STDOUT ---\n"
                f"{stdout}\n\n"
                f"--- STDERR ---\n"
                f"{stderr}\n"
            )
            evidence_file.write_text(evidence_content, encoding="utf-8")

            tc = parse_unittest_output(stdout, stderr, exit_code)
            results.append((name, exit_code, evidence_file, tc))

            if exit_code != 0:
                any_failed = True
            if tc["discovery_failure"] or tc["tests_run"] == 0:
                any_failed = True

            status_note = ""
            if tc["no_output"]:
                status_note = " [NO OUTPUT]"
            elif tc["discovery_failure"]:
                status_note = " [DISCOVERY FAIL]"

            print(
                f"  -> Exit: {exit_code}, Run: {tc['tests_run']}, "
                f"Passed: {tc['passed']}, Skipped: {tc['skipped']}, "
                f"Failures: {tc['failures']}, Errors: {tc['errors']}{status_note}"
            )

        # Step 4: Print summary table
        print("\n" + "=" * 82)
        print("TEST PACKAGE SUMMARY")
        print("=" * 82)
        header = (
            f"{'Package':<16} "
            f"{'Tests Run':<11} "
            f"{'Passed':<8} "
            f"{'Skipped':<9} "
            f"{'Failures':<10} "
            f"{'Errors':<8} "
            f"{'Exit':<6} "
            f"{'Status'}"
        )
        print(header)
        print("-" * 82)

        for name, exit_code, evidence_file, tc in results:
            if tc["no_output"]:
                status = "FAIL (no output)"
            elif tc["discovery_failure"]:
                status = "FAIL (discovery)"
            elif exit_code == 0:
                status = "PASS"
            else:
                status = "FAIL"

            row = (
                f"{name:<16} "
                f"{tc['tests_run']:<11} "
                f"{tc['passed']:<8} "
                f"{tc['skipped']:<9} "
                f"{tc['failures']:<10} "
                f"{tc['errors']:<8} "
                f"{exit_code:<6} "
                f"{status}"
            )
            print(row)

        print("-" * 82)
        total_run = sum(r[3]["tests_run"] for r in results)
        total_passed = sum(r[3]["passed"] for r in results)
        total_skipped = sum(r[3]["skipped"] for r in results)
        total_failures = sum(r[3]["failures"] for r in results)
        total_errors = sum(r[3]["errors"] for r in results)
        overall_exit = 0 if not any_failed else 1
        overall_status = "PASS" if not any_failed else "FAIL"

        total_row = (
            f"{'TOTAL':<16} "
            f"{total_run:<11} "
            f"{total_passed:<8} "
            f"{total_skipped:<9} "
            f"{total_failures:<10} "
            f"{total_errors:<8} "
            f"{overall_exit:<6} "
            f"{overall_status}"
        )
        print(total_row)
        print("=" * 82)
        print(f"Overall: {overall_status}")
        print(f"Evidence stored in: {evidence_dir.as_posix()}/\n")

        # Step 5: Inform about NOT RUN categories
        print("NOTE: The following matrix categories are NOT RUN by this script:")
        print("  functional unit tests only (the suites listed above)")
        print("  Latency, soak, scale, migration, protocol, security")
        print("  Fuzz, protocol, concurrency soak, crash/power-loss, corruption/recovery")
        print("  Dependency failure, 1k-100k scale, migration, time, security, mutation")
        print("  Latency/RAM benchmarks, and soak")
        print("Only the functional test packages listed above were executed.")
        print("(Add corresponding automated suites to include them.)")

        return 1 if any_failed else 0
    finally:
        # Step 6: Always release lock
        release_lock(owner_token)

if __name__ == "__main__":
    sys.exit(main())
