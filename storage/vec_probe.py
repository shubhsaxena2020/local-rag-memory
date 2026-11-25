"""
Probe script to check if sqlite-vec extension can be loaded.
This script is intended to be run in a clean environment (e.g., a virtual environment)
to avoid interfering with the system Python.
"""

import sys
import importlib

def check_sqlite_vec():
    """Check if sqlite-vec can be imported and loaded."""
    try:
        import sqlite_vec
        print("SUCCESS: sqlite-vec imported successfully")
        # Try to load the extension in a temporary SQLite connection
        import sqlite3
        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            print("SUCCESS: sqlite-vec loaded into SQLite connection")
            return True
        finally:
            try:
                conn.enable_load_extension(False)
            except Exception:
                pass
            conn.close()
    except ImportError as e:
        print(f"INFO: sqlite-vec not installed: {e}")
        return False
    except Exception as e:
        print(f"WARNING: sqlite-vec imported but failed to load: {e}")
        return False

def check_numpy():
    """Check if numpy is available as a fallback."""
    try:
        import numpy
        print("SUCCESS: numpy available for fallback vector operations")
        return True
    except ImportError as e:
        print(f"INFO: numpy not available: {e}")
        return False

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    vec_available = check_sqlite_vec()
    numpy_available = check_numpy()
    if vec_available:
        print("RESULT: sqlite-vec is available and loadable")
    elif numpy_available:
        print("RESULT: sqlite-vec not available, but numpy is available for fallback")
    else:
        print("RESULT: Neither sqlite-vec nor numpy is available; vector operations will be limited")
