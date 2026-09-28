#!/usr/bin/env python3
"""CUDA integration check for complete checkpoint restore and history rewind."""
import argparse
from pathlib import Path
import subprocess
import tempfile


def run(executable, *args, success=True):
    result = subprocess.run([str(executable), *map(str, args)],
                            capture_output=True, text=True)
    if success and result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    if not success and result.returncode == 0:
        raise AssertionError("Invalid checkpoint was accepted")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    with tempfile.TemporaryDirectory(prefix="neural-checkpoint-") as directory:
        root = Path(directory)
        continuous = root / "continuous.csv"
        resumed = root / "resumed.csv"
        checkpoint = root / "world.checkpoint.gz"
        options = ("--population", 1, "--seed", 2, "--block-brains")
        run(executable, *options, "--steps", 200, "--output", continuous)
        run(executable, *options, "--steps", 100, "--output", resumed,
            "--checkpoint", checkpoint, "--checkpoint-every", 100)
        for suffix in ("", ".behavior.csv", ".lives.csv"):
            with Path(str(resumed) + suffix).open("ab") as history:
                history.write(b"999999,discarded-after-crash\n")
        run(executable, "--resume", checkpoint, "--steps", 200)
        for suffix in ("", ".behavior.csv", ".lives.csv",
                       ".neurons.csv", ".synapses.csv", ".png"):
            expected = Path(str(continuous) + suffix).read_bytes()
            actual = Path(str(resumed) + suffix).read_bytes()
            if actual != expected:
                raise AssertionError(f"Checkpoint round trip differs: {suffix or '.csv'}")
        corrupt = root / "corrupt.gz"
        payload = checkpoint.read_bytes()
        corrupt.write_bytes(payload[:len(payload) // 2])
        before = resumed.read_bytes()
        run(executable, "--resume", corrupt, "--steps", 300,
            "--output", root / "corrupt-output.csv", success=False)
        if resumed.read_bytes() != before:
            raise AssertionError("Failed resume changed existing history")
    print("Checkpoint round trip, history rewind, and corruption rejection passed")


if __name__ == "__main__":
    main()
