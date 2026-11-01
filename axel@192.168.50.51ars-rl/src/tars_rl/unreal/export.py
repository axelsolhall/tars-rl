"""Write the files the Unreal project reads: the BodySpec JSON and the scripted sequence (A57).

Run after any body or sequence change: python -m tars_rl.unreal.export
(tests fail if the committed copies under unreal/TarsRl/Content/TarsData are stale).
"""

from __future__ import annotations

from pathlib import Path

from tars_rl.body.mjcf import load_spec
from tars_rl.body.spec import BodySpec
from tars_rl.unreal.sequence import Sequence, default_sequence

REPO = Path(__file__).resolve().parents[3]
UNREAL_DATA = REPO / "unreal" / "TarsRl" / "Content" / "TarsData"
SPEC_FILE, SEQUENCE_FILE = "body_spec.json", "sequence.json"


def export_files(spec: BodySpec, seq: Sequence) -> dict[str, str]:
    return {SPEC_FILE: spec.to_json(), SEQUENCE_FILE: seq.to_json()}


def write(out: Path = UNREAL_DATA) -> list[Path]:
    spec = load_spec()
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in export_files(spec, default_sequence(spec)).items():
        (out / name).write_text(text)
        paths.append(out / name)
    return paths


if __name__ == "__main__":
    for p in write():
        print(p)
