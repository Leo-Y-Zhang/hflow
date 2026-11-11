"""Opt-in exact pixel comparisons, run in the separate official-loader environment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("HFLOW_LEROBOT_TESTS") != "1",
        reason="set HFLOW_LEROBOT_TESTS=1 in the official-loader environment",
    ),
    pytest.mark.requires_system_ffmpeg,
]

PUBLIC_DATASETS = [
    ("lerobot/columbia_cairlab_pusht_real", "66482aa15878b5362c98620f7f60ee23ef1d5a2c"),
    ("lerobot/ucsd_pick_and_place_dataset", "5af7ee806291f5b95eafafe29eaf1d31012fcc03"),
    ("lerobot/xarm_push_medium", "50352b1fd09fd7cb440a3f09c0c0f0b362e45876"),
]


def _export_python() -> str:
    return os.environ.get("HFLOW_EXPORT_PYTHON", sys.executable)


def _compare_frames(destination: Path, source: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    loader = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    torch = pytest.importorskip("torch")
    av = pytest.importorskip("av")
    pq = pytest.importorskip("pyarrow.parquet")
    provenance = json.loads((destination / "export-provenance.json").read_text())
    info = json.loads((source / "meta/info.json").read_text())
    exported_info = json.loads((destination / "meta/info.json").read_text())
    cameras = [
        name for name, feature in exported_info["features"].items() if feature["dtype"] == "video"
    ]
    metadata = {
        row["episode_index"]: row
        for path in sorted((source / "meta/episodes").rglob("*.parquet"))
        for row in pq.read_table(path).to_pylist()
    }
    count = 0

    def refuse_source_download(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("partial source cache; use local PyAV decoding")

    for episode, source_index in enumerate(provenance["source_episode_indexes"]):
        exported = loader.LeRobotDataset(
            "local/export", root=destination, episodes=[episode], video_backend="pyav"
        )
        # Try the official source loader first, but never let it complete a
        # partial cache by downloading the remainder of a public dataset.
        with monkeypatch.context() as patch:
            patch.setattr(loader.LeRobotDatasetMetadata, "pull_from_repo", refuse_source_download)
            patch.setattr(loader.LeRobotDataset, "download", refuse_source_download)
            try:
                original = loader.LeRobotDataset(
                    "local/source", root=source, episodes=[source_index], video_backend="pyav"
                )
            except (FileNotFoundError, KeyError, ValueError) as error:
                original = None
                print(
                    f"source episode {source_index}: PyAV fallback ({type(error).__name__}: {error})"
                )
            else:
                print(f"source episode {source_index}: official LeRobotDataset with pyav")
        source_rows = []
        for path in sorted((source / "data").rglob("*.parquet")):
            source_rows.extend(
                pq.read_table(path, filters=[("episode_index", "=", source_index)]).to_pylist()
            )
        source_rows.sort(key=lambda row: row["frame_index"])
        assert len(exported) == len(source_rows)
        if original is not None:
            assert len(original) == len(source_rows)
        # Decode each source video sequentially with PyAV to avoid sharing the
        # export's seek/window selection logic. RGB CHW float32 / 255 is the
        # official PyAV backend's tensor conversion.
        decoded = {}
        if original is None:
            for camera in cameras:
                row = metadata[source_index]
                video = source / info["video_path"].format(
                    video_key=camera,
                    chunk_index=row[f"videos/{camera}/chunk_index"],
                    file_index=row[f"videos/{camera}/file_index"],
                )
                start = row[f"videos/{camera}/from_timestamp"]
                targets = [start + frame["timestamp"] for frame in source_rows]
                frames = []
                with av.open(str(video)) as container:
                    for frame in container.decode(video=0):
                        timestamp = float(frame.pts * frame.time_base)
                        # Retain only the episode window and one frame either
                        # side, enough to check the nearest timestamp at edges.
                        if timestamp >= targets[0] - 1 / info["fps"]:
                            tensor = (
                                torch.from_numpy(frame.to_ndarray(format="rgb24"))
                                .permute(2, 0, 1)
                                .float()
                                / 255
                            )
                            frames.append((timestamp, tensor))
                        if timestamp > targets[-1] + 1 / info["fps"]:
                            break
                decoded[camera] = frames
        for frame_index, row in enumerate(source_rows):
            actual = exported[frame_index]
            expected = original[frame_index] if original is not None else None
            for camera in cameras:
                if expected is not None:
                    wanted = expected[camera]
                else:
                    target = (
                        metadata[source_index][f"videos/{camera}/from_timestamp"] + row["timestamp"]
                    )
                    timestamp, wanted = min(decoded[camera], key=lambda pair: abs(pair[0] - target))
                    assert abs(timestamp - target) < exported.tolerance_s, (
                        camera,
                        frame_index,
                        target,
                        timestamp,
                    )
                if not torch.equal(actual[camera], wanted):
                    pixel = tuple((actual[camera] != wanted).nonzero()[0].tolist())
                    pytest.fail(
                        f"{provenance['source_repository']}: episode {episode} (source {source_index}), camera {camera}, frame {frame_index}, pixel {pixel}: export {actual[camera][pixel].item()} != source {wanted[pixel].item()}"
                    )
                count += 1
    return count


def test_synthetic_frame_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    subprocess.run(
        [
            _export_python(),
            "-c",
            "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "from test_lerobot_export import _export_real_video_fixture; "
            "_export_real_video_fixture(Path(sys.argv[2]), (0, 1))",
            str(Path(__file__).parent),
            str(tmp_path),
        ],
        check=True,
    )
    destination = tmp_path / "exported"
    assert json.loads((destination / "export-provenance.json").read_text())[
        "source_episode_indexes"
    ] == [1, 0]
    assert _compare_frames(destination, tmp_path / "source", monkeypatch) == 14


@pytest.mark.skipif(
    os.environ.get("HFLOW_NETWORK_TESTS") != "1",
    reason="set HFLOW_NETWORK_TESTS=1 to download pinned public datasets",
)
@pytest.mark.parametrize(("repository", "revision"), PUBLIC_DATASETS)
def test_public_frame_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repository: str, revision: str
) -> None:
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    hub = pytest.importorskip("huggingface_hub")
    started = time.monotonic()
    info_path = hub.hf_hub_download(
        repository,
        "meta/info.json",
        repo_type="dataset",
        revision=revision,
        local_dir=tmp_path / "source-info",
    )
    info = json.loads(Path(info_path).read_text())
    cameras = [name for name, feature in info["features"].items() if feature["dtype"] == "video"]
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "repository": repository,
                "revision": revision,
                "episodes": [0, 1, 2],
                "cameras": cameras,
            }
        )
    )
    data_dir = tmp_path / "workflow"
    subprocess.run(
        [
            _export_python(),
            "examples/lerobot/workflow.py",
            "--data-dir",
            str(data_dir),
            "--source-manifest",
            str(manifest),
        ],
        check=True,
    )
    destination = data_dir / "v3"
    source = data_dir / "_lerobot_cache" / revision
    provenance = json.loads((destination / "export-provenance.json").read_text())
    assert sorted(provenance["source_episode_indexes"]) == [0, 1, 2], (
        "curation must retain all three episodes for full coverage"
    )
    # Run the column verifier explicitly too; the workflow also runs it.
    subprocess.run(
        [
            _export_python(),
            "examples/lerobot/verify.py",
            str(destination),
            "--expect",
            "0,1,2",
            "--source",
            str(source),
        ],
        check=True,
    )
    frames = _compare_frames(destination, source, monkeypatch)
    downloaded = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    print(
        f"{repository}: {frames} camera frames compared, source cache {downloaded} bytes, elapsed {time.monotonic() - started:.2f}s"
    )
