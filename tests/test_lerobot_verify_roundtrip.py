"""Content checks run without torch, media decoding, or network access."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_lerobot_export import (
    CAMS,
    _fake_corpus,
    _fake_manifest,
    _install_fake_import,
    _provenance_meta,
)

from examples.lerobot import export, verify


@pytest.fixture()
def roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    source = tmp_path / "source"
    corpus = _fake_corpus(source)
    metadata_path = source / "meta/episodes/chunk-000/file-000.parquet"
    metadata = pq.read_table(metadata_path)
    for name in metadata.column_names:
        if name.endswith(("from_timestamp", "to_timestamp")):
            metadata = metadata.set_column(
                metadata.schema.get_field_index(name), name, metadata[name].cast(pa.float64())
            )
    pq.write_table(metadata, metadata_path)
    path = source / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    for name in ("observation.state", "action"):
        field = pa.field(name, pa.list_(pa.float32(), 6), metadata={b"unit": b"radian"})
        table = table.set_column(
            table.schema.get_field_index(name), field, table[name].cast(field.type)
        )
    table = table.append_column("task_index", table["episode_index"].cast(pa.int64()))
    pq.write_table(table, path)
    pq.write_table(
        pa.table({"task_index": range(4), "task": [f"task-{i}" for i in range(4)]}),
        source / "meta/tasks.parquet",
    )
    _install_fake_import(corpus, source, monkeypatch)
    monkeypatch.setattr(export, "_fetch_task_registry", lambda *_args: None)
    manifest = _fake_manifest(
        tmp_path,
        [{"metadata_json": _provenance_meta(2, "task-2")}, {"metadata_json": _provenance_meta(0)}],
    )
    destination = tmp_path / "exported"
    export.export(destination, manifest=manifest, camera_keys=CAMS)
    return destination, source


def _replace_value(path: Path, column: str, row: int, value: object) -> None:
    table = pq.read_table(path)
    values = table[column].to_pylist()
    values[row] = value
    field = table.schema.field(column)
    table = table.set_column(
        table.schema.get_field_index(column), field, pa.array(values, type=field.type)
    )
    pq.write_table(table, path)


def test_roundtrip_untouched(roundtrip: tuple[Path, Path]) -> None:
    facts = verify.verify_roundtrip(*roundtrip)
    assert facts == {"episodes": 2, "rows": 130, "columns_compared": 4, "columns_remapped": 3}


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("float", r"episode 1 .*observation.state.*frame 59"),
        ("drop", r"episode 1 .*row count"),
        ("duplicate", r"episode 1 .*row count"),
        ("swap", r"episode 0 .*row count"),
        ("dtype", r"episode 1 .*observation.state.*schema"),
        ("field_metadata", r"episode 1 .*observation.state.*schema"),
        ("task", r"episode 1 .*task_index.*frame 59"),
        ("task_string", r"episode 1 .*task_index.*frame 0"),
        ("window", r"episode 1 .*observation.images.up/from_timestamp"),
        ("tasks", r"episode 1 .*tasks"),
        ("length", r"episode 1 .*length"),
        ("provenance", "provenance"),
        ("mapping_length", "source_episode_indexes"),
        ("index", r"episode 1 .*index.*frame 59"),
        ("frame_index", r"episode 1 .*frame_index.*frame 59"),
        ("nan", r"episode 1 .*observation.state.*frame 59"),
        ("extra_episode", "unexpected episode_index"),
        ("source_registry", "source task registry"),
    ],
)
def test_roundtrip_corruption(roundtrip: tuple[Path, Path], corruption: str, message: str) -> None:
    destination, source = roundtrip
    path = destination / "data/chunk-000/file-001.parquet"
    table = pq.read_table(path)
    match corruption:
        case "float" | "nan":
            value = float("nan") if corruption == "nan" else 999.0
            _replace_value(path, "observation.state", -1, [value] * 6)
        case "drop":
            pq.write_table(table.slice(0, len(table) - 1), path)
        case "duplicate":
            pq.write_table(pa.concat_tables([table, table.slice(len(table) - 1)]), path)
        case "swap":
            other = path.with_name("file-000.parquet")
            first = pq.read_table(other)
            # Keep exported IDs and swap the episode payloads.
            for target, payload, index in ((path, first, 1), (other, table, 0)):
                field = payload.schema.field("episode_index")
                payload = payload.set_column(
                    payload.schema.get_field_index("episode_index"),
                    field,
                    pa.array([index] * len(payload), type=field.type),
                )
                pq.write_table(payload, target)
        case "dtype" | "field_metadata":
            name = "observation.state"
            field = table.schema.field(name)
            field = (
                field.with_type(pa.list_(pa.float64(), 6))
                if corruption == "dtype"
                else field.with_metadata({b"unit": b"wrong"})
            )
            table = table.set_column(
                table.schema.get_field_index(name), field, table[name].cast(field.type)
            )
            pq.write_table(table, path)
        case "task":
            _replace_value(path, "task_index", -1, 2)
        case "task_string":
            _replace_value(destination / "meta/tasks.parquet", "task", 0, "different task")
        case "window" | "tasks" | "length":
            column, value = {
                "window": (f"videos/{CAMS[0]}/from_timestamp", 1 / 30),
                "tasks": ("tasks", ["different task"]),
                "length": ("length", 61),
            }[corruption]
            _replace_value(
                destination / "meta/episodes/chunk-000/file-000.parquet", column, 1, value
            )
        case "provenance":
            (destination / "export-provenance.json").unlink()
        case "mapping_length":
            provenance = destination / "export-provenance.json"
            record = json.loads(provenance.read_text())
            record["source_episode_indexes"] = [2]
            provenance.write_text(json.dumps(record))
        case "index" | "frame_index":
            _replace_value(path, corruption, -1, 999)
        case "extra_episode":
            _replace_value(path, "episode_index", -1, 7)
        case "source_registry":
            for name in ("tasks.parquet", "tasks.jsonl"):
                (source / "meta" / name).unlink(missing_ok=True)
    with pytest.raises(ValueError, match=message):
        verify.verify_roundtrip(*roundtrip)


def test_roundtrip_nan_and_null(roundtrip: tuple[Path, Path]) -> None:
    destination, source = roundtrip
    for path in (
        source / "data/chunk-000/file-000.parquet",
        destination / "data/chunk-000/file-001.parquet",
    ):
        _replace_value(path, "observation.state", 59, [float("nan"), None, 1.0, 2.0, 3.0, 4.0])
        _replace_value(path, "action", 58, None)
    assert verify.verify_roundtrip(*roundtrip)["rows"] == 130
    _replace_value(
        destination / "data/chunk-000/file-001.parquet",
        "observation.state",
        59,
        [float("nan"), 0.0, 1.0, 2.0, 3.0, 4.0],
    )
    with pytest.raises(ValueError, match=r"observation.state.*frame 59"):
        verify.verify_roundtrip(*roundtrip)
