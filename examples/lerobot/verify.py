"""Verify an exported LeRobot Dataset v3 repository in a clean process.

Runs without importing HFlow: reads ``meta/info.json``, opens every
episode and data parquet, resolves every episode's data and camera
references against the files that exist, and -- when the official
``lerobot`` package is importable in this interpreter -- also loads the
repository through ``LeRobotDataset`` and enumerates its episodes. The
exporter renumbers episodes in selection order, so a three-episode
selection is verified as ``--expect 0,1,2``.

    python examples/lerobot/verify.py ./data/lerobot_workflow/v3 --expect 0,1,2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _format_ref(template: str, **values: object) -> Path:
    try:
        return Path(template.format(**values))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"reference template {template!r} cannot format {values!r}: {error}"
        ) from None


def verify(destination: Path, expect: list[int]) -> dict:
    """Return verification facts; raise ValueError on the first defect."""
    meta_path = destination / "meta" / "info.json"
    if not meta_path.exists():
        raise ValueError(f"exported dataset has no meta/info.json: {destination}")
    info = json.loads(meta_path.read_text())
    if info.get("code") != "LeRobotDataset/v3":
        raise ValueError(f"exported dataset is not LeRobotDataset/v3: {info.get('code')}")

    expected = sorted(expect)
    if int(info.get("total_episodes", -1)) != len(expected):
        raise ValueError(
            f"info.json declares {info.get('total_episodes')} episodes, expected {len(expected)}"
        )
    video_features = [
        key
        for key, value in (info.get("features") or {}).items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    ]
    if not video_features:
        raise ValueError("info.json declares no video features")

    episode_parquets = sorted((destination / "meta" / "episodes").rglob("*.parquet"))
    data_parquets = sorted((destination / "data").rglob("*.parquet"))
    if not episode_parquets:
        raise ValueError(f"exported dataset has no episode parquets under {destination}")
    if not data_parquets:
        raise ValueError(f"exported dataset has no data parquets under {destination}")

    import duckdb

    connection = duckdb.connect()
    try:
        episode_indexes: list[int] = []
        total_frames = 0
        for episode_parquet in episode_parquets:
            quoted = str(episode_parquet).replace("'", "''")
            rows = connection.execute(f"SELECT * FROM read_parquet('{quoted}')").fetchall()
            columns = [d[0] for d in connection.description]
            for row in rows:
                record = dict(zip(columns, row, strict=True))
                episode_index = int(record["episode_index"])
                length = int(record["length"])
                if length < 1:
                    raise ValueError(f"episode {episode_index} has no frames")
                data_from = int(record.get("dataset_from_index", -1))
                data_to = int(record.get("dataset_to_index", -1))
                if data_to - data_from != length:
                    raise ValueError(
                        f"episode {episode_index} window {data_from}-{data_to} != length {length}"
                    )
                data_ref = destination / _format_ref(
                    info["data_path"],
                    chunk_index=record.get("data/chunk_index"),
                    file_index=record.get("data/file_index"),
                )
                if not data_ref.exists():
                    raise ValueError(f"episode {episode_index} references missing data {data_ref}")
                for feature in video_features:
                    video_ref = destination / _format_ref(
                        info["video_path"],
                        video_key=feature,
                        chunk_index=record.get(f"videos/{feature}/chunk_index"),
                        file_index=record.get(f"videos/{feature}/file_index"),
                    )
                    if not video_ref.exists():
                        raise ValueError(
                            f"episode {episode_index} references missing video {video_ref}"
                        )
                episode_indexes.append(episode_index)
                total_frames += length

        if sorted(set(episode_indexes)) != expected:
            raise ValueError(
                f"exported episodes {sorted(set(episode_indexes))} != expected {expected}"
            )

        data_rows = 0
        for data_parquet in data_parquets:
            quoted = str(data_parquet).replace("'", "''")
            count_row = connection.execute(
                f"SELECT COUNT(*) FROM read_parquet('{quoted}')"
            ).fetchone()
            if count_row is None:
                raise ValueError(f"count query returned no row for {data_parquet}")
            data_rows += int(count_row[0])
    finally:
        connection.close()

    if data_rows != total_frames:
        raise ValueError(f"data rows {data_rows} != declared episode frames {total_frames}")
    expected_frames = int(info.get("total_frames", -1))
    if expected_frames >= 0 and expected_frames != total_frames:
        raise ValueError(f"info.json total_frames {expected_frames} != counted {total_frames}")

    lerobot_load = _load_with_lerobot(destination, expected)
    return {
        "episodes": expected,
        "frames": total_frames,
        "video_features": video_features,
        "lerobot_load": lerobot_load,
    }


def _load_with_lerobot(destination: Path, expected: list[int]) -> str | None:
    """Load through the official package when importable; None when not installed."""
    try:
        from lerobot import LeRobotDataset  # ty: ignore[unresolved-import]
    except (ImportError, ModuleNotFoundError):
        return None
    dataset = LeRobotDataset(
        dataset_id=destination.name,
        root=str(destination.parent),
        local_files_only=True,
    )
    episodes = list(dataset.episodes)
    indexes = sorted(
        int(getattr(episode, "index", getattr(episode, "episode_index", -1)))
        for episode in episodes
    )
    if indexes != sorted(expected):
        raise ValueError(f"LeRobotDataset loaded episodes {indexes}, expected {sorted(expected)}")
    return f"LeRobotDataset loaded {len(indexes)} episode(s) (indexes {indexes})"


# _write_v3_repository rewrites exactly these data columns. Task IDs are preserved.
ROUNDTRIP_REMAP = {
    "episode_index": "position in source_episode_indexes (selection order)",  # New episode ID.
    "index": "contiguous global frame index starting at zero",  # New dataset row offset.
    "frame_index": "contiguous episode-local frame index starting at zero",  # Enumerated rows.
}


def _check_schema(actual: pa.Schema, expected: pa.Schema, context: str) -> None:
    if actual.names != expected.names:
        raise ValueError(f"{context}: data column schema {actual.names} != {expected.names}")
    for field, source_field in zip(actual, expected, strict=True):
        if not pa.schema([field]).equals(pa.schema([source_field]), check_metadata=True):
            raise ValueError(f"{context}: column {field.name} schema {field} != {source_field}")
    if not actual.equals(expected, check_metadata=True):
        raise ValueError(f"{context}: data schema metadata differs from source")


def _episode_data(root: Path, index: int, context: str) -> pa.Table:
    tables = []
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, filters=[("episode_index", "=", index)])
        if len(table):
            if tables:
                _check_schema(table.schema, tables[0].schema, context)
            tables.append(table)
    if not tables:
        raise ValueError(f"{context}: row count is zero under {root}")
    return pa.concat_tables(tables).sort_by("frame_index")


def _episode_metadata(root: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            index = row["episode_index"]
            if index in records:
                raise ValueError(f"{root.name}: duplicate episode metadata {index}")
            records[index] = row
    return records


def _tasks(root: Path) -> dict[int, str]:
    path = root / "meta/tasks.parquet"
    if path.exists():
        table = pq.read_table(path)
        metadata = json.loads((table.schema.metadata or {}).get(b"pandas", b"{}"))
        name = "task"
        if name not in table.column_names:
            name = next(
                (
                    key
                    for key in metadata.get("index_columns", [])
                    if isinstance(key, str) and key in table.column_names
                ),
                "task",
            )
        if name not in table.column_names:
            raise ValueError(f"{root.name}: task_index registry has no task strings")
        return {
            row.get("task_index", index): row[name]
            for index, row in enumerate(table.to_pylist())
            if row.get("task_index", index) is not None
        }
    path = root / "meta/tasks.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return {row.get("task_index", index): row["task"] for index, row in enumerate(rows)}
    return {}


def _scalar_equal(actual: pa.Scalar, expected: pa.Scalar) -> bool:
    """Arrow equality with matching NaNs, recursively including nested list values."""
    if actual.equals(expected):
        return True
    if not actual.is_valid or not expected.is_valid:
        return False
    if pa.types.is_floating(actual.type):
        # Arrow has no nans_equal option. Canonicalize only the NaN buffers;
        # all finite values retain Arrow's exact equality (no float tolerance).
        left, right = pa.array([actual]), pa.array([expected])
        if (
            pc.call_function("is_nan", [left])[0].as_py()
            and pc.call_function("is_nan", [right])[0].as_py()
        ):
            left = pc.call_function(
                "if_else", [pc.call_function("is_nan", [left]), pa.scalar(0, type=left.type), left]
            )
            right = pc.call_function(
                "if_else",
                [pc.call_function("is_nan", [right]), pa.scalar(0, type=right.type), right],
            )
            return left.buffers()[1] == right.buffers()[1]
    if (
        pa.types.is_list(actual.type)
        or pa.types.is_large_list(actual.type)
        or pa.types.is_fixed_size_list(actual.type)
    ):
        return len(actual.values) == len(expected.values) and all(
            _scalar_equal(a, b) for a, b in zip(actual.values, expected.values, strict=True)
        )
    if pa.types.is_struct(actual.type):
        return all(_scalar_equal(actual[field.name], expected[field.name]) for field in actual.type)
    return False


def verify_roundtrip(destination: Path, source_root: Path) -> dict:
    """Compare every selected source row and window, without importing HFlow or torch."""
    info = json.loads((destination / "meta/info.json").read_text())
    path = destination / "export-provenance.json"
    try:
        provenance = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{destination.name}: missing or invalid provenance: {error}") from None
    mapping = provenance.get("source_episode_indexes")
    if (
        not isinstance(mapping, list)
        or len(mapping) != info["total_episodes"]
        or any(type(index) is not int or index < 0 for index in mapping)
    ):
        raise ValueError(
            f"{destination.name}: provenance source_episode_indexes must match total_episodes"
        )
    dataset = provenance.get("source_repository", source_root.name)
    exported_meta, source_meta = _episode_metadata(destination), _episode_metadata(source_root)
    if set(exported_meta) != set(range(len(mapping))):
        raise ValueError(f"{dataset}: episode metadata does not match provenance")
    total_rows = 0
    for path in sorted((destination / "data").rglob("*.parquet")):
        indexes = pq.read_table(path, columns=["episode_index"])["episode_index"]
        if any(index not in exported_meta for index in indexes.to_pylist()):
            raise ValueError(f"{dataset}: unexpected episode_index in {path.name}")
        total_rows += len(indexes)
    cameras = [key for key, value in info["features"].items() if value.get("dtype") == "video"]
    exported_tasks, source_tasks = _tasks(destination), _tasks(source_root)
    offset = 0
    compared: set[str] = set()
    remapped: set[str] = set()
    dataset_schema = None
    for episode, source_index in enumerate(mapping):
        context = f"{dataset}: episode {episode} (source {source_index})"
        if source_index not in source_meta:
            raise ValueError(f"{context}: missing source episode metadata")
        actual_meta, expected_meta = exported_meta[episode], source_meta[source_index]
        for key in [
            "length",
            "tasks",
            *(
                f"videos/{camera}/{bound}"
                for camera in cameras
                for bound in ("from_timestamp", "to_timestamp")
            ),
        ]:
            if (
                key not in actual_meta
                or key not in expected_meta
                or actual_meta[key] != expected_meta[key]
            ):
                raise ValueError(
                    f"{context}: {key} {actual_meta.get(key)!r} != source {expected_meta.get(key)!r}"
                )
        actual = _episode_data(destination, episode, context)
        expected = _episode_data(source_root, source_index, context)
        if len(actual) != len(expected) or len(actual) != actual_meta["length"]:
            raise ValueError(
                f"{context}: row count {len(actual)} != source {len(expected)} or length {actual_meta['length']}"
            )
        _check_schema(actual.schema, expected.schema, context)
        if dataset_schema is not None:
            _check_schema(actual.schema, dataset_schema, context)
        dataset_schema = actual.schema
        for name in actual.column_names:
            wanted = expected[name]
            if name in ROUNDTRIP_REMAP:
                match name:
                    case "episode_index":
                        values = [episode] * len(actual)
                    case "index":
                        values = range(offset, offset + len(actual))
                    case "frame_index":
                        values = range(len(actual))
                    case _:
                        raise AssertionError(f"undocumented remap: {name}")
                wanted = pa.chunked_array([pa.array(values, type=actual[name].type)])
                remapped.add(name)
            else:
                compared.add(name)
            if not actual[name].equals(wanted):
                for frame, (value, source_value) in enumerate(
                    zip(actual[name], wanted, strict=True)
                ):
                    if not _scalar_equal(value, source_value):
                        raise ValueError(
                            f"{context}: column {name} first differing frame {frame}: {value.as_py()!r} != expected {source_value.as_py()!r}"
                        )
        if "task_index" in actual.column_names:
            if not source_tasks:
                raise ValueError(
                    f"{dataset}: source task registry (meta/tasks.parquet or meta/tasks.jsonl) "
                    f"missing under {source_root}"
                )
            for frame, task in enumerate(actual["task_index"].to_pylist()):
                if task is not None and (
                    task not in source_tasks
                    or task not in exported_tasks
                    or exported_tasks[task] != source_tasks[task]
                ):
                    raise ValueError(
                        f"{context}: column task_index first differing frame {frame}: task {task} resolves to {exported_tasks.get(task)!r} != source {source_tasks.get(task)!r}"
                    )
        offset += len(actual)
    if offset != total_rows or offset != info["total_frames"]:
        raise ValueError(
            f"{dataset}: total row count {total_rows} != compared {offset} or declared {info['total_frames']}"
        )
    return {
        "episodes": len(mapping),
        "rows": offset,
        "columns_compared": len(compared),
        "columns_remapped": len(remapped),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="exported v3 repository directory")
    parser.add_argument("--expect", required=True, help="comma-separated episode indexes")
    parser.add_argument("--source", type=Path, help="source cache for exact content comparison")
    args = parser.parse_args(argv)

    try:
        expected = sorted(int(part) for part in args.expect.split(",") if part.strip())
        facts = verify(args.destination, expected)
        roundtrip = verify_roundtrip(args.destination, args.source) if args.source else None
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(
        f"verify: {len(facts['episodes'])} episode(s) (indexes "
        f"{','.join(str(index) for index in facts['episodes'])}), "
        f"{facts['frames']} frame(s), {len(facts['video_features'])} camera feature(s)"
    )
    if facts["lerobot_load"] is None:
        print("verify: official LeRobotDataset load skipped (lerobot not importable here)")
    else:
        print(f"verify: {facts['lerobot_load']}")
    if roundtrip is not None:
        print(
            f"roundtrip: {roundtrip['episodes']} episodes, {roundtrip['rows']} rows, "
            f"{roundtrip['columns_compared']} columns compared, "
            f"{roundtrip['columns_remapped']} columns remapped"
        )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
