"""Reusable quality-control pipeline for Match Charting Project data."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

MATCH_COLUMNS = {"match_id", "Date", "Player 1", "Player 2"}
POINT_COLUMNS = {"match_id", "Pt", "Svr", "1st", "2nd", "PtWinner"}
LEDGER_COLUMNS = ["reason", "scope", "match_id", "Pt", "detail"]
WARNING_COLUMNS = ["reason", "match_id", "value", "detail"]

FH = set("frvhjulo")
BH = set("bszikymp")
LABELS = ["f1", "f2", "f3", "b1", "b2", "b3"]
SHOT = re.compile(r"([fbrsvzopuylmhijktq])[+\-=;^]*([0-3])?[789]?[nwdxe!g]*[*#@C]?")
SERVE_ONLY = re.compile(r"^c*[0456]?\+?[nwdxegV!]+$")
SERVE_PREFIX = re.compile(r"^c*[0456]?\+?")
TERMINAL_SERVE = re.compile(r"^c*[0456]?\+?(?:[nwdxegV!]+)?[*#@C]$")


@dataclass
class QCResult:
    """Clean tables and an auditable record of exclusions and warnings."""

    match_meta: pd.DataFrame
    player_points: pd.DataFrame
    exclusions: pd.DataFrame
    warnings: pd.DataFrame
    player: str

    @property
    def exclusion_summary(self) -> pd.DataFrame:
        if self.exclusions.empty:
            return pd.DataFrame(columns=["reason", "scope", "rows"])
        return (
            self.exclusions.groupby(["reason", "scope"], dropna=False)
            .size()
            .rename("rows")
            .reset_index()
            .sort_values(["scope", "reason"], ignore_index=True)
        )


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


def _name_token(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _is_terminal_serve(value: object) -> bool:
    return isinstance(value, str) and bool(TERMINAL_SERVE.fullmatch(value.strip()))


def rally(first: object, second: object) -> list[str | None]:
    """Parse the six analysis buckets using the same notation as the PoC."""

    code = first.strip() if isinstance(first, str) else ""
    if SERVE_ONLY.fullmatch(code):
        code = second.strip() if isinstance(second, str) else ""
    code = SERVE_PREFIX.sub("", code)
    shots: list[str | None] = []
    for stroke, direction in SHOT.findall(code):
        hand = "f" if stroke in FH else "b" if stroke in BH else None
        shots.append(
            hand + direction if hand and direction in {"1", "2", "3"} else None
        )
    return shots


def _eligible_transitions(
    first: object, second: object, server: int, player_side: int
) -> tuple[int, int, list[tuple[str, str]]]:
    shots = rally(first, second)
    hitter = 3 - server
    hitters = []
    for _ in shots:
        hitters.append(hitter)
        hitter = 3 - hitter
    opponent = 3 - player_side
    candidates = []
    eligible = []
    for index in range(len(shots) - 1):
        if hitters[index] == opponent and hitters[index + 1] == player_side:
            pair = (shots[index], shots[index + 1])
            candidates.append(pair)
            if pair[0] is not None and pair[1] is not None:
                eligible.append((pair[0], pair[1]))
    return len(candidates), len(eligible), eligible


def load_raw_data(
    matches_path: str | Path, point_paths: Iterable[str | Path]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load source CSVs with stable import behavior and source provenance."""

    matches = pd.read_csv(matches_path, low_memory=False)
    _require_columns(matches, MATCH_COLUMNS, str(matches_path))

    point_frames = []
    for point_path in point_paths:
        frame = pd.read_csv(point_path, low_memory=False)
        _require_columns(frame, POINT_COLUMNS, str(point_path))
        frame["_source_file"] = Path(point_path).name
        point_frames.append(frame)
    if not point_frames:
        raise ValueError("At least one point CSV is required")
    return matches, pd.concat(point_frames, ignore_index=True)


def run_qc(
    matches: pd.DataFrame,
    points: pd.DataFrame,
    player: str,
    *,
    earliest_match_date: str | date | None = None,
    future_date_buffer_days: int = 30,
    coverage_warning_threshold: float = 0.80,
) -> QCResult:
    """Validate and clean data for one player.

    Tier-A point defects drop their entire match when a match id is available.
    Orphan rows are the exception: they can only be dropped at row level. Metadata
    defects drop a match, while parse/outcome defects drop only the affected point.
    """

    _require_columns(matches, MATCH_COLUMNS, "matches")
    _require_columns(points, POINT_COLUMNS, "points")
    if not player.strip():
        raise ValueError("player must not be blank")
    if not 0 <= coverage_warning_threshold <= 1:
        raise ValueError("coverage_warning_threshold must be between 0 and 1")

    matches = matches.copy()
    points = points.copy()
    ledger: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []

    def log(
        reason: str,
        scope: str,
        match_id: object = pd.NA,
        point: object = pd.NA,
        detail: str = "",
    ) -> None:
        ledger.append(
            {
                "reason": reason,
                "scope": scope,
                "match_id": match_id,
                "Pt": point,
                "detail": detail,
            }
        )

    # A2: rows without a match key cannot be attributed to a match.
    blank_match_points = _blank(points["match_id"])
    for row in points.loc[blank_match_points, ["Pt"]].itertuples(index=False):
        log("A2_blank_match_id", "point", point=row.Pt)
    points = points.loc[~blank_match_points].copy()
    blank_match_rows = _blank(matches["match_id"])
    for _ in matches.loc[blank_match_rows].itertuples(index=False):
        log("A2_blank_match_id", "match")
    matches = matches.loc[~blank_match_rows].copy()

    match_drop_reasons: dict[object, set[str]] = {}

    def mark_matches(ids: Iterable[object], reason: str) -> None:
        for match_id in set(ids):
            match_drop_reasons.setdefault(match_id, set()).add(reason)

    # A6: conflicting metadata for one match id.
    metadata_cols = ["Date", "Player 1", "Player 2"]
    metadata_conflicts = (
        matches.groupby("match_id", dropna=False)[metadata_cols]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    mark_matches(
        metadata_conflicts[metadata_conflicts].index, "A6_conflicting_match_metadata"
    )
    matches = matches.drop_duplicates("match_id", keep="first")

    player_cf = player.casefold()
    p1 = matches["Player 1"].fillna("").astype(str).str.casefold().eq(player_cf)
    p2 = matches["Player 2"].fillna("").astype(str).str.casefold().eq(player_cf)
    focus_mask = p1 ^ p2
    # "Neither" is the expected state for most corpus rows, not an exclusion.
    for row in matches.loc[p1 & p2, ["match_id"]].itertuples(index=False):
        log("B2_player_not_on_exactly_one_side", "match", row.match_id)
    player_matches = matches.loc[focus_mask].copy()
    player_matches["player_side"] = p1.loc[focus_mask].map({True: 1, False: 2})

    # B1: all three normalized names should be represented in the documented id.
    def slug_is_consistent(row: pd.Series) -> bool:
        slug = _name_token(row["match_id"])
        return all(
            _name_token(row[column]) in slug for column in ("Player 1", "Player 2")
        )

    slug_ok = player_matches.apply(slug_is_consistent, axis=1)
    mark_matches(
        player_matches.loc[~slug_ok, "match_id"], "B1_player_match_id_mismatch"
    )

    # B4/B5: parse once and keep a normalized date for downstream splits.
    parsed_dates = pd.to_datetime(
        player_matches["Date"].astype("string").str.replace(".", "-", regex=False),
        errors="coerce",
    )
    mark_matches(player_matches.loc[parsed_dates.isna(), "match_id"], "B4_invalid_date")
    if earliest_match_date is not None:
        minimum = pd.Timestamp(earliest_match_date)
        mark_matches(
            player_matches.loc[parsed_dates < minimum, "match_id"],
            "B5_date_before_minimum",
        )
    maximum = pd.Timestamp(date.today() + timedelta(days=future_date_buffer_days))
    mark_matches(
        player_matches.loc[parsed_dates > maximum, "match_id"], "B5_future_date"
    )

    # A8: point rows not represented in any metadata row are row-level orphans.
    known_ids = set(matches["match_id"])
    orphan = ~points["match_id"].isin(known_ids)
    for row in points.loc[orphan, ["match_id", "Pt"]].itertuples(index=False):
        log("A8_orphan_point", "point", row.match_id, row.Pt)
    points = points.loc[~orphan].copy()

    # D1/A7: identical duplicate point rows are corpus overlap; conflicts drop match.
    key = ["match_id", "Pt"]
    duplicate_mask = points.duplicated(key, keep=False)
    comparison_cols = ["1st", "2nd", "Svr", "PtWinner"]
    duplicate_groups = points.loc[duplicate_mask].groupby(key, dropna=False)
    conflicting_keys = []
    for point_key, group in duplicate_groups:
        if group[comparison_cols].nunique(dropna=False).gt(1).any():
            conflicting_keys.append(point_key)
            mark_matches([point_key[0]], "A7_conflicting_duplicate_point")
        else:
            for row in group.iloc[1:].itertuples(index=False):
                log("D1_identical_cross_file_duplicate", "point", row.match_id, row.Pt)
    points = points.drop_duplicates(key, keep="first")

    # A2-A5: any invalid Tier-A point invalidates its charted match.
    points["_Pt_numeric"] = pd.to_numeric(points["Pt"], errors="coerce")
    points["_Svr_numeric"] = pd.to_numeric(points["Svr"], errors="coerce")
    tier_a_masks = {
        "A2_blank_point_number": _blank(points["Pt"]),
        "A3_non_numeric_point_or_server": points["_Pt_numeric"].isna()
        | points["_Svr_numeric"].isna(),
        "A4_invalid_server": ~points["_Svr_numeric"].isin([1, 2]),
        "A5_empty_rally": _blank(points["1st"]) & _blank(points["2nd"]),
    }
    for reason, mask in tier_a_masks.items():
        mark_matches(points.loc[mask, "match_id"], reason)

    # Materialize all match drops in the ledger before removing rows.
    for match_id, reasons in match_drop_reasons.items():
        for reason in sorted(reasons):
            log(reason, "match", match_id)
    dropped_match_ids = set(match_drop_reasons)
    player_matches = player_matches.loc[
        ~player_matches["match_id"].isin(dropped_match_ids)
    ].copy()
    points = points.loc[~points["match_id"].isin(dropped_match_ids)].copy()

    match_meta = player_matches[["match_id", "Date", "player_side"]].copy()
    player_points = points.merge(
        match_meta, on="match_id", how="inner", validate="many_to_one"
    )

    # A9: metadata-only focus-player matches are retained in the ledger, not output.
    represented = set(player_points["match_id"])
    hollow_ids = set(match_meta["match_id"]) - represented
    for match_id in sorted(hollow_ids):
        log("A9_hollow_match", "match", match_id)
    match_meta = match_meta.loc[~match_meta["match_id"].isin(hollow_ids)].copy()

    # C1/C2/C6: parser and outcome defects are isolated to one point.
    first = player_points["1st"].fillna("").astype(str).str.strip()
    second_blank = _blank(player_points["2nd"])
    c1 = (
        first.map(lambda value: bool(SERVE_ONLY.fullmatch(value)))
        .astype(bool)
        .reindex(player_points.index, fill_value=False)
        & second_blank
    )
    c1 &= ~first.map(_is_terminal_serve).astype(bool)

    parsed_shots = [
        rally(first_code, second_code)
        for first_code, second_code in player_points[["1st", "2nd"]].itertuples(
            index=False, name=None
        )
    ]
    nonempty_code = ~(_blank(player_points["1st"]) & _blank(player_points["2nd"]))
    first_serve_only = first.map(
        lambda value: bool(SERVE_ONLY.fullmatch(value))
    ).astype(bool)
    second_text = player_points["2nd"].fillna("").astype(str).str.strip()
    second_serve_only = second_text.map(
        lambda value: bool(SERVE_ONLY.fullmatch(value))
    ).astype(bool)
    terminal = (
        player_points["1st"].map(_is_terminal_serve).astype(bool)
        | player_points["2nd"].map(_is_terminal_serve).astype(bool)
        | (first_serve_only & second_serve_only)
    )
    c2 = pd.Series([not shots for shots in parsed_shots], index=player_points.index)
    c2 &= nonempty_code & ~terminal
    winner = pd.to_numeric(player_points["PtWinner"], errors="coerce")
    c6 = ~winner.isin([1, 2])

    point_drop = c1 | c2 | c6
    for reason, mask in {
        "C1_missing_second_serve_record": c1,
        "C2_unparseable_rally": c2,
        "C6_invalid_point_winner": c6,
    }.items():
        for row in player_points.loc[mask, ["match_id", "Pt"]].itertuples(index=False):
            log(reason, "point", row.match_id, row.Pt)
    player_points = player_points.loc[~point_drop].copy()

    # C4/C5 are post-extraction invariants. Repeated shot pairs within one rally are
    # valid; only duplicate transition identities (including ordinal) are invalid.
    transition_rows = []
    coverage_rows = []
    transition_columns = [
        "match_id",
        "Pt",
        "1st",
        "2nd",
        "_Svr_numeric",
        "player_side",
    ]
    for match_id, point, first_code, second_code, server, side in player_points[
        transition_columns
    ].itertuples(index=False, name=None):
        candidate_n, eligible_n, eligible = _eligible_transitions(
            first_code,
            second_code,
            int(server),
            int(side),
        )
        coverage_rows.append((match_id, candidate_n, eligible_n))
        for transition_number, pair in enumerate(eligible, start=1):
            transition_rows.append(
                (
                    match_id,
                    point,
                    transition_number,
                    pair[0],
                    pair[1],
                )
            )
    transitions = pd.DataFrame(
        transition_rows,
        columns=["match_id", "Pt", "transition_number", "incoming", "response"],
    )
    if not transitions.empty:
        transition_key = ["match_id", "Pt", "transition_number"]
        duplicate_transitions = transitions.duplicated(transition_key, keep=False)
        if duplicate_transitions.any():
            bad_points = transitions.loc[
                duplicate_transitions, ["match_id", "Pt"]
            ].drop_duplicates()
            for row in bad_points.itertuples(index=False):
                log("C5_duplicate_transition_identity", "point", row.match_id, row.Pt)
            bad_keys = pd.MultiIndex.from_frame(bad_points)
            point_keys = pd.MultiIndex.from_frame(player_points[["match_id", "Pt"]])
            player_points = player_points.loc[~point_keys.isin(bad_keys)].copy()
        contiguous = transitions.groupby(["match_id", "Pt"])["transition_number"].apply(
            lambda values: list(values) == list(range(1, len(values) + 1))
        )
        for match_id, point in contiguous[~contiguous].index:
            log("C4_noncontiguous_transition_numbers", "point", match_id, point)

    # E1/E2 warnings are informational only.
    coverage = pd.DataFrame(
        coverage_rows, columns=["match_id", "candidates", "eligible"]
    )
    if coverage.empty:
        match_coverage = pd.DataFrame(
            {"candidates": 0, "eligible": 0}, index=match_meta["match_id"]
        )
    else:
        match_coverage = coverage.groupby("match_id")[["candidates", "eligible"]].sum()
        match_coverage = match_coverage.reindex(match_meta["match_id"], fill_value=0)
    for match_id, row in match_coverage.iterrows():
        if row["eligible"] == 0:
            warning_rows.append(
                {
                    "reason": "E1_zero_eligible_transitions",
                    "match_id": match_id,
                    "value": 0,
                    "detail": "",
                }
            )
        ratio = (
            row["eligible"] / row["candidates"] if row["candidates"] else float("nan")
        )
        if pd.notna(ratio) and ratio < coverage_warning_threshold:
            warning_rows.append(
                {
                    "reason": "E2_low_transition_coverage",
                    "match_id": match_id,
                    "value": ratio,
                    "detail": f"threshold={coverage_warning_threshold:.3f}",
                }
            )

    player_points = player_points.drop(
        columns=["_Pt_numeric", "_Svr_numeric"], errors="ignore"
    ).reset_index(drop=True)
    match_meta = match_meta.reset_index(drop=True)
    return QCResult(
        match_meta=match_meta,
        player_points=player_points,
        exclusions=pd.DataFrame(ledger, columns=LEDGER_COLUMNS),
        warnings=pd.DataFrame(warning_rows, columns=WARNING_COLUMNS),
        player=player,
    )


def run_qc_from_csv(
    matches_path: str | Path,
    point_paths: Iterable[str | Path],
    player: str,
    **kwargs: object,
) -> QCResult:
    matches, points = load_raw_data(matches_path, point_paths)
    return run_qc(matches, points, player, **kwargs)


def player_slug(player: str) -> str:
    return _name_token(player)


def save_artifacts(
    result: QCResult,
    output_dir: str | Path,
    *,
    source_paths: Iterable[str | Path] = (),
) -> Path:
    """Write portable CSV artifacts and return the manifest path."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.match_meta.to_csv(output / "match_meta.csv", index=False)
    result.player_points.to_csv(output / "player_points.csv", index=False)
    result.exclusions.to_csv(output / "qc_exclusions.csv", index=False)
    result.warnings.to_csv(output / "qc_warnings.csv", index=False)
    manifest = {
        "player": result.player,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "sources": [str(path) for path in source_paths],
        "row_counts": {
            "matches": len(result.match_meta),
            "points": len(result.player_points),
            "exclusions": len(result.exclusions),
            "warnings": len(result.warnings),
        },
        "files": {
            "match_meta": "match_meta.csv",
            "player_points": "player_points.csv",
            "exclusions": "qc_exclusions.csv",
            "warnings": "qc_warnings.csv",
        },
    }
    manifest_path = output / "qc_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def load_artifacts(output_dir: str | Path) -> QCResult:
    """Load a previously saved QC snapshot."""

    output = Path(output_dir)
    manifest = json.loads((output / "qc_manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    return QCResult(
        match_meta=pd.read_csv(output / files["match_meta"], low_memory=False),
        player_points=pd.read_csv(output / files["player_points"], low_memory=False),
        exclusions=pd.read_csv(output / files["exclusions"], low_memory=False),
        warnings=pd.read_csv(output / files["warnings"], low_memory=False),
        player=manifest["player"],
    )
