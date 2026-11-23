"""Checks for the annotation quality policy (``quality_flag``).

Run from the repository root::

    python -m tests.test_quality_flag
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotators.quality import assess_annotation_quality
from annotators.rules import ANNOTATION_RULES_ACTIVITYNET

GOOD = {
    "scene_env": "outdoor",
    "scene_type": "street",
    "people_count": "crowd",
    "lighting": "natural_light",
    "time_of_day": "morning",
    "semantic_summary": "A busy street with several pedestrians crossing",
}


def assess(annotation: dict, **kwargs) -> "object":
    return assess_annotation_quality(annotation, ANNOTATION_RULES_ACTIVITYNET, **kwargs)


def test_good_annotation_is_ok() -> None:
    report = assess(dict(GOOD))
    assert report.flag == "ok", report
    assert report.reasons == (), report.reasons
    print("ok  clean annotation -> ok")


def test_unknown_enum_field_is_bad() -> None:
    annotation = dict(GOOD, time_of_day="unknown")
    report = assess(annotation)
    assert report.flag == "bad"
    assert report.reasons == ("unknown:time_of_day",), report.reasons

    # Empty / alias values count as unknown too.
    for alias in ("", "None", "N/A", "unclear"):
        assert "unknown:lighting" in assess(dict(GOOD, lighting=alias)).reasons, alias
    print("ok  unknown enum value -> bad")


def test_unknown_fields_can_be_tolerated() -> None:
    annotation = dict(GOOD, time_of_day="unknown")
    report = assess(annotation, ignore_unknown_fields=["time_of_day"])
    assert report.flag == "ok", report

    report = assess(annotation, strict_unknown=False)
    assert report.flag == "ok", report
    print("ok  unknown tolerance is configurable per field")


def test_bad_flag_value_is_configurable() -> None:
    report = assess(dict(GOOD, scene_env="unknown"), bad_flag="not_ok")
    assert report.flag == "not_ok", report

    try:
        assess(GOOD, ok_flag="same", bad_flag="same")
    except ValueError:
        pass
    else:
        raise AssertionError("equal ok/bad flags must be rejected")
    print("ok  bad flag value is configurable (e.g. not_ok)")


def test_invalid_enum_value_is_bad() -> None:
    report = assess(dict(GOOD, scene_type="street_at_night"))
    assert "invalid:scene_type" in report.reasons, report.reasons
    print("ok  out-of-vocabulary enum value -> bad")


def test_meaningless_summary_is_bad() -> None:
    cases = {
        "": "summary_empty",
        "   ": "summary_empty",
        "unknown": "summary_unknown",
        "N/A": "summary_unknown",
        "\ufffd\ufffd\ufffd": "summary_mojibake",
        "[short plain-text summary]": "summary_placeholder",
        "!!! ??? ###": "summary_no_text",
        "aaa aaa": "summary_degenerate_chars",
        "car car car": "summary_repeated_tokens",
        "car @@@ #### $$$$ %%%% !!": "summary_garbage_chars",
    }
    for summary, expected in cases.items():
        report = assess(dict(GOOD, semantic_summary=summary))
        assert expected in report.reasons, (summary, report.reasons)
        assert report.flag == "bad", summary
    print("ok  meaningless semantic_summary -> bad")


def test_meaningless_short_phrase_is_bad() -> None:
    cases = {
        "unknown scene": "summary_meaningless_phrase",
        "a video clip": "summary_meaningless_phrase",
        "the scene shows a video": "summary_meaningless_phrase",
        "some objects": "summary_meaningless_phrase",
        "nothing is visible": "summary_meaningless_phrase",
        "no information available": "summary_meaningless_phrase",
        "unknown man walking": "summary_uncertain",
        "the footage is unclear": "summary_meaningless_phrase",
        "car": "summary_too_short",
        "街道": "summary_meaningless_phrase",
    }
    for summary, expected in cases.items():
        report = assess(dict(GOOD, semantic_summary=summary))
        assert expected in report.reasons, (summary, report.reasons)
        assert report.flag == "bad", (summary, report.reasons)

    # strict_unknown=False also tolerates hedging wording in the summary.
    report = assess(
        dict(GOOD, semantic_summary="unknown man walking"), strict_unknown=False
    )
    assert report.flag == "ok", report.reasons
    print("ok  meaningless short phrases (incl. unknown vocabulary) -> bad")


def test_summary_with_text_is_ok() -> None:
    for summary in (
        "Two men playing guitar on a stage",
        "夜晚的街道上有很多行人",
        "People: 3; cars: 2",
        "Street (outdoor)",
    ):
        report = assess(dict(GOOD, semantic_summary=summary))
        assert report.flag == "ok", (summary, report.reasons)
    print("ok  informative summary -> ok")


def test_multiple_reasons_are_collected() -> None:
    report = assess(
        {
            "scene_env": "unknown",
            "scene_type": "street",
            "people_count": "unknown",
            "lighting": "dim",
            "time_of_day": "night",
            "semantic_summary": "???",
        }
    )
    assert report.flag == "bad"
    assert "unknown:scene_env" in report.reasons
    assert "unknown:people_count" in report.reasons
    assert "summary_no_text" in report.reasons
    assert report.reason_text.count(",") == 2
    print("ok  all reasons are collected and reported")


def main() -> None:
    tests = [
        test_good_annotation_is_ok,
        test_unknown_enum_field_is_bad,
        test_unknown_fields_can_be_tolerated,
        test_bad_flag_value_is_configurable,
        test_invalid_enum_value_is_bad,
        test_meaningless_summary_is_bad,
        test_meaningless_short_phrase_is_bad,
        test_summary_with_text_is_ok,
        test_multiple_reasons_are_collected,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc!r}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - failed}/{len(tests)} checks passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
