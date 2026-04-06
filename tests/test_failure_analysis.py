"""Tests for the failure analysis engine (aggregation, categorization, excerpts)."""

from __future__ import annotations

import pytest

from test_runner.agents.troubleshooter.failure_analysis import (
    CategorizedFailure,
    ExcerptConfig,
    FailureAnalysisConfig,
    FailureAnalysisReport,
    FailureGroup,
    LogExcerpt,
    analyze_failures,
    extract_failure_excerpts,
    normalize_error_pattern,
    _extract_excerpt,
    _find_important_lines,
    _build_groups,
    _select_representative_excerpt,
)
from test_runner.agents.troubleshooter.models import FailureCategory
from test_runner.models.summary import FailureDetail, TestOutcome


# ---------------------------------------------------------------------------
# Helpers — factory functions for test data
# ---------------------------------------------------------------------------


def _make_failure(
    test_id: str = "tests/test_foo.py::test_bar",
    test_name: str = "test_bar",
    outcome: TestOutcome = TestOutcome.FAILED,
    error_message: str = "AssertionError: expected 1, got 2",
    error_type: str = "AssertionError",
    stack_trace: str = "",
    stdout: str = "",
    stderr: str = "",
    log_output: str = "",
    file_path: str = "tests/test_foo.py",
    line_number: int | None = 42,
    framework: str = "pytest",
) -> FailureDetail:
    return FailureDetail(
        test_id=test_id,
        test_name=test_name,
        outcome=outcome,
        error_message=error_message,
        error_type=error_type,
        stack_trace=stack_trace,
        stdout=stdout,
        stderr=stderr,
        log_output=log_output,
        file_path=file_path,
        line_number=line_number,
        framework=framework,
    )


def _make_import_failure(
    test_id: str = "tests/test_imports.py::test_module",
    module: str = "requests",
) -> FailureDetail:
    return _make_failure(
        test_id=test_id,
        test_name="test_module",
        error_message=f"ModuleNotFoundError: No module named '{module}'",
        error_type="ModuleNotFoundError",
        stack_trace=(
            f'  File "tests/test_imports.py", line 3, in test_module\n'
            f"    import {module}\n"
            f"ModuleNotFoundError: No module named '{module}'"
        ),
        stderr=f"ERROR: No module named '{module}'",
        file_path="tests/test_imports.py",
    )


def _make_syntax_failure(
    test_id: str = "tests/test_syntax.py::test_parse",
) -> FailureDetail:
    return _make_failure(
        test_id=test_id,
        test_name="test_parse",
        error_message="SyntaxError: invalid syntax",
        error_type="SyntaxError",
        stack_trace=(
            '  File "src/broken.py", line 10\n'
            "    def foo(:\n"
            "            ^\n"
            "SyntaxError: invalid syntax"
        ),
        file_path="src/broken.py",
        line_number=10,
    )


def _make_timeout_failure(
    test_id: str = "tests/test_slow.py::test_timeout",
) -> FailureDetail:
    return _make_failure(
        test_id=test_id,
        test_name="test_timeout",
        error_message="TimeoutError: test exceeded 30s limit",
        error_type="TimeoutError",
        file_path="tests/test_slow.py",
    )


# ---------------------------------------------------------------------------
# LogExcerpt model tests
# ---------------------------------------------------------------------------


class TestLogExcerpt:
    def test_empty_excerpt(self):
        exc = LogExcerpt(source="stdout")
        assert exc.line_count == 0
        assert exc.text == ""
        assert exc.has_content is False

    def test_non_empty_excerpt(self):
        exc = LogExcerpt(source="stderr", lines=["error line 1", "error line 2"])
        assert exc.line_count == 2
        assert "error line 1" in exc.text
        assert exc.has_content is True

    def test_truncated_flag(self):
        exc = LogExcerpt(source="logs", lines=["a"], total_lines=100, truncated=True)
        assert exc.truncated is True

    def test_important_line_numbers(self):
        exc = LogExcerpt(
            source="traceback",
            lines=["line1", "line2"],
            important_line_numbers=[1, 5, 10],
        )
        assert exc.important_line_numbers == [1, 5, 10]


# ---------------------------------------------------------------------------
# find_important_lines tests
# ---------------------------------------------------------------------------


class TestFindImportantLines:
    def test_finds_error_lines(self):
        lines = ["normal output", "ERROR: something failed", "more output"]
        result = _find_important_lines(lines)
        assert 2 in result  # 1-based

    def test_finds_traceback_lines(self):
        lines = ["Traceback (most recent call last):", "  File \"test.py\""]
        result = _find_important_lines(lines)
        assert 1 in result
        assert 2 in result

    def test_finds_assertion_lines(self):
        lines = ["setup", "assert x == 1", "teardown"]
        result = _find_important_lines(lines)
        assert 2 in result

    def test_no_important_lines(self):
        lines = ["normal output", "all good", "nothing to see"]
        result = _find_important_lines(lines)
        assert result == []

    def test_finds_exception_types(self):
        lines = [
            "ModuleNotFoundError: No module named 'foo'",
            "ImportError: cannot import name 'bar'",
            "SyntaxError: invalid syntax",
            "TypeError: expected str",
            "AttributeError: has no attribute 'baz'",
        ]
        result = _find_important_lines(lines)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# extract_excerpt tests
# ---------------------------------------------------------------------------


class TestExtractExcerpt:
    def test_empty_text(self):
        exc = _extract_excerpt("", "stdout", ExcerptConfig())
        assert exc.has_content is False

    def test_whitespace_only(self):
        exc = _extract_excerpt("   \n  \n  ", "stdout", ExcerptConfig())
        assert exc.has_content is False

    def test_important_lines_with_context(self):
        lines = [f"line {i}" for i in range(20)]
        lines[10] = "ERROR: critical failure"
        text = "\n".join(lines)
        cfg = ExcerptConfig(context_lines=1, max_lines_per_source=50)
        exc = _extract_excerpt(text, "stderr", cfg)
        assert exc.has_content is True
        # Should include line 10 (the error) and lines 9, 11 (context)
        assert "ERROR: critical failure" in exc.text
        assert "line 9" in exc.text
        assert "line 11" in exc.text

    def test_tail_behavior_when_no_important_lines(self):
        lines = [f"normal line {i}" for i in range(50)]
        text = "\n".join(lines)
        cfg = ExcerptConfig(max_lines_per_source=10)
        exc = _extract_excerpt(text, "stdout", cfg)
        assert exc.has_content is True
        # Should have the last 10 lines
        assert "normal line 49" in exc.text
        assert "normal line 40" in exc.text

    def test_truncation(self):
        lines = [f"ERROR line {i}" for i in range(100)]
        text = "\n".join(lines)
        cfg = ExcerptConfig(max_lines_per_source=5, context_lines=0)
        exc = _extract_excerpt(text, "logs", cfg)
        assert exc.line_count <= 5
        assert exc.truncated is True

    def test_total_lines_count(self):
        text = "a\nb\nc\nd\ne"
        exc = _extract_excerpt(text, "stdout", ExcerptConfig())
        assert exc.total_lines == 5


# ---------------------------------------------------------------------------
# extract_failure_excerpts tests
# ---------------------------------------------------------------------------


class TestExtractFailureExcerpts:
    def test_no_log_sources(self):
        failure = _make_failure(
            stack_trace="", stdout="", stderr="", log_output=""
        )
        excerpts = extract_failure_excerpts(failure)
        assert excerpts == []

    def test_extracts_from_all_sources(self):
        failure = _make_failure(
            stack_trace="Traceback:\n  File test.py",
            stdout="test output here",
            stderr="ERROR: something went wrong",
            log_output="WARNING: low memory",
        )
        excerpts = extract_failure_excerpts(failure)
        sources = {e.source for e in excerpts}
        assert "traceback" in sources
        assert "stdout" in sources
        assert "stderr" in sources
        assert "logs" in sources

    def test_respects_total_line_budget(self):
        long_trace = "\n".join(f"trace line {i}" for i in range(100))
        long_stderr = "\n".join(f"ERROR line {i}" for i in range(100))
        failure = _make_failure(stack_trace=long_trace, stderr=long_stderr)
        cfg = ExcerptConfig(max_total_lines=20, max_lines_per_source=15)
        excerpts = extract_failure_excerpts(failure, cfg)
        total_lines = sum(e.line_count for e in excerpts)
        assert total_lines <= 20

    def test_prioritizes_traceback_first(self):
        failure = _make_failure(
            stack_trace="Traceback here",
            stderr="stderr here",
        )
        excerpts = extract_failure_excerpts(failure)
        assert excerpts[0].source == "traceback"

    def test_skips_empty_sources(self):
        failure = _make_failure(
            stack_trace="", stdout="some output", stderr="", log_output=""
        )
        excerpts = extract_failure_excerpts(failure)
        assert len(excerpts) == 1
        assert excerpts[0].source == "stdout"


# ---------------------------------------------------------------------------
# normalize_error_pattern tests
# ---------------------------------------------------------------------------


class TestNormalizeErrorPattern:
    def test_empty_message(self):
        assert normalize_error_pattern("") == ""

    def test_replaces_quoted_strings(self):
        result = normalize_error_pattern("No module named 'requests'")
        assert "'<value>'" in result
        assert "requests" not in result

    def test_replaces_numbers(self):
        result = normalize_error_pattern("expected 42, got 100")
        assert "<N>" in result
        assert "42" not in result
        assert "100" not in result

    def test_replaces_hex_addresses(self):
        result = normalize_error_pattern("object at 0x7f1234abcd")
        assert "<addr>" in result

    def test_collapses_whitespace(self):
        result = normalize_error_pattern("too   many   spaces")
        assert "  " not in result

    def test_takes_first_line_only(self):
        result = normalize_error_pattern("first line\nsecond line\nthird line")
        assert "second" not in result

    def test_similar_errors_normalize_same(self):
        err1 = "No module named 'requests'"
        err2 = "No module named 'flask'"
        assert normalize_error_pattern(err1) == normalize_error_pattern(err2)

    def test_different_errors_normalize_differently(self):
        err1 = "AssertionError: expected True"
        err2 = "ModuleNotFoundError: No module named 'foo'"
        assert normalize_error_pattern(err1) != normalize_error_pattern(err2)


# ---------------------------------------------------------------------------
# CategorizedFailure tests
# ---------------------------------------------------------------------------


class TestCategorizedFailure:
    def test_basic_properties(self):
        failure = _make_failure()
        cf = CategorizedFailure(
            failure=failure,
            category=FailureCategory.ASSERTION,
        )
        assert cf.has_excerpts is False
        assert cf.excerpt_text == ""
        assert "FAIL" in cf.summary_line
        assert "assertion" in cf.summary_line

    def test_with_excerpts(self):
        failure = _make_failure()
        exc = LogExcerpt(source="stderr", lines=["ERROR: bad"])
        cf = CategorizedFailure(
            failure=failure,
            category=FailureCategory.ASSERTION,
            excerpts=[exc],
        )
        assert cf.has_excerpts is True
        assert "stderr" in cf.excerpt_text
        assert "ERROR: bad" in cf.excerpt_text

    def test_summary_line_with_location(self):
        failure = _make_failure(file_path="test.py", line_number=10)
        cf = CategorizedFailure(
            failure=failure,
            category=FailureCategory.IMPORT_ERROR,
        )
        assert "test.py:10" in cf.summary_line

    def test_summary_line_error_outcome(self):
        failure = _make_failure(outcome=TestOutcome.ERROR)
        cf = CategorizedFailure(
            failure=failure,
            category=FailureCategory.RUNTIME,
        )
        assert "[ERROR]" in cf.summary_line


# ---------------------------------------------------------------------------
# FailureGroup tests
# ---------------------------------------------------------------------------


class TestFailureGroup:
    def _make_group(self, count: int = 3) -> FailureGroup:
        failures = []
        for i in range(count):
            f = _make_failure(
                test_id=f"test_{i}",
                file_path=f"file_{i % 2}.py",
            )
            failures.append(
                CategorizedFailure(
                    failure=f,
                    category=FailureCategory.ASSERTION,
                )
            )
        return FailureGroup(
            key="assertion",
            group_type="category",
            failures=failures,
        )

    def test_count(self):
        group = self._make_group(5)
        assert group.count == 5

    def test_test_ids(self):
        group = self._make_group(3)
        assert group.test_ids == ["test_0", "test_1", "test_2"]

    def test_affected_files_unique(self):
        group = self._make_group(4)
        files = group.affected_files
        assert len(files) == 2  # file_0.py and file_1.py

    def test_summary_line(self):
        group = self._make_group(2)
        line = group.summary_line()
        assert "category:assertion" in line
        assert "2 failure(s)" in line


# ---------------------------------------------------------------------------
# _build_groups tests
# ---------------------------------------------------------------------------


class TestBuildGroups:
    def test_groups_by_category(self):
        cfs = [
            CategorizedFailure(
                failure=_make_failure(test_id="t1"),
                category=FailureCategory.ASSERTION,
            ),
            CategorizedFailure(
                failure=_make_failure(test_id="t2"),
                category=FailureCategory.ASSERTION,
            ),
            CategorizedFailure(
                failure=_make_failure(test_id="t3"),
                category=FailureCategory.IMPORT_ERROR,
            ),
        ]
        groups = _build_groups(cfs, lambda cf: cf.category.value, "category")
        assert len(groups) == 2
        # Sorted by count descending
        assert groups[0].key == "assertion"
        assert groups[0].count == 2
        assert groups[1].key == "import_error"
        assert groups[1].count == 1

    def test_skips_empty_keys(self):
        cfs = [
            CategorizedFailure(
                failure=_make_failure(test_id="t1", file_path=""),
                category=FailureCategory.UNKNOWN,
            ),
        ]
        groups = _build_groups(cfs, lambda cf: cf.failure.file_path, "file")
        assert len(groups) == 0

    def test_min_group_size(self):
        cfs = [
            CategorizedFailure(
                failure=_make_failure(test_id="t1"),
                category=FailureCategory.ASSERTION,
            ),
            CategorizedFailure(
                failure=_make_failure(test_id="t2"),
                category=FailureCategory.IMPORT_ERROR,
            ),
        ]
        groups = _build_groups(
            cfs, lambda cf: cf.category.value, "category", min_group_size=2
        )
        assert len(groups) == 0  # Each has only 1


# ---------------------------------------------------------------------------
# analyze_failures (main entry point) tests
# ---------------------------------------------------------------------------


class TestAnalyzeFailures:
    def test_empty_failures(self):
        report = analyze_failures([])
        assert report.total_failures == 0
        assert report.has_failures is False
        assert report.by_category == []
        assert report.by_file == []

    def test_single_failure(self):
        failure = _make_failure()
        report = analyze_failures([failure])
        assert report.total_failures == 1
        assert report.has_failures is True
        assert len(report.categorized_failures) == 1
        assert report.categorized_failures[0].category == FailureCategory.ASSERTION

    def test_multiple_categories(self):
        failures = [
            _make_failure(
                test_id="t1",
                error_message="AssertionError",
                error_type="AssertionError",
            ),
            _make_import_failure(test_id="t2"),
            _make_syntax_failure(test_id="t3"),
            _make_timeout_failure(test_id="t4"),
        ]
        report = analyze_failures(failures)
        assert report.total_failures == 4
        assert report.category_count >= 3  # At least assertion, import, syntax/timeout

    def test_category_counts(self):
        failures = [
            _make_failure(test_id="t1"),
            _make_failure(test_id="t2"),
            _make_import_failure(test_id="t3"),
        ]
        report = analyze_failures(failures)
        assert report.category_counts.get("assertion", 0) == 2
        assert report.category_counts.get("import_error", 0) == 1

    def test_most_common_category(self):
        failures = [
            _make_import_failure(test_id="t1"),
            _make_import_failure(test_id="t2"),
            _make_failure(test_id="t3"),
        ]
        report = analyze_failures(failures)
        assert report.most_common_category == "import_error"

    def test_groups_by_file(self):
        failures = [
            _make_failure(test_id="t1", file_path="file_a.py"),
            _make_failure(test_id="t2", file_path="file_a.py"),
            _make_failure(test_id="t3", file_path="file_b.py"),
        ]
        report = analyze_failures(failures)
        assert report.file_count == 2
        # First group should be the one with 2 failures
        assert report.by_file[0].count == 2
        assert report.by_file[0].key == "file_a.py"

    def test_groups_by_error_type(self):
        failures = [
            _make_failure(test_id="t1", error_type="AssertionError"),
            _make_failure(test_id="t2", error_type="AssertionError"),
            _make_failure(test_id="t3", error_type="ValueError"),
        ]
        report = analyze_failures(failures)
        assert len(report.by_error_type) >= 2

    def test_groups_by_error_pattern(self):
        failures = [
            _make_failure(
                test_id="t1",
                error_message="No module named 'requests'",
                error_type="ModuleNotFoundError",
            ),
            _make_failure(
                test_id="t2",
                error_message="No module named 'flask'",
                error_type="ModuleNotFoundError",
            ),
        ]
        report = analyze_failures(failures)
        # These should normalize to the same pattern
        assert len(report.by_error_pattern) >= 1
        # The pattern group should have 2 failures
        patterns_with_2 = [g for g in report.by_error_pattern if g.count == 2]
        assert len(patterns_with_2) == 1

    def test_budget_cap(self):
        failures = [_make_failure(test_id=f"t{i}") for i in range(100)]
        cfg = FailureAnalysisConfig(max_failures=5)
        report = analyze_failures(failures, cfg)
        assert report.total_failures == 5
        assert any("Budget cap" in n for n in report.analysis_notes)

    def test_dominant_category_note(self):
        # 4 out of 5 are assertions — should trigger dominant note
        failures = [
            _make_failure(test_id=f"t{i}") for i in range(4)
        ] + [_make_import_failure(test_id="t_import")]
        report = analyze_failures(failures)
        assert any("Dominant" in n for n in report.analysis_notes)

    def test_hotspot_note(self):
        failures = [
            _make_failure(test_id=f"t{i}", file_path="hotspot.py")
            for i in range(3)
        ]
        report = analyze_failures(failures)
        assert any("hotspot" in n.lower() for n in report.analysis_notes)

    def test_recurring_pattern_note(self):
        failures = [
            _make_failure(
                test_id=f"t{i}",
                error_message="No module named 'missing'",
                error_type="ModuleNotFoundError",
            )
            for i in range(3)
        ]
        report = analyze_failures(failures)
        assert any("Recurring" in n for n in report.analysis_notes)

    def test_excerpts_included(self):
        failure = _make_failure(
            stderr="ERROR: assertion failed\nexpected 1 got 2",
            stack_trace="Traceback:\n  File test.py, line 42",
        )
        report = analyze_failures([failure])
        cf = report.categorized_failures[0]
        assert cf.has_excerpts is True
        sources = {e.source for e in cf.excerpts}
        assert "traceback" in sources or "stderr" in sources

    def test_pattern_grouping_disabled(self):
        failures = [_make_failure(test_id="t1")]
        cfg = FailureAnalysisConfig(group_by_pattern=False)
        report = analyze_failures(failures, cfg)
        assert report.by_error_pattern == []

    def test_get_category_group(self):
        failures = [_make_failure(), _make_import_failure()]
        report = analyze_failures(failures)
        group = report.get_category_group(FailureCategory.ASSERTION)
        assert group is not None
        assert group.key == "assertion"

    def test_get_category_group_missing(self):
        report = analyze_failures([_make_failure()])
        assert report.get_category_group(FailureCategory.TIMEOUT) is None

    def test_get_file_group(self):
        failures = [_make_failure(file_path="foo.py")]
        report = analyze_failures(failures)
        group = report.get_file_group("foo.py")
        assert group is not None
        assert group.count == 1

    def test_get_file_group_missing(self):
        report = analyze_failures([_make_failure()])
        assert report.get_file_group("nonexistent.py") is None


# ---------------------------------------------------------------------------
# FailureAnalysisReport output tests
# ---------------------------------------------------------------------------


class TestFailureAnalysisReportOutput:
    def test_summary_lines(self):
        failures = [_make_failure(), _make_import_failure()]
        report = analyze_failures(failures)
        lines = report.summary_lines()
        assert len(lines) > 0
        assert "2 failure(s)" in lines[0]

    def test_to_report_dict(self):
        failures = [_make_failure(), _make_import_failure()]
        report = analyze_failures(failures)
        d = report.to_report_dict()
        assert d["total_failures"] == 2
        assert "categories" in d
        assert "files" in d
        assert "failures" in d
        assert len(d["failures"]) == 2
        assert d["failures"][0]["test_id"] == "tests/test_foo.py::test_bar"

    def test_report_dict_has_category_info(self):
        failures = [_make_failure()]
        report = analyze_failures(failures)
        d = report.to_report_dict()
        assert len(d["categories"]) >= 1
        cat = d["categories"][0]
        assert "category" in cat
        assert "count" in cat
        assert "test_ids" in cat

    def test_report_dict_has_excerpt_info(self):
        failure = _make_failure(stderr="ERROR: fail")
        report = analyze_failures([failure])
        d = report.to_report_dict()
        assert d["failures"][0]["has_excerpts"] is True
        assert len(d["failures"][0]["excerpt_sources"]) > 0


# ---------------------------------------------------------------------------
# select_representative_excerpt tests
# ---------------------------------------------------------------------------


class TestSelectRepresentativeExcerpt:
    def test_selects_longest(self):
        short = CategorizedFailure(
            failure=_make_failure(test_id="t1"),
            category=FailureCategory.ASSERTION,
            excerpts=[LogExcerpt(source="stderr", lines=["short"])],
        )
        long = CategorizedFailure(
            failure=_make_failure(test_id="t2"),
            category=FailureCategory.ASSERTION,
            excerpts=[
                LogExcerpt(source="stderr", lines=["long " * 20]),
                LogExcerpt(source="traceback", lines=["trace " * 10]),
            ],
        )
        result = _select_representative_excerpt([short, long])
        assert "long" in result
        assert "trace" in result

    def test_empty_group(self):
        result = _select_representative_excerpt([])
        assert result == ""


# ---------------------------------------------------------------------------
# Integration: imports from __init__
# ---------------------------------------------------------------------------


class TestImportsFromInit:
    def test_analyze_failures_importable(self):
        from test_runner.agents.troubleshooter import analyze_failures as af
        assert af is not None

    def test_models_importable(self):
        from test_runner.agents.troubleshooter import (
            CategorizedFailure,
            ExcerptConfig,
            FailureAnalysisConfig,
            FailureAnalysisReport,
            FailureGroup,
            LogExcerpt,
        )
        assert all([
            CategorizedFailure,
            ExcerptConfig,
            FailureAnalysisConfig,
            FailureAnalysisReport,
            FailureGroup,
            LogExcerpt,
        ])

    def test_in_all_exports(self):
        import test_runner.agents.troubleshooter as ts
        expected = [
            "analyze_failures",
            "CategorizedFailure",
            "ExcerptConfig",
            "FailureAnalysisConfig",
            "FailureAnalysisReport",
            "FailureGroup",
            "LogExcerpt",
            "extract_failure_excerpts",
            "normalize_error_pattern",
        ]
        for name in expected:
            assert name in ts.__all__, f"{name} not in __all__"
