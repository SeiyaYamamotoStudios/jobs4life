"""Unit tests for the over-claim / over-flag arithmetic (jfl_evals.scoring).

Pure functions, hand-built `ItemResult` lists -- no gate call, no dataset, no
Inspect. The degenerate cases matter more than the typical ones here: a scorer
that cannot tell "flags everything" and "passes everything" apart from a good
gate is worse than no scorer at all.
"""

from __future__ import annotations

from jfl_evals.scoring import (
    ItemResult,
    SentenceKind,
    Verdict,
    aggregate,
    is_over_claim,
    is_over_flag,
    per_expected_breakdown,
)


def _result(
    item_id: str,
    expected: Verdict,
    actual: Verdict | None,
    kind: SentenceKind | None = "claim",
) -> ItemResult:
    return ItemResult(item_id=item_id, expected=expected, actual=actual, kind=kind)


class TestIsOverClaim:
    def test_supported_when_expected_unsupported_is_an_over_claim(self) -> None:
        assert is_over_claim("unsupported", "supported") is True

    def test_supported_when_expected_review_is_an_over_claim(self) -> None:
        assert is_over_claim("review", "supported") is True

    def test_supported_when_expected_supported_is_not_an_over_claim(self) -> None:
        assert is_over_claim("supported", "supported") is False

    def test_unsupported_verdict_is_never_an_over_claim(self) -> None:
        assert is_over_claim("supported", "unsupported") is False
        assert is_over_claim("review", "unsupported") is False

    def test_review_verdict_is_never_an_over_claim(self) -> None:
        assert is_over_claim("unsupported", "review") is False


class TestIsOverFlag:
    def test_unsupported_when_expected_supported_is_an_over_flag(self) -> None:
        assert is_over_flag("supported", "unsupported") is True

    def test_unsupported_when_expected_unsupported_is_not_an_over_flag(self) -> None:
        assert is_over_flag("unsupported", "unsupported") is False

    def test_review_verdict_is_never_an_over_flag(self) -> None:
        assert is_over_flag("supported", "review") is False


class TestAggregateDegenerateCases:
    """The three gates CLAUDE.md's testing section calls out by name."""

    def test_a_gate_that_flags_everything_has_zero_over_claim_rate_and_is_worthless(self) -> None:
        results = [
            _result("supported-1", "supported", "unsupported"),
            _result("unsupported-1", "unsupported", "unsupported"),
            _result("review-1", "review", "unsupported"),
        ]
        summary = aggregate(results)

        # Perfect over-claim rate...
        assert summary.over_claim_rate == 0.0
        # ...but a gate this bad is caught by the *other* number: it flags every
        # single grounded claim too.
        assert summary.over_flag_rate == 1.0

    def test_a_gate_that_passes_everything_has_zero_over_flag_rate_and_is_worse_than_worthless(
        self,
    ) -> None:
        results = [
            _result("supported-1", "supported", "supported"),
            _result("unsupported-1", "unsupported", "supported"),
            _result("review-1", "review", "supported"),
        ]
        summary = aggregate(results)

        assert summary.over_flag_rate == 0.0
        # Every single non-grounded claim was passed as supported.
        assert summary.over_claim_rate == 1.0

    def test_a_perfect_gate_has_zero_on_both_rates(self) -> None:
        results = [
            _result("supported-1", "supported", "supported"),
            _result("unsupported-1", "unsupported", "unsupported"),
            _result("review-1", "review", "review"),
        ]
        summary = aggregate(results)

        assert summary.over_claim_rate == 0.0
        assert summary.over_flag_rate == 0.0


class TestAggregateArithmetic:
    def test_over_claim_rate_denominator_excludes_expected_supported_items(self) -> None:
        results = [
            _result("a", "supported", "supported"),
            _result("b", "unsupported", "supported"),  # over-claim
            _result("c", "review", "unsupported"),  # not an over-claim
        ]
        summary = aggregate(results)
        assert summary.over_claim_denominator == 2  # "b" and "c": expected != supported
        assert summary.over_claim_count == 1
        assert summary.over_claim_rate == 0.5

    def test_over_flag_rate_denominator_is_only_expected_supported_items(self) -> None:
        results = [
            _result("a", "supported", "unsupported"),  # over-flag
            _result("b", "supported", "supported"),
            _result("c", "unsupported", "unsupported"),  # not eligible: expected != supported
        ]
        summary = aggregate(results)
        assert summary.over_flag_denominator == 2  # "a" and "b"
        assert summary.over_flag_count == 1
        assert summary.over_flag_rate == 0.5

    def test_empty_denominator_is_none_not_zero(self) -> None:
        """A rate with no eligible items is undefined, not a real zero -- an
        empty golden-set slice must never look identical to a perfect score.
        """
        results = [_result("a", "supported", "supported")]  # no expected-non-supported items
        summary = aggregate(results)
        assert summary.over_claim_denominator == 0
        assert summary.over_claim_rate is None

    def test_no_results_at_all_yields_none_rates_and_zero_counts(self) -> None:
        summary = aggregate([])
        assert summary.n_items == 0
        assert summary.over_claim_rate is None
        assert summary.over_flag_rate is None
        assert summary.framing_rate is None
        assert summary.framing_over_claim_rate is None

    def test_errored_items_are_excluded_from_scored_counts_and_every_rate(self) -> None:
        results = [
            _result("a", "supported", "supported"),
            _result("b", "unsupported", None),  # the gate call itself failed
        ]
        summary = aggregate(results)
        assert summary.n_items == 2
        assert summary.n_scored == 1
        assert summary.n_errors == 1
        # "b" never enters over_claim_denominator despite expected != supported --
        # it was never actually scored.
        assert summary.over_claim_denominator == 0
        assert summary.over_claim_rate is None

    def test_confusion_matrix_counts_every_scored_pair(self) -> None:
        results = [
            _result("a", "supported", "supported"),
            _result("b", "supported", "supported"),
            _result("c", "supported", "unsupported"),
            _result("d", "review", "supported"),
        ]
        summary = aggregate(results)
        assert summary.confusion == {
            ("supported", "supported"): 2,
            ("supported", "unsupported"): 1,
            ("review", "supported"): 1,
        }

    def test_framing_rate_counts_claim_sentences_the_gate_called_framing(self) -> None:
        results = [
            _result("a", "supported", "supported", kind="claim"),
            _result("b", "supported", "supported", kind="framing"),
        ]
        summary = aggregate(results)
        assert summary.framing_count == 1
        assert summary.framing_rate == 0.5

    def test_framing_never_enters_over_claim_or_over_flag(self) -> None:
        """Framing rate is diagnostic-only (module docstring) -- it must not shift
        either headline rate just by being present in the same batch.
        """
        matched = [_result("a", "supported", "supported", kind="claim")]
        with_framing = matched + [_result("b", "supported", "supported", kind="framing")]
        assert aggregate(matched).over_claim_rate == aggregate(with_framing).over_claim_rate
        assert aggregate(matched).over_flag_rate == aggregate(with_framing).over_flag_rate


class TestFramingOverClaim:
    """`framing_over_claim_rate`/`_count`: how often a `kind="framing"`
    misclassification specifically caused an over-claim -- the failure mode with
    no defence anywhere in the gate (see the module docstring). Distinct from
    `framing_rate` (how often the gate calls anything framing at all) and from
    `over_claim_rate` (how often any over-claim happens, framing-caused or not).
    """

    def test_no_framing_at_all_yields_zero_count_and_zero_rate(self) -> None:
        results = [
            _result("a", "supported", "supported", kind="claim"),
            _result("b", "unsupported", "unsupported", kind="claim"),
        ]
        summary = aggregate(results)
        assert summary.framing_over_claim_count == 0
        assert summary.framing_over_claim_rate == 0.0

    def test_framing_that_did_not_over_claim_does_not_count(self) -> None:
        """A framing item whose expected label is "supported" -- the gate's
        framing contract forces verdict="supported", which happens to match, so
        this is not an over-claim even though it is still a kind misclassification
        (framing_rate is nonzero; framing_over_claim_rate stays at zero).
        """
        results = [
            _result("a", "supported", "supported", kind="claim"),
            _result("b", "supported", "supported", kind="framing"),
        ]
        summary = aggregate(results)
        assert summary.framing_count == 1
        assert summary.framing_over_claim_count == 0
        assert summary.framing_over_claim_rate == 0.0

    def test_framing_that_over_claimed_counts_and_computes_a_rate(self) -> None:
        """The exact shape of the real finding this metric exists for: a
        NOT-ENOUGH-INFO item (expected "review") the gate called framing, which
        the framing contract forces to verdict "supported" -- an over-claim with
        no corpus check anywhere in the path that produced it.
        """
        results = [
            _result("a", "supported", "supported", kind="claim"),
            _result("b", "supported", "supported", kind="claim"),
            _result("c", "review", "supported", kind="framing"),
        ]
        summary = aggregate(results)
        assert summary.framing_count == 1
        assert summary.framing_over_claim_count == 1
        assert summary.framing_over_claim_rate == 1 / 3

    def test_framing_over_claim_rate_denominator_is_n_scored_not_over_claim_denominator(
        self,
    ) -> None:
        """Deliberately not the same denominator as over_claim_rate (see the field's
        own docstring on ScoreSummary) -- this answers "how often does a framing
        miss silently pass a claim", scaled against everything scored, not just
        against the items eligible to be an over-claim in the first place.
        """
        results = [
            _result("a", "supported", "supported", kind="claim"),  # not over-claim-eligible
            _result("b", "review", "supported", kind="framing"),  # framing + over-claim
        ]
        summary = aggregate(results)
        assert summary.over_claim_denominator == 1  # only "b" is expected != supported
        assert summary.framing_over_claim_count == 1
        assert summary.framing_over_claim_rate == 1 / 2  # over n_scored (2), not over 1

    def test_framing_over_claim_rate_is_none_not_zero_when_nothing_was_scored(self) -> None:
        results = [_result("a", "supported", None, kind=None)]  # gate call itself failed
        summary = aggregate(results)
        assert summary.n_scored == 0
        assert summary.framing_over_claim_count == 0
        assert summary.framing_over_claim_rate is None


class TestPerExpectedBreakdown:
    def test_groups_confusion_by_expected_label(self) -> None:
        results = [
            _result("a", "supported", "supported"),
            _result("b", "supported", "unsupported"),
            _result("c", "review", "review"),
        ]
        summary = aggregate(results)
        breakdown = per_expected_breakdown(summary.confusion)
        assert breakdown == {
            "supported": {"supported": 1, "unsupported": 1},
            "review": {"review": 1},
        }

    def test_empty_confusion_yields_empty_breakdown(self) -> None:
        assert per_expected_breakdown({}) == {}
