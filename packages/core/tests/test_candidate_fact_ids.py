"""Fact identity: what makes two CV lines the same fact, and what does not.

The number this protects is the one the owner will actually see: thirty-three
generated CVs saying much the same thing must collapse into one list to confirm,
or onboarding is unusable. The opposite failure matters more, though -- two
genuinely different facts folded into one means a fact the user never got to
confirm, silently missing from a screen whose whole job is to show it.
"""

from __future__ import annotations

from jfl_core.ids import fact_fingerprint, fold, normalise, role_key


class TestFold:
    def test_case_punctuation_and_spacing_all_fold_away(self) -> None:
        assert fold("Led a team of 8.") == fold("led  a team of 8")
        assert fold("Acme Ltd -- Engineering Manager") == fold("acme ltd  engineering manager")

    def test_separators_become_spaces_rather_than_vanishing(self) -> None:
        """ "Engineer,Platform" must not fuse into one word -- the same rule
        `jfl_intake.normalise` follows for job titles.
        """
        assert fold("Engineer,Platform") == "engineer platform"

    def test_unicode_forms_compare_equal(self) -> None:
        assert fold("Ｅngineer") == fold("Engineer")

    def test_blank_folds_to_empty(self) -> None:
        assert fold("") == ""
        assert fold("   \n ") == ""

    def test_it_is_harsher_than_normalise_and_deliberately_so(self) -> None:
        """`normalise` keeps case because case can change a claim's meaning;
        that is right for span identity and wrong for "have I proposed this
        already?".
        """
        assert normalise("Led a Team") != normalise("led a team")
        assert fold("Led a Team") == fold("led a team")


class TestFingerprint:
    def test_the_same_fact_written_differently_is_one_fingerprint(self) -> None:
        a = fact_fingerprint("Acme Ltd -- Engineering Manager, 2021-2024", "Led a team of 8.")
        b = fact_fingerprint("acme ltd – engineering manager, 2021–2024", "led a  team of 8")
        assert a == b

    def test_the_same_claim_under_a_different_role_is_a_different_fact(self) -> None:
        """Confirming "led a team of six" at one employer must not confirm it at
        another -- that would attach a confirmation to a claim about a different
        job.
        """
        assert fact_fingerprint("Acme -- EM", "Led a team of six") != fact_fingerprint(
            "Northwind -- EM", "Led a team of six"
        )

    def test_a_different_number_is_a_different_fact(self) -> None:
        assert fact_fingerprint("Acme -- EM", "Led a team of six") != fact_fingerprint(
            "Acme -- EM", "Led a team of sixteen"
        )

    def test_no_stemming_and_no_synonyms(self) -> None:
        """Both would be a guess about what counts as the same fact, and a wrong
        guess hides a fact the user never got to confirm.
        """
        assert fact_fingerprint("Acme -- Sr Engineer", "x") != fact_fingerprint(
            "Acme -- Senior Engineer", "x"
        )


class TestRoleKey:
    def test_the_same_role_spelled_differently_groups_together(self) -> None:
        assert role_key("Acme Ltd — Engineering Manager, 2021-2024") == role_key(
            "acme ltd  engineering manager 2021 2024"
        )

    def test_different_roles_stay_apart(self) -> None:
        assert role_key("Acme -- EM") != role_key("Acme -- Staff Engineer")
