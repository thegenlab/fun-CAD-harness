from __future__ import annotations

import unittest

from ..intake.prompt_clause_binding import prompt_clause_is_bound


PROMPT = (
    "Build a hand-operated screwdriver as one authoritative STEP assembly, "
    "with a grounded handle body and a rotating bit shaft that holds a "
    "screwdriver bit. The tool must drive a screw with a back-and-forth wrist "
    "motion: rotating the handle in the drive direction must rotate the bit "
    "shaft together with it and transmit torque to it, and rotating the handle "
    "back must leave the bit shaft essentially stationary and essentially "
    "torque-free, so that repeated back-and-forth cranking keeps advancing the "
    "bit shaft in one direction only, without regripping the tool. The tool "
    "must keep working after repeated use. The one-way drive principle is the "
    "designer's choice."
)


class PromptClauseBindingTests(unittest.TestCase):
    def assert_bound(self, clause: str) -> None:
        self.assertTrue(
            prompt_clause_is_bound(
                PROMPT,
                clause,
                target_role="bit shaft",
                input_role="handle",
            )
        )

    def assert_unbound(self, clause: str) -> None:
        self.assertFalse(
            prompt_clause_is_bound(
                PROMPT,
                clause,
                target_role="bit shaft",
                input_role="handle",
            )
        )

    def test_literal_clause_remains_bound(self) -> None:
        self.assert_bound(
            "rotating the handle back must leave the bit shaft essentially "
            "stationary"
        )

    def test_explicit_causal_connector_binds_ordered_source_spans(self) -> None:
        self.assert_bound(
            "rotating the handle in the drive direction must rotate the bit "
            "shaft together with it, so that repeated back-and-forth cranking "
            "keeps advancing the bit shaft in one direction only"
        )

    def test_semicolon_binds_two_ordered_source_propositions(self) -> None:
        self.assert_bound(
            "repeated back-and-forth cranking keeps advancing the bit shaft in "
            "one direction only; the tool must keep working after repeated use"
        )

    def test_short_coordinated_property_may_be_elided(self) -> None:
        self.assert_bound(
            "rotating the handle back must leave the bit shaft essentially "
            "torque-free"
        )

    def test_role_coreference_binds_scoped_context_and_claim(self) -> None:
        self.assert_bound(
            "transmit torque to the bit shaft while driving a screw"
        )

    def test_unsupported_object_is_rejected(self) -> None:
        self.assert_unbound("transmit torque to the housing while driving a screw")

    def test_invented_threshold_is_rejected(self) -> None:
        self.assert_unbound(
            "transmit 3 N.m of torque to the bit shaft while driving a screw"
        )

    def test_polarity_inversion_is_rejected(self) -> None:
        self.assert_unbound(
            "do not transmit torque to the bit shaft while driving a screw"
        )

    def test_reversed_role_is_rejected(self) -> None:
        self.assert_unbound(
            "the bit shaft transmits torque to the handle while driving a screw"
        )

    def test_false_backstroke_context_is_rejected(self) -> None:
        self.assert_unbound(
            "transmit torque to the bit shaft while rotating the handle back"
        )

    def test_reversed_multi_span_order_is_rejected(self) -> None:
        self.assert_unbound(
            "the tool must keep working after repeated use; repeated "
            "back-and-forth cranking keeps advancing the bit shaft in one "
            "direction only"
        )

    def test_absent_clause_remains_rejected(self) -> None:
        self.assert_unbound("quietly rotate only ten degrees")


if __name__ == "__main__":
    unittest.main()
