from __future__ import annotations

import inspect
import unittest
from dataclasses import fields
from pathlib import Path

import decepticons as opc

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "decepticons"


def _field_names(cls: type[object]) -> set[str]:
    return {field.name for field in fields(cls)}


def _public_class_names() -> set[str]:
    names: set[str] = set()
    for name in dir(opc):
        if name.startswith("_"):
            continue
        value = getattr(opc, name)
        if inspect.isclass(value):
            names.add(name)
    return names


class FrontierBoundaryTests(unittest.TestCase):
    def test_probability_diagnostics_stays_family_neutral(self) -> None:
        public_names = set(dir(opc))

        self.assertTrue(hasattr(opc, "bits_per_byte_from_probabilities"))
        self.assertTrue(hasattr(opc, "bridge_feature_arrays"))
        self.assertFalse(
            any(
                ("prob" in name.lower() or "diagn" in name.lower() or "artifact" in name.lower())
                and any(fragment in name.lower() for fragment in ("payload", "teacher", "program", "controller"))
                for name in public_names
            )
        )

    def test_noncausal_payload_wire_format_policy_stays_out_of_src(self) -> None:
        public_names = set(dir(opc))
        public_classes = _public_class_names()
        src_module_names = {path.stem for path in SRC_ROOT.rglob("*.py")}

        forbidden_fragments = (
            "Payload",
            "payload",
            "WireFormat",
            "wire_format",
            "wireformat",
            "PayloadPolicy",
        )

        for fragment in forbidden_fragments:
            self.assertFalse(any(fragment in name for name in public_names))
            self.assertFalse(any(fragment in name for name in public_classes))
            self.assertFalse(any(fragment in name for name in src_module_names))

        bridge_params = tuple(inspect.signature(opc.BridgeExportAdapter.__init__).parameters)
        bridge_forbidden = {
            "payload",
            "payload_format",
            "payload_policy",
            "payload_wire_format",
            "wire_format",
            "wire_payload",
        }
        self.assertFalse(bridge_forbidden & set(bridge_params))

        bridge_report_fields = _field_names(opc.BridgeExportReport)
        self.assertFalse(any(fragment.lower() in field for field in bridge_report_fields for fragment in forbidden_fragments))

    def test_bridge_teacher_policy_internals_stay_project_local(self) -> None:
        public_names = set(dir(opc))
        public_classes = _public_class_names()
        bridge_params = tuple(inspect.signature(opc.BridgeExportAdapter.__init__).parameters)
        bridge_report_fields = _field_names(opc.BridgeExportReport)
        bridge_fit_fields = _field_names(opc.BridgeExportFitReport)
        teacher_params = tuple(inspect.signature(opc.TeacherExportAdapter.__init__).parameters)
        teacher_report_fields = _field_names(opc.TeacherExportReport)

        allowed_teacher_names = {
            "teacher_export",
            "TeacherExportAdapter",
            "TeacherExportConfig",
            "TeacherExportRecord",
            "TeacherExportReport",
        }

        self.assertEqual({name for name in public_names if "teacher" in name.lower()}, allowed_teacher_names)
        self.assertTrue(
            {
                "TeacherExportAdapter",
                "TeacherExportConfig",
                "TeacherExportRecord",
                "TeacherExportReport",
            }
            <= public_classes
        )
        self.assertNotIn("teacher", bridge_params)
        self.assertNotIn("teacher_policy", bridge_params)
        self.assertNotIn("teacher_forcing", bridge_params)
        self.assertNotIn("teacher_export", bridge_params)

        bridge_field_names = bridge_report_fields | bridge_fit_fields
        self.assertFalse(any("teacher" in field.lower() for field in bridge_field_names))

        forbidden_teacher_params = {
            "attack",
            "attack_policy",
            "teacher_policy",
            "teacher_forcing",
            "payload",
            "payload_policy",
            "wire_format",
        }
        self.assertFalse(forbidden_teacher_params & set(teacher_params))
        self.assertFalse(any("attack" in field.lower() for field in teacher_report_fields))
        self.assertFalse(any("payload" in field.lower() for field in teacher_report_fields))

    def test_higher_order_causal_program_controller_policy_stays_out_of_src(self) -> None:
        causal_params = tuple(inspect.signature(opc.CausalPredictiveAdapter.__init__).parameters)
        causal_fit_fields = _field_names(opc.CausalPredictiveFitReport)
        causal_score_fields = _field_names(opc.CausalPredictiveScore)
        public_names = set(dir(opc))
        public_classes = _public_class_names()

        forbidden_params = {
            "program",
            "program_controller",
            "controller_policy",
            "higher_order",
            "higher_order_program",
            "route_policy",
            "lag_policy",
            "policy_stack",
        }
        self.assertFalse(forbidden_params & set(causal_params))

        self.assertFalse(any("program" in name.lower() for name in public_names))
        self.assertFalse(any("program" in name.lower() for name in public_classes))
        self.assertFalse(any("policy" in name.lower() and "controller" in name.lower() for name in public_classes))

        causal_field_names = causal_fit_fields | causal_score_fields
        self.assertFalse(any("program" in field.lower() for field in causal_field_names))
        self.assertFalse(any("policy" in field.lower() for field in causal_field_names))
        self.assertFalse(any("higher" in field.lower() for field in causal_field_names))


if __name__ == "__main__":
    unittest.main()
