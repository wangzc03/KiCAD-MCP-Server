"""
Regression tests for (mirror x|y) semantics in pin_locator.

Background:
  Early versions of apply_symbol_transform() copied the (buggy) mirror
  convention from WireDragger.pin_world_xy, which treats `(mirror x)` as
  "flip local X" and omits the lib y-up → schematic y-down conversion.

  KiCad's actual sexp convention (matching
  schematic_analysis._transform_local_point and wire_connectivity
  inline code):

      (mirror x) = reflect across the X axis = flip LOCAL Y coordinate
      (mirror y) = reflect across the Y axis = flip LOCAL X coordinate

  Plus: lib_symbols uses y-up, schematic uses y-down, so pin y must
  always be negated before mirror/rotate/translate.

These tests pin down the correct behaviour so future edits to the
transform pipeline can't silently regress.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PYTHON_DIR = Path(__file__).parent.parent / "python"
sys.path.insert(0, str(PYTHON_DIR))

# commands/__init__.py chain-loads board/view.py, which imports PIL.
# Stub PIL so the import chain succeeds in minimal test environments.
for _pil_mod in ("PIL", "PIL.Image"):
    sys.modules.setdefault(_pil_mod, types.ModuleType(_pil_mod))


def _stub_symbol(ref: str, at: list, lib_id: str = "Device:R") -> MagicMock:
    """Build a minimal kicad-skip symbol stub (no mirror clause)."""
    sym = MagicMock()
    sym.property.Reference.value = ref
    sym.at.value = at
    sym.lib_id.value = lib_id
    del sym.mirror
    return sym


def _stub_symbol_with_mirror(
    ref: str, at: list, mirror_axis: str, lib_id: str = "Device:R"
) -> MagicMock:
    """Build a kicad-skip symbol stub carrying (mirror x) or (mirror y)."""
    sym = MagicMock()
    sym.property.Reference.value = ref
    sym.at.value = at
    sym.lib_id.value = lib_id
    sym.mirror.value = mirror_axis
    return sym


# ===========================================================================
# apply_symbol_transform — pure function, no mocks needed
# ===========================================================================


@pytest.mark.unit
class TestApplySymbolTransform:
    """
    The canonical transform pipeline:
        (lx, ly) = (px, -py)                # lib y-up → schematic y-down
        if mirror_x: ly = -ly               # (mirror x) flips Y
        if mirror_y: lx = -lx               # (mirror y) flips X
        rotate(rotation, CCW)
        translate by (sym_x, sym_y)
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from commands.pin_locator import PinLocator

        self.fn = PinLocator.apply_symbol_transform

    def test_no_mirror_no_rotation_applies_y_negate(self):
        """
        Pin at lib (2, 3) on a symbol at schematic (10, 20) should land at
        (12, 17): the lib y=+3 is 'up' in y-up, which is 'lower y' in y-down.
        """
        ax, ay = self.fn(2.0, 3.0, 10.0, 20.0, 0.0, False, False)
        assert (ax, ay) == (12.0, 17.0)

    def test_mirror_x_flips_local_y(self):
        """
        (mirror x) = reflect across X axis = flip Y.
        Pin at lib (2, 3), sym at (10, 20): without mirror y lands at 17,
        with (mirror x) it lands at 23 (reflected across sym_y).
        """
        ax, ay = self.fn(2.0, 3.0, 10.0, 20.0, 0.0, True, False)
        assert ay == 23.0, f"(mirror x) must flip Y: expected y=23, got {ay}"
        assert ax == 12.0, f"(mirror x) must NOT flip X: expected x=12, got {ax}"

    def test_mirror_y_flips_local_x(self):
        """
        (mirror y) = reflect across Y axis = flip X.
        Pin at lib (2, 3), sym at (10, 20): without mirror x lands at 12,
        with (mirror y) it lands at 8 (reflected across sym_x).
        """
        ax, ay = self.fn(2.0, 3.0, 10.0, 20.0, 0.0, False, True)
        assert ax == 8.0, f"(mirror y) must flip X: expected x=8, got {ax}"
        assert ay == 17.0, f"(mirror y) must NOT flip Y: expected y=17, got {ay}"

    def test_rotation_90_ccw_no_mirror(self):
        """
        Rotation is applied AFTER y-negate.
        Lib (2, 3) → y-negate → (2, -3) → 90° CCW → (3, 2) → +sym (13, 22).
        """
        ax, ay = self.fn(2.0, 3.0, 10.0, 20.0, 90.0, False, False)
        assert ax == 13.0
        assert ay == 22.0

    def test_mirror_x_then_rotation(self):
        """
        Transform order: y-negate → mirror → rotate → translate.
        Lib (2, 3) → y-negate → (2, -3) → mirror_x → (2, 3) → rot 90 → (-3, 2)
                 → +sym (10, 20) → (7, 22).
        """
        ax, ay = self.fn(2.0, 3.0, 10.0, 20.0, 90.0, True, False)
        assert ax == 7.0
        assert ay == 22.0

    def test_matches_schematic_analysis_transform_local_point(self):
        """
        apply_symbol_transform must produce the same result as the upstream
        reference implementation schematic_analysis._transform_local_point
        (which uses the correct semantics). If they diverge, one of them
        has been broken.
        """
        from commands.schematic_analysis import _transform_local_point

        cases = [
            (2.0, 3.0, 10.0, 20.0, 0.0, False, False),
            (2.0, 3.0, 10.0, 20.0, 0.0, True, False),
            (2.0, 3.0, 10.0, 20.0, 0.0, False, True),
            (2.0, 3.0, 10.0, 20.0, 90.0, False, False),
            (2.0, 3.0, 10.0, 20.0, 180.0, True, False),
            (2.0, 3.0, 10.0, 20.0, 270.0, False, True),
            (-1.5, 4.5, 50.0, 50.0, 90.0, True, True),
        ]
        for px, py, sx, sy, rot, mx, my in cases:
            mine = self.fn(px, py, sx, sy, rot, mx, my)
            theirs = _transform_local_point(px, py, sx, sy, rot, mx, my)
            assert mine == pytest.approx(theirs, abs=1e-6), (
                f"Diverged from _transform_local_point for "
                f"(px={px}, py={py}, rot={rot}, mx={mx}, my={my}): "
                f"apply_symbol_transform={mine}, _transform_local_point={theirs}"
            )


# ===========================================================================
# get_pin_angle — mirror semantics end-to-end
# ===========================================================================


@pytest.mark.unit
class TestGetPinAngleMirrorSemantics:
    """
    Outward pin direction in SCHEMATIC (y-down) coordinates:
      0°   = right
      90°  = down     (because schematic y increases downward)
      180° = left
      270° = up

    Pin angles stored in lib_symbols use the y-up convention, so we apply:
      y-negate  →  -angle
      mirror_x (reflect X, flip Y)  →  -angle
      mirror_y (reflect Y, flip X)  →  180 - angle
      + symbol rotation
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from commands.pin_locator import PinLocator

        self.locator = PinLocator()

    def _call(self, sym, pin_def: dict, pin_number: str = "1"):
        self.locator._schematic_cache["t.kicad_sch"] = MagicMock(symbol=[sym])
        with patch.object(
            self.locator, "get_symbol_pins", return_value={pin_number: pin_def}
        ):
            return self.locator.get_pin_angle(Path("t.kicad_sch"), sym.property.Reference.value, pin_number)

    def test_horizontal_pin_no_mirror_points_right_in_schematic(self):
        """
        Lib angle=180 means 'pin body extends to the LEFT, so the outward pin
        endpoint points to the right (angle=0 in schematic)'.
        With no mirror / no rotation, y-negate alone gives (360-180)%360 = 180.
        """
        sym = _stub_symbol("R1", at=[0.0, 0.0, 0.0])
        angle = self._call(sym, {"x": 0.0, "y": 0.0, "angle": 180, "name": "~"})
        assert angle == 180.0

    def test_horizontal_pin_angle0_with_mirror_x_stays_horizontal(self):
        """
        A pin pointing at lib angle=0 (to the right), on a symbol with
        (mirror x) — reflect across X axis flips Y only, so a horizontal
        pin's direction is unchanged. Expected schematic angle = 0.
        """
        sym = _stub_symbol_with_mirror("R1", at=[0.0, 0.0, 0.0], mirror_axis="x")
        angle = self._call(sym, {"x": 0.0, "y": 0.0, "angle": 0, "name": "~"})
        assert angle == 0.0

    def test_horizontal_pin_angle0_with_mirror_y_becomes_left(self):
        """
        A pin pointing at lib angle=0 (right), on a symbol with (mirror y)
        — reflect across Y axis flips X, so right becomes left. Expected
        schematic angle = 180.
        """
        sym = _stub_symbol_with_mirror("R1", at=[0.0, 0.0, 0.0], mirror_axis="y")
        angle = self._call(sym, {"x": 0.0, "y": 0.0, "angle": 0, "name": "~"})
        assert angle == 180.0

    def test_vertical_pin_angle90_with_mirror_x_becomes_down(self):
        """
        Lib angle=90 means 'up' in lib y-up coords. In schematic y-down,
        that direction (0, +1 in lib) becomes (0, -1) (going UP in schematic)
        — angle 270. Apply (mirror x) (flip Y) to get (0, +1) in schematic
        → angle 90 (DOWN in schematic).

        Walk-through with our formula:
          pin_def_angle = 90
          y-negate: (360 - 90) = 270
          mirror_x: (360 - 270) = 90
          + rotation 0 = 90
        """
        sym = _stub_symbol_with_mirror("R1", at=[0.0, 0.0, 0.0], mirror_axis="x")
        angle = self._call(sym, {"x": 0.0, "y": 0.0, "angle": 90, "name": "~"})
        assert angle == 90.0

    def test_symbol_rotation_is_additive(self):
        """
        No mirror, lib angle=0 → schematic 0; then symbol rotation 90° CCW
        is simply added: expected 90.
        """
        sym = _stub_symbol("R1", at=[0.0, 0.0, 90.0])
        angle = self._call(sym, {"x": 0.0, "y": 0.0, "angle": 0, "name": "~"})
        assert angle == 90.0
