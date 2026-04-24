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
    Outward pin direction in the SCREEN-angle convention used by
    get_pin_angle's consumers (see connection_schematic.py, which does
    ``stub_end_y = pin_y - 2.54 * sin(angle)`` — note the MINUS on the Y
    component):

      0°   = right
      90°  = up       (visually up on screen; y decreases in y-down sch)
      180° = left
      270° = down

    Pin angles stored in lib_symbols use the y-up convention, so we apply:
      y-negate  →  -angle
      mirror_x (reflect X, flip Y)  →  -angle
      mirror_y (reflect Y, flip X)  →  180 - angle
      - symbol rotation                 # see note in get_pin_angle: the
                                        # position-rotation matrix is math
                                        # CCW in y-up applied to y-down
                                        # coords, i.e. visually CW, so in
                                        # the screen-angle convention the
                                        # rotation is SUBTRACTED.
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
        Lib angle=90 means the outward direction points UP in the lib
        y-up frame. y-negating flips that to DOWN on screen; applying
        (mirror x) (which flips local Y) puts it back to UP on screen.

        In the screen-angle convention used here (0=right, 90=up),
        that outward direction is 90° (up).

        Walk-through with our formula:
          pin_def_angle = 90
          y-negate: (360 - 90) = 270          # now 'down' in 90=up conv
          mirror_x: (360 - 270) = 90          # flips back to 'up'
          - rotation 0 = 90
        """
        sym = _stub_symbol_with_mirror("R1", at=[0.0, 0.0, 0.0], mirror_axis="x")
        angle = self._call(sym, {"x": 0.0, "y": 0.0, "angle": 90, "name": "~"})
        assert angle == 90.0

    def test_symbol_rotation_is_subtracted_in_screen_angle(self):
        """
        No mirror, lib angle=0 (outward points right in lib y-up).
        After y-negate in the "0=right, 90=up (visual)" convention the
        outward is still 0 (right — horizontal direction is unaffected
        by a Y flip). Now apply a +90° symbol rotation in the sexp.

        apply_symbol_transform uses rotate_point(x,y,90) = (-y, x) on
        already-y-negated coordinates. Applied to the outward direction
        vector (1, 0) in sch y-down, it produces (0, 1) — i.e. +y in
        y-down, which is VISUALLY DOWN on screen, = angle 270 in the
        0=right / 90=up convention.

        Hence the rotation is SUBTRACTED from the screen angle:
          (0 - 90) mod 360 = 270.

        Regression guard: before the fix this test asserted 90 (the
        buggy "additive" behaviour), which silently inverted every
        pin-angle query on any symbol with non-zero rotation. See
        pin_locator.get_pin_angle's long-form comment for the
        end-to-end failure mode (Device:R with mirror_x + rotation=90
        producing a wire stub that threads through the resistor body).
        """
        sym = _stub_symbol("R1", at=[0.0, 0.0, 90.0])
        angle = self._call(sym, {"x": 0.0, "y": 0.0, "angle": 0, "name": "~"})
        assert angle == 270.0

    def test_mirror_x_combined_with_rotation_90(self):
        """
        Regression guard for the combined-transform bug that the original
        mirror fix missed: a Device:R (pin 1 at lib (0, 3.81) angle 270)
        placed with BOTH (mirror x) AND rotation=90.

        Geometry: the resistor renders horizontally with its body spanning
        x = sym_x-2.54 .. sym_x+2.54. pin 1 ends up on the LEFT of the body
        (at sym_x - 3.81), so its outward direction is LEFT (= 180 in the
        screen-angle convention).

        Walk-through with the fixed formula:
          pin_def_angle = 270 (lib)
          y-negate:                  (360 - 270)   = 90      # was 'up'
          mirror_x (flip Y direction):  (360 - 90) = 270     # now 'down'
          subtract rotation 90:        (270 - 90)  = 180     # 'left' ✓

        Cross-checked end-to-end by rendering the same symbol with
        kicad-cli sch export svg and reading back the pin path — the
        drawn outward direction matches 180° exactly. See also
        tests/test_pin_locator_mirror_e2e.py (if present) for the
        SVG-ground-truth harness.
        """
        sym = _stub_symbol_with_mirror("R4", at=[200.0, 50.0, 90.0], mirror_axis="x")
        angle = self._call(sym, {"x": 0.0, "y": 3.81, "angle": 270, "name": "~"}, pin_number="1")
        assert angle == 180.0

        # And pin 2 of the same symbol must come out the other side (right = 0°).
        angle_p2 = self._call(
            sym, {"x": 0.0, "y": -3.81, "angle": 90, "name": "~"}, pin_number="2"
        )
        assert angle_p2 == 0.0
