"""Reconstruct 3D tool positions in a simulated vitreoretinal-surgery eye.

Everything is expressed in a normalized eye frame:

    - origin  = eye center
    - scale   = eye radius (so the eye surface is the unit sphere, radius 1)
    - x axis  = left/right in the microscope image
    - y axis  = top/bottom in the microscope image
    - z axis  = depth along the microscope view axis
                (z = -1 is the top pole the camera looks in through,
                 z = +1 is the far/deep point of the retina)

Two trocars (entry ports) are fixed on the eye's surface and are described
by a pair of angles:

    - ``rot_up``    angle (radians) between the insertion direction and the
                    vertical ("straight down") insertion axis. ``0`` means
                    the trocar sits at the top pole.
    - ``rot_clock`` azimuth (radians), measured clockwise as seen in the
                    microscope image, with ``0`` pointing toward +x.

The light source is modeled as entering its trocar and then travelling in
some aim direction by a given depth to reach its tip, which acts as a
point light source. By default that aim direction points straight at the
eye center, but it can be tilted away from that default using two more
angles, so the light does not have to be aimed exactly at the center:

    - ``aim_tilt``  angle (radians) between the aim direction and the
                    default "straight at the eye center" direction. ``0``
                    (the default) reproduces the original fixed behavior.
    - ``aim_clock`` azimuth (radians) of that tilt around the default aim
                    direction, with ``0`` tilting toward increasing
                    ``rot_up`` and positive values sweeping the same way
                    ``rot_clock`` does.

A depth of 0 leaves the tip at the trocar (the eye's surface); with no
tilt, a depth of 1 (one eye radius) would put the tip exactly at the eye
center. The public API takes this depth (and the jaw length, below) in
millimeters, together with the eye's radius in millimeters, and converts
them to the normalized frame internally.

The forceps has its own trocar. Its two jaw tips are not directly visible
in 3D - only their 2D microscope-image positions and the 2D positions of
the shadows they cast on the retina are known. Because the light source
position is known, each tip must lie on the 3D ray from the light tip
through the shadow point on the retina; the observed 2D tip position (in
the microscope's x/y image plane) pins down exactly where on that ray the
real, floating tip is. The point where the two jaws physically meet is then
recovered from the two reconstructed tip positions, the forceps trocar, and
the known length of a single jaw.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Small vector helpers (kept local so this module has no dependencies).
# ---------------------------------------------------------------------------

def _sub3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add3(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale3(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot2(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _sub2(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def _norm3(a: Vec3) -> float:
    return math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)


def _normalize3(a: Vec3) -> Vec3:
    n = _norm3(a)
    if n == 0.0:
        raise ValueError("cannot normalize a zero-length vector")
    return _scale3(a, 1.0 / n)


def _xy(a: Vec3) -> Vec2:
    return (a[0], a[1])


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

def pixel_to_normalized(
    point_px: Vec2, eye_center_px: Vec2, eye_radius_px: float
) -> Vec2:
    """Convert a microscope-image pixel coordinate to normalized x/y.

    The result is expressed in units of eye radii, with the origin at the
    eye center as seen in the image.
    """
    px, py = point_px
    cx, cy = eye_center_px
    return ((px - cx) / eye_radius_px, (py - cy) / eye_radius_px)


def trocar_position(rot_up: float, rot_clock: float) -> Vec3:
    """Position of a trocar on the unit eye sphere.

    ``rot_up`` is the angle away from the vertical ("straight down")
    insertion axis; ``rot_clock`` is the clockwise azimuth (as seen in the
    microscope image) around that axis, with 0 pointing toward +x.
    """
    return (
        math.sin(rot_up) * math.cos(rot_clock),
        math.sin(rot_up) * math.sin(rot_clock),
        -math.cos(rot_up),
    )


def light_aim_direction(
    rot_up: float, rot_clock: float, aim_tilt: float = 0.0, aim_clock: float = 0.0
) -> Vec3:
    """Direction the light points from its trocar, as a unit vector.

    With ``aim_tilt = 0`` (the default) this is simply the direction
    straight back to the eye center, matching the light's original,
    fixed behavior. A nonzero ``aim_tilt`` swings the direction away from
    that default by ``aim_tilt`` radians, with ``aim_clock`` choosing
    which way it swings (using the same local tangent frame that
    ``rot_up``/``rot_clock`` use to place the trocar itself), giving full
    freedom to aim the light anywhere rather than only at the center.
    """
    trocar = trocar_position(rot_up, rot_clock)
    default_aim = _scale3(trocar, -1.0)

    # Local tangent frame at the trocar: e_up is the direction of
    # increasing rot_up, e_clock is the direction of increasing
    # rot_clock. Together with default_aim they form an orthonormal
    # basis, so tilting default_aim toward them stays a unit vector.
    e_up = (
        math.cos(rot_up) * math.cos(rot_clock),
        math.cos(rot_up) * math.sin(rot_clock),
        math.sin(rot_up),
    )
    e_clock = (-math.sin(rot_clock), math.cos(rot_clock), 0.0)

    swing = _add3(
        _scale3(e_up, math.cos(aim_clock)), _scale3(e_clock, math.sin(aim_clock))
    )
    return _add3(
        _scale3(default_aim, math.cos(aim_tilt)), _scale3(swing, math.sin(aim_tilt))
    )


def light_tip_position(
    rot_up: float,
    rot_clock: float,
    depth: float,
    aim_tilt: float = 0.0,
    aim_clock: float = 0.0,
) -> Vec3:
    """Tip of the light probe: its trocar position, then along its aim direction.

    Travelling ``depth`` along the light's aim direction (see
    ``light_aim_direction``) from the trocar gives the tip position. With
    the default ``aim_tilt = 0``, this aims straight back at the eye
    center, so ``depth = 1`` (one eye radius) puts the tip exactly at the
    center, matching the original fixed behavior.
    """
    trocar = trocar_position(rot_up, rot_clock)
    direction = light_aim_direction(rot_up, rot_clock, aim_tilt, aim_clock)
    return _add3(trocar, _scale3(direction, depth))


def shadow_point_on_retina(
    shadow_px: Vec2, eye_center_px: Vec2, eye_radius_px: float
) -> Vec3:
    """Place an observed shadow pixel onto the far (deep) retinal surface."""
    sx, sy = pixel_to_normalized(shadow_px, eye_center_px, eye_radius_px)
    radicand = 1.0 - sx ** 2 - sy ** 2
    sz = math.sqrt(radicand) if radicand > 0.0 else 0.0
    return (sx, sy, sz)


def reconstruct_tip_from_shadow(
    light_tip: Vec3,
    tip_px: Vec2,
    shadow_px: Vec2,
    eye_center_px: Vec2,
    eye_radius_px: float,
) -> Vec3:
    """Recover the 3D position of a tool tip from its shadow.

    The tip lies on the ray from the light tip through the shadow point on
    the retina. Its 2D microscope-image position fixes how far along that
    ray it sits (solved as a least-squares projection in the image plane,
    since the ray's own image-plane projection is a single line and the
    observed tip offset only constrains that 2D component).
    """
    shadow_3d = shadow_point_on_retina(shadow_px, eye_center_px, eye_radius_px)
    ray_dir = _sub3(shadow_3d, light_tip)

    tip_xy = pixel_to_normalized(tip_px, eye_center_px, eye_radius_px)
    delta = _sub2(tip_xy, _xy(light_tip))
    dir_xy = _xy(ray_dir)

    denom = _dot2(dir_xy, dir_xy)
    if denom == 0.0:
        raise ValueError(
            "light tip and shadow project to the same image point; "
            "cannot solve for the ray parameter"
        )
    t = _dot2(delta, dir_xy) / denom

    return _add3(light_tip, _scale3(ray_dir, t))


def jaw_meeting_point(
    trocar_forceps: Vec3, left_tip: Vec3, right_tip: Vec3, jaw_length: float
) -> Vec3:
    """Where the two forceps jaws physically meet (the hinge point).

    The hinge lies on the forceps shaft axis (trocar -> tip midpoint),
    behind the midpoint of the two tips by whatever distance is needed so
    that each jaw has the given length.
    """
    midpoint = _scale3(_add3(left_tip, right_tip), 0.5)
    half_spread = _norm3(_sub3(left_tip, right_tip)) / 2.0

    if jaw_length < half_spread:
        raise ValueError(
            f"jaw_length ({jaw_length}) is shorter than half the tip "
            f"spread ({half_spread}); tip reconstruction is inconsistent "
            "with the given jaw length"
        )

    axis_dir = _normalize3(_sub3(midpoint, trocar_forceps))
    back_distance = math.sqrt(jaw_length ** 2 - half_spread ** 2)
    return _sub3(midpoint, _scale3(axis_dir, back_distance))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_tool_positions(
    light_rot_up: float,
    light_rot_clock: float,
    light_depth_mm: float,
    forceps_rot_up: float,
    forceps_rot_clock: float,
    left_tip_px: Vec2,
    left_shadow_px: Vec2,
    right_tip_px: Vec2,
    right_shadow_px: Vec2,
    eye_center_px: Vec2,
    eye_radius_px: float,
    eye_radius_mm: float,
    jaw_length_mm: float,
    light_aim_tilt: float = 0.0,
    light_aim_clock: float = 0.0,
) -> Dict[str, Vec3]:
    """Reconstruct all tool positions in the normalized eye frame.

    Parameters
    ----------
    light_rot_up, light_rot_clock:
        Orientation (radians) of the light trocar.
    light_depth_mm:
        How far the light is inserted past its trocar, in millimeters,
        measured along its aim direction from the trocar.
    light_aim_tilt, light_aim_clock:
        Optional angles (radians) tilting the light's aim direction away
        from the default of pointing straight at the eye center. Both
        default to ``0``, which reproduces the original fixed behavior.
    forceps_rot_up, forceps_rot_clock:
        Orientation (radians) of the forceps trocar.
    left_tip_px, right_tip_px:
        Microscope-image pixel coordinates of the two visible forceps tips.
    left_shadow_px, right_shadow_px:
        Microscope-image pixel coordinates of the shadows cast by the two
        forceps tips onto the retina.
    eye_center_px, eye_radius_px:
        Pixel coordinates of the eye's center and its radius in pixels, as
        measured in the microscope image.
    eye_radius_mm:
        The eye's radius in millimeters. Used only to convert
        ``light_depth_mm`` and ``jaw_length_mm`` into the normalized
        (fraction-of-eye-radius) frame used everywhere else.
    jaw_length_mm:
        Length of a single forceps jaw, in millimeters.

    Returns
    -------
    A dict with six 3D points (each an (x, y, z) tuple in the normalized
    eye frame, i.e. in units of eye radii): ``trocar_light``, ``tip_light``,
    ``trocar_forceps``, ``left_tip_forceps``, ``right_tip_forceps``,
    ``jaw_meet_forceps``.
    """
    light_depth = light_depth_mm / eye_radius_mm
    jaw_length = jaw_length_mm / eye_radius_mm

    trocar_light = trocar_position(light_rot_up, light_rot_clock)
    tip_light = light_tip_position(
        light_rot_up, light_rot_clock, light_depth, light_aim_tilt, light_aim_clock
    )

    trocar_forceps = trocar_position(forceps_rot_up, forceps_rot_clock)

    left_tip_forceps = reconstruct_tip_from_shadow(
        tip_light, left_tip_px, left_shadow_px, eye_center_px, eye_radius_px
    )
    right_tip_forceps = reconstruct_tip_from_shadow(
        tip_light, right_tip_px, right_shadow_px, eye_center_px, eye_radius_px
    )

    jaw_meet_forceps = jaw_meeting_point(
        trocar_forceps, left_tip_forceps, right_tip_forceps, jaw_length
    )

    return {
        "trocar_light": trocar_light,
        "tip_light": tip_light,
        "trocar_forceps": trocar_forceps,
        "left_tip_forceps": left_tip_forceps,
        "right_tip_forceps": right_tip_forceps,
        "jaw_meet_forceps": jaw_meet_forceps,
    }


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Verification against the worked example (left/right tip
    # substitutions). The example's derivation used a frame where axes are
    # ordered (x, depth, z_horizontal) with the depth axis flipped relative
    # to this module's (x, y, z) frame (y = image-y, z = depth). Concretely,
    # for a point (x, y_depth, z_horiz) in the example's frame, the
    # equivalent point here is (x, -z_horiz, -y_depth), all divided by the
    # eye radius in mm to normalize. The relabeling is a reflection (it
    # only swaps/negates coordinate labels), so distances, dot products and
    # the sphere equation are unaffected - only which axis is called what
    # changes.
    # ------------------------------------------------------------------

    def relabel_mm_to_normalized(x: float, y_depth: float, z_horiz: float, r_mm: float) -> Vec3:
        return (x / r_mm, -z_horiz / r_mm, -y_depth / r_mm)

    r_mm = 12.00
    eye_center_px = (120.00, 120.00)
    # eye_radius_px reconstructed from the example's sigma = 0.0870 mm/px:
    # eye_radius_px = r_mm / sigma = m * N / 2 = 1.15 * 240 / 2 = 138.0
    eye_radius_px = 138.0

    light_tip = relabel_mm_to_normalized(-2.53, 1.47, -2.68, r_mm)

    def check_tip(name: str, tip_px: Vec2, shadow_px: Vec2, expected_mm: Vec3) -> None:
        got = reconstruct_tip_from_shadow(
            light_tip, tip_px, shadow_px, eye_center_px, eye_radius_px
        )
        expected = relabel_mm_to_normalized(*expected_mm, r_mm=r_mm)
        err = _norm3(_sub3(got, expected)) * r_mm
        print(f"{name}: got={got}, expected={expected}, error={err:.4f} mm-equiv")
        assert err < 0.02, f"{name} reconstruction error too large: {err}"

    check_tip("left tip", (133.4, 112.9), (209.2, 45.1), (1.16, -1.36, 0.62))
    check_tip("right tip", (157.2, 99.5), (227.7, 44.9), (3.24, -1.04, 1.78))

    # Sanity checks for the trocar/light geometry helpers.
    north_pole = trocar_position(0.0, 0.0)
    assert math.isclose(north_pole[2], -1.0, abs_tol=1e-9)
    assert math.isclose(_norm3(north_pole), 1.0, abs_tol=1e-9)

    equator_point = trocar_position(math.pi / 2, 0.0)
    assert math.isclose(equator_point[0], 1.0, abs_tol=1e-9)

    # depth=0 leaves the tip at the trocar; depth=1 puts it at the eye center.
    trocar_sample = trocar_position(math.pi / 4, math.pi / 3)
    assert light_tip_position(math.pi / 4, math.pi / 3, 0.0) == trocar_sample
    tip_at_center = light_tip_position(math.pi / 4, math.pi / 3, 1.0)
    assert math.isclose(_norm3(tip_at_center), 0.0, abs_tol=1e-9)
    tip_half = light_tip_position(math.pi / 4, math.pi / 3, 0.5)
    assert math.isclose(_norm3(_sub3(tip_half, _scale3(trocar_sample, 0.5))), 0.0, abs_tol=1e-9)

    # aim_tilt=0 must reproduce the original "always toward center" direction.
    default_aim = light_aim_direction(math.pi / 4, math.pi / 3)
    assert math.isclose(_norm3(_sub3(default_aim, _scale3(trocar_sample, -1.0))), 0.0, abs_tol=1e-9)

    # A tilted aim direction must still be a unit vector, and must actually
    # differ from the default once tilted.
    tilted_aim = light_aim_direction(math.pi / 4, math.pi / 3, aim_tilt=0.3, aim_clock=1.0)
    assert math.isclose(_norm3(tilted_aim), 1.0, abs_tol=1e-9)
    assert _norm3(_sub3(tilted_aim, default_aim)) > 1e-6
    tilted_tip = light_tip_position(math.pi / 4, math.pi / 3, 0.5, aim_tilt=0.3, aim_clock=1.0)
    assert math.isclose(_norm3(_sub3(tilted_tip, trocar_sample)), 0.5, abs_tol=1e-9)

    # End-to-end smoke test with the jaw-meeting-point solve.
    # light_depth_mm/jaw_length_mm are given in mm here and converted
    # internally using eye_radius_mm; 0.9*12=10.8mm and 0.3*12=3.6mm
    # reproduce the same normalized values used before this change.
    result = compute_tool_positions(
        light_rot_up=0.2,
        light_rot_clock=0.4,
        light_depth_mm=10.8,
        forceps_rot_up=0.3,
        forceps_rot_clock=-1.2,
        left_tip_px=(133.4, 112.9),
        left_shadow_px=(209.2, 45.1),
        right_tip_px=(157.2, 99.5),
        right_shadow_px=(227.7, 44.9),
        eye_center_px=eye_center_px,
        eye_radius_px=eye_radius_px,
        eye_radius_mm=r_mm,
        jaw_length_mm=3.6,
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("All checks passed.")
