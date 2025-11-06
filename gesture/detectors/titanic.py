""""Gesture recognition: Titanic pose"""

import itertools
import time
from datetime import datetime
from math import sqrt

from senseSpaceLib.senseSpace.protocol import Frame, Person
from senseSpaceLib.senseSpace.enums import Body34Joint as J

from OpenGL.GL import *
from OpenGL.GLU import *


class TitanicDetector:
    """
    Detect Titanic pose (A envelops, B spreads into the space).
    Uses joint indices from the SenseSpace body model (mm units).
    """

    # Joint indices (match your other detectors)
    LEFT_HAND = 8
    RIGHT_HAND = 15
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 12
    PELVIS = 0
    CHEST = J.SPINE_CHEST.value if hasattr(J, "SPINE_CHEST") else J.NECK.value

    def __init__(
        self,
        hands_close_threshold_mm: float = 200.0,        # tighter: hands within 20cm => closed
        pelvis_proximity_threshold_mm: float = 300.0,  # bodies overlap within 30cm
        pelvis_height_threshold_mm: float = 200.0,     # pelvis vertical alignment tolerance
        arm_spread_margin_mm: float = 150.0,           # B hand must be this much further than shoulder->pelvis
        hand_shoulder_height_tol_mm: float = 150.0,    # hands roughly at shoulder height
    ):
        self._hands_close_threshold = hands_close_threshold_mm
        self._pelvis_proximity_threshold = pelvis_proximity_threshold_mm
        self._pelvis_height_threshold = pelvis_height_threshold_mm
        self._arm_spread_margin = arm_spread_margin_mm
        self._hand_shoulder_height_tol = hand_shoulder_height_tol_mm

        # state for rising-edge detection
        self._prev_titanic = set()

    # ---- helpers to read joint safely ----
    def _get_joint(self, person: Person, idx: int):
        skel = getattr(person, "skeleton", None)
        if not skel or len(skel) <= idx:
            return None
        node = skel[idx]
        return getattr(node, "pos", None)  # Position-like object with .x, .y, .z

    def _person_id(self, person: Person):
        return getattr(person, "id", getattr(person, "track_id", id(person)))

    # ---- simple geometry in mm (pure floats) ----
    def _dist(self, a, b):
        if a is None or b is None:
            return float("inf")
        try:
            dx = float(a.x) - float(b.x)
            dy = float(a.y) - float(b.y)
            dz = float(a.z) - float(b.z)
            return sqrt(dx * dx + dy * dy + dz * dz)
        except Exception:
            return float("inf")

    def _vec(self, a, b):
        # returns tuple vector a -> b
        return (float(b.x - a.x), float(b.y - a.y), float(b.z - a.z))

    def _normalize(self, v):
        mag = sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if mag < 1e-6:
            return (0.0, 0.0, 0.0)
        return (v[0] / mag, v[1] / mag, v[2] / mag)

    def _dot(self, u, v):
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    # ---- pose tests ----
    def _arms_closed(self, left, right) -> bool:
        return left is not None and right is not None and self._dist(left, right) <= self._hands_close_threshold

    def _arms_spread(self, person: Person):
        """Return (is_spread:bool, hand_midpoint_position)"""
        lw = self._get_joint(person, self.LEFT_HAND)
        rw = self._get_joint(person, self.RIGHT_HAND)
        chest = self._get_joint(person, self.CHEST)
        lsh = self._get_joint(person, self.LEFT_SHOULDER)
        rsh = self._get_joint(person, self.RIGHT_SHOULDER)

        if not (lw and rw and chest and lsh and rsh):
            return False, None

        # height check: wrists roughly at shoulder y
        shoulder_y = (float(lsh.y) + float(rsh.y)) / 2.0
        cond_height = abs(float(lw.y) - shoulder_y) < self._hand_shoulder_height_tol and abs(
            float(rw.y) - shoulder_y
        ) < self._hand_shoulder_height_tol

        # distance from pelvis: must be wider than shoulder->pelvis + margin
        left_shoulder_to_pelvis = self._dist(lsh, chest)  # approximate radius using chest as center
        right_shoulder_to_pelvis = self._dist(rsh, chest)
        cond_distance = (self._dist(lw, chest) > left_shoulder_to_pelvis + self._arm_spread_margin) and (
            self._dist(rw, chest) > right_shoulder_to_pelvis + self._arm_spread_margin
        )

        is_spread = cond_height and cond_distance

        # compute midpoint (as simple tuple of floats)
        mid = type(lw)  # just to detect structure; we'll return a simple dict-like object
        hand_mid = None
        try:
            # return a lightweight object with x,y,z attributes by using a small inner class
            class P:
                pass

            hand_mid = P()
            hand_mid.x = (float(lw.x) + float(rw.x)) / 2.0
            hand_mid.y = (float(lw.y) + float(rw.y)) / 2.0
            hand_mid.z = (float(lw.z) + float(rw.z)) / 2.0
        except Exception:
            hand_mid = None

        return is_spread, hand_mid

    # ---- main API used by main.py ----
    def process(self, frame: Frame, gl_context=False):
        events = []
        if not hasattr(frame, "people") or not frame.people:
            self._prev_titanic.clear()
            return events

        # Build dictionary of people by id and raw joints
        people = {}
        for p in frame.people:
            pid = self._person_id(p)
            people[pid] = {
                "person": p,
                "left_hand": self._get_joint(p, self.LEFT_HAND),
                "right_hand": self._get_joint(p, self.RIGHT_HAND),
                "left_shoulder": self._get_joint(p, self.LEFT_SHOULDER),
                "right_shoulder": self._get_joint(p, self.RIGHT_SHOULDER),
                "pelvis": self._get_joint(p, self.PELVIS),
                "chest": self._get_joint(p, self.CHEST),
            }

        current = set()

        # Evaluate ordered pairs (A=enveloper, B=spreader)
        ids = list(people.keys())
        for a_id, b_id in itertools.permutations(ids, 2):
            A = people[a_id]
            B = people[b_id]

            # availability
            if A["left_hand"] is None or A["right_hand"] is None or A["pelvis"] is None:
                continue
            if B["left_hand"] is None or B["right_hand"] is None or B["pelvis"] is None:
                continue

            # 1) A arms closed
            if not self._arms_closed(A["left_hand"], A["right_hand"]):
                continue

            # 2) pelvis proximity & vertical alignment
            if self._dist(A["pelvis"], B["pelvis"]) > self._pelvis_proximity_threshold:
                continue
            if abs(float(A["pelvis"].y) - float(B["pelvis"].y)) > self._pelvis_height_threshold:
                continue

            # 3) B arms spread
            b_person_obj = B["person"]
            is_spread, hand_mid = self._arms_spread(b_person_obj)
            if not is_spread:
                continue

            # 4) ensure B pelvis is between A pelvis and A hand-midpoint in front depth
            # define hand midpoint for A
            try:
                A_hand_mid_x = (float(A["left_hand"].x) + float(A["right_hand"].x)) / 2.0
                A_hand_mid_y = (float(A["left_hand"].y) + float(A["right_hand"].y)) / 2.0
                A_hand_mid_z = (float(A["left_hand"].z) + float(A["right_hand"].z)) / 2.0
            except Exception:
                continue

            # represent as simple points
            class P:
                pass

            a_mid = P()
            a_mid.x, a_mid.y, a_mid.z = A_hand_mid_x, A_hand_mid_y, A_hand_mid_z
            a_pel = A["pelvis"]
            b_pel = B["pelvis"]

            # Project B pelvis onto segment A_pelvis -> A_hand_mid using param t
            # Compute vector arithmetic with tuples
            vx = a_mid.x - float(a_pel.x)
            vy = a_mid.y - float(a_pel.y)
            vz = a_mid.z - float(a_pel.z)
            wx = float(b_pel.x) - float(a_pel.x)
            wy = float(b_pel.y) - float(a_pel.y)
            wz = float(b_pel.z) - float(a_pel.z)

            v_len2 = vx * vx + vy * vy + vz * vz
            if v_len2 == 0:
                continue
            dot = vx * wx + vy * wy + vz * wz
            t = dot / v_len2
            # clamp
            if t < 0.0 or t > 1.0:
                continue

            # closest point coordinates
            cx = float(a_pel.x) + vx * t
            cy = float(a_pel.y) + vy * t
            cz = float(a_pel.z) + vz * t

            # perpendicular distance
            perp_dx = float(b_pel.x) - cx
            perp_dy = float(b_pel.y) - cy
            perp_dz = float(b_pel.z) - cz
            perp_dist = sqrt(perp_dx * perp_dx + perp_dy * perp_dy + perp_dz * perp_dz)

            if perp_dist > self._pelvis_proximity_threshold:  # too far from A's embrace line
                continue

            # passed all checks -> titanic detected for pair (A enveloper, B spreader)
            pair = (a_id, b_id)
            current.add(pair)

            if pair not in self._prev_titanic:
                ts = getattr(frame, "timestamp", time.time())
                dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                # report B pelvis as event pos
                pos = (int(float(b_pel.x)), int(float(b_pel.y)), int(float(b_pel.z)))
                events.append(
                    {
                        "type": "titanic",
                        "timestamp": ts,
                        "time_str": dt,
                        "people": pair,
                        "pos": pos,
                        "detail": {"perp_dist_mm": float(perp_dist), "t_param": float(t)},
                    }
                )

                # debug visual (sphere at B pelvis) if requested
                if gl_context:
                    try:
                        glPushMatrix()
                        glTranslatef(float(b_pel.x), float(b_pel.y), float(b_pel.z))
                        glColor4f(1.0, 0.6, 0.0, 0.9)
                        quad = gluNewQuadric()
                        gluSphere(quad, 90.0, 18, 18)
                        gluDeleteQuadric(quad)
                        glPopMatrix()
                    except Exception as e:
                        print(f"[OpenGL error while drawing titanic debug] {e}")

        # update rising-edge state
        self._prev_titanic = current
        return events
