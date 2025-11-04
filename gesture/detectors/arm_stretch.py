"""Gesture recognition: Raised arm stretch (both wrists above head)"""

import time
from datetime import datetime
from senseSpaceLib.senseSpace.protocol import Frame, Person
from senseSpaceLib.senseSpace.enums import Body34Joint as J
from PyQt5.QtGui import QVector3D
from OpenGL.GL import *
from OpenGL.GLU import *


class ArmStretchDetector:
    """Detect when a person raises both hands above their head."""

    def __init__(self):
        self._prev_stretch = {}  # Track previous frame states per person
        self._last_trigger = {}  # Track last trigger time per person to avoid rapid repeats
        self._cooldown = 0.5     # seconds between allowed triggers per person

    def _get_joint_pos(self, person: Person, joint_index: int):
        skel = getattr(person, 'skeleton', None)
        if not skel or len(skel) <= joint_index:
            return None
        node = skel[joint_index]
        return getattr(node, 'pos', None)

    def process(self, frame: Frame, gl_context=None):
        """Return a list of detected 'arm_stretch' events."""
        events = []
        if not hasattr(frame, 'people') or not frame.people:
            self._prev_stretch.clear()
            return events

        for p in frame.people:
            pid = getattr(p, 'id', id(p))
            skel = getattr(p, 'skeleton', None) or []
            if not skel:
                continue

            lw = self._get_joint_pos(p, J.LEFT_WRIST.value)
            rw = self._get_joint_pos(p, J.RIGHT_WRIST.value)
            head = self._get_joint_pos(p, J.HEAD.value if hasattr(J, 'HEAD') else J.NOSE.value)

            if not (lw and rw and head):
                continue

            # 1️⃣ both wrists higher than head by 100 mm
            cond1 = lw.y > head.y + 100 and rw.y > head.y + 100
            # 2️⃣ wrists close together (<150 mm)
            dist = ((lw.x - rw.x)**2 + (lw.y - rw.y)**2 + (lw.z - rw.z)**2) ** 0.5
            cond2 = dist < 150

            is_stretch = cond1 and cond2
            prev = self._prev_stretch.get(pid, False)

            if is_stretch and not prev:
                ts = getattr(frame, 'timestamp', time.time())
                dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                midpoint = QVector3D((lw.x + rw.x) / 2, (lw.y + rw.y) / 2, (lw.z + rw.z) / 2)
                events.append({
                    "type": "arm_stretch",
                    "timestamp": ts,
                    "time_str": dt,
                    "people": pid,
                    "pos": (int(midpoint.x()), int(midpoint.y()), int(midpoint.z()))
                })

            # draw marker if active
            if is_stretch and gl_context:
                try:
                    midpoint = QVector3D((lw.x + rw.x) / 2, (lw.y + rw.y) / 2, (lw.z + rw.z) / 2)
                    glPushMatrix()
                    glTranslatef(midpoint.x(), midpoint.y(), midpoint.z())
                    glColor4f(1.0, 0.6, 0.0, 0.8)  # warm orange
                    quad = gluNewQuadric()
                    gluSphere(quad, 60.0, 16, 16)
                    gluDeleteQuadric(quad)
                    glPopMatrix()
                except Exception as e:
                    print(f"[OpenGL error while drawing stretch sphere] {e}")

            self._prev_stretch[pid] = is_stretch

        return events
