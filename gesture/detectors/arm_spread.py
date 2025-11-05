"""Gesture recognition: Arms Spread (open / embrace pose)"""

import time
from datetime import datetime
from senseSpaceLib.senseSpace.protocol import Frame, Person
from senseSpaceLib.senseSpace.enums import Body34Joint as J
from PyQt5.QtGui import QVector3D
from OpenGL.GL import *
from OpenGL.GLU import *


class ArmSpreadDetector:
    """Detect when a person spreads both arms wide — open / embrace gesture."""

    def __init__(self):
        self._prev_spread = {}
        self._last_trigger = {}
        #self._cooldown = 0.5  # seconds between triggers (prevents spam)

    def _get_joint_pos(self, person: Person, joint_index: int):
        """Safely extract a joint position (QVector3D) from a skeleton."""
        skel = getattr(person, 'skeleton', None)
        if not skel or len(skel) <= joint_index:
            return None
        node = skel[joint_index]
        return getattr(node, 'pos', None)

    def process(self, frame: Frame, gl_context=None):
        """Return a list of detected 'arm_spread' events."""
        events = []
        if not hasattr(frame, 'people') or not frame.people:
            self._prev_spread.clear()
            return events

        #now = time.time()

        for p in frame.people:
            pid = getattr(p, 'id', id(p))
            skel = getattr(p, 'skeleton', None) or []
            if not skel:
                continue

            lw = self._get_joint_pos(p, J.LEFT_WRIST.value)
            rw = self._get_joint_pos(p, J.RIGHT_WRIST.value)
            chest = self._get_joint_pos(p, J.SPINE_CHEST.value if hasattr(J, 'SPINE_CHEST') else J.NECK.value)
            lshoulder = self._get_joint_pos(p, J.LEFT_SHOULDER.value)
            rshoulder = self._get_joint_pos(p, J.RIGHT_SHOULDER.value)

            if not (lw and rw and chest and lshoulder and rshoulder):
                continue

            # 1️⃣ Both wrists roughly level with shoulders (±150mm)
            shoulder_y = (lshoulder.y + rshoulder.y) / 2.0
            cond_height = abs(lw.y - shoulder_y) < 150 and abs(rw.y - shoulder_y) < 150

            # 2️⃣ Wrists far from body center (>600mm from chest center)
            dist_left = abs(lw.x - chest.x)
            dist_right = abs(rw.x - chest.x)
            cond_distance = dist_left > 500 and dist_right > 500

            is_spread = cond_height and cond_distance
            prev = self._prev_spread.get(pid, False)

            # Trigger only if cooldown passed
            #last_time = self._last_trigger.get(pid, 0)
            #if is_spread and (now - last_time) >= self._cooldown:
            if is_spread and not prev:
                ts = getattr(frame, 'timestamp', time.time())
                dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                midpoint = QVector3D(
                    (lw.x + rw.x) / 2, (lw.y + rw.y) / 2, (lw.z + rw.z) / 2
                )
                events.append({
                    "type": "arm_spread",
                    "timestamp": ts,
                    "time_str": dt,
                    "people": pid,
                    "pos": (int(midpoint.x()), int(midpoint.y()), int(midpoint.z()))
                })
                #self._last_trigger[pid] = now

            # Draw a cyan visual marker when active
            if is_spread and gl_context:
                try:
                    midpoint = QVector3D((lw.x + rw.x) / 2, (lw.y + rw.y) / 2, (lw.z + rw.z) / 2)
                    glPushMatrix()
                    glTranslatef(midpoint.x(), midpoint.y(), midpoint.z())
                    glColor4f(0.0, 1.0, 1.0, 0.6)  # cyan aura
                    quad = gluNewQuadric()
                    gluSphere(quad, 80.0, 16, 16)
                    gluDeleteQuadric(quad)
                    glPopMatrix()
                except Exception as e:
                    print(f"[OpenGL error while drawing arm spread sphere] {e}")

            self._prev_spread[pid] = is_spread

        return events
