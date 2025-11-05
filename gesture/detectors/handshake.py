"""Gesture recognition: Handshake detection (right hand proximity)."""

import itertools
import time
from datetime import datetime
from senseSpaceLib.senseSpace.protocol import Frame, Person
from senseSpaceLib.senseSpace.enums import Body34Joint as J
from PyQt5.QtGui import QVector3D
from OpenGL.GL import *
from OpenGL.GLU import *


class HandshakeDetector:
    """Detect when two people's right hands come close together."""

    def __init__(self):
        self._prev_touching = set()   # Track pairs currently in contact
        self._last_trigger = {}       # Track last trigger per pair
        self._threshold_mm = 200.0    # 20 cm proximity
        self._cooldown = 1.0          # seconds between allowed triggers per pair

    def _get_hand_pos(self, person: Person):
        """Get the right-hand (or wrist) position as QVector3D."""
        skel = getattr(person, 'skeleton', None) or []
        if not skel:
            return None

        try:
            idx = J.RIGHT_HAND.value
        except Exception:
            idx = J.RIGHT_WRIST.value

        if idx < len(skel):
            node = skel[idx]
            pos = getattr(node, 'pos', None)
            if pos:
                return QVector3D(float(pos.x), float(pos.y), float(pos.z))
        return None

    def process(self, frame: Frame, gl_context=None):
        """Return a list of detected 'handshake' events."""
        events = []
        if not hasattr(frame, 'people') or len(frame.people) < 2:
            self._prev_touching.clear()
            return events

        # Collect hand positions
        hands = {}
        for p in frame.people:
            hp = self._get_hand_pos(p)
            if hp is not None:
                hands[getattr(p, 'id', id(p))] = hp

        current_touching = set()

        # Check every pair of hands
        for a, b in itertools.combinations(hands.keys(), 2):
            pa, pb = hands[a], hands[b]
            if pa is None or pb is None:
                continue

            d = (pa - pb).length()
            if d <= self._threshold_mm:
                pair = tuple(sorted((a, b)))
                current_touching.add(pair)

                # cooldown check
                now = time.time()
                last_time = self._last_trigger.get(pair, 0)
                cooldown_ok = (now - last_time) >= self._cooldown

                if pair not in self._prev_touching and cooldown_ok:
                    ts = getattr(frame, 'timestamp', now)
                    dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    contact = (pa + pb) * 0.5

                    events.append({
                        "type": "handshake",
                        "timestamp": ts,
                        "time_str": dt,
                        "people": pair,
                        "pos": (int(contact.x()), int(contact.y()), int(contact.z()))
                    })

                    self._last_trigger[pair] = now

                # Optional visual sphere
                if gl_context:
                    try:
                        contact = (pa + pb) * 0.5
                        glPushMatrix()
                        glTranslatef(contact.x(), contact.y(), contact.z())
                        glColor4f(0.0, 1.0, 0.0, 0.8)  # green
                        quad = gluNewQuadric()
                        gluSphere(quad, 40.0, 12, 12)
                        gluDeleteQuadric(quad)
                        glPopMatrix()
                    except Exception as e:
                        print(f"[OpenGL error while drawing handshake sphere] {e}")

        self._prev_touching = current_touching
        return events
