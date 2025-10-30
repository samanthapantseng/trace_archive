""""Gesture recognition for head bump"""

import itertools
import time
from datetime import datetime# Transfer to TouchDesigner via OSC

from senseSpaceLib.senseSpace.protocol import Frame, Person        

from PyQt5.QtGui import QVector3D
from OpenGL.GL import *
from OpenGL.GLU import *

class HeadBumpDetector:
    """Detect head-to-head touches and log them."""

    def __init__(self):
        self._prev_touching = set()  # set of pair keys currently touching
        self._touch_threshold_mm = 250.0  # ~25cm between head centers

    def _get_head_pos(self, person: Person):
        skel = getattr(person, 'skeleton', None) 
        if not skel or len(skel) <= 26:
            return None
        pos = getattr(skel[26], 'pos', None)
        if not pos:
            return None
        return QVector3D(float(pos.x), float(pos.y), float(pos.z))
    
    def process(self, frame: Frame, gl_context=None):
        """"Return a list of detected events"""
        events = []
        if not hasattr(frame, 'people') or len(frame.people) < 2:
            self._prev_touching.clear()
            return events
        
        heads = {}
        for p in frame.people:
            hp = self._get_head_pos(p)
            if hp is not None:
                heads[getattr(p, 'id', id(p))] = hp

        current_touching = set()
        for a, b in itertools.combinations(heads.keys(), 2):
            pa, pb = heads[a], heads[b]
            d = (pa - pb).length()
            if d <= self._touch_threshold_mm:
                pair = tuple(sorted((a, b)))
                current_touching.add(pair)

                # Always define contact (used in both event + OpenGL)
                contact = (pa + pb) * 0.5

                # Only trigger event on *new* touches
                if pair not in self._prev_touching:
                    ts = getattr(frame, 'timestamp', time.time())
                    dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    events.append({
                        "type": "head_touch",
                        "timestamp": ts,
                        "time_str": dt,
                        "people": pair,
                        "pos": (int(contact.x()), int(contact.y()), int(contact.z()))
                    })

                # Draw visual marker (safe because contact is always defined)
                if gl_context:
                    try:
                        glPushMatrix()
                        glTranslatef(contact.x(), contact.y(), contact.z())
                        glColor4f(1.0, 0.6, 0.0, 0.8)
                        quad = gluNewQuadric()
                        gluSphere(quad, 60.0, 16, 16)
                        gluDeleteQuadric(quad)
                        glPopMatrix()
                    except Exception as e:
                        print(f"[OpenGL error while drawing sphere] {e}")


        self._prev_touching = current_touching
        return events