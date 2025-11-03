""""Gesture recognition for high five"""

import itertools
import time
from datetime import datetime# Transfer to TouchDesigner via OSC

from senseSpaceLib.senseSpace.protocol import Frame, Person        

from PyQt5.QtGui import QVector3D
from OpenGL.GL import *
from OpenGL.GLU import *

class HighFiveDetector:
    """Detect high fives and log them (the hand to hand contact must be above shoulder level)"""

    RIGHT_HAND = 15
    LEFT_HAND = 8
    RIGHT_SHOULDER = 12
    LEFT_SHOULDER = 5

    def __init__(self):
        self._prev_touching = set()  # set of pair keys currently touching
        self._touch_threshold_mm = 200.0  # ~20cm between hands

    def _get_joint_pos(self, person: Person, index: int):
        """Extract joint position as QVector3D"""
        skel = getattr(person, 'skeleton', None) 
        if not skel or len(skel) <= index:
            return None
        pos = getattr(skel[index], 'pos', None)
        if not pos:
            return None
        return QVector3D(float(pos.x), float(pos.y), float(pos.z))
    
    def _get_hand_pos(self, person: Person, side: str):
        """Return left or right hand position"""
        if side == "left":
            return self._get_joint_pos(person, self.LEFT_HAND)
        elif side == "right":
            return self._get_joint_pos(person, self.RIGHT_HAND)
        return None
    
    def _get_shoulder_pos(self, person: Person, side: str):
        """Return left or right shoulder position"""
        if side == "left":
            return self._get_joint_pos(person, self.LEFT_SHOULDER)
        elif side == "right":
            return self._get_joint_pos(person, self.RIGHT_SHOULDER)
        return None
    
    def process(self, frame: Frame, gl_context=None):
        """"Detect high fives among people in frame"""
        events = []
        if not hasattr(frame, 'people') or len(frame.people) < 2:
            self._prev_touching.clear()
            return events
        
        #Store hands
        hands = {}
        for p in frame.people:
            pid = getattr(p, 'id', id(p))
            hands[pid] = {
                "left": self._get_hand_pos(p, "left"),
                "right": self._get_hand_pos(p, "right"),
                "left_shoulder": self._get_shoulder_pos(p, "left"),
                "right_shoulder": self._get_shoulder_pos(p, "right"),
            }

        current_touching = set()

        # Iterate over all pairs of people
        for a, b in itertools.combinations(hands.keys(), 2):
            pa, pb = hands[a], hands[b]

            # All 4 combinations possible 
            combos = [
                ("left", "left"),
                ("right", "right"),
                ("left", "right"),
                ("right", "left"),
            ]

            for side_a, side_b in combos:
                ha = pa[side_a]
                hb = pb[side_b]
                sa = pa[f"{side_a}_shoulder"]
                sb = pb[f"{side_b}_shoulder"]

                #Ensure valid keypoints exist
                if ha is None or hb is None or sa is None or sb is None:
                    continue

                #Both hands MUST be above their own shoulders
                if not (ha.y() > sa.y() and hb.y() > sb.y()):
                    continue

                #Distance between hands
                d = (ha - hb).length()
                if d <= self._touch_threshold_mm:
                    pair = tuple(sorted((a, b)))
                    combo_tag = f"{side_a}-{side_b}"
                    contact_id = (pair, combo_tag)
                    current_touching.add(contact_id)

                    #Average contact position
                    contact = (ha + hb) * 0.5

                    # Trigger only on new contact
                    if contact_id not in self._prev_touching:
                        ts = getattr(frame, 'timestamp', time.time())
                        dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        events.append({
                            "type": "high_five",
                            "timestamp": ts,
                            "time_str": dt,
                            "pos": (int(contact.x()), int(contact.y()), int(contact.z())),
                            "detail": combo_tag,
                        })

                    # Draw visual marker
                    if gl_context:
                        try:
                            glPushMatrix()
                            glTranslatef(contact.x(), contact.y(), contact.z())
                            glColor4f(0.2,0.6,1.0,0.9) #blue
                            quad = gluNewQuadric()
                            gluSphere(quad, 60.0, 16, 16)
                            gluDeleteQuadric(quad)
                            glPopMatrix()
                        except Exception as e:
                            print(f"[OpenGL error while drawing high five sphere] {e}")

        self._prev_touching = current_touching
        return events