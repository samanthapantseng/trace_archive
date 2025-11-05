""""Gesture recognition for mutual hugs (consensual)"""

import itertools
import time
from datetime import datetime

from senseSpaceLib.senseSpace.protocol import Frame, Person

from PyQt5.QtGui import QVector3D
from OpenGL.GL import *
from OpenGL.GLU import *

class HugDetector:
    """
    Detect mutual (consensual) hugs.

    Logic:
      - A person is considered "arms-closed" if their left and right hands are close.
      - Compute a person's "hug center" as the average of (left_hand + right_hand + pelvis).
      - Another person's pelvis must be within `embrace_radius_mm` of that hug center
        for them to be considered "embraced" by the first person.
      - A hug is confirmed only when both persons are "arms-closed" and each person's
        pelvis is inside the other's embrace zone (mutual).
    """

    # Joint index
    LEFT_HAND = 8
    RIGHT_HAND = 15
    PELVIS = 0

    def __init__(self, 
                 arm_closure_threshold_mm: float = 300.0,
                 embrace_radius_mm: float = 400.0):
        # Parameters
        self._arm_closure_threshold_mm = arm_closure_threshold_mm
        self._embrace_radius_mm = embrace_radius_mm

        # State
        self._prev_hugs = set()

    def _safe_joint(self, person: Person, idx: int):
        """Return QVector3D for joint index"""
        skel = getattr(person, 'skeleton', None) 
        if not skel or len(skel) <= idx:
            return None
        j = getattr(skel[idx], 'pos', None)
        if not j:
            return None
        return QVector3D(float(j.x), float(j.y), float(j.z))
    
    def _left_hand(self, person: Person):
        return self._safe_joint(person, self.LEFT_HAND)
    
    def _right_hand(self, person: Person):
        return self._safe_joint(person, self.RIGHT_HAND)
    
    def _pelvis(self, person: Person):
        return self._safe_joint(person, self.PELVIS)
    
    def _person_id(self, person: Person):
        return getattr(person, 'id', getattr(person, 'track_id', id(person)))

    def _is_arms_closed(self, left: QVector3D, right: QVector3D) -> bool:
        """Return true if person's arms are closed"""
        if left is None or right is None:
            return False
        return (left - right).length() <= self._arm_closure_threshold_mm
    
    def _hug_center(self, left: QVector3D, right: QVector3D, pelvis: QVector3D) -> QVector3D:
        """Compute average point used as hug center"""
        return (left + right + pelvis) * (1.0 / 3.0)
    
    def process(self, frame: Frame, gl_context=None):
        """Return list of hug events"""
        events = []
        if not hasattr(frame, 'people') or len(frame.people) < 2:
            self._prev_hugs.clear()
            return events
        
        # Build map of person ID
        people_map = {}

        # Store original Person objects
        person_obj_map = {}
        for p in frame.people:
            pid = self._person_id(p)
            left = self._left_hand(p)
            right = self._right_hand(p)
            pelvis = self._pelvis(p)
            people_map[pid] = {
                "left": left,
                "right": right,
                "pelvis": pelvis,
                "arms_closed": self._is_arms_closed(left, right),
            }
            person_obj_map[pid] = p

        current_hugs = set()

        # Iterate pairs
        for a, b in itertools.combinations(people_map.keys(), 2):
            A = people_map[a]
            B = people_map[b]

            # Need both pelvis and hands available for checking
            if A["pelvis"] is None or B["pelvis"] is None:
                continue

            # Both persons must have hands available to evaluate "arms closed"
            if not (A["arms_closed"] and B["arms_closed"]):
                continue

            # Compute hug centers
            if A["left"] is None or A["right"] is None or B["left"] is None or B["right"] is None:
                continue

            hug_center_A = self._hug_center(A["left"], A["right"], A["pelvis"])
            hug_center_B = self._hug_center(B["left"], B["right"], B["pelvis"])

            # Check if each person's pelvis is within the other's embrace radius
            dist_Bpelvis_to_Acenter = (B["pelvis"] - hug_center_A).length()
            dist_Apelvis_to_Bcenter = (A["pelvis"] - hug_center_B).length()

            inside_A = dist_Bpelvis_to_Acenter <= self._embrace_radius_mm
            inside_B = dist_Apelvis_to_Bcenter <= self._embrace_radius_mm

            if inside_A and inside_B:
                pair = tuple(sorted((a, b)))
                current_hugs.add(pair)

                # Determine contact
                contact = (hug_center_A + hug_center_B) * 0.5

                # Only trigger on "new" hug
                if pair not in self._prev_hugs:
                    ts = getattr(frame, 'timestamp', time.time())
                    dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    events.append({
                        "type": "hug",
                        "timestamp": ts,
                        "time_str": dt,
                        "people": pair,
                        "pos": (int(contact.x()), int(contact.y()), int(contact.z())),
                        "detail": {
                            "hug_center_A": (int(hug_center_A.x()), int(hug_center_A.y()), int(hug_center_A.z())),
                            "hug_center_B": (int(hug_center_B.x()), int(hug_center_B.y()), int(hug_center_B.z())),
                            "dist_Bpelvis_to_Acenter": float(dist_Bpelvis_to_Acenter),
                            "dist_Apelvis_to_Bcenter": float(dist_Apelvis_to_Bcenter),
                        }
                    })

                # Debug visualisation
                if gl_context:
                    try:
                        glPushMatrix()
                        glTranslatef(contact.x(), contact.y(), contact.z())
                        glColor4f(0.0, 0.9, 0.2, 0.95)  # green for confirmed hug
                        quad = gluNewQuadric()
                        gluSphere(quad, 80.0, 18, 18)
                        gluDeleteQuadric(quad)
                        glPopMatrix()
                    except Exception as e:
                        print(f"[OpenGL error while drawing confirmed hug] {e}")

            # Draw debug visuals for this pair (pelvis -> pelvis line, hug centers):
            if gl_context:
                try:
                    pa = A["pelvis"]
                    pb = B["pelvis"]
                    """ if pa is not None and pb is not None:
                        glBegin(GL_LINES)
                        glColor4f(1.0, 0.2, 0.8, 0.8)  # magenta line
                        glVertex3f(pa.x(), pa.y(), pa.z())
                        glVertex3f(pb.x(), pb.y(), pb.z())
                        glEnd() """

                    # small cyan spheres at hug centers (if available)
                    if A["left"] is not None and A["right"] is not None and A["pelvis"] is not None:
                        hA = hug_center_A
                        glPushMatrix()
                        glTranslatef(hA.x(), hA.y(), hA.z())
                        glColor4f(0.2, 0.8, 0.9, 0.9)  # cyan
                        quad = gluNewQuadric()
                        gluSphere(quad, 35.0, 12, 12)
                        gluDeleteQuadric(quad)
                        glPopMatrix()

                    if B["left"] is not None and B["right"] is not None and B["pelvis"] is not None:
                        hB = hug_center_B
                        glPushMatrix()
                        glTranslatef(hB.x(), hB.y(), hB.z())
                        glColor4f(0.2, 0.8, 0.9, 0.9)  # cyan
                        quad = gluNewQuadric()
                        gluSphere(quad, 35.0, 12, 12)
                        gluDeleteQuadric(quad)
                        glPopMatrix()

                except Exception as e:
                    print(f"[OpenGL error while drawing debug hug visuals] {e}")

        # Update state
        self._prev_hugs = current_hugs
        return events