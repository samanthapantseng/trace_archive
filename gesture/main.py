""""Gesture recognition main module."""

import argparse
import sys
import os
import itertools
import time
from datetime import datetime

# Ensure local 'libs' folder is on sys.path when running from repo
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
libs_path = os.path.join(repo_root, 'libs')
if os.path.isdir(libs_path) and libs_path not in sys.path:
    sys.path.insert(0, libs_path)

# Import from shared library
from senseSpaceLib.senseSpace.vizClient import VisualizationClient  # noqa: E402
from senseSpaceLib.senseSpace.protocol import Frame, Person         # noqa: E402
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget     # noqa: E402
from senseSpaceLib.senseSpace.enums import Body34Joint as J         # noqa: E402

from PyQt5.QtGui import QVector3D
from OpenGL.GL import *
from OpenGL.GLU import *


def _person_key(person):
    """Stable key for a person object."""
    return getattr(person, 'id', getattr(person, 'track_id', id(person)))


def _qvec_from_pos(pos) -> QVector3D:
    """Convert Position or dict to QVector3D."""
    if pos is None:
        return None
    if hasattr(pos, 'x'):
        return QVector3D(float(pos.x), float(pos.y), float(pos.z))
    return QVector3D(float(pos["x"]), float(pos["y"]), float(pos["z"]))


def _get_head_pos(person: Person) -> QVector3D:
    """
    Get the head position from joint #26 (HEAD).
    """
    skel = getattr(person, 'skeleton', None) or []
    head_idx = 26  # J.HEAD
    
    if head_idx < len(skel):
        return _qvec_from_pos(getattr(skel[head_idx], 'pos', None))
    
    return None


class CustomSkeletonWidget(SkeletonGLWidget):
    """Detect head-to-head touch and log it."""

    def onInit(self):
        self._prev_touching = set()  # set of pair keys currently touching
        self._touch_threshold_mm = 300.0  # ~30cm between head centers

    def onClose(self):
        pass

    def draw_custom(self, frame: Frame):
        """Detect touches and optionally draw markers."""
        if not hasattr(frame, 'people') or len(frame.people) < 2:
            self._prev_touching.clear()
            return

        # Build head positions for all people this frame
        heads = {}  # key -> QVector3D
        for person in frame.people:
            key = _person_key(person)
            hp = _get_head_pos(person)
            if hp is not None:
                heads[key] = hp

        current_touching = set()

        # Check all unique pairs
        for a_key, b_key in itertools.combinations(heads.keys(), 2):
            pa = heads[a_key]
            pb = heads[b_key]
            # distance in mm
            d = (pa - pb).length()
            if d <= self._touch_threshold_mm:
                pair = tuple(sorted((a_key, b_key)))
                current_touching.add(pair)
                # Rising edge: newly touching this frame
                if pair not in self._prev_touching:
                    contact = (pa + pb) * 0.5
                    # Use frame timestamp (seconds)
                    ts = getattr(frame, 'timestamp', time.time())
                    dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    print(f"[HEAD-TOUCH] epoch={ts:.3f}s, time={dt}, people=({a_key},{b_key}), pos=({contact.x():.1f}, {contact.y():.1f}, {contact.z():.1f}) mm")

                # Optional: draw a small marker at contact point
                try:
                    contact = (pa + pb) * 0.5
                    glPushMatrix()
                    glTranslatef(contact.x(), contact.y(), contact.z())
                    glColor4f(1.0, 0.6, 0.0, 0.8)  # orange
                    quad = gluNewQuadric()
                    gluSphere(quad, 60.0, 16, 16)
                    gluDeleteQuadric(quad)
                    glPopMatrix()
                except Exception:
                    pass

        # Update for next frame
        self._prev_touching = current_touching


def main():
    parser = argparse.ArgumentParser(description="Detect head-to-head touches and log events")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP address")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    args = parser.parse_args()

    client = VisualizationClient(
        viewer_class=CustomSkeletonWidget,
        server_ip=args.server,
        server_port=args.port,
        window_title="Head-Touch Detector"
    )
    
    ok = client.run()
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()

