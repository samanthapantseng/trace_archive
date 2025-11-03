#!/usr/bin/env python3
"""
Example: Raised arm detection with universal joint enum
"""

import argparse
import sys
import os

# Ensure local 'libs' folder is on sys.path when running from repo
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
libs_path = os.path.join(repo_root, 'libs')
if os.path.isdir(libs_path) and libs_path not in sys.path:
    sys.path.insert(0, libs_path)

# Import from shared library
from senseSpaceLib.senseSpace.vizClient import VisualizationClient
from senseSpaceLib.senseSpace.protocol import Frame, Person
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget
from senseSpaceLib.senseSpace.enums import Body34Joint as J

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QVector3D
import time
from datetime import datetime


def _person_key(person):
    """Stable key for a person object."""
    return getattr(person, 'id', getattr(person, 'track_id', id(person)))
from OpenGL.GL import *
from OpenGL.GLU import *


def check_stretch(person: Person, body_model: str = "BODY_34") -> tuple[bool, QVector3D]:
    """Detect "arms over head with hands crossed" pose for a single person.

    Conditions:
    1. Both wrists higher than head by at least 100 mm.
    2. Distance between wrists less than 150 mm.

    Returns: (is_stretch, midpoint_qvector)
    """
    # Access joints directly from the BODY_34 skeleton array using enum values
    skel = getattr(person, 'skeleton', None) or []
    # indices from Body34Joint
    li = J.LEFT_WRIST.value
    ri = J.RIGHT_WRIST.value
    head_i = J.HEAD.value if hasattr(J, 'HEAD') else None
    nose_i = J.NOSE.value if hasattr(J, 'NOSE') else None

    def _pos_from_idx(idx):
        if idx is None:
            return None
        if idx < len(skel):
            node = skel[idx]
            return getattr(node, 'pos', None)
        return None

    lw_pos = _pos_from_idx(li)
    rw_pos = _pos_from_idx(ri)
    head_pos = _pos_from_idx(head_i) or _pos_from_idx(nose_i)

    if not (lw_pos and rw_pos and head_pos):
        return False, None

    # Condition 1: both wrists higher than head by 100mm
    if not (lw_pos.y > head_pos.y + 100 and rw_pos.y > head_pos.y + 100):
        return False, None

    # Condition 2: wrists close together (<150mm)
    dx = lw_pos.x - rw_pos.x
    dy = lw_pos.y - rw_pos.y
    dz = lw_pos.z - rw_pos.z
    dist = (dx*dx + dy*dy + dz*dz) ** 0.5
    if dist > 150.0:
        return False, None

    # midpoint
    midx = (lw_pos.x + rw_pos.x) / 2.0
    midy = (lw_pos.y + rw_pos.y) / 2.0
    midz = (lw_pos.z + rw_pos.z) / 2.0
    midpoint = QVector3D(float(midx), float(midy), float(midz))
    return True, midpoint


class CustomSkeletonWidget(SkeletonGLWidget):
    """Custom visualization with raised arm detection"""
    
    def onInit(self):
        """Initialize custom state"""
        self.sphere_radius = 80.0  # Normal size
        self.sphere_size_large = False  # Track size state
        self.quadric = gluNewQuadric()  # Create quadric once
        # track previous stretch state per person key
        self._prev_stretch = {}
    
    def onClose(self):
        """Cleanup resources"""
        if hasattr(self, 'quadric') and self.quadric:
            gluDeleteQuadric(self.quadric)
    
    def keyPressEvent(self, event):
        """Handle keyboard input"""
        if event.key() == Qt.Key_Space:
            # Toggle sphere size
            self.sphere_size_large = not self.sphere_size_large
            if self.sphere_size_large:
                self.sphere_radius = 160.0  # 2x bigger
                print("[INFO] Sphere size: LARGE (160mm)")
            else:
                self.sphere_radius = 80.0  # Normal
                print("[INFO] Sphere size: NORMAL (80mm)")
        else:
            # Pass other keys to parent
            super().keyPressEvent(event)
       
    def draw_custom(self, frame: Frame):
        """Draw red spheres at raised hands"""
        if not hasattr(frame, 'people') or not frame.people:
            return
        
        # Get body model from frame
        body_model = frame.body_model if frame.body_model else "BODY_34"
        
        # Check each person for the 'stretch' pose
        for person in frame.people:
            key = _person_key(person)
            is_stretch, midpoint = check_stretch(person, body_model)

            # Rising edge: just detected stretch
            prev = self._prev_stretch.get(key, False)
            if is_stretch and not prev:
                ts = getattr(frame, 'timestamp', time.time())
                dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                data = f"[STRETCH] epoch={ts:.3f}s, time={dt}, person={key}, pos=({midpoint.x():.1f}, {midpoint.y():.1f}, {midpoint.z():.1f}) mm"
                print(data)

            # Draw single red sphere at midpoint when stretching
            if is_stretch and midpoint is not None:
                try:
                    glPushMatrix()
                    glTranslatef(midpoint.x(), midpoint.y(), midpoint.z())
                    glColor4f(1.0, 0.0, 0.0, 0.8)
                    rad = self.sphere_radius * (2.0 if self.sphere_size_large else 1.0)
                    gluSphere(self.quadric, rad, 32, 32)
                    glPopMatrix()
                except Exception:
                    pass

            self._prev_stretch[key] = is_stretch


def main():
    parser = argparse.ArgumentParser(description="SenseSpace Raised Arm Detection")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP address")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    
    args = parser.parse_args()
    
    # Create and run visualization client
    client = VisualizationClient(
        viewer_class=CustomSkeletonWidget,
        server_ip=args.server,
        server_port=args.port,
        window_title="Raised Arm Detection - Press SPACE to toggle sphere size"
    )
    
    success = client.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
