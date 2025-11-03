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

# Add OSC import
from pythonosc import udp_client


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


def _get_right_hand_pos(person: Person) -> QVector3D:
    """
    Get the right-hand position using the BODY_34 enum.
    Falls back to RIGHT_WRIST if RIGHT_HAND not available in this enum.
    """
    skel = getattr(person, 'skeleton', None) or []
    try:
        hand_idx = J.RIGHT_HAND.value
    except Exception:
        hand_idx = J.RIGHT_WRIST.value
    if hand_idx < len(skel):
        return _qvec_from_pos(getattr(skel[hand_idx], 'pos', None))
    return None

 
class CustomSkeletonWidget(SkeletonGLWidget):
    """Detect right-hand handshakes and log/send events."""

    def onInit(self):
        # handshake detection state
        self._prev_handshakes = set()
        self._handshake_threshold_mm = 200.0  # 20cm
        # OSC sender for TouchDesigner (optional)
        self.TDsender = TDsender("127.0.0.1", 8000)

    def onClose(self):
        pass

    def draw_custom(self, frame: Frame):
        # self.TDsender.sendData("/test",0)
        """Detect touches and optionally draw markers."""
        if not hasattr(frame, 'people') or len(frame.people) < 2:
            # Clear handshake state when fewer than two people are present
            self._prev_handshakes.clear()
            return

        # --- Handshake detection (right-hand) ---
        hands = {}  # key -> QVector3D
        for person in frame.people:
            key = _person_key(person)
            hp = _get_right_hand_pos(person)
            if hp is not None:
                hands[key] = hp

        current_handshakes = set()
        for a_key, b_key in itertools.combinations(hands.keys(), 2):
            ra = hands[a_key]
            rb = hands[b_key]
            d = (ra - rb).length()
            if d <= self._handshake_threshold_mm:
                pair = tuple(sorted((a_key, b_key)))
                current_handshakes.add(pair)
                if pair not in self._prev_handshakes:
                    # rising edge: new handshake
                    contact = (ra + rb) * 0.5
                    ts = getattr(frame, 'timestamp', time.time())
                    dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    data = f"[HANDSHAKE] epoch={ts:.3f}s, time={dt}, people=({a_key},{b_key}), pos=({contact.x():.1f}, {contact.y():.1f}, {contact.z():.1f}) mm"
                    print(data)
                    # send to TouchDesigner if available
                    try:
                        self.TDsender.sendData("/handshake", data)
                    except Exception:
                        pass
                    # Optional visual marker at contact
                    try:
                        glPushMatrix()
                        glTranslatef(contact.x(), contact.y(), contact.z())
                        glColor4f(0.0, 1.0, 0.0, 0.8)  # green
                        quad = gluNewQuadric()
                        gluSphere(quad, 40.0, 12, 12)
                        gluDeleteQuadric(quad)
                        glPopMatrix()
                    except Exception:
                        pass

        self._prev_handshakes = current_handshakes
        

class TDsender:
    """Simple OSC sender for TouchDesigner - NOT a QWidget."""

    def __init__(self, td_ip="127.0.0.1", td_port=7000):
        # Initialize OSC client for TouchDesigner
        self.td_ip = td_ip
        self.td_port = td_port
        self.osc_client = udp_client.SimpleUDPClient(self.td_ip, self.td_port)
        print(f"[OSC] Sending to TouchDesigner at {self.td_ip}:{self.td_port}")
        
    def sendData(self, address, data):
        """Send OSC message to TouchDesigner."""
        try:
            self.osc_client.send_message(address, data)
            print(f"[OSC] Sent to {address}: {data}")
        except Exception as e:
            print(f"[OSC] Error sending: {e}")


def main():
    parser = argparse.ArgumentParser(description="Detect handshakes (right-hand) and log events")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP address")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    args = parser.parse_args()

    # def create_widget(*widget_args, **widget_kwargs):
    #     widget = CustomSkeletonWidget(*widget_args, **widget_kwargs)
    #     widget.sender = sender  # Share the same sender instance
    #     return widget

    client = VisualizationClient(
        viewer_class=CustomSkeletonWidget,
        server_ip=args.server,
        server_port=args.port,
        window_title="Handshake Detector"
    )

    ok = client.run()
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()

