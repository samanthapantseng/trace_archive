import sys
import argparse
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
libs_path = os.path.join(repo_root, 'libs')
if os.path.isdir(libs_path) and libs_path not in sys.path:
    sys.path.insert(0, libs_path)

from senseSpaceLib.senseSpace.vizClient import VisualizationClient
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget
from senseSpaceLib.senseSpace.enums import UniversalJoint
from senseSpaceLib.senseSpace.vecMathHelper import insidePoly, getPlaneIntersection, getNormal
from PyQt5.QtGui import QVector3D
#from senseSpaceLib.senseSpace.visualization import draw_skeletons_with_bones
from pythonosc import udp_client
from OpenGL.GL import *
import json
from glob import glob

# Import detectors 
#from detectors.head_bump import HeadBumpDetector
#from detectors.high_five import HighFiveDetector
from detectors.hug import HugDetector
#from detectors.arm_stretch import ArmStretchDetector
#from detectors.arm_spread import ArmSpreadDetector 
#from detectors.handshake import HandshakeDetector
#from detectors.titanic import TitanicDetector
from detectors.trail import TrailDetector

class MultiSender:
    def __init__(self, targets):
        self.clients = []
        for ip, port in targets:
            try:
                client = udp_client.SimpleUDPClient(ip, port)
                print(f"[OSC] Sending events to {ip}:{port}")
                self.clients.append(client)
            except Exception as e:
                print(f"[OSC] Error setting up client for {ip}:{port} - {e}")

    def send(self, address, data):
        for client in self.clients:
            try:
                client.send_message(address, data)
            except Exception as e:
                print(f"[OSC] Error sending to {client._address}:{client._port} - {e}")

class CustomSkeletonWidget(SkeletonGLWidget):
    def __init__(self, *args, calibration_polygon=None, **kwargs):
        self.calibration_polygon = calibration_polygon
        super().__init__(*args, **kwargs)
    
    def onInit(self):
        if self.calibration_polygon:
            print(f"[INIT] Calibration polygon active {len(self.calibration_polygon)}")

        self.sender = MultiSender([
            # Change IP depending on computer being used
            ("192.168.1.18", 8000), # Receiving for Sound 
            ("192.168.1.13", 8001), # Receiving for TD
        ])

        # Load detectors
        self.detectors = [
            #HeadBumpDetector(),
            #HighFiveDetector(),
            HugDetector(),
            #ArmStretchDetector(),
            #ArmSpreadDetector(),
            #HandshakeDetector(),
            #TitanicDetector(),
            TrailDetector()
        ]

        print(f"[INIT] Loaded {len(self.detectors)} detectors.")
    
    def draw_polygon_floor(self):
        # Draw calibration polygon on the floor
        if not self.calibration_polygon:
            return
        
        glLineWidth(3)
        glColor4f(0.1, 1.0, 0.4, 0.6)
        glBegin(GL_LINE_LOOP)
        for (x, z) in self.calibration_polygon:
            glVertex3f(x, 0, z)
        glEnd()
        glLineWidth(1)

    def project_pelvis_to_floor(self, person, frame):
        body_model = frame.body_model if frame.body_model else "BODY_34"
        joint = person.get_joint(UniversalJoint.PELVIS, body_model)
        if not joint:
            return None

        pos, _ = joint
        pelvis = QVector3D(pos.x, pos.y, pos.z)

        floor_p1 = QVector3D(0, 0, 0)
        floor_p2 = QVector3D(1000, 0, 0)
        floor_p3 = QVector3D(0, 0, 1000)

        ray_dir = QVector3D(0, -1, 0)
        return getPlaneIntersection(ray_dir, floor_p1, floor_p2, floor_p3, pelvis)

    def draw_custom(self, frame):
        """Process frame with all detectors and draw polygon."""
        # ------------------------------------
        # FILTER PEOPLE INSIDE POLYGON
        # ------------------------------------
        if self.calibration_polygon:
            self.draw_polygon_floor()

            filtered_people = []
            polygon_plane_normal = getNormal(
                QVector3D(0,0,0), QVector3D(1000,0,0), QVector3D(0,0,1000)
            )
            floor_polygon = [QVector3D(x, 0, z) for (x, z) in self.calibration_polygon]

            for p in frame.people:
                proj = self.project_pelvis_to_floor(p, frame)
                if proj and insidePoly(proj, polygon_plane_normal, floor_polygon):
                    filtered_people.append(p)
        else:
            filtered_people = frame.people

        # ------------------------------------
        # RUN DETECTORS USING FILTERED PEOPLE
        # ------------------------------------
        original_people = frame.people          # save original
        frame.people = filtered_people          # replace for detectors

        events = []
        for detector in self.detectors:
            result = detector.process(frame, gl_context=True)
            if result:
                events.extend(result)

        frame.people = original_people          # restore original list

        # ------------------------------------
        # SEND EVENTS
        # ------------------------------------
        for e in events:
            addr = f"/{e['type']}"
            msg = f"time={e['time_str']}|pos={e['pos']}"
            print(addr, msg)
            self.sender.send(addr, msg)

def main():
    parser = argparse.ArgumentParser(description="SenseSpace Detection Sphere")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP address")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--rec", type=str, default=None, help="Playback mode: path to .ssrec file")
    parser.add_argument("--auto-record", type=int, default=0, 
                       help="Auto-record for N seconds (0 = manual control with 'R' key)")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Calibration keyword")
    
    args = parser.parse_args()
    
    calibration_polygon = None
    if args.calibration:
        calib_dir = os.path.join(os.path.dirname(__file__), "calibrations")
        matches = sorted(glob(os.path.join(calib_dir, f"*_{args.calibration}*.json")))
        if matches:
            with open(matches[-1], 'r') as f:
                calibration_polygon = json.load(f)
            print(f"[CALIBRATION] Loaded from {matches[-1]}")
        else:
            print(f"[CALIBRATION] No calibration found for '{args.calibration}'")

    def widget_factory():
        return CustomSkeletonWidget(calibration_polygon=calibration_polygon)

    client = VisualizationClient(
        viewer_class=widget_factory,
        server_ip=args.server,
        server_port=args.port,
        playback_file=args.rec,
        window_title="Gesture Detectors",
    )
    
    client.run()

if __name__ == "__main__":
    main()
