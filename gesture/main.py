import sys
import argparse
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
libs_path = os.path.join(repo_root, 'libs')
if os.path.isdir(libs_path) and libs_path not in sys.path:
    sys.path.insert(0, libs_path)

from senseSpaceLib.senseSpace.vizClient import VisualizationClient
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget
#from senseSpaceLib.senseSpace.visualization import draw_skeletons_with_bones
from pythonosc import udp_client

# Import detectors 
from detectors.head_bump import HeadBumpDetector
from detectors.high_five import HighFiveDetector
from detectors.hug import HugDetector
from detectors.arm_stretch import ArmStretchDetector
from detectors.arm_spread import ArmSpreadDetector 
from detectors.handshake import HandshakeDetector

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
    def onInit(self):
        self.sender = MultiSender([
            # Change IP depending on computer being used
            ("192.168.1.18", 8000), # Receiving for TD 
            ("192.168.1.13", 8001), # Receiving for Sound
        ])
        # self.min_confidence = 60.0 # Threshold for considering a person valid

        # Load detectors
        self.detectors = [
            HeadBumpDetector(),
            HighFiveDetector(),
            HugDetector(),
            ArmStretchDetector(),
            ArmSpreadDetector(),
            HandshakeDetector()
        ]

        print(f"[INIT] Loaded {len(self.detectors)} detectors.")

    """def draw_skeletons(self, frame):
        #Override the filter of low confidence skeletons
        if not hasattr(frame, 'people') or not frame.people:
            return
        
        # Filter low confidence people
        filtered_people = [
            person for person in frame.people
            if person.confidence >= self.min_confidence
        ]

        if not filtered_people:
            return
    
        draw_skeletons_with_bones(
            filtered_people,
            joint_color=(0.2, 0.8, 1.0),
            bone_color=(0.8, 0.2, 0.2),
            show_orientation=self.show_orientation,
        )"""

    def draw_custom(self, frame):
        #print(f"[FRAME] Processing frame at timestamp {getattr(frame, 'timestamp', 'N/A')}")

        for detector in self.detectors:
            events = detector.process(frame, gl_context=True)
            for e in events:
                msg = f"time={e['time_str']}|pos={e['pos']}".strip() # Add for people tag - {e['people']}
                addr = f"/{e['type']}".strip()
                print(msg) #Print to console
                self.sender.send(addr, msg) #Send via OSC

def main():
    parser = argparse.ArgumentParser(description="SenseSpace Detection Sphere")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP address")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    parser.add_argument("--rec", type=str, default=None, help="Playback mode: path to .ssrec file")
    parser.add_argument("--auto-record", type=int, default=0, 
                       help="Auto-record for N seconds (0 = manual control with 'R' key)")
    
    args = parser.parse_args()

    client = VisualizationClient(
        viewer_class=CustomSkeletonWidget,
        server_ip=args.server,
        server_port=args.port,
        playback_file=args.rec,
        window_title="Gesture Detectors"
    )
    client.run()

if __name__ == "__main__":
    main()
