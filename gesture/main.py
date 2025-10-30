import sys
import argparse
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
libs_path = os.path.join(repo_root, 'libs')
if os.path.isdir(libs_path) and libs_path not in sys.path:
    sys.path.insert(0, libs_path)

from senseSpaceLib.senseSpace.vizClient import VisualizationClient
from senseSpaceLib.senseSpace.vizWidget import SkeletonGLWidget
from pythonosc import udp_client

# Import detectors 
from detectors.head_bump import HeadBumpDetector

class TDsender:
    def __init__(self, ip="127.0.0.1", port=7000):
        self.osc_client = udp_client.SimpleUDPClient(ip, port)
        print(f"[OSC] Sending all events to TouchDesigner at {ip}:{port}")

    def send(self, address, data):
        try:
            self.osc_client.send_message(address, data)
        except Exception as e:
            print(f"[OSC] Error sending: {e}")


class CustomSkeletonWidget(SkeletonGLWidget):
    def onInit(self):
        self.sender = TDsender("127.0.0.1", 7000)


        # Load detectors
        self.detectors = [
            HeadBumpDetector(),
        ]

        print(f"[INIT] Loaded {len(self.detectors)} detectors.")

    def draw_custom(self, frame):
        for detector in self.detectors:
            events = detector.process(frame, gl_context=True)
            for e in events:
                msg = f"[{e['type'].upper()}] {e['time_str']} - {e['pos']}" # Add for people tag - {e['people']}
                print(msg) #Print to console
                self.sender.send(f"/{e['type']}", msg) #Send to TouchDesigner

def main():
    parser = argparse.ArgumentParser(description="SenseSpace Detection Sphere")
    parser.add_argument("--server", "-s", default="localhost", help="Server IP address")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Server port")
    
    args = parser.parse_args()

    client = VisualizationClient(
        viewer_class=CustomSkeletonWidget,
        server_ip=args.server,
        server_port=args.port,
        window_title="Gesture Detectors"
    )
    client.run()

if __name__ == "__main__":
    main()
