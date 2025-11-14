import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os
from datetime import datetime

WATCH_DIR = "svgs"  
INKSCAPE_PATH = "/Applications/Inkscape.app/Contents/MacOS/inkscape" 
env = os.environ.copy()
env["HOME"] = os.path.expanduser("~")
env["PATH"] = "/opt/homebrew/bin:/Library/Frameworks/Python.framework/Versions/3.13/bin:/Library/Frameworks/Python.framework/Versions/3.13/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin:/usr/local/share/dotnet:~/.dotnet/tools"

class SVGHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(".svg"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
            print(f"[{timestamp}] Detected SVG change: {event.src_path}") 
            subprocess.run([
                INKSCAPE_PATH,
                event.src_path,
                "--batch-process",
                "--actions=command.idraw2.0-manager.noprefs"
                ], env=env, capture_output=True, text=True)

observer = Observer()
observer.schedule(SVGHandler(), WATCH_DIR, recursive=False)
observer.start()

try:
    print("Watching for SVG changes...")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping SVG watcher...")
    observer.stop()
observer.join()