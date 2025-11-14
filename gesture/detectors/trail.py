"""Trail detector - tracks hand movement on vertical plane and exports SVG trails."""

import time
from datetime import datetime
from PyQt5.QtGui import QVector3D, QColor
from OpenGL.GL import *
from senseSpaceLib.senseSpace.protocol import Frame, Person
from senseSpaceLib.senseSpace.enums import UniversalJoint, Body34Joint
from senseSpaceLib.senseSpace.vecMathHelper import getPlaneIntersection
import svgwrite
import os
import random
import subprocess


class TrailDetector:
    """
    Tracks both hands of all users on a vertical plane (wall).
    Handles skeleton disappearance/reappearance by proximity matching.
    Each person gets a unique color.
    """
    
    def __init__(self):
        # stable_id -> {'left': [(x,y,t)...], 'right': [(x,y,t)...], 'color': (r,g,b)}
        self._trails = {}
        self._last_positions = {}  # stable_id -> {'left': (x,y,t), 'right': (x,y,t)}
        self._person_mapping = {}  # current_key -> stable_id
        self._next_stable_id = 0
        self._start_time = time.time()
        self._duration_seconds = 20
        self._proximity_threshold = 500.0  # 500mm = 50cm for reappearance matching
        self._exported = False
        self._colors = self._generate_color_palette()
        
        # Plotter configuration
        self._plotter_enabled = False  # DISABLED: Use separate plot_svg.py script instead
        self._inkscape_path = "/Applications/Inkscape.app/Contents/MacOS/inkscape"
    
    def _generate_color_palette(self):
        """Generate distinct colors for different people."""
        colors = [
            (1.0, 0.2, 0.6),  # Hot pink
            (0.2, 0.8, 1.0),  # Cyan
            (1.0, 0.8, 0.0),  # Gold
            (0.6, 0.2, 1.0),  # Purple
            (0.2, 1.0, 0.4),  # Green
            (1.0, 0.4, 0.0),  # Orange
            (0.4, 0.6, 1.0),  # Light blue
            (1.0, 0.2, 0.2),  # Red
        ]
        return colors
    
    def _get_color(self, stable_id):
        """Get color for a stable_id."""
        return self._colors[stable_id % len(self._colors)]
        
    def _get_stable_id(self, current_key, hand_positions, timestamp):
        """
        Match current person to previous trail by proximity.
        hand_positions: {'left': (x,y) or None, 'right': (x,y) or None}
        Returns stable_id for trail continuity.
        """
        # Check if we already know this person from this frame
        if current_key in self._person_mapping:
            return self._person_mapping[current_key]
        
        # Try to find closest previous person within threshold
        best_match = None
        best_dist = self._proximity_threshold
        
        for stable_id, prev_hands in self._last_positions.items():
            # Calculate average distance between matching hands
            total_dist = 0
            count = 0
            
            for side in ['left', 'right']:
                if hand_positions[side] is not None and side in prev_hands:
                    prev_x, prev_y, prev_time = prev_hands[side]
                    # Only consider recent positions (within last 2 seconds)
                    if timestamp - prev_time > 2.0:
                        continue
                    curr_x, curr_y = hand_positions[side]
                    dist = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2) ** 0.5
                    total_dist += dist
                    count += 1
            
            if count > 0:
                avg_dist = total_dist / count
                if avg_dist < best_dist:
                    best_dist = avg_dist
                    best_match = stable_id
        
        if best_match is not None:
            # Reuse existing trail
            stable_id = best_match
        else:
            # Create new trail
            stable_id = self._next_stable_id
            self._next_stable_id += 1
            color = self._get_color(stable_id)
            self._trails[stable_id] = {'left': [], 'right': [], 'color': color}
            self._last_positions[stable_id] = {}
        
        self._person_mapping[current_key] = stable_id
        return stable_id
    
    def _project_hand_to_wall(self, person, frame, hand_side):
        """
        Get hand position projected onto vertical wall plane (Z=constant).
        Returns QVector3D position or None.
        hand_side: 'LEFT' or 'RIGHT'
        """
        skel = getattr(person, 'skeleton', None)
        if not skel:
            return None
        
        # Body34 joint indices
        LEFT_HAND_IDX = 8
        RIGHT_HAND_IDX = 15
        
        # Get the appropriate hand joint index
        if hand_side == 'LEFT':
            index = LEFT_HAND_IDX
        else:
            index = RIGHT_HAND_IDX
        
        if index >= len(skel):
            return None
        
        pos = getattr(skel[index], 'pos', None)
        if not pos:
            return None
        
        # Return x, y coordinates as QVector3D
        return QVector3D(pos.x, pos.y, pos.z)
    
    def _draw_trails(self):
        """Draw all hand trails on vertical plane (wall view)."""
        if not self._trails:
            return
        
        glDisable(GL_DEPTH_TEST)
        glLineWidth(3.0)
        
        for stable_id, trail_data in self._trails.items():
            color = trail_data['color']
            glColor4f(color[0], color[1], color[2], 0.8)
            
            # Draw left hand trail
            left_trail = trail_data['left']
            if len(left_trail) >= 2:
                glBegin(GL_LINE_STRIP)
                for x, y, _ in left_trail:
                    glVertex3f(x, y, 0)  # Project to wall at Z=0
                glEnd()
            
            # Draw right hand trail
            right_trail = trail_data['right']
            if len(right_trail) >= 2:
                glBegin(GL_LINE_STRIP)
                for x, y, _ in right_trail:
                    glVertex3f(x, y, 0)  # Project to wall at Z=0
                glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)
    
    def _send_to_plotter(self, svg_file):
        """Send SVG file to plotter via Inkscape and iDraw extension."""
        if not self._plotter_enabled:
            return
        
        if not os.path.exists(self._inkscape_path):
            print(f"[PLOTTER] Inkscape not found at {self._inkscape_path}")
            return
        
        if not os.path.exists(svg_file):
            print(f"[PLOTTER] SVG file not found: {svg_file}")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[PLOTTER] [{timestamp}] Queuing for plotter: {os.path.basename(svg_file)}")
            
            # Write a shell script to execute Inkscape in the background
            # This avoids fork/exec crashes when calling GUI apps from within GUI contexts
            script_dir = os.path.join(os.path.dirname(svg_file), '.plotter_queue')
            os.makedirs(script_dir, exist_ok=True)
            
            script_file = os.path.join(script_dir, f'plot_{os.path.basename(svg_file)}.sh')
            svg_file_abs = os.path.abspath(svg_file)
            
            # Create a shell script that will run Inkscape with proper environment
            with open(script_file, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write(f'# Auto-generated script to plot {os.path.basename(svg_file)}\n')
                f.write(f'export PATH="/opt/homebrew/bin:/Library/Frameworks/Python.framework/Versions/3.13/bin:/usr/local/bin:$PATH"\n')
                f.write(f'cd "{os.path.dirname(svg_file_abs)}"\n')
                f.write(f'"{self._inkscape_path}" "{svg_file_abs}" --batch-process --actions="command.idraw2.0-manager.noprefs" 2>&1 | tee "{script_file}.log"\n')
                f.write(f'echo "Plot completed at $(date)" >> "{script_file}.log"\n')
            
            # Make script executable
            os.chmod(script_file, 0o755)
            
            print(f"[PLOTTER] Created plot script: {script_file}")
            print(f"[PLOTTER] To plot manually, run: bash {script_file}")
            print(f"[PLOTTER] Or set _plotter_enabled = False to disable auto-plotting")
            
            # Try to execute the script in background (non-blocking)
            # Using subprocess.Popen with proper detachment
            subprocess.Popen(
                ['bash', script_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # Detach from parent process
                cwd=os.path.dirname(svg_file_abs)
            )
            
            print(f"[PLOTTER] Launched plotting process in background")
                    
        except Exception as e:
            print(f"[PLOTTER] Error: {e}")
    
    def _export_svg(self):
        """Export hand trails to SVG file (A5 size, wall/front view)."""
        if not self._trails or self._exported:
            return
        
        # A5 size in mm (landscape for wall view)
        page_width_mm = 210
        page_height_mm = 148
        
        # Collect all points from both hands
        all_points = []
        for trail_data in self._trails.values():
            for x, y, _ in trail_data['left']:
                all_points.append((x, y))
            for x, y, _ in trail_data['right']:
                all_points.append((x, y))
        
        if not all_points:
            print("[TRAIL] No points to export")
            return
        
        # Find bounding box
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        print(f"[TRAIL DEBUG] BBox: width={bbox_width:.1f}mm, height={bbox_height:.1f}mm")
        print(f"[TRAIL DEBUG] Range X: {min_x:.1f} to {max_x:.1f}")
        print(f"[TRAIL DEBUG] Range Y: {min_y:.1f} to {max_y:.1f}")
        
        if bbox_width == 0 or bbox_height == 0:
            print("[TRAIL] Bounding box too small to export")
            return
        
        # FIXED SCALE: Define maximum gesture size in real world (mm)
        # This ensures all exports use the same scale for comparison
        max_gesture_height_mm = 2000  
        max_gesture_width_mm = 2000  
        
        # Calculate fixed scale based on fitting max gesture to page
        # Use 90% of page to leave some margin
        scale_for_max_height = (page_height_mm * 0.9) / max_gesture_height_mm
        scale_for_max_width = (page_width_mm * 0.9) / max_gesture_width_mm
        fixed_scale = min(scale_for_max_height, scale_for_max_width)
        
        print(f"[TRAIL DEBUG] Fixed scale: {fixed_scale:.6f} (max gesture: {max_gesture_width_mm}x{max_gesture_height_mm}mm)")
        
        # Calculate actual size after scaling with fixed scale
        scaled_width = bbox_width * fixed_scale
        scaled_height = bbox_height * fixed_scale
        
        print(f"[TRAIL DEBUG] Scaled size: {scaled_width:.1f}mm x {scaled_height:.1f}mm")
        print(f"[TRAIL DEBUG] Original bbox: {bbox_width:.1f}mm x {bbox_height:.1f}mm")
        
        # Center on A5
        offset_x = (page_width_mm - scaled_width) / 2
        offset_y = (page_height_mm - scaled_height) / 2
        
        print(f"[TRAIL DEBUG] Offsets: x={offset_x:.1f}mm, y={offset_y:.1f}mm")
        
        # Calculate center point of the gesture for positioning
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        def transform(x, y):
            """Transform from world coords to SVG coords with fixed scale."""
            # Apply fixed scale and center on page
            svg_x = (x - center_x) * fixed_scale + page_width_mm / 2
            svg_y = page_height_mm - ((y - center_y) * fixed_scale + page_height_mm / 2)  # Flip Y
            return svg_x, svg_y
        
        # Create SVG
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'trails')
        
        # Create a folder for this export session
        session_dir = os.path.join(output_dir, f'trail_{timestamp}')
        os.makedirs(session_dir, exist_ok=True)
        
        # File for combined trail with all people
        combined_file = os.path.join(session_dir, f'all_trails_{timestamp}.svg')
        
        # ===== COMBINED SVG WITH ALL TRAILS =====
        dwg = svgwrite.Drawing(
            combined_file, 
            size=(f'{page_width_mm}mm', f'{page_height_mm}mm'),
            viewBox=f'0 0 {page_width_mm} {page_height_mm}'
        )
        
        # Draw each person's trails
        for stable_id, trail_data in self._trails.items():
            color = trail_data['color']
            color_hex = '#{:02x}{:02x}{:02x}'.format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
            
            # Draw left hand trail
            left_trail = trail_data['left']
            if len(left_trail) >= 2:
                points = [transform(x, y) for x, y, _ in left_trail]
                if stable_id == 0:
                    print(f"[TRAIL DEBUG] First point: {points[0]}, Last point: {points[-1]}")
                polyline = dwg.polyline(points, stroke=color_hex, fill='none', 
                                       stroke_width=1, stroke_linecap='round')
                dwg.add(polyline)
            
            # Draw right hand trail
            right_trail = trail_data['right']
            if len(right_trail) >= 2:
                points = [transform(x, y) for x, y, _ in right_trail]
                polyline = dwg.polyline(points, stroke=color_hex, fill='none', 
                                       stroke_width=1, stroke_linecap='round')
                dwg.add(polyline)
        
        # Add metadata
        info_text = f"Duration: {self._duration_seconds}s | People: {len(self._trails)} | {timestamp}"
        dwg.add(dwg.text(info_text, insert=(5, page_height_mm - 5), 
                        font_size='8px', fill='gray'))
        
        dwg.save()
        print(f"[TRAIL] Exported combined SVG to {combined_file}")
        
        # ===== INDIVIDUAL SVG FOR EACH PERSON'S EACH HAND =====
        for stable_id, trail_data in self._trails.items():
            color = trail_data['color']
            color_hex = '#{:02x}{:02x}{:02x}'.format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
            
            # LEFT HAND SVG
            left_trail = trail_data['left']
            if len(left_trail) >= 2:
                left_file = os.path.join(session_dir, f'person_{stable_id}_left_{timestamp}.svg')
                
                dwg_left = svgwrite.Drawing(
                    left_file, 
                    size=(f'{page_width_mm}mm', f'{page_height_mm}mm'),
                    viewBox=f'0 0 {page_width_mm} {page_height_mm}'
                )
                
                # Draw left hand trail
                points = [transform(x, y) for x, y, _ in left_trail]
                polyline = dwg_left.polyline(points, stroke=color_hex, fill='none', 
                                             stroke_width=1, stroke_linecap='round')
                dwg_left.add(polyline)
                
                dwg_left.save()
                print(f"[TRAIL] Exported left hand SVG for Person {stable_id}")
                
                # Send to plotter
                self._send_to_plotter(left_file)
            
            # RIGHT HAND SVG
            right_trail = trail_data['right']
            if len(right_trail) >= 2:
                right_file = os.path.join(session_dir, f'person_{stable_id}_right_{timestamp}.svg')
                
                dwg_right = svgwrite.Drawing(
                    right_file, 
                    size=(f'{page_width_mm}mm', f'{page_height_mm}mm'),
                    viewBox=f'0 0 {page_width_mm} {page_height_mm}'
                )
                
                # Draw right hand trail
                points = [transform(x, y) for x, y, _ in right_trail]
                polyline = dwg_right.polyline(points, stroke=color_hex, fill='none', 
                                              stroke_width=1, stroke_linecap='round')
                dwg_right.add(polyline)
                
                dwg_right.save()
                print(f"[TRAIL] Exported right hand SVG for Person {stable_id}")
                
                # Send to plotter
                self._send_to_plotter(right_file)
        
        print(f"[TRAIL] All exports saved to folder: {session_dir}")
        print(f"[TRAIL] Total people: {len(self._trails)}, Total points: {len(all_points)}")
        self._exported = True
    
    def process(self, frame: Frame, gl_context=None):
        """Process each frame to track hand trails."""
        if not hasattr(frame, 'people'):
            return []
        
        current_time = time.time()
        elapsed = current_time - self._start_time
        timestamp = getattr(frame, 'timestamp', current_time)
        
        # Clear person mapping for this frame
        self._person_mapping.clear()
        
        # Track each person's hands
        for person in frame.people:
            current_key = getattr(person, 'id', getattr(person, 'track_id', id(person)))
            
            # Get both hand positions
            left_hand = self._project_hand_to_wall(person, frame, 'LEFT')
            right_hand = self._project_hand_to_wall(person, frame, 'RIGHT')
            
            # Skip if no hands detected
            if left_hand is None and right_hand is None:
                continue
            
            # Prepare hand positions for matching
            hand_positions = {
                'left': (left_hand.x(), left_hand.y()) if left_hand else None,
                'right': (right_hand.x(), right_hand.y()) if right_hand else None
            }
            
            # Match to stable trail
            stable_id = self._get_stable_id(current_key, hand_positions, timestamp)
            trail_data = self._trails[stable_id]
            
            # Add left hand point to trail
            if hand_positions['left'] is not None:
                x, y = hand_positions['left']
                left_trail = trail_data['left']
                
                # Avoid duplicates too close together (100mm² threshold)
                if len(left_trail) == 0 or \
                   ((x - left_trail[-1][0])**2 + (y - left_trail[-1][1])**2) > 100:
                    left_trail.append((x, y, timestamp))
                
                # Update last known position
                if stable_id not in self._last_positions:
                    self._last_positions[stable_id] = {}
                self._last_positions[stable_id]['left'] = (x, y, timestamp)
            
            # Add right hand point to trail
            if hand_positions['right'] is not None:
                x, y = hand_positions['right']
                right_trail = trail_data['right']
                
                # Avoid duplicates too close together
                if len(right_trail) == 0 or \
                   ((x - right_trail[-1][0])**2 + (y - right_trail[-1][1])**2) > 100:
                    right_trail.append((x, y, timestamp))
                
                # Update last known position
                if stable_id not in self._last_positions:
                    self._last_positions[stable_id] = {}
                self._last_positions[stable_id]['right'] = (x, y, timestamp)
        
        # Draw trails if we have OpenGL context
        if gl_context:
            self._draw_trails()
        
        # Check if time to export
        if elapsed >= self._duration_seconds and not self._exported:
            self._export_svg()
        
        return []  # No events to report
