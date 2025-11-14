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


class TrailDetector:
    """
    Tracks both hands of all users on a vertical plane (wall).
    Handles skeleton disappearance/reappearance by proximity matching.
    Each person gets a unique color.
    Exports SVG after 60 seconds.
    """
    
    def __init__(self):
        # stable_id -> {'left': [(x,y,t)...], 'right': [(x,y,t)...], 'color': (r,g,b)}
        self._trails = {}
        self._last_positions = {}  # stable_id -> {'left': (x,y,t), 'right': (x,y,t)}
        self._person_mapping = {}  # current_key -> stable_id
        self._next_stable_id = 0
        self._start_time = time.time()
        self._duration_seconds = 60
        self._proximity_threshold = 500.0  # 500mm = 50cm for reappearance matching
        self._exported = False
        self._colors = self._generate_color_palette()
    
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
    
    def _export_svg(self):
        """Export hand trails to SVG file (A4 size, wall/front view)."""
        if not self._trails or self._exported:
            return
        
        # A4 size in mm (landscape for wall view)
        a4_width_mm = 297
        a4_height_mm = 210
        
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
        
        if bbox_width == 0 or bbox_height == 0:
            print("[TRAIL] Bounding box too small to export")
            return
        
        # Add 10% margin
        margin = 0.1
        bbox_width *= (1 + margin * 2)
        bbox_height *= (1 + margin * 2)
        min_x -= bbox_width * margin
        min_y -= bbox_height * margin
        
        # Scale to fit A4 (keep aspect ratio)
        scale_x = a4_width_mm / bbox_width
        scale_y = a4_height_mm / bbox_height
        scale = min(scale_x, scale_y)
        
        # Center on A4
        scaled_width = bbox_width * scale
        scaled_height = bbox_height * scale
        offset_x = (a4_width_mm - scaled_width) / 2
        offset_y = (a4_height_mm - scaled_height) / 2
        
        def transform(x, y):
            """Transform from world coords to SVG coords (flip Y for SVG)."""
            svg_x = (x - min_x) * scale + offset_x
            svg_y = a4_height_mm - ((y - min_y) * scale + offset_y)  # Flip Y
            return svg_x, svg_y
        
        # Create SVG
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'trails')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'hand_trail_{timestamp}.svg')
        
        dwg = svgwrite.Drawing(output_file, size=(f'{a4_width_mm}mm', f'{a4_height_mm}mm'))
        
        # Add background
        dwg.add(dwg.rect(insert=(0, 0), size=(f'{a4_width_mm}mm', f'{a4_height_mm}mm'), 
                         fill='white'))
        
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
                polyline = dwg.polyline(points, stroke=color_hex, fill='none', 
                                       stroke_width=2, stroke_linecap='round',
                                       stroke_dasharray='5,2')  # Dashed for left
                dwg.add(polyline)
            
            # Draw right hand trail (solid line)
            right_trail = trail_data['right']
            if len(right_trail) >= 2:
                points = [transform(x, y) for x, y, _ in right_trail]
                polyline = dwg.polyline(points, stroke=color_hex, fill='none', 
                                       stroke_width=2, stroke_linecap='round')
                dwg.add(polyline)
        
        # Add legend
        legend_y = 10
        dwg.add(dwg.text("Hand Trails", insert=(5, legend_y), 
                        font_size='10px', fill='black', font_weight='bold'))
        dwg.add(dwg.text("Dashed = Left Hand | Solid = Right Hand", 
                        insert=(5, legend_y + 12), font_size='8px', fill='gray'))
        
        # Add metadata
        info_text = f"Duration: {self._duration_seconds}s | People: {len(self._trails)} | {timestamp}"
        dwg.add(dwg.text(info_text, insert=(5, a4_height_mm - 5), 
                        font_size='8px', fill='gray'))
        
        dwg.save()
        print(f"[TRAIL] Exported SVG to {output_file}")
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
