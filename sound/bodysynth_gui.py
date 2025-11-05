import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox, QCheckBox, QTextEdit, QPushButton, QGroupBox
from PyQt6.QtCore import QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import threading
import math

pg.setConfigOption('background', 'k')
pg.setConfigOption('foreground', 'w')

class ReverbWidget(QWidget):
    """Visual indicator for plane angle (0°=plane normal along X/min reverb, 90°=plane normal along Z/max reverb)"""
    def __init__(self, color='yellow'):
        super().__init__()
        self.angle = 0.0  # Angle in degrees from plane normal (0-90)
        self.reverb = 0.0  # Reverb wetness 0.0-1.0
        self.color = color
        self.setMinimumSize(50, 50)
        self.setMaximumSize(50, 50)

    def set_angle_and_reverb(self, angle, reverb):
        """Set plane angle (degrees, 0-90) and reverb wetness (0.0-1.0)"""
        self.angle = angle
        self.reverb = reverb
        self.update()

    def paintEvent(self, event):
        if self.angle is None:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        pen = QPen(QColor(self.color), 2)
        painter.setPen(pen)

        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2
        length = min(width, height) / 2 * 0.8

        # Angle is already 0-90: 0° = horizontal (min reverb), 90° = vertical (max reverb)
        rad_angle = math.radians(self.angle)
        
        end_x = center_x + length * math.cos(rad_angle)
        end_y = center_y - length * math.sin(rad_angle)  # Negative because screen Y is inverted

        painter.drawLine(int(center_x), int(center_y), int(end_x), int(end_y))
        
        # Draw angle text
        painter.setPen(QColor(self.color))
        painter.setFont(QFont('Arial', 8))
        text = f"{int(self.angle)}°"
        painter.drawText(5, height - 5, text)

class SkeletonWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(100, 100)
        self.setMaximumSize(100, 100)
        self.skeleton_data = None
        self.armline_data = None  # Store armline for projection calculation
        self.line_info = None  # Store fitted line info for camera direction
        self.color = 'cyan'  # Default color
        
        # BODY_34 bone connections (joint index pairs)
        self.bones = [
            # Spine
            (0, 1), (1, 2), (2, 3), (3, 26),
            # Left arm
            (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (8, 10),
            # Right arm  
            (3, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (15, 17),
            # Left leg
            (0, 18), (18, 19), (19, 20), (20, 21), (20, 32),
            # Right leg
            (0, 22), (22, 23), (23, 24), (24, 25), (24, 33),
            # Face
            (26, 27), (26, 28), (26, 29), (28, 30), (29, 31),
        ]
        
    def set_skeleton(self, joints, armline=None, line_info=None):
        """
        joints: list of (x, y, z) tuples for all skeleton joints
        armline: list of (x, y, z) tuples for armline joints (for plane calculation)
        line_info: dict with 'centroid', 'direction' (x,z), and 'angle' for camera view
        """
        self.skeleton_data = joints
        self.armline_data = armline
        self.line_info = line_info
        self.update()
    
    def _best_fit_plane(self, points):
        """Calculate best fit plane through points"""
        if len(points) < 3:
            # Not enough points, return identity
            return np.array([0, 0, 0]), np.array([0, 0, 1])
        
        centroid = points.mean(axis=0)
        centered = points - centroid
        
        # Check if points are degenerate
        if np.linalg.norm(centered) < 1e-6:
            return centroid, np.array([0, 0, 1])
        
        uu, dd, vv = np.linalg.svd(centered, full_matrices=False)
        normal = vv[2]
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        return centroid, normal
        
    def _project_onto_plane(self, points, centroid, normal):
        """Project points onto plane defined by centroid and normal"""
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        displacement = np.dot((points - centroid), normal)[:, None] * normal
        return points - displacement
    
    def paintEvent(self, event):
        if self.skeleton_data is None or len(self.skeleton_data) == 0:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        
        skeleton_points = np.array(self.skeleton_data)
        
        # Use fitted line info to determine camera view if available
        if self.line_info and self.armline_data and len(self.armline_data) >= 3:
            armline_points = np.array(self.armline_data)
            
            # Get line direction in XZ plane
            line_direction_xz = self.line_info['direction']  # (x, z)
            centroid = self.line_info['centroid']  # (x, y, z)
            
            # Camera views perpendicular to the line in XZ plane
            # Normal to the line in XZ: rotate 90 degrees
            normal_xz = np.array([-line_direction_xz[1], line_direction_xz[0]])  # Perpendicular in XZ
            
            # Create 3D viewing direction (camera points towards skeleton from this direction)
            # Normal points in XZ plane, Y component is 0
            normal = np.array([normal_xz[0], 0, normal_xz[1]])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            
            # Project skeleton onto plane perpendicular to viewing direction
            proj_skeleton = self._project_onto_plane(skeleton_points, centroid, normal)
            
            # Build in-plane basis with Y axis staying vertical
            # v should align with world Y axis (vertical, up direction)
            world_y = np.array([0, 1, 0])
            
            # Project world Y onto the plane to get our vertical axis v
            v = world_y - np.dot(world_y, normal) * normal
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-6:
                # If Y is perpendicular to plane, use armline direction
                proj_armline = self._project_onto_plane(armline_points, centroid, normal)
                v = proj_armline[1] - proj_armline[0]
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-6:
                    raise ValueError("Cannot create basis")
            v = v / v_norm
            
            # u is perpendicular to both normal and v (horizontal in the view)
            u = np.cross(v, normal)
            u_norm = np.linalg.norm(u)
            if u_norm < 1e-6:
                raise ValueError("Cannot create perpendicular basis")
            u = u / u_norm
            
            # Ensure consistent orientation: right arm should be on left side of view
            # Armline goes from left wrist (index 0) to right wrist (index 8)
            # Project left and right wrist positions
            proj_armline = self._project_onto_plane(armline_points, centroid, normal)
            if len(proj_armline) >= 9:
                left_wrist = proj_armline[0]  # First point is left wrist
                right_wrist = proj_armline[8]  # Last point is right wrist
                
                # Get horizontal positions in current u direction
                left_u = np.dot(left_wrist - centroid, u)
                right_u = np.dot(right_wrist - centroid, u)
                
                # If right wrist is on the right side (positive u), flip u
                # We want right wrist on the left (negative u)
                if right_u < left_u:
                    u = -u
            
            # Project skeleton points onto the u-v plane (2D coordinates)
            points_2d = []
            for p in proj_skeleton:
                x_coord = np.dot(p - centroid, u)
                y_coord = np.dot(p - centroid, v)
                points_2d.append((x_coord, y_coord))
        elif self.armline_data and len(self.armline_data) >= 3:
            # Old behavior: use best fit plane through armline
            armline_points = np.array(self.armline_data)
            
            if np.any(np.isnan(armline_points)) or np.any(np.isinf(armline_points)):
                raise ValueError("Invalid armline data")
            
            centroid, normal = self._best_fit_plane(armline_points)
            proj_skeleton = self._project_onto_plane(skeleton_points, centroid, normal)
            
            # Build in-plane basis with Y axis staying vertical
            world_y = np.array([0, 1, 0])
            v = world_y - np.dot(world_y, normal) * normal
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-6:
                proj_armline = self._project_onto_plane(armline_points, centroid, normal)
                v = proj_armline[1] - proj_armline[0]
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-6:
                    raise ValueError("Cannot create basis")
            v = v / v_norm
            
            u = np.cross(v, normal)
            u_norm = np.linalg.norm(u)
            if u_norm < 1e-6:
                raise ValueError("Cannot create perpendicular basis")
            u = u / u_norm
            
            proj_armline = self._project_onto_plane(armline_points, centroid, normal)
            if len(proj_armline) >= 9:
                left_wrist = proj_armline[0]
                right_wrist = proj_armline[8]
                left_u = np.dot(left_wrist - centroid, u)
                right_u = np.dot(right_wrist - centroid, u)
                if right_u < left_u:
                    u = -u
            
            points_2d = []
            for p in proj_skeleton:
                x_coord = np.dot(p - centroid, u)
                y_coord = np.dot(p - centroid, v)
                points_2d.append((x_coord, y_coord))
        else:
            # Fallback: simple (x, y) projection
            points_2d = [(p[0], p[1]) for p in skeleton_points]

        
        if len(points_2d) < 2:
            return
        
        # Find bounds
        xs = [p[0] for p in points_2d]
        ys = [p[1] for p in points_2d]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Calculate scale to fit in widget
        width = self.width()
        height = self.height()
        margin = 10
        
        range_x = max_x - min_x if max_x != min_x else 1
        range_y = max_y - min_y if max_y != min_y else 1
        
        scale = min((width - 2*margin) / range_x, (height - 2*margin) / range_y)
        
        # Transform to screen coordinates (flip y axis since screen y goes down)
        def to_screen(x, y):
            screen_x = margin + (x - min_x) * scale
            screen_y = height - margin - (y - min_y) * scale  # Flip y
            return int(screen_x), int(screen_y)
        
        # Draw bone connections using the widget's color
        pen_line = QPen(QColor(self.color), 2)
        painter.setPen(pen_line)
        
        for bone in self.bones:
            if bone[0] < len(points_2d) and bone[1] < len(points_2d):
                x1, y1 = to_screen(points_2d[bone[0]][0], points_2d[bone[0]][1])
                x2, y2 = to_screen(points_2d[bone[1]][0], points_2d[bone[1]][1])
                painter.drawLine(x1, y1, x2, y2)
        
        # Draw joints as white circles
        pen_joint = QPen(QColor('white'), 1)
        painter.setPen(pen_joint)
        painter.setBrush(QColor('white'))
        
        for p in points_2d:
            x, y = to_screen(p[0], p[1])
            painter.drawEllipse(x-3, y-3, 6, 6)

class HeadPositionWidget(QWidget):
    def __init__(self, num_lanes=3, top_left=None, bottom_right=None):
        super().__init__()
        self.setMinimumSize(300, 600)  # Larger size for right column
        # No maximum size - let it expand
        self.head_positions = {}  # {person_id: (x, y, z)}
        self.armlines = {}  # {person_id: [(x,y,z), ...]}
        self.fitted_lines = {}  # {person_id: {'centroid': (x,y,z), 'direction': (x,z), 'angle': float}}
        self.loop_position = 0.0  # 0.0 to 1.0
        self.num_lanes = num_lanes  # Number of sample lanes
        # Calibration space boundaries [x, z]
        self.top_left = top_left if top_left else [1000, 1000]
        self.bottom_right = bottom_right if bottom_right else [-1000, -1000]
        
    def update_positions(self, positions_dict, armlines_dict=None, fitted_lines_dict=None):
        """
        positions_dict: {person_id: (x, y, z)}
        armlines_dict: {person_id: [(x,y,z), ...]} - list of joint positions
        fitted_lines_dict: {person_id: {'centroid': (x,y,z), 'direction': (x,z), 'angle': float}}
        """
        self.head_positions = positions_dict
        if armlines_dict is not None:
            self.armlines = armlines_dict
        if fitted_lines_dict is not None:
            self.fitted_lines = fitted_lines_dict
        self.update()
    
    def update_loop_position(self, position):
        """position: 0.0 to 1.0 representing position in loop"""
        self.loop_position = position
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        margin = 20
        
        # Fill entire background with dark gray
        painter.fillRect(0, 0, width, height, QColor(30, 30, 30))
        
        # Dynamically determine number of lanes from number of people
        num_lanes = self.num_lanes  # Default
        if self.head_positions:
            valid_positions = {pid: pos for pid, pos in self.head_positions.items() if pos is not None}
            if valid_positions:
                num_lanes = len(valid_positions)
        
        # Calculate calibration space boundaries
        min_x = self.bottom_right[0]
        max_x = self.top_left[0]
        min_z = self.bottom_right[1]
        max_z = self.top_left[1]
        range_x = max_x - min_x if max_x != min_x else 1000
        range_z = max_z - min_z if max_z != min_z else 1000
        
        # Helper to convert world coords to screen coords
        def coord_to_screen_x(x_coord):
            return margin + (x_coord - min_x) / range_x * (width - 2 * margin)
        
        def coord_to_screen_z(z_coord):
            return height - margin - (z_coord - min_z) / range_z * (height - 2 * margin)
        
        # Draw grid background
        pen_grid = QPen(QColor(80, 80, 80), 1)
        painter.setPen(pen_grid)
        
        # Draw horizontal lane dividers (num_lanes - 1 interior lines)
        # This creates num_lanes regions between the top and bottom
        for i in range(1, num_lanes):
            y_pos = margin + i * (height - 2 * margin) / num_lanes
            painter.drawLine(margin, int(y_pos), width - margin, int(y_pos))
        
        # Draw grid lines with labels
        pen_grid = QPen(QColor(80, 80, 80), 1, Qt.PenStyle.DotLine)
        pen_axis = QPen(QColor(150, 150, 150), 2)
        painter.setFont(QFont('Arial', 8))
        
        # X gridlines (vertical) - every 500mm
        x_step = 500
        for x in range(int(min_x), int(max_x) + 1, x_step):
            screen_x = coord_to_screen_x(x)
            if x == 0:
                painter.setPen(pen_axis)
            else:
                painter.setPen(pen_grid)
            painter.drawLine(int(screen_x), margin, int(screen_x), height - margin)
            # Label at bottom
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(int(screen_x) - 15, height - margin + 15, f"{x}")
        
        # Z gridlines (horizontal) - every 500mm
        z_step = 500
        for z in range(int(min_z), int(max_z) + 1, z_step):
            screen_z = coord_to_screen_z(z)
            if z == 0:
                painter.setPen(pen_axis)
            else:
                painter.setPen(pen_grid)
            painter.drawLine(margin, int(screen_z), width - margin, int(screen_z))
            # Label at left
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(5, int(screen_z) + 4, f"{z}")
        
        # Draw colored lane backgrounds ON TOP of the grid (semi-transparent)
        lane_colors = [
            QColor(255, 80, 80, 100),   # Red with alpha (kick)
            QColor(80, 255, 80, 100),   # Green with alpha (snare)
            QColor(80, 120, 255, 100),  # Blue with alpha (cymbal)
            QColor(255, 255, 80, 100),  # Yellow (extra lanes)
            QColor(255, 80, 255, 100),  # Magenta (extra lanes)
            QColor(80, 255, 255, 100),  # Cyan (extra lanes)
        ]
        
        for lane_idx in range(num_lanes):
            # Calculate Z boundaries for this lane in world coordinates
            lane_width_z = range_z / num_lanes
            lane_z_min = min_z + lane_idx * lane_width_z
            lane_z_max = min_z + (lane_idx + 1) * lane_width_z
            
            # Convert to screen coordinates
            screen_z_top = coord_to_screen_z(lane_z_max)
            screen_z_bottom = coord_to_screen_z(lane_z_min)
            
            # Draw filled rectangle
            color = lane_colors[lane_idx % len(lane_colors)]
            rect_x = int(margin)
            rect_y = int(screen_z_top)
            rect_w = int(width - 2 * margin)
            rect_h = int(screen_z_bottom - screen_z_top)
            
            painter.fillRect(rect_x, rect_y, rect_w, rect_h, color)
            
            # Draw lane label
            lane_names = ['KICK', 'SNARE', 'CYMBAL', 'DRUM4', 'DRUM5', 'DRUM6']
            lane_name = lane_names[lane_idx % len(lane_names)]
            painter.setPen(QColor(255, 255, 255, 255))
            painter.setFont(QFont('Arial', 18, QFont.Weight.Bold))
            text_y = int((screen_z_top + screen_z_bottom) / 2) + 7
            painter.drawText(int(margin + 10), text_y, f"Lane {lane_idx}: {lane_name}")
            
            # Draw Z range info
            painter.setFont(QFont('Arial', 11))
            painter.setPen(QColor(255, 255, 255, 220))
            range_text = f"Z: {int(lane_z_min)} to {int(lane_z_max)}"
            painter.drawText(int(margin + 10), text_y + 22, range_text)
        
        # Draw loop position bar (vertical line moving left to right)
        if self.loop_position >= 0:
            loop_x = margin + self.loop_position * (width - 2 * margin)
            pen_loop = QPen(QColor('yellow'), 3)
            painter.setPen(pen_loop)
            painter.drawLine(int(loop_x), margin, int(loop_x), height - margin)
        
        # Draw border
        pen_border = QPen(QColor(150, 150, 150), 2)
        painter.setPen(pen_border)
        painter.drawRect(margin, margin, width - 2*margin, height - 2*margin)
        
        # Get all positions to determine scale
        if not self.head_positions:
            return
        
        # Filter out None positions
        valid_positions = {pid: pos for pid, pos in self.head_positions.items() if pos is not None}
        if not valid_positions:
            return
            
        positions = list(valid_positions.values())
        ys = [p[1] for p in positions]
        
        # Use calibration space boundaries for X and Z (fixed scale)
        min_x = self.bottom_right[0]
        max_x = self.top_left[0]
        min_z = self.bottom_right[1]
        max_z = self.top_left[1]
        
        range_x = max_x - min_x if max_x != min_x else 1000
        range_z = max_z - min_z if max_z != min_z else 1000
        
        # Y range from actual data for dot sizing
        min_y, max_y = min(ys), max(ys)
        range_y = max_y - min_y if max_y != min_y else 1000
        
        # Scale to fit in widget
        scale_x = (width - 2 * margin) / range_x
        scale_z = (height - 2 * margin) / range_z
        
        def to_screen(x, z):
            screen_x = margin + (x - min_x) * scale_x
            screen_z = height - margin - (z - min_z) * scale_z  # Flip Z
            return int(screen_x), int(screen_z)
        
        def size_from_y(y):
            """Map Y position to dot size (higher = bigger)"""
            # Normalize Y to 0-1 range
            y_norm = (y - min_y) / range_y if range_y > 0 else 0.5
            # Map to radius 5-15 (reduced from 5-25 to prevent dots getting too large)
            return int(5 + y_norm * 10)
        
        # Draw head positions as circles
        colors = [QColor('red'), QColor('green'), QColor('blue'), QColor('yellow'), QColor('magenta'), QColor('cyan')]
        
        for idx, (person_id, pos) in enumerate(valid_positions.items()):
            x, y, z = pos
            screen_x, screen_z = to_screen(x, z)
            radius = size_from_y(y)
            
            color = colors[idx % len(colors)]
            
            # Draw armline if available for this person
            if person_id in self.armlines and self.armlines[person_id]:
                armline = self.armlines[person_id]
                # Draw line connecting all armline joints
                painter.setPen(QPen(color, 3))
                for i in range(len(armline) - 1):
                    x1, y1, z1 = armline[i]
                    x2, y2, z2 = armline[i + 1]
                    sx1, sz1 = to_screen(x1, z1)
                    sx2, sz2 = to_screen(x2, z2)
                    painter.drawLine(sx1, sz1, sx2, sz2)
                
                # Draw small circles at each joint
                painter.setBrush(color)
                for joint_pos in armline:
                    jx, jy, jz = joint_pos
                    sjx, sjz = to_screen(jx, jz)
                    joint_radius = 3
                    painter.drawEllipse(sjx - joint_radius, sjz - joint_radius, 
                                      joint_radius * 2, joint_radius * 2)
            
            # Draw fitted line if available for this person (in top-down XZ plane)
            if person_id in self.fitted_lines and self.fitted_lines[person_id]:
                line_info = self.fitted_lines[person_id]
                centroid = line_info['centroid']
                direction = line_info['direction']  # (x, z) direction vector
                
                # Draw line extending from centroid in both directions
                line_length = 400  # mm, extend in each direction
                cx, cy, cz = centroid
                dx, dz = direction
                
                # Start and end points of the fitted line
                x1 = cx - dx * line_length
                z1 = cz - dz * line_length
                x2 = cx + dx * line_length
                z2 = cz + dz * line_length
                
                sx1, sz1 = to_screen(x1, z1)
                sx2, sz2 = to_screen(x2, z2)
                
                # Draw thick white dashed line
                pen_fitted = QPen(QColor('white'), 4, Qt.PenStyle.DashLine)
                painter.setPen(pen_fitted)
                painter.drawLine(sx1, sz1, sx2, sz2)
                
                # Draw centroid as a small white square
                scx, scz = to_screen(cx, cz)
                painter.setBrush(QColor('white'))
                painter.drawRect(scx - 4, scz - 4, 8, 8)
            
            # Draw head position as larger circle
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            painter.drawEllipse(screen_x - radius, screen_z - radius, radius * 2, radius * 2)
            
            # Draw person ID
            painter.setPen(QColor('white'))
            painter.drawText(screen_x + radius + 3, screen_z + 5, str(person_id))

class VoiceWidget(QWidget):
    def __init__(self, voice_id, color='yellow'):
        super().__init__()
        self.voice_id = voice_id
        self.color = color
        self.initUI()
        self.setFixedHeight(200)

    def initUI(self):
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # Left side: Skeleton
        self.skeleton_widget = SkeletonWidget()
        self.skeleton_widget.color = self.color  # Set skeleton color
        
        # Middle: Column with ID, Freq, Reverb info
        info_layout = QVBoxLayout()
        
        self.id_label = QLabel(f"Person: {self.voice_id}")
        self.freq_label = QLabel("Freq: N/A")
        self.reverb_widget = ReverbWidget(color=self.color)
        self.reverb_label = QLabel("Reverb: N/A")
        
        info_layout.addWidget(self.id_label)
        info_layout.addWidget(self.freq_label)
        info_layout.addWidget(self.reverb_widget)
        info_layout.addWidget(self.reverb_label)
        info_layout.addStretch()
        
        # Right side: Wavetable plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setYRange(-1.1, 1.1)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_curve = self.plot_widget.plot(pen=self.color)  # Use the voice color
        
        layout.addWidget(self.skeleton_widget, stretch=3)  # 30%
        layout.addLayout(info_layout, stretch=1)  # 10%
        layout.addWidget(self.plot_widget, stretch=6)  # 60%


    def update_data(self, data):
        if data:
            wavetable = data.get('wavetable', np.array([]))
            skeleton_joints = data.get('skeleton', [])
            armline_joints = data.get('armline', [])  # Get armline for projection
            line_info = data.get('line_info')  # Get fitted line info for camera view
            self.skeleton_widget.set_skeleton(skeleton_joints, armline_joints, line_info)
            
            # Add a point at y=0 at the end to complete the cycle
            if len(wavetable) > 0:
                wavetable_with_end = np.append(wavetable, 0.0)
                self.plot_curve.setData(wavetable_with_end)
                self.plot_widget.setXRange(0, len(wavetable_with_end))
            else:
                # Empty wavetable - clear the plot
                self.plot_curve.setData(np.array([]))
                self.plot_widget.setXRange(0, 1)
            
            freq = data.get('freq', 0)
            angle = data.get('angle', 0)
            reverb = data.get('reverb', 0)
            
            self.freq_label.setText(f"Freq: {freq:.1f} Hz" if freq > 0 else "Freq: N/A")
            
            if angle is not None and reverb is not None and reverb > 0:
                self.reverb_widget.set_angle_and_reverb(angle, reverb)
                self.reverb_label.setText(f"Reverb: {reverb:.2f}")
            else:
                self.reverb_widget.set_angle_and_reverb(0, 0)
                self.reverb_label.setText("Reverb: N/A")
                
            self.id_label.setText(f"Person: {self.voice_id}")
        else:
            self.clear_data()

    def clear_data(self):
        self.skeleton_widget.set_skeleton(None, None, None)
        self.plot_curve.setData(np.array([]))
        self.freq_label.setText("Freq: N/A")
        self.reverb_widget.set_angle_and_reverb(0, 0)
        self.reverb_label.setText("Reverb: N/A")
        self.id_label.setText(f"Person: -")


class SynthMonitor(QMainWindow):
    update_signal = pyqtSignal(dict)
    loop_length_changed = pyqtSignal(float)  # Signal for loop length changes
    scale_changed = pyqtSignal(str)  # Signal for scale changes
    click_enabled_changed = pyqtSignal(bool)  # Signal for click enable/disable
    drum_gain_changed = pyqtSignal(float)  # Signal for drum gain changes
    wave_gain_changed = pyqtSignal(float)  # Signal for wave gain changes
    
    # Drum effect signals
    drum_distortion_changed = pyqtSignal(float)  # Signal for drum distortion changes
    drum_compression_changed = pyqtSignal(float)  # Signal for drum compression changes
    
    # Wave effect signals
    wave_distortion_changed = pyqtSignal(float)  # Signal for wave distortion changes
    wave_lfo_changed = pyqtSignal(float)  # Signal for wave LFO amount changes
    wave_lfo_slowdown_changed = pyqtSignal(float)  # Signal for wave LFO slowdown changes
    wave_min_reverb_changed = pyqtSignal(float)  # Signal for wave min reverb changes
    wave_max_armlen_changed = pyqtSignal(float)  # Signal for wave max arm length changes
    wave_min_armlen_changed = pyqtSignal(float)  # Signal for wave min arm length changes
    wave_octave_high_changed = pyqtSignal(float)  # Signal for wave high octave changes
    wave_octave_low_changed = pyqtSignal(float)  # Signal for wave low octave changes
    
    # LLM Singer signals
    llm_prompt_changed = pyqtSignal(str)  # Signal for base prompt changes
    llm_interval_changed = pyqtSignal(float)  # Signal for playback interval changes (how often to play pre-cached phrases)
    llm_autotune_changed = pyqtSignal(float)  # Signal for auto-tune amount changes
    llm_reverb_changed = pyqtSignal(float)  # Signal for reverb amount changes
    llm_transpose_changed = pyqtSignal(float)  # Signal for transpose changes
    llm_gain_changed = pyqtSignal(float)  # Signal for LLM gain changes
    llm_lyrics_update = pyqtSignal(str)  # Signal for lyrics updates
    
    # Settings save signal
    settings_changed = pyqtSignal()  # Signal to trigger settings save
    
    # Define colors for each voice (matches head position dots)
    VOICE_COLORS = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']

    def __init__(self, num_voices=6, num_lanes=3, top_left=None, bottom_right=None, initial_loop_length=2.4):
        super().__init__()
        self.num_voices = num_voices
        self.setWindowTitle("Synth Monitor")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main vertical layout
        main_v_layout = QVBoxLayout(self.central_widget)
        
        # Top: Controls (loop length slider and scale selector)
        controls_layout = QHBoxLayout()
        
        # Loop length slider
        loop_label = QLabel("Loop Length:")
        self.loop_slider = QSlider(Qt.Orientation.Horizontal)
        self.loop_slider.setMinimum(5)  # 0.5 seconds
        self.loop_slider.setMaximum(100)  # 10 seconds
        self.loop_slider.setValue(int(initial_loop_length * 10))  # 2.4s default
        self.loop_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.loop_slider.setTickInterval(5)
        self.loop_value_label = QLabel(f"{initial_loop_length:.1f}s")
        self.loop_slider.valueChanged.connect(self._on_loop_slider_changed)
        
        controls_layout.addWidget(loop_label)
        controls_layout.addWidget(self.loop_slider, stretch=2)
        controls_layout.addWidget(self.loop_value_label)
        
        # Scale selector
        scale_label = QLabel("Scale:")
        self.scale_combo = QComboBox()
        self.scale_combo.addItems([
            "Chromatic (None)",
            "Major (Ionian)",
            "Minor (Aeolian)",
            "Dorian",
            "Phrygian",
            "Lydian",
            "Mixolydian",
            "Pentatonic Major",
            "Pentatonic Minor",
            "Blues",
            "Whole Tone"
        ])
        self.scale_combo.setCurrentText("Dorian")  # Set default to match WavetableSynth
        self.scale_combo.currentTextChanged.connect(self._on_scale_changed)
        
        controls_layout.addWidget(scale_label)
        controls_layout.addWidget(self.scale_combo, stretch=1)
        
        # Root note display
        self.root_note_label = QLabel("Root: A (440Hz)")
        self.root_note_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        controls_layout.addWidget(self.root_note_label)
        
        # Click enable checkbox
        self.click_checkbox = QCheckBox("Click")
        self.click_checkbox.setChecked(False)  # Disabled by default
        self.click_checkbox.stateChanged.connect(self._on_click_changed)
        controls_layout.addWidget(self.click_checkbox)
        
        main_v_layout.addLayout(controls_layout)
        
        # Gain controls row
        gain_layout = QHBoxLayout()
        
        # Wave gain slider
        wave_gain_label = QLabel("Wave Gain:")
        self.wave_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.wave_gain_slider.setMinimum(0)  # 0.0
        self.wave_gain_slider.setMaximum(100)  # 1.0
        self.wave_gain_slider.setValue(30)  # 0.3 default
        self.wave_gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.wave_gain_slider.setTickInterval(10)
        self.wave_gain_value_label = QLabel("0.30")
        self.wave_gain_slider.valueChanged.connect(self._on_wave_gain_changed)
        
        gain_layout.addWidget(wave_gain_label)
        gain_layout.addWidget(self.wave_gain_slider, stretch=2)
        gain_layout.addWidget(self.wave_gain_value_label)
        
        # Drum gain slider
        drum_gain_label = QLabel("Drum Gain:")
        self.drum_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.drum_gain_slider.setMinimum(0)  # 0.0
        self.drum_gain_slider.setMaximum(100)  # 1.0
        self.drum_gain_slider.setValue(80)  # 0.8 default
        self.drum_gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.drum_gain_slider.setTickInterval(10)
        self.drum_gain_value_label = QLabel("0.80")
        self.drum_gain_slider.valueChanged.connect(self._on_drum_gain_changed)
        
        gain_layout.addWidget(drum_gain_label)
        gain_layout.addWidget(self.drum_gain_slider, stretch=2)
        gain_layout.addWidget(self.drum_gain_value_label)
        
        # LLM gain slider
        llm_gain_label = QLabel("LLM Gain:")
        self.llm_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.llm_gain_slider.setMinimum(0)  # 0.0
        self.llm_gain_slider.setMaximum(100)  # 1.0
        self.llm_gain_slider.setValue(60)  # 0.6 default
        self.llm_gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.llm_gain_slider.setTickInterval(10)
        self.llm_gain_value_label = QLabel("0.60")
        self.llm_gain_slider.valueChanged.connect(self._on_llm_gain_changed)
        
        gain_layout.addWidget(llm_gain_label)
        gain_layout.addWidget(self.llm_gain_slider, stretch=2)
        gain_layout.addWidget(self.llm_gain_value_label)
        
        main_v_layout.addLayout(gain_layout)
        
        # Effect parameters row - 3 equal columns (Wave, Drum, LLM)
        effects_layout = QHBoxLayout()
        
        # Wave Effects GroupBox (1/3 width)
        wave_effects_group = QGroupBox("Wave Effects")
        wave_effects_layout = QVBoxLayout()
        
        # Wave Distortion
        wave_dist_h = QHBoxLayout()
        wave_dist_h.addWidget(QLabel("Distortion:"))
        self.wave_distortion_slider = QSlider(Qt.Orientation.Horizontal)
        self.wave_distortion_slider.setMinimum(0)
        self.wave_distortion_slider.setMaximum(100)
        self.wave_distortion_slider.setValue(0)  # 0.0 default
        self.wave_distortion_value_label = QLabel("0.00")
        self.wave_distortion_slider.valueChanged.connect(self._on_wave_distortion_changed)
        wave_dist_h.addWidget(self.wave_distortion_slider, stretch=2)
        wave_dist_h.addWidget(self.wave_distortion_value_label)
        wave_effects_layout.addLayout(wave_dist_h)
        
        # Wave LFO Amount
        wave_lfo_h = QHBoxLayout()
        wave_lfo_h.addWidget(QLabel("LFO:"))
        self.wave_lfo_slider = QSlider(Qt.Orientation.Horizontal)
        self.wave_lfo_slider.setMinimum(0)
        self.wave_lfo_slider.setMaximum(100)
        self.wave_lfo_slider.setValue(50)  # 0.5 default
        self.wave_lfo_value_label = QLabel("0.50")
        self.wave_lfo_slider.valueChanged.connect(self._on_wave_lfo_changed)
        wave_lfo_h.addWidget(self.wave_lfo_slider, stretch=2)
        wave_lfo_h.addWidget(self.wave_lfo_value_label)
        wave_effects_layout.addLayout(wave_lfo_h)
        
        # Wave LFO Slowdown
        wave_lfo_slowdown_h = QHBoxLayout()
        wave_lfo_slowdown_h.addWidget(QLabel("LFO Slowdown:"))
        self.wave_lfo_slowdown_slider = QSlider(Qt.Orientation.Horizontal)
        self.wave_lfo_slowdown_slider.setMinimum(10)  # Min 10x slower
        self.wave_lfo_slowdown_slider.setMaximum(1000)  # Max 1000x slower
        self.wave_lfo_slowdown_slider.setValue(200)  # 200 default
        self.wave_lfo_slowdown_value_label = QLabel("200")
        self.wave_lfo_slowdown_slider.valueChanged.connect(self._on_wave_lfo_slowdown_changed)
        wave_lfo_slowdown_h.addWidget(self.wave_lfo_slowdown_slider, stretch=2)
        wave_lfo_slowdown_h.addWidget(self.wave_lfo_slowdown_value_label)
        wave_effects_layout.addLayout(wave_lfo_slowdown_h)
        
        # Wave Min Reverb
        wave_reverb_h = QHBoxLayout()
        wave_reverb_h.addWidget(QLabel("Min Reverb:"))
        self.wave_min_reverb_slider = QSlider(Qt.Orientation.Horizontal)
        self.wave_min_reverb_slider.setMinimum(0)
        self.wave_min_reverb_slider.setMaximum(100)
        self.wave_min_reverb_slider.setValue(20)  # 0.2 default
        self.wave_min_reverb_value_label = QLabel("0.20")
        self.wave_min_reverb_slider.valueChanged.connect(self._on_wave_min_reverb_changed)
        wave_reverb_h.addWidget(self.wave_min_reverb_slider, stretch=2)
        wave_reverb_h.addWidget(self.wave_min_reverb_value_label)
        wave_effects_layout.addLayout(wave_reverb_h)
        
        # Max Arm Length
        wave_max_arm_h = QHBoxLayout()
        wave_max_arm_h.addWidget(QLabel("Max Arm:"))
        self.wave_max_armlen_slider = QSlider(Qt.Orientation.Horizontal)
        self.wave_max_armlen_slider.setMinimum(1000)
        self.wave_max_armlen_slider.setMaximum(4000)
        self.wave_max_armlen_slider.setValue(2400)  # 2400mm default
        self.wave_max_armlen_value_label = QLabel("2400mm")
        self.wave_max_armlen_slider.valueChanged.connect(self._on_wave_max_armlen_changed)
        wave_max_arm_h.addWidget(self.wave_max_armlen_slider, stretch=2)
        wave_max_arm_h.addWidget(self.wave_max_armlen_value_label)
        wave_effects_layout.addLayout(wave_max_arm_h)
        
        # Min Arm Length
        wave_min_arm_h = QHBoxLayout()
        wave_min_arm_h.addWidget(QLabel("Min Arm:"))
        self.wave_min_armlen_slider = QSlider(Qt.Orientation.Horizontal)
        self.wave_min_armlen_slider.setMinimum(200)
        self.wave_min_armlen_slider.setMaximum(1000)
        self.wave_min_armlen_slider.setValue(450)  # 450mm default
        self.wave_min_armlen_value_label = QLabel("450mm")
        self.wave_min_armlen_slider.valueChanged.connect(self._on_wave_min_armlen_changed)
        wave_min_arm_h.addWidget(self.wave_min_armlen_slider, stretch=2)
        wave_min_arm_h.addWidget(self.wave_min_armlen_value_label)
        wave_effects_layout.addLayout(wave_min_arm_h)
        
        # High Octave Range
        wave_octave_high_h = QHBoxLayout()
        wave_octave_high_h.addWidget(QLabel("High Octave:"))
        self.wave_octave_high_slider = QSlider(Qt.Orientation.Horizontal)
        self.wave_octave_high_slider.setMinimum(0)  # 0 octaves
        self.wave_octave_high_slider.setMaximum(1000)   # +1 octaves
        self.wave_octave_high_slider.setValue(0)     # +0 octaves default
        self.wave_octave_high_value_label = QLabel("+0.00")
        self.wave_octave_high_slider.valueChanged.connect(self._on_wave_octave_high_changed)
        wave_octave_high_h.addWidget(self.wave_octave_high_slider, stretch=2)
        wave_octave_high_h.addWidget(self.wave_octave_high_value_label)
        wave_effects_layout.addLayout(wave_octave_high_h)
        
        # Low Octave Range
        wave_octave_low_h = QHBoxLayout()
        wave_octave_low_h.addWidget(QLabel("Low Octave:"))
        self.wave_octave_low_slider = QSlider(Qt.Orientation.Horizontal)
        self.wave_octave_low_slider.setMinimum(-1000)  # -1 octaves
        self.wave_octave_low_slider.setMaximum(0)   # 0 octaves
        self.wave_octave_low_slider.setValue(0)    # +0 octaves default
        self.wave_octave_low_value_label = QLabel("0.00")
        self.wave_octave_low_slider.valueChanged.connect(self._on_wave_octave_low_changed)
        wave_octave_low_h.addWidget(self.wave_octave_low_slider, stretch=2)
        wave_octave_low_h.addWidget(self.wave_octave_low_value_label)
        wave_effects_layout.addLayout(wave_octave_low_h)
        
        wave_effects_group.setLayout(wave_effects_layout)
        effects_layout.addWidget(wave_effects_group, stretch=1)
        
        # Drum Effects GroupBox (1/3 width)
        drum_effects_group = QGroupBox("Drum Effects")
        drum_effects_layout = QVBoxLayout()
        
        # Drum Distortion
        drum_dist_h = QHBoxLayout()
        drum_dist_h.addWidget(QLabel("Distortion:"))
        self.drum_distortion_slider = QSlider(Qt.Orientation.Horizontal)
        self.drum_distortion_slider.setMinimum(0)
        self.drum_distortion_slider.setMaximum(100)
        self.drum_distortion_slider.setValue(100)  # 1.0 default
        self.drum_distortion_value_label = QLabel("1.00")
        self.drum_distortion_slider.valueChanged.connect(self._on_drum_distortion_changed)
        drum_dist_h.addWidget(self.drum_distortion_slider, stretch=2)
        drum_dist_h.addWidget(self.drum_distortion_value_label)
        drum_effects_layout.addLayout(drum_dist_h)
        
        # Drum Compression
        drum_comp_h = QHBoxLayout()
        drum_comp_h.addWidget(QLabel("Compression:"))
        self.drum_compression_slider = QSlider(Qt.Orientation.Horizontal)
        self.drum_compression_slider.setMinimum(0)
        self.drum_compression_slider.setMaximum(100)
        self.drum_compression_slider.setValue(100)  # 1.0 default
        self.drum_compression_value_label = QLabel("1.00")
        self.drum_compression_slider.valueChanged.connect(self._on_drum_compression_changed)
        drum_comp_h.addWidget(self.drum_compression_slider, stretch=2)
        drum_comp_h.addWidget(self.drum_compression_value_label)
        drum_effects_layout.addLayout(drum_comp_h)
        
        drum_effects_group.setLayout(drum_effects_layout)
        effects_layout.addWidget(drum_effects_group, stretch=1)
        
        # LLM Effects GroupBox (1/3 width)
        llm_effects_group = QGroupBox("LLM Singer")
        llm_effects_layout = QVBoxLayout()
        
        # Auto-tune Amount
        llm_autotune_h = QHBoxLayout()
        llm_autotune_h.addWidget(QLabel("Auto-tune:"))
        self.llm_autotune_slider = QSlider(Qt.Orientation.Horizontal)
        self.llm_autotune_slider.setMinimum(0)
        self.llm_autotune_slider.setMaximum(100)
        self.llm_autotune_slider.setValue(50)  # 0.5 default
        self.llm_autotune_value_label = QLabel("0.50")
        self.llm_autotune_slider.valueChanged.connect(self._on_llm_autotune_changed)
        llm_autotune_h.addWidget(self.llm_autotune_slider, stretch=2)
        llm_autotune_h.addWidget(self.llm_autotune_value_label)
        llm_effects_layout.addLayout(llm_autotune_h)
        
        # Reverb Amount
        llm_reverb_h = QHBoxLayout()
        llm_reverb_h.addWidget(QLabel("Reverb:"))
        self.llm_reverb_slider = QSlider(Qt.Orientation.Horizontal)
        self.llm_reverb_slider.setMinimum(0)
        self.llm_reverb_slider.setMaximum(100)
        self.llm_reverb_slider.setValue(70)  # 0.7 default
        self.llm_reverb_value_label = QLabel("0.70")
        self.llm_reverb_slider.valueChanged.connect(self._on_llm_reverb_changed)
        llm_reverb_h.addWidget(self.llm_reverb_slider, stretch=2)
        llm_reverb_h.addWidget(self.llm_reverb_value_label)
        llm_effects_layout.addLayout(llm_reverb_h)
        
        # Transpose
        llm_transpose_h = QHBoxLayout()
        llm_transpose_h.addWidget(QLabel("Transpose:"))
        self.llm_transpose_slider = QSlider(Qt.Orientation.Horizontal)
        self.llm_transpose_slider.setMinimum(-400)
        self.llm_transpose_slider.setMaximum(0)
        self.llm_transpose_slider.setValue(0)
        self.llm_transpose_value_label = QLabel("0.00")
        self.llm_transpose_slider.valueChanged.connect(self._on_llm_transpose_changed)
        llm_transpose_h.addWidget(self.llm_transpose_slider, stretch=2)
        llm_transpose_h.addWidget(self.llm_transpose_value_label)
        llm_effects_layout.addLayout(llm_transpose_h)
        
        # Playback Interval (how often to play pre-cached phrases)
        llm_interval_h = QHBoxLayout()
        llm_interval_h.addWidget(QLabel("Play Interval:"))
        self.llm_interval_slider = QSlider(Qt.Orientation.Horizontal)
        self.llm_interval_slider.setMinimum(10)
        self.llm_interval_slider.setMaximum(120)
        self.llm_interval_slider.setValue(15)
        self.llm_interval_value_label = QLabel("15s")
        self.llm_interval_slider.valueChanged.connect(self._on_llm_interval_changed)
        llm_interval_h.addWidget(self.llm_interval_slider, stretch=2)
        llm_interval_h.addWidget(self.llm_interval_value_label)
        llm_effects_layout.addLayout(llm_interval_h)
        
        llm_effects_group.setLayout(llm_effects_layout)
        effects_layout.addWidget(llm_effects_group, stretch=1)
        
        main_v_layout.addLayout(effects_layout)
        
        # Bottom: Main content (voice widgets, head position, and LLM controls)
        self.main_layout = QHBoxLayout()
        
        # Left side: voice widgets in vertical layout (33%)
        self.voices_layout = QVBoxLayout()
        self.main_layout.addLayout(self.voices_layout, stretch=1)
        
        # Middle: head position widget (sequencer) (33%)
        self.head_widget = HeadPositionWidget(num_lanes=num_lanes, top_left=top_left, bottom_right=bottom_right)
        self.main_layout.addWidget(self.head_widget, stretch=1)
        
        # Right side: LLM Singer controls (33%)
        self.llm_layout = self._create_llm_controls()
        self.main_layout.addLayout(self.llm_layout, stretch=1)
        
        main_v_layout.addLayout(self.main_layout)
        
        self.voice_widgets = []
        
        for i in range(self.num_voices):
            color = self.VOICE_COLORS[i % len(self.VOICE_COLORS)]
            widget = VoiceWidget(i + 1, color=color)
            self.voice_widgets.append(widget)
            self.voices_layout.addWidget(widget)

        self.update_signal.connect(self.handle_update)
    
    def _create_llm_controls(self):
        """Create LLM Singer lyrics display panel"""
        llm_layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("LLM Singer Lyrics")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        llm_layout.addWidget(title_label)
        
        # Current Lyrics Display
        self.llm_lyrics_display = QLabel("")
        self.llm_lyrics_display.setWordWrap(True)
        self.llm_lyrics_display.setStyleSheet("background-color: #2a2a2a; padding: 10px; border-radius: 5px; color: #00ff00; font-size: 12px;")
        self.llm_lyrics_display.setAlignment(Qt.AlignmentFlag.AlignTop)
        llm_layout.addWidget(self.llm_lyrics_display, stretch=1)
        
        # Prompt text in the lyrics area
        llm_prompt_label = QLabel("Prompt:")
        llm_layout.addWidget(llm_prompt_label)
        self.llm_prompt_text_bottom = QTextEdit()
        self.llm_prompt_text_bottom.setPlainText("You are an alchemist trying to find a dance to create a bad spell on a kingdom. Describe what you see in poetic, mystical terms in one short sentence.")
        self.llm_prompt_text_bottom.setMaximumHeight(80)
        llm_layout.addWidget(self.llm_prompt_text_bottom)
        
        # Apply button
        apply_prompt_btn = QPushButton("Apply")
        apply_prompt_btn.clicked.connect(self._on_llm_prompt_changed_bottom)
        llm_layout.addWidget(apply_prompt_btn)
        
        # Connect lyrics update signal
        self.llm_lyrics_update.connect(self._on_lyrics_update)
        
        return llm_layout
    
    def _on_loop_slider_changed(self, value):
        """Handle loop length slider changes"""
        loop_length = value / 10.0  # Convert to seconds
        self.loop_value_label.setText(f"{loop_length:.1f}s")
        self.loop_length_changed.emit(loop_length)
        self.settings_changed.emit()
    
    def _on_scale_changed(self, scale_name):
        """Handle scale selection changes"""
        self.scale_changed.emit(scale_name)
        self.settings_changed.emit()
    
    def _on_click_changed(self, state):
        """Handle click checkbox changes"""
        enabled = (state == Qt.CheckState.Checked.value)
        self.click_enabled_changed.emit(enabled)
        self.settings_changed.emit()
    
    def _on_drum_gain_changed(self, value):
        """Handle drum gain slider changes"""
        gain = value / 100.0  # Convert to 0.0-1.0
        self.drum_gain_value_label.setText(f"{gain:.2f}")
        self.drum_gain_changed.emit(gain)
        self.settings_changed.emit()
    
    def _on_wave_gain_changed(self, value):
        """Handle wave gain slider changes"""
        gain = value / 100.0  # Convert to 0.0-1.0
        self.wave_gain_value_label.setText(f"{gain:.2f}")
        self.wave_gain_changed.emit(gain)
        self.settings_changed.emit()
    
    def _on_llm_prompt_changed(self):
        """Handle LLM prompt changes (legacy - kept for compatibility)"""
        # Use bottom prompt text field if it exists
        if hasattr(self, 'llm_prompt_text_bottom'):
            prompt = self.llm_prompt_text_bottom.toPlainText()
        else:
            prompt = ""
        self.llm_prompt_changed.emit(prompt)
        self.settings_changed.emit()
    
    def _on_llm_prompt_changed_bottom(self):
        """Handle LLM prompt changes from bottom text area"""
        prompt = self.llm_prompt_text_bottom.toPlainText()
        self.llm_prompt_changed.emit(prompt)
        self.settings_changed.emit()
    
    def _on_llm_interval_changed(self, value):
        """Handle LLM interval slider changes"""
        self.llm_interval_value_label.setText(f"{value}s")
        self.llm_interval_changed.emit(float(value))
        self.settings_changed.emit()
    
    def _on_llm_autotune_changed(self, value):
        """Handle LLM auto-tune slider changes"""
        amount = value / 100.0
        self.llm_autotune_value_label.setText(f"{amount:.2f}")
        self.llm_autotune_changed.emit(amount)
        self.settings_changed.emit()
    
    def _on_llm_reverb_changed(self, value):
        """Handle LLM reverb slider changes"""
        amount = value / 100.0
        self.llm_reverb_value_label.setText(f"{amount:.2f}")
        self.llm_reverb_changed.emit(amount)
        self.settings_changed.emit()
    
    def _on_llm_transpose_changed(self, value):
        """Handle LLM transpose slider changes"""
        octaves = value / 100.0  # -2.00 to +2.00
        self.llm_transpose_value_label.setText(f"{octaves:+.2f}")
        self.llm_transpose_changed.emit(octaves)
        self.settings_changed.emit()
    
    def _on_llm_gain_changed(self, value):
        """Handle LLM gain slider changes"""
        gain = value / 100.0
        self.llm_gain_value_label.setText(f"{gain:.3f}")
        self.llm_gain_changed.emit(gain)
        self.settings_changed.emit()
    
    def _on_drum_distortion_changed(self, value):
        """Handle drum distortion slider changes"""
        amount = value / 100.0
        self.drum_distortion_value_label.setText(f"{amount:.2f}")
        self.drum_distortion_changed.emit(amount)
        self.settings_changed.emit()
    
    def _on_drum_compression_changed(self, value):
        """Handle drum compression slider changes"""
        amount = value / 100.0
        self.drum_compression_value_label.setText(f"{amount:.2f}")
        self.drum_compression_changed.emit(amount)
        self.settings_changed.emit()
    
    def _on_wave_distortion_changed(self, value):
        """Handle wave distortion slider changes"""
        amount = value / 100.0
        self.wave_distortion_value_label.setText(f"{amount:.2f}")
        self.wave_distortion_changed.emit(amount)
        self.settings_changed.emit()
    
    def _on_wave_lfo_changed(self, value):
        """Handle wave LFO slider changes"""
        amount = value / 100.0
        self.wave_lfo_value_label.setText(f"{amount:.2f}")
        self.wave_lfo_changed.emit(amount)
        self.settings_changed.emit()
    
    def _on_wave_lfo_slowdown_changed(self, value):
        """Handle wave LFO slowdown slider changes"""
        self.wave_lfo_slowdown_value_label.setText(f"{value}")
        self.wave_lfo_slowdown_changed.emit(float(value))
        self.settings_changed.emit()
    
    def _on_wave_min_reverb_changed(self, value):
        """Handle wave min reverb slider changes"""
        amount = value / 100.0
        self.wave_min_reverb_value_label.setText(f"{amount:.2f}")
        self.wave_min_reverb_changed.emit(amount)
        self.settings_changed.emit()
    
    def _on_wave_max_armlen_changed(self, value):
        """Handle wave max arm length slider changes"""
        self.wave_max_armlen_value_label.setText(f"{value}mm")
        self.wave_max_armlen_changed.emit(float(value))
        self.settings_changed.emit()
    
    def _on_wave_min_armlen_changed(self, value):
        """Handle wave min arm length slider changes"""
        self.wave_min_armlen_value_label.setText(f"{value}mm")
        self.wave_min_armlen_changed.emit(float(value))
        self.settings_changed.emit()
    
    def _on_wave_octave_high_changed(self, value):
        """Handle wave high octave range slider changes"""
        octave = value / 1000.0
        self.wave_octave_high_value_label.setText(f"{octave:+.2f}")
        self.wave_octave_high_changed.emit(octave)
        self.settings_changed.emit()
    
    def _on_wave_octave_low_changed(self, value):
        """Handle wave low octave range slider changes"""
        octave = value / 1000.0
        self.wave_octave_low_value_label.setText(f"{octave:+.2f}")
        self.wave_octave_low_changed.emit(octave)
        self.settings_changed.emit()
    
    def _on_lyrics_update(self, lyrics):
        """Handle lyrics updates from LLM Singer"""
        self.llm_lyrics_display.setText(lyrics)
    
    def update_root_note_display(self, note_name, frequency):
        """Update the root note display label"""
        self.root_note_label.setText(f"Root: {note_name} ({frequency:.0f}Hz)")
    
    def get_all_settings(self):
        """Get all current GUI settings as a dictionary"""
        return {
            "loop_length": self.loop_slider.value() / 10.0,
            "scale": self.scale_combo.currentText(),
            "click_enabled": self.click_checkbox.isChecked(),
            "drum_gain": self.drum_gain_slider.value() / 100.0,
            "drum_distortion": self.drum_distortion_slider.value() / 100.0,
            "drum_compression": self.drum_compression_slider.value() / 100.0,
            "wave_gain": self.wave_gain_slider.value() / 100.0,
            "wave_distortion": self.wave_distortion_slider.value() / 100.0,
            "wave_lfo_amount": self.wave_lfo_slider.value() / 100.0,
            "wave_lfo_slowdown": float(self.wave_lfo_slowdown_slider.value()),
            "wave_min_reverb": self.wave_min_reverb_slider.value() / 100.0,
            "wave_max_armlen": float(self.wave_max_armlen_slider.value()),
            "wave_min_armlen": float(self.wave_min_armlen_slider.value()),
            "wave_octave_high": self.wave_octave_high_slider.value() / 1000.0,
            "wave_octave_low": self.wave_octave_low_slider.value() / 1000.0,
            "llm_gain": self.llm_gain_slider.value() / 100.0,
            "llm_prompt": self.llm_prompt_text_bottom.toPlainText() if hasattr(self, 'llm_prompt_text_bottom') else "",
            "llm_interval": float(self.llm_interval_slider.value()),
            "llm_autotune": self.llm_autotune_slider.value() / 100.0,
            "llm_reverb": self.llm_reverb_slider.value() / 100.0,
            "llm_transpose": self.llm_transpose_slider.value() / 100.0,
        }
    
    def set_all_settings(self, settings):
        """Set all GUI settings from a dictionary"""
        # Block signals during bulk update to avoid triggering multiple saves
        self.blockSignals(True)
        
        if "loop_length" in settings:
            self.loop_slider.setValue(int(settings["loop_length"] * 10))
        if "scale" in settings:
            self.scale_combo.setCurrentText(settings["scale"])
        if "click_enabled" in settings:
            self.click_checkbox.setChecked(settings["click_enabled"])
        if "drum_gain" in settings:
            self.drum_gain_slider.setValue(int(settings["drum_gain"] * 100))
        if "drum_distortion" in settings:
            self.drum_distortion_slider.setValue(int(settings["drum_distortion"] * 100))
        if "drum_compression" in settings:
            self.drum_compression_slider.setValue(int(settings["drum_compression"] * 100))
        if "wave_gain" in settings:
            self.wave_gain_slider.setValue(int(settings["wave_gain"] * 100))
        if "wave_distortion" in settings:
            self.wave_distortion_slider.setValue(int(settings["wave_distortion"] * 100))
        if "wave_lfo_amount" in settings:
            self.wave_lfo_slider.setValue(int(settings["wave_lfo_amount"] * 100))
        if "wave_lfo_slowdown" in settings:
            self.wave_lfo_slowdown_slider.setValue(int(settings["wave_lfo_slowdown"]))
        if "wave_min_reverb" in settings:
            self.wave_min_reverb_slider.setValue(int(settings["wave_min_reverb"] * 100))
        if "wave_max_armlen" in settings:
            self.wave_max_armlen_slider.setValue(int(settings["wave_max_armlen"]))
        if "wave_min_armlen" in settings:
            self.wave_min_armlen_slider.setValue(int(settings["wave_min_armlen"]))
        if "wave_octave_high" in settings:
            self.wave_octave_high_slider.setValue(int(settings["wave_octave_high"] * 1000))
        if "wave_octave_low" in settings:
            self.wave_octave_low_slider.setValue(int(settings["wave_octave_low"] * 1000))
        if "llm_gain" in settings:
            self.llm_gain_slider.setValue(int(settings["llm_gain"] * 100))
        if "llm_prompt" in settings and hasattr(self, 'llm_prompt_text_bottom'):
            self.llm_prompt_text_bottom.setPlainText(settings["llm_prompt"])
        if "llm_interval" in settings:
            self.llm_interval_slider.setValue(int(settings["llm_interval"]))
        if "llm_autotune" in settings:
            self.llm_autotune_slider.setValue(int(settings["llm_autotune"] * 100))
        if "llm_reverb" in settings:
            self.llm_reverb_slider.setValue(int(settings["llm_reverb"] * 100))
        if "llm_transpose" in settings:
            self.llm_transpose_slider.setValue(int(settings["llm_transpose"] * 100))
        
        self.blockSignals(False)

    def handle_update(self, all_voice_data):
        active_ids = list(all_voice_data.keys())
        
        # Update head positions, armlines, and fitted lines
        head_positions = {}
        armlines = {}
        fitted_lines = {}
        for voice_id, data in all_voice_data.items():
            head_pos = data.get('head_pos')
            if head_pos:
                head_positions[voice_id] = head_pos
            armline = data.get('armline')
            if armline:
                armlines[voice_id] = armline
            line_info = data.get('line_info')
            if line_info:
                fitted_lines[voice_id] = line_info
        self.head_widget.update_positions(head_positions, armlines, fitted_lines)

        # Update widgets with data
        for i in range(self.num_voices):
            widget = self.voice_widgets[i]
            if i < len(active_ids):
                voice_id = active_ids[i]
                data = all_voice_data[voice_id]
                widget.voice_id = voice_id # Re-assign person ID
                widget.update_data(data)
            else:
                widget.update_data(None)


def run_gui(monitor_instance):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    monitor_instance.show()
    app.exec()
