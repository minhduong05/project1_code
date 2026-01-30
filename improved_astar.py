import numpy as np
import math
from traditional_astar import TraditionalAStar

class ImprovedAStar:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        self.M_spline = np.array([
            [-1,  3, -3,  1],
            [ 3, -6,  3,  0],
            [-3,  0,  3,  0],
            [ 1,  4,  1,  0]
        ]) / 6.0

    def _is_valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] == 0

    def _LOS(self, p1, p2):
        x0, y0 = p1
        x1, y1 = p2
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if not self._is_valid(x, y): return False
                err -= dy
                if err < 0:
                    if not self._is_valid(x + sx, y) or not self._is_valid(x, y + sy):
                        return False
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if not self._is_valid(x, y): return False
                err -= dx
                if err < 0:
                    if not self._is_valid(x + sx, y) or not self._is_valid(x, y + sy):
                        return False
                    x += sx
                    err += dy
                y += sy

        if not self._is_valid(x1, y1): return False
        return True

    def GreedyRaycast(self, raw_path):
        if not raw_path or len(raw_path) < 3:
            return raw_path
        
        final_path = [raw_path[0]]
        current_idx = 0
        
        while current_idx < len(raw_path) - 1:
            found_next = False
            
            for check_idx in range(len(raw_path) - 1, current_idx, -1):
                target_node = raw_path[check_idx]
                if self._LOS(raw_path[current_idx], target_node):
                    final_path.append(target_node)
                    current_idx = check_idx
                    found_next = True
                    break

            if not found_next:
                current_idx += 1
                final_path.append(raw_path[current_idx])

        return final_path

    def _calculate_turning_angle(self, p_prev, p_curr, p_next):
        v1 = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]])
        v2 = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]])
        norm_v1 = np.linalg.norm(v1); norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0
        dot_val = np.dot(v1, v2)
        cos_angle = np.clip(dot_val / (norm_v1 * norm_v2), -1.0, 1.0)
        return np.arccos(cos_angle)

    def _generate_b_spline(self, p0, p1, p2, p3, num_points=10):
        P = np.array([p0, p1, p2, p3])
        curve_points = []
        for t in np.linspace(0, 1, num_points):
            T = np.array([t**3, t**2, t, 1])
            Q = T @ self.M_spline @ P
            curve_points.append(tuple(Q))
        return curve_points

    def _get_control_points(self, p_prev, p_curr, p_next):
        def get_pt(start, end, dist=1000.0):
            vec = np.array([end[0] - start[0], end[1] - start[1]])
            length = np.linalg.norm(vec)
            if length == 0: return start
            real_dist = min(dist, length)
            unit_vec = vec / length
            return (start[0] + unit_vec[0] * real_dist, start[1] + unit_vec[1] * real_dist)
        return get_pt(p_curr, p_prev, 1000.0), get_pt(p_curr, p_next, 1000.0)

    def smooth_path(self, final_nodes):
        if not final_nodes or len(final_nodes) < 3:
            return final_nodes

        smooth_result = [final_nodes[0]]
        i = 1
        while i < len(final_nodes) - 1:
            p_prev = final_nodes[i-1]
            p_curr = final_nodes[i]
            p_next = final_nodes[i+1]
            
            angle = self._calculate_turning_angle(p_prev, p_curr, p_next)
            if angle > 0.08: 
                cp1, cp2 = self._get_control_points(p_prev, p_curr, p_next)
                curve = self._generate_b_spline(cp1, p_curr, p_curr, cp2)
                smooth_result.extend(curve)
            else:
                smooth_result.append(p_curr)
            i += 1
            
        smooth_result.append(final_nodes[-1])
        return smooth_result

    def search(self, start, end):
        traditional_finder = TraditionalAStar(self.grid)
        raw_path = traditional_finder.search(start, end)
        
        if not raw_path:
            return None, None
            
        optimized_path = self.GreedyRaycast(raw_path)
        final_smooth_path = self.smooth_path(optimized_path)
        
        return optimized_path, final_smooth_path