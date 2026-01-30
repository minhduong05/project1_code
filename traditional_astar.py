import heapq
import math

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

class TraditionalAStar:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] == 0

    def _heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def search(self, start, end):
        if not self._is_valid(start) or not self._is_valid(end):
            return None 

        start_node = Node(start)
        open_list = []
        heapq.heappush(open_list, start_node)
        
        closed_set = set()
        g_score = {start: 0}

        while open_list:
            current = heapq.heappop(open_list)
            
            if current.position in closed_set:
                continue
            
            closed_set.add(current.position)

            if current.position == end:
                path = []
                while current:
                    path.append(current.position)
                    current = current.parent

                return path[::-1] 

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]

            for dr, dc in directions:
                neighbor_pos = (current.position[0] + dr, current.position[1] + dc)

                if not self._is_valid(neighbor_pos):
                    continue

                if abs(dr) == 1 and abs(dc) == 1:
                    if self.grid[current.position[0] + dr][current.position[1]] == 1 and \
                       self.grid[current.position[0]][current.position[1] + dc] == 1:
                        continue

                if neighbor_pos in closed_set:
                    continue

                new_g = current.g + math.sqrt(dr**2 + dc**2)

                if new_g < g_score.get(neighbor_pos, float('inf')):
                    g_score[neighbor_pos] = new_g
                    neighbor = Node(neighbor_pos, current)
                    neighbor.g = new_g
                    neighbor.f = new_g + self._heuristic(neighbor_pos, end)
                    heapq.heappush(open_list, neighbor)

        return None 