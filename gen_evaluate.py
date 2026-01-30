import numpy as np
import time
import math
import matplotlib.pyplot as plt

# Đảm bảo bạn đã có file traditional_astar.py và improved_astar.py cùng thư mục
try:
    from traditional_astar import TraditionalAStar
    from improved_astar import ImprovedAStar
except ImportError:
    print("Lỗi: Không tìm thấy file 'traditional_astar.py' hoặc 'improved_astar.py'")
    exit()

# ==========================================
# 1. CÁC HÀM TÍNH TOÁN METRICS
# ==========================================

def calculate_path_length(path):
    """Tính độ dài Euclidean"""
    if not path or len(path) < 2: return 0.0
    length = 0.0
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i+1])
        length += np.linalg.norm(p2 - p1)
    return length

def calculate_total_turning_angle(path):
    """Tính tổng góc quay (độ mượt)"""
    if not path or len(path) < 3: return 0.0
    total_angle = 0.0
    for i in range(1, len(path) - 1):
        p_prev = np.array(path[i-1])
        p_curr = np.array(path[i])
        p_next = np.array(path[i+1])
        
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 > 0 and norm_v2 > 0:
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            total_angle += np.degrees(angle)
    return total_angle

# ==========================================
# 2. HÀM TẠO MAP 100x100
# ==========================================
def generate_constrained_grid(rows, cols, obstacle_prob=0.45):
    grid = np.zeros((rows, cols), dtype=int)
    
    # Đổ vật cản vào phần bụng
    inner_shape = (rows - 2, cols - 2)
    if inner_shape[0] > 0 and inner_shape[1] > 0:
        inner_obstacles = np.random.choice(
            [0, 1], 
            size=inner_shape, 
            p=[1 - obstacle_prob, obstacle_prob]
        )
        grid[1:rows-1, 1:cols-1] = inner_obstacles
        
    # Đảm bảo viền và Start/End an toàn
    grid[0, :] = 0; grid[rows-1, :] = 0
    grid[:, 0] = 0; grid[:, cols-1] = 0
    grid[rows-1, 0] = 0
    grid[0, cols-1] = 0
    
    return grid

# ==========================================
# 3. HÀM BENCHMARK 100x100
# ==========================================
def run_benchmark(num_tests=50, map_size=(100, 100), obstacle_prob=0.45):
    print(f"\nDang chay Benchmark: {num_tests} test cases, Map {map_size}, Obstacle {obstacle_prob*100}%...")
    print("Metrics Improved A* duoc do tren OPTIMIZED PATH (Key Nodes).")
    print("-" * 90)
    
    results = {
        'trad': {'time': [], 'length': [], 'smoothness': [], 'nodes': [], 'success': 0},
        'imp':  {'time': [], 'length': [], 'smoothness': [], 'nodes': [], 'success': 0}
    }

    start_node = (map_size[0] - 1, 0)
    end_node = (0, map_size[1] - 1)

    for i in range(num_tests):
        grid = generate_constrained_grid(map_size[0], map_size[1], obstacle_prob)
        
        # --- TEST TRADITIONAL ---
        t_start = time.perf_counter()
        trad_solver = TraditionalAStar(grid)
        trad_path = trad_solver.search(start_node, end_node)
        t_end = time.perf_counter()
        
        # --- TEST IMPROVED ---
        i_start = time.perf_counter()
        imp_solver = ImprovedAStar(grid)
        # Lấy 'imp_path' là biến đầu tiên (Optimized Path / Key Nodes)
        imp_path, _ = imp_solver.search(start_node, end_node)
        i_end = time.perf_counter()

        if trad_path and imp_path:
            results['trad']['success'] += 1
            results['trad']['time'].append((t_end - t_start) * 1000)
            results['trad']['length'].append(calculate_path_length(trad_path))
            results['trad']['smoothness'].append(calculate_total_turning_angle(trad_path))
            results['trad']['nodes'].append(len(trad_path))

            results['imp']['success'] += 1
            results['imp']['time'].append((i_end - i_start) * 1000)
            results['imp']['length'].append(calculate_path_length(imp_path))
            results['imp']['smoothness'].append(calculate_total_turning_angle(imp_path))
            results['imp']['nodes'].append(len(imp_path))
        
        if (i+1) % 10 == 0:
            print(f" -> Xong {i+1}/{num_tests} maps...")

    avg = lambda lst: sum(lst) / len(lst) if lst else 0

    print("\n" + "="*100)
    print(f"{'METRIC':<25} | {'TRADITIONAL A*':<20} | {'IMPROVED A*':<20} | {'CAI THIEN':<10}")
    print("="*100)

    # 1. Thời gian
    t1, t2 = avg(results['trad']['time']), avg(results['imp']['time'])
    diff_t = ((t1 - t2)/t1 * 100) if t1 > 0 else 0
    print(f"{'Thoi gian (ms)':<25} | {t1:<20.4f} | {t2:<20.4f} | {diff_t:+.2f}%")

    # 2. Độ dài
    l1, l2 = avg(results['trad']['length']), avg(results['imp']['length'])
    diff_l = ((l1 - l2)/l1 * 100) if l1 > 0 else 0
    print(f"{'Do dai (Euclidean)':<25} | {l1:<20.4f} | {l2:<20.4f} | {diff_l:+.2f}%")

    # 3. Số Node
    n1, n2 = avg(results['trad']['nodes']), avg(results['imp']['nodes'])
    diff_n = ((n1 - n2)/n1 * 100) if n1 > 0 else 0
    print(f"{'So luong Node':<25} | {n1:<20.2f} | {n2:<20.2f} | {diff_n:+.2f}%")

    # 4. Độ mượt
    s1, s2 = avg(results['trad']['smoothness']), avg(results['imp']['smoothness'])
    diff_s = ((s1 - s2)/s1 * 100) if s1 > 0 else 0
    print(f"{'Tong goc quay (Deg)':<25} | {s1:<20.2f} | {s2:<20.2f} | {diff_s:+.2f}%")
    print("-" * 100)
    print(f"Success Rate: {results['trad']['success']}/{num_tests}")
    
    return results

# ==========================================
# 4. HÀM VẼ 1 CASE 
# ==========================================
def visualize_one_case(map_size=(100, 100), obstacle_prob=0.45):
    grid = generate_constrained_grid(map_size[0], map_size[1], obstacle_prob)
    start = (map_size[0] - 1, 0)
    end = (0, map_size[1] - 1)
    
    t_solver = TraditionalAStar(grid)
    t_path = t_solver.search(start, end)
    
    i_solver = ImprovedAStar(grid)
    geo_path, smooth_path = i_solver.search(start, end) 
    
    if not t_path or not geo_path:
        print("Map ngau nhien nay khong co duong, thu lai...")
        return visualize_one_case(map_size, obstacle_prob)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='binary', origin='upper') 
    
    if t_path:
        ty, tx = zip(*t_path)
        plt.plot(tx, ty, 'y--', linewidth=1, label='Traditional A*', alpha=0.5)
        
    if geo_path:
        gy, gx = zip(*geo_path)
        plt.plot(gx, gy, 'bo', markersize=4, linestyle='-', linewidth=1, label='Geometric Key Nodes')

    if smooth_path:
        sy, sx = zip(*smooth_path)
        plt.plot(sx, sy, 'r-', linewidth=2, label='B-Spline Illustration')
        
    plt.plot(start[1], start[0], 'go', markersize=10, label='Start')
    plt.plot(end[1], end[0], 'ro', markersize=10, label='End')
    
    plt.title(f"Comparison (Map {map_size} - Obs {obstacle_prob*100}%)")
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    # 1. Chạy Benchmark
    run_benchmark(num_tests=50, map_size=(100, 100), obstacle_prob=0.3)
    
    # 2. Vẽ hình minh họa
    visualize_one_case(map_size=(100, 100), obstacle_prob=0.3)
