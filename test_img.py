import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- IMPORT CLASS THUẬT TOÁN ---
try:
    from improved_astar import ImprovedAStar
    from traditional_astar import TraditionalAStar
except ImportError:
    print("LỖI: Không tìm thấy file thuật toán (improved_astar.py, traditional_astar.py)")
    exit()

# ==========================================
# 1. HÀM ĐỌC MAP TỪ ẢNH
# ==========================================
def load_map(image_path, target_size=(20, 20)):
    if not os.path.exists(image_path):
        print(f"LỖI: Không tìm thấy file {image_path}")
        return None

    img = Image.open(image_path).convert('L')
    img = img.resize(target_size, Image.Resampling.NEAREST)
    img_arr = np.array(img)
    
    grid = np.where(img_arr > 128, 0, 1)
    return grid

# ==========================================
# 2. HÀM VẼ KẾT QUẢ (TRỰC TIẾP TRÊN MATRIX)
# ==========================================
def plot_direct(grid, start, end, raw_path, key_nodes, smooth_path):
    h, w = grid.shape
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap='binary', origin='upper')

    if raw_path:
        rows, cols = zip(*raw_path)
        ax.plot(cols, rows, 'y--', linewidth=2, label='Traditional A*', alpha=0.7)
        
    if key_nodes:
        rows, cols = zip(*key_nodes)
        ax.plot(cols, rows, 'bo', markersize=6, linestyle='-', linewidth=1, label='Key Nodes')
        
    if smooth_path:
        rows, cols = zip(*smooth_path)
        ax.plot(cols, rows, 'r-', linewidth=3, label='Improved (Smooth)')

    ax.plot(start[1], start[0], 'go', markersize=12, label='Start') 
    ax.plot(end[1], end[0], 'ro', markersize=12, label='End')    
    ax.set_title(f"Kết quả Map {w}x{h} (Gốc 0,0 ở góc trên trái)", fontsize=12)
    ax.legend(loc='lower right')
    plt.show()

# ==========================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    img_path = "data_img/20x20.png"
    size = (20, 20)

    print(f"Đang xử lý: {img_path}...")
    grid = load_map(img_path, target_size=size)

    if grid is not None:
        rows, cols = grid.shape
        start_node = (rows - 1, 0)
        end_node = (0, cols - 1)
        
        if grid[start_node] == 1: 
            print("Start chạm tường! Dời sang (rows-2, 1)")
            start_node = (rows - 2, 1)
            
        if grid[end_node] == 1:
            print("End chạm tường! Dời sang (1, cols-2)")
            end_node = (1, cols - 2)

        print(f"Start: {start_node} (Row, Col)")
        print(f"End:   {end_node} (Row, Col)")

        # --- CHẠY THUẬT TOÁN ---
        
        # 1. Traditional
        print("-> Running Traditional A*...")
        t_solver = TraditionalAStar(grid)
        raw_path = t_solver.search(start_node, end_node)

        # 2. Improved
        print("-> Running Improved A*...")
        i_solver = ImprovedAStar(grid)
        key_nodes, smooth_path = i_solver.search(start_node, end_node)

        # --- VẼ ---
        if raw_path and smooth_path:
            print("Đã tìm thấy đường! Đang vẽ...")
            plot_direct(grid, start_node, end_node, raw_path, key_nodes, smooth_path)
        else:
            print("Không tìm thấy đường đi!")