import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import math
import copy
from collections import deque
import time
import psutil


def load_octave_column_matrix(file_path):
    """
    Read a column-format file where the first 5 lines are header and the following
    lines are integer values of a 512x512 matrix in column-major order.
    Returns a 512x512 numpy array (dtype=np.uint8).
    """
    values = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip the first 5 header lines (if they exist)
    matrix_lines = lines[5:] if len(lines) > 5 else lines

    for line in matrix_lines:
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line))
        except ValueError:
            # ignore non-integer lines
            continue

    values = np.array(values, dtype=np.int32)

    if values.size != 512 * 512:
        raise ValueError(f"Expected 262144 elements, but got {values.size} elements.")

    # The file claims column-major; the original script reshaped to (512,512) and then transposed;
    # to preserve the original behaviour we will reshape in column-major and then transpose.
    reshaped = values.reshape((512, 512), order="F")
    # Convert to uint8 for image operations (clamp to 0..255)
    reshaped = np.clip(reshaped, 0, 255).astype(np.uint8)
    return reshaped


def create_patches(image, patch_size=128):
    """
    Split a square image into equal non-overlapping patches of given patch_size.
    Returns: patches (list indexed 0..num_patches-1) and patch_grid_shape (rows, cols).
    """
    h, w = image.shape[:2]
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError("Image dimensions must be divisible by patch_size")

    rows = h // patch_size
    cols = w // patch_size
    patches = []
    grid_indices = []
    idx = 0
    for i in range(rows):
        row_idx = []
        for j in range(cols):
            patch = image[i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size]
            patches.append(patch.copy())
            row_idx.append(idx)
            idx += 1
        grid_indices.append(row_idx)

    return patches, grid_indices  # patches is list of arrays, grid_indices is layout of indices for original order


def reconstruct_image(patches, grid):
    """
    Reconstruct image from patches using grid of indices.
    patches: list of numpy arrays (patch_h, patch_w)
    grid: 2D list of patch indices
    """
    patch_h, patch_w = patches[0].shape[:2]
    grid_h = len(grid)
    grid_w = len(grid[0])
    full = np.zeros((grid_h * patch_h, grid_w * patch_w), dtype=np.uint8)

    for i, row in enumerate(grid):
        for j, idx in enumerate(row):
            full[i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w] = patches[idx]

    return full


def get_value(candidate_indices, parent_patch, patches, direction):
    """
    Among candidate_indices (list of patch indices), return the index whose border best matches parent_patch
    in the provided direction.
    direction: (dx, dy) where parent -> child is (dx,dy)
    """
    best_score = float("inf")
    best_idx = None
    parent = np.array(parent_patch)

    for idx in candidate_indices:
        child = np.array(patches[idx])
        score = 0
        if direction == (0, 1):  # child is to the right of parent
            # compare parent's right column with child's left column
            score = np.sum(np.abs(parent[:, -1].astype(int) - child[:, 0].astype(int)))
        elif direction == (0, -1):  # child is to the left of parent
            score = np.sum(np.abs(parent[:, 0].astype(int) - child[:, -1].astype(int)))
        elif direction == (1, 0):  # child is below parent
            score = np.sum(np.abs(parent[-1, :].astype(int) - child[0, :].astype(int)))
        elif direction == (-1, 0):  # child is above parent
            score = np.sum(np.abs(parent[0, :].astype(int) - child[-1, :].astype(int)))
        else:
            continue

        if score < best_score:
            best_score = score
            best_idx = idx

    return best_idx


def bfs_fill(grid, patches, remaining_indices):
    """
    Fill grid starting from (0,0) using greedy neighbor selection.
    grid: 4x4 list of -1 entries except grid[0][0] preset to a chosen index.
    remaining_indices: list of available patch indices to place (does NOT include the seed index).
    """
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    q = deque([(0, 0)])
    visited = set([(0, 0)])

    while q and remaining_indices:
        x, y = q.popleft()
        # For each neighbor place the best-fitting remaining patch
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 4 and 0 <= ny < 4 and (nx, ny) not in visited:
                # find best patch from remaining_indices to place at (nx, ny) relative to (x,y)
                parent_idx = grid[x][y]
                if parent_idx == -1:
                    # if the parent's index hasn't been set for some reason, skip
                    visited.add((nx, ny))
                    q.append((nx, ny))
                    continue

                best_idx = get_value(remaining_indices, patches[parent_idx], patches, (dx, dy))
                if best_idx is None:
                    visited.add((nx, ny))
                    q.append((nx, ny))
                    continue

                grid[nx][ny] = best_idx
                remaining_indices.remove(best_idx)
                visited.add((nx, ny))
                q.append((nx, ny))


def get_neighbors(i, j, grid):
    neighbors = []
    rows, cols = len(grid), len(grid[0])
    if i - 1 >= 0:
        neighbors.append((i - 1, j))
    if i + 1 < rows:
        neighbors.append((i + 1, j))
    if j - 1 >= 0:
        neighbors.append((i, j - 1))
    if j + 1 < cols:
        neighbors.append((i, j + 1))
    return neighbors


def value_function(grid, patches):
    """
    Compute an aggregate boundary mismatch score for the grid.
    Lower is better. We compute absolute differences on all adjacent borders.
    """
    score = 0
    H = 128
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            idx = grid[i][j]
            if idx < 0:
                # missing patch (shouldn't happen in final grid) — penalize heavily
                score += 255 * H
                continue
            for ni, nj in get_neighbors(i, j, grid):
                # to avoid double counting, only sum for neighbors that are "greater" in index order
                if (ni, nj) <= (i, j):
                    continue
                img_a = patches[idx].astype(int)
                img_b = patches[grid[ni][nj]].astype(int)
                if ni == i and nj == j + 1:
                    # right neighbor: compare a[:, -1] with b[:, 0]
                    score += np.sum(np.abs(img_a[:, -1] - img_b[:, 0]))
                elif ni == i and nj == j - 1:
                    score += np.sum(np.abs(img_b[:, -1] - img_a[:, 0]))
                elif ni == i + 1 and nj == j:
                    score += np.sum(np.abs(img_a[-1, :] - img_b[0, :]))
                elif ni == i - 1 and nj == j:
                    score += np.sum(np.abs(img_b[-1, :] - img_a[0, :]))

    # return sqrt to keep same scale idea as original
    return math.sqrt(score)


def calculate_gradients(image, threshold=100):
    """
    Compute Sobel gradient magnitudes, threshold weak gradients, and return a scalar score.
    """
    gray = image.astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x*2 + grad_y*2)

    # zero-out small magnitudes
    grad_x[magnitude < threshold] = 0
    grad_y[magnitude < threshold] = 0

    # combine absolute sums into a single scalar
    grad_sum = math.sqrt(np.sum(np.abs(grad_x)) + np.sum(np.abs(grad_y)))
    return grad_sum


def simulated_annealing(grid, patches, initial_score):
    """
    Improve arrangement using simulated annealing by random swaps.
    Returns best_grid, best_score, iterations_taken.
    """
    current_grid = copy.deepcopy(grid)
    best_grid = copy.deepcopy(grid)
    current_score = initial_score
    best_score = current_score

    initial_temp = 1000.0
    final_temp = 1.0
    alpha = 0.995
    temp = initial_temp
    iterations = 0

    while temp > final_temp:
        iterations += 1
        # pick two distinct cells and swap their patch indices
        x1, y1 = random.randrange(4), random.randrange(4)
        x2, y2 = random.randrange(4), random.randrange(4)
        while x1 == x2 and y1 == y2:
            x2, y2 = random.randrange(4), random.randrange(4)

        # perform swap
        current_grid[x1][y1], current_grid[x2][y2] = current_grid[x2][y2], current_grid[x1][y1]

        new_score = value_function(current_grid, patches)

        # accept if better or with probability exp((current - new)/T)
        delta = current_score - new_score
        if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-9)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_grid = copy.deepcopy(current_grid)
        else:
            # revert swap
            current_grid[x1][y1], current_grid[x2][y2] = current_grid[x2][y2], current_grid[x1][y1]

        temp *= alpha

    return best_grid, best_score, iterations


def show_image(image, title="Image"):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


if _name_ == "_main_":
    # Load input (expects the specific column-format file used previously)
    imgs = load_octave_column_matrix("jigsaww.mat")
    print("Loaded image shape:", imgs.shape)  # Should be (512,512)

    # Original code transposed the matrix; keep same behaviour (if needed)
    imgs = imgs.T

    patches, original_grid = create_patches(imgs, patch_size=128)  # patches list and original mapping

    all_indices = list(range(len(patches)))  # 0..15

    final_grid = None
    final_score = float("inf")
    total_iterations = 0

    start_time = time.time()
    process = psutil.Process()

    # Try all possible seeds for position (0,0) as initial greedy fill
    for seed in range(len(patches)):
        # create empty 4x4 grid and set seed in (0,0)
        grid = [[-1 for _ in range(4)] for _ in range(4)]
        grid[0][0] = seed
        remaining = all_indices.copy()
        remaining.remove(seed)

        # Greedy BFS fill
        bfs_fill(grid, patches, remaining)

        # reconstruct and evaluate via gradient (edge-based)
        reconstructed = reconstruct_image(patches, grid)
        grad_score = calculate_gradients(reconstructed)

        # run simulated annealing starting from this greedy arrangement
        sa_grid, sa_score, iters = simulated_annealing(grid, patches, grad_score)

        # choose best among greedy and simulated annealing results
        if sa_score < final_score:
            final_score = sa_score
            final_grid = sa_grid
            total_iterations = iters
        if grad_score < final_score:
            final_score = grad_score
            final_grid = grid
            total_iterations = iters

    total_time = time.time() - start_time
    mem_mb = process.memory_info().rss / (1024 * 1024)

    print(f"Iterations in last SA run: {total_iterations}")
    print(f"Time required: {total_time:.2f} seconds")
    print(f"Memory usage: {mem_mb:.2f} MB")

    # Reconstruct and display the best image found
    best_img = reconstruct_image(patches, final_grid)
    show_image(best_img, title=f"Best reconstructed image (score={final_score:.2f})")