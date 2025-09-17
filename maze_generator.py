import matplotlib.pyplot as plt
import numpy as np
import random
from queue import Queue
import time

def create_maze(dim):
    # Create a grid filled with walls
    maze = np.ones((dim*2, dim*2))

    # Define the starting point
    x, y = (0, 0)
    maze[2*x, 2*y] = 0

    # Initialize the stack with the starting point
    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack[-1]

        # Define possible directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx >= 0 and ny >= 0 and nx < dim and ny < dim and maze[2*nx, 2*ny] == 1:
                maze[2*nx, 2*ny] = 0
                maze[2*x+dx, 2*y+dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()
            
    # Create an entrance and an exit
    maze[-1, 0] = 0
    maze[-2, -1] = 0

    return maze

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_maze(maze, path=None, passages=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display the maze
    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
    
    rows, cols = maze.shape
    half_row = rows // 2
    half_col = cols // 2

    # Draw the solution path if it exists
    if path is not None:
        x_coords = [x[1] for x in path]
        y_coords = [x[0] for x in path]
        ax.plot(x_coords, y_coords, color='red', linewidth=2)
    
    if passages is not None:
        for section_coords in passages.values():       # left, right, top, bottom
            for pair in section_coords:               # each passage is a pair of coords
                for (r, c) in pair:
                    ax.scatter(c, r, color='green', s=2000, zorder=3)
    
    # Draw cross lines shifted
    ax.axhline(y=half_row-.5, color='blue', linewidth=2, linestyle='--', zorder=5)  # horizontal slightly up
    ax.axvline(x=half_col-.5, color='blue', linewidth=2, linestyle='--', zorder=5)  # vertical slightly left
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Draw entry and exit arrows
    ax.arrow(0, 0, .4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)
    ax.arrow(
        cols-1, rows-2, 0.4, 0,
        fc='blue', ec='blue', head_width=0.3, head_length=0.3
    )
    
    plt.show()

def find_path(maze):
    # BFS algorithm to find the shortest path
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    start = (1, 1)
    end = (maze.shape[0]-2, maze.shape[1]-2)
    visited = np.zeros_like(maze, dtype=bool)
    visited[start] = True
    queue = Queue()
    queue.put((start, []))
    while not queue.empty():
        (node, path) = queue.get()
        for dx, dy in directions:
            next_node = (node[0]+dx, node[1]+dy)
            if (next_node == end):
                return path + [next_node]
            if (next_node[0] >= 0 and next_node[1] >= 0 and 
                next_node[0] < maze.shape[0] and next_node[1] < maze.shape[1] and 
                maze[next_node] == 0 and not visited[next_node]):
                visited[next_node] = True
                queue.put((next_node, path + [next_node]))



import numpy as np

def search(maze):
    rows = len(maze)
    maze[0][0] = 2
    maze[rows-2][-1] = 2
    half = rows // 2

    # Coordinate storage
    leftcoords, rightcoords, topcoords, bottomcoords = [], [], [], []

    # Storage + counters
    left, right, top, bottom = [], [], [], []
    leftcounter = rightcounter = topcounter = bottomcounter = 0

    # === LEFT (first half of elements from middle 2 rows) ===
    row1 = [int(i) if isinstance(i, np.float64) else i for i in maze[half-1][:half]]
    row2 = [int(i) if isinstance(i, np.float64) else i for i in maze[half][:half]]
    left.append(row1)
    left.append(row2)

    for i in range(half):
        if left[0][i] == 0 and left[1][i] == 0:
            leftcounter += 1
            leftcoords.append(((half-1, i), (half, i)))

    # === RIGHT (last half of elements from middle 2 rows) ===
    row1 = [int(i) if isinstance(i, np.float64) else i for i in maze[half-1][-half:]]
    row2 = [int(i) if isinstance(i, np.float64) else i for i in maze[half][-half:]]
    right.append(row1)
    right.append(row2)

    for i in range(half):
        if right[0][i] == 0 and right[1][i] == 0:
            rightcounter += 1
            col = i + (rows - half)   # adjust column index because of slicing
            rightcoords.append(((half-1, col), (half, col)))

    # === TOP (first half rows, middle 2 columns) ===
    col1 = []
    col2 = []
    for r in maze[:half]:
        i1 = int(r[half-1]) if isinstance(r[half-1], np.float64) else r[half-1]
        i2 = int(r[half]) if isinstance(r[half], np.float64) else r[half]
        col1.append(i1)
        col2.append(i2)
    top.append(col1)
    top.append(col2)

    for i in range(half):
        if top[0][i] == 0 and top[1][i] == 0:
            topcounter += 1
            topcoords.append(((i, half-1), (i, half)))

    # === BOTTOM (last half rows, middle 2 columns) ===
    col1 = []
    col2 = []
    for r in maze[-half:]:
        i1 = int(r[half-1]) if isinstance(r[half-1], np.float64) else r[half-1]
        i2 = int(r[half]) if isinstance(r[half], np.float64) else r[half]
        col1.append(i1)
        col2.append(i2)
    bottom.append(col1)
    bottom.append(col2)

    for i in range(half):
        if bottom[0][i] == 0 and bottom[1][i] == 0:
            bottomcounter += 1
            row = i + (rows - half)   # adjust row index because of slicing
            bottomcoords.append(((row, half-1), (row, half)))

    q1 = leftcounter + topcounter
    q2 = topcounter + rightcounter
    q3 = rightcounter + bottomcounter
    q4 = bottomcounter + leftcounter
    # Return passages in dictionary form
    return {
        "left": leftcoords,
        "right": rightcoords,
        "top": topcoords,
        "bottom": bottomcoords
    }





maze = create_maze(4)
#starttime = time.perf_counter()
#path = find_path(maze)
#endtime = time.perf_counter()
path = search(maze)
draw = draw_maze(maze, passages=path)
#print(f"Took {endtime - starttime} seconds to solve the maze.")