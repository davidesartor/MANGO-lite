def add_indent(s: str, indent=2, skip_first=True) -> str:
    """Add indentation to all lines in a string."""
    s = "\n".join(" " * indent + line for line in s.splitlines())
    if skip_first:
        s = s[indent:]
    return s


def torch_style_repr(class_name: str, params: dict[str, str]) -> str:
    repr_str = class_name + "(\n"
    for k, v in params.items():
        repr_str += f"({k}): {v}\n"
    repr_str = add_indent(repr_str) + "\n)"
    return repr_str


from collections import deque
import numpy.typing as npt
import numpy as np

def is_traversable(matrix: npt.NDArray[np.int32], start: tuple[int, int], end: tuple[int, int]) -> bool:
    """
    Determine if it's possible to traverse from start to end in the matrix.
    
    :param matrix: 3D numpy array of shape (n, n, 3)
    :param start: Tuple (i, j) indicating starting cell
    :param end: Tuple (i, j) indicating ending cell
    :return: Boolean indicating whether end is reachable from start
    """
    n = matrix.shape[1]
    visited = [[False for _ in range(n)] for _ in range(n)]
    queue = deque([start])
    
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]  # Right, Down, Up, Left
    
    while queue:
        current = queue.popleft()
        i, j = current
        
        # Mark as visited
        visited[i][j] = True
        
        # Check if we've reached the end
        if current == end:
            return True
        
        # Explore neighbors
        for di, dj in directions:
            ni, nj = i + di, j + dj
            
            # Check bounds and if the cell is traversable and not visited
            if 0 <= ni < n and 0 <= nj < n and not matrix[1,ni, nj] and not visited[ni][nj]:
                queue.append((ni, nj))
                visited[ni][nj] = True     
    return False