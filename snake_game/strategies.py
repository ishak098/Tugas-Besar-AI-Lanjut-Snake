from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from random import Random
from collections import deque

from .models import Direction, GameSnapshot


class MoveStrategy(ABC):
    """Base class for every automated snake movement method."""

    @abstractmethod
    def choose_move(
        self, snapshot: GameSnapshot, snake_id: str, rng: Random
    ) -> Direction:
        """Return the direction for ``snake_id`` for the current turn."""


STRATEGY_REGISTRY: dict[str, type[MoveStrategy]] = {}


def register_strategy(name: str) -> Callable[[type[MoveStrategy]], type[MoveStrategy]]:
    """Register a strategy class so it appears in the setup-screen dropdown."""

    def decorator(strategy_class: type[MoveStrategy]) -> type[MoveStrategy]:
        if not name.strip():
            raise ValueError("Strategy names cannot be empty.")
        STRATEGY_REGISTRY[name] = strategy_class
        return strategy_class

    return decorator


def create_strategy(name: str) -> MoveStrategy:
    try:
        return STRATEGY_REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown strategy: {name}") from exc


@register_strategy("Safe Random")
class SafeRandomStrategy(MoveStrategy):
    def choose_move(
        self, snapshot: GameSnapshot, snake_id: str, rng: Random
    ) -> Direction:
        legal = snapshot.legal_moves_for(snake_id)
        if legal:
            return rng.choice(legal)
        return snapshot.snake(snake_id).direction


@register_strategy("Greedy")
class GreedyStrategy(MoveStrategy):
    def choose_move(
        self, snapshot: GameSnapshot, snake_id: str, rng: Random
    ) -> Direction:
        legal = snapshot.legal_moves_for(snake_id)
        snake = snapshot.snake(snake_id)
        if not legal or not snapshot.apples:
            return snake.direction

        head_x, head_y = snake.body[0]

        def distance(direction: Direction) -> int:
            dx, dy = direction.vector
            new_x, new_y = head_x + dx, head_y + dy
            return min(
                abs(new_x - apple_x) + abs(new_y - apple_y)
                for apple_x, apple_y in snapshot.apples
            )

        best_distance = min(distance(direction) for direction in legal)
        best_moves = [direction for direction in legal if distance(direction) == best_distance]
        return rng.choice(best_moves)


# Assignment template -------------------------------------------------------
# 1. Copy this class and give it a unique name.
# 2. Implement choose_move using only the immutable snapshot.
# 3. Add @register_strategy("My Strategy") above the class. It will then appear
#    automatically in the setup screen.
#
# @register_strategy("My Strategy")
# class MyStrategy(MoveStrategy):
#     def choose_move(self, snapshot, snake_id, rng):
#         legal_moves = snapshot.legal_moves_for(snake_id)
#         return legal_moves[0] if legal_moves else snapshot.snake(snake_id).direction

@register_strategy("AStar")
class AStarStrategy(MoveStrategy):
    def choose_move(
        self, snapshot: GameSnapshot, snake_id: str, rng: Random
    ) -> Direction:
        legal = snapshot.legal_moves_for(snake_id)
        snake = snapshot.snake(snake_id)
        if not legal or not snapshot.apples:
            return snake.direction

        head_x, head_y = snake.body[0]
        start = (head_x, head_y)
        offset = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        obstacles = set(snake.body)

        def hiuristic(node: tuple[int, int]) -> int:
            node_x, node_y = node
            # Cari jarak apel paling dekat dengan snake
            return min(
                abs(node_x - apple_x) + abs(node_y - apple_y)
                for apple_x, apple_y in snapshot.apples
            )
        open_list = [start]
        closed_set = set()
        parent_map = {} #jejak selama dipapan game

        g_score = {start: 0}

        f_score = {start : g_score[start] + hiuristic(start)}

        while open_list:
            # ambil kordinat dengan nilai terkecil perhitungan dari f_score dengan key yang ada di open_list
            current = min(open_list, key=f_score.get)
            if current in snapshot.apples:
                # untuk mendaftar jalur menuju apel
                path = []
                curr = current
                while curr in parent_map:
                    # tambah jalur dari apel ke kepala ular
                    path.append(curr)
                    # menelurusi jalur asal titik
                    curr = parent_map[curr]
                path.reverse()

                if path:
                    next_x, next_y = path[0]
                    # selisih dari kordinat kepala ular dan kordinat perpindahan
                    dx, dy = next_x - head_x, next_y - head_y
                    for move in legal:
                        # selisihnya ada yang sama dengan arah (UDLR) atau tidak
                        if move.vector == (dx, dy):
                            return move
            open_list.remove(current)
            closed_set.add(current)

            for dx, dy in offset:
                new_x, new_y = current[0] + dx, current[1] + dy
                new_node = (new_x, new_y)
                if new_x < 0 or new_x >= snapshot.columns or new_y < 0 or new_y >= snapshot.rows:
                    continue
                if new_node in obstacles or new_node in closed_set:
                    # Lewati jika node sudah dikunjungi
                    continue
                new_g_score = g_score[current] + 1
                if new_node not in open_list or new_g_score < g_score[new_node]:
                    parent_map[new_node] = current
                    g_score[new_node] = new_g_score
                    f_score[new_node] = new_g_score + hiuristic(new_node)
                    if new_node not in open_list:
                        open_list.append(new_node)

        return rng.choice(legal)

@register_strategy("Uniform Cost Search")
class UCSStrategy(MoveStrategy):
    def choose_move(self, snapshot: GameSnapshot, snake_id: str, rng: Random) -> Direction:
        legal = snapshot.legal_moves_for(snake_id)
        snake = snapshot.snake(snake_id)
        if not legal or not snapshot.apples:
            return snake.direction

        head_x, head_y = snake.body[0]
        start = (head_x, head_y)
        obstacles = set(snake.body)
        
        open_list = [start]
        closed_set = set()
        parent_map = {}

        g_score = {start: 0} 

        while open_list:
            current = min(open_list, key=g_score.get)
            
            if current in snapshot.apples:
                path = []
                curr = current
                while curr in parent_map:
                    path.append(curr)
                    curr = parent_map[curr]
                path.reverse()
                
                if path:
                    next_x, next_y = path[0]
                    dx, dy = next_x - head_x, next_y - head_y
                    for move in legal:
                        if move.vector == (dx, dy):
                            return move
                            
            open_list.remove(current)
            closed_set.add(current)

            for d in Direction:
                dx, dy = d.vector
                new_x, new_y = current[0] + dx, current[1] + dy
                new_node = (new_x, new_y)
                
                if not (0 <= new_x < snapshot.columns and 0 <= new_y < snapshot.rows):
                    continue
                if new_node in obstacles or new_node in closed_set:
                    continue
                    
                new_g_score = g_score[current] + 1
                
                if new_node not in open_list or new_g_score < g_score[new_node]:
                    parent_map[new_node] = current
                    g_score[new_node] = new_g_score
                    
                    if new_node not in open_list:
                        open_list.append(new_node)
                        
        return rng.choice(legal)


@register_strategy("Greedy BFS")
class GBFSStrategy(MoveStrategy):
    def choose_move(self, snapshot: GameSnapshot, snake_id: str, rng: Random) -> Direction:
        legal = snapshot.legal_moves_for(snake_id)
        snake = snapshot.snake(snake_id)
        if not legal or not snapshot.apples:
            return snake.direction

        head_x, head_y = snake.body[0]
        start = (head_x, head_y)
        obstacles = set(snake.body)
        
        def heuristic(node: tuple[int, int]) -> int:
            node_x, node_y = node
            return min(abs(node_x - a_x) + abs(node_y - a_y) for a_x, a_y in snapshot.apples)

        open_list = [start]
        closed_set = set()
        parent_map = {}

        h_score = {start: heuristic(start)} 

        while open_list:
            current = min(open_list, key=h_score.get)
            
            if current in snapshot.apples:
                path = []
                curr = current
                while curr in parent_map:
                    path.append(curr)
                    curr = parent_map[curr]
                path.reverse()
                
                if path:
                    next_x, next_y = path[0]
                    dx, dy = next_x - head_x, next_y - head_y
                    for move in legal:
                        if move.vector == (dx, dy):
                            return move
                            
            open_list.remove(current)
            closed_set.add(current)

            for d in Direction:
                dx, dy = d.vector
                new_x, new_y = current[0] + dx, current[1] + dy
                new_node = (new_x, new_y)
                
                if not (0 <= new_x < snapshot.columns and 0 <= new_y < snapshot.rows):
                    continue
                if new_node in obstacles or new_node in closed_set:
                    continue
                
                if new_node not in open_list:
                    parent_map[new_node] = current
                    h_score[new_node] = heuristic(new_node)
                    open_list.append(new_node)
                        
        return rng.choice(legal)

@register_strategy("Iterative Deepening Search")
class IDSStrategy(MoveStrategy):
    def choose_move(self, snapshot: GameSnapshot, snake_id: str, rng: Random) -> Direction:
        legal = snapshot.legal_moves_for(snake_id)
        if not legal or not snapshot.apples:
            return snapshot.snake(snake_id).direction

        snake = snapshot.snake(snake_id)
        head_x, head_y = snake.body[0]
        body_set = set(snake.body)
        
        MAX_NODES = 4000
        nodes_visited = [0] 
        
        def dfs_explore(pos, current_depth_limit, visited):
            nodes_visited[0] += 1
            
            if nodes_visited[0] > MAX_NODES:
                return False
                
            if pos in snapshot.apples:
                return True
                
            if current_depth_limit <= 0:
                return False
                
            curr_x, curr_y = pos
            for d in Direction:
                dx, dy = d.vector
                next_pos = (curr_x + dx, curr_y + dy)
                
                if (0 <= next_pos[0] < snapshot.columns) and (0 <= next_pos[1] < snapshot.rows):
                    if next_pos not in body_set and next_pos not in visited:
                        visited.add(next_pos)
                        
                        if dfs_explore(next_pos, current_depth_limit - 1, visited):
                            return True
                            
                        visited.remove(next_pos)
                        
            return False

        MAX_VISION_RADIUS = 15 
        
        for bfs_radius in range(1, MAX_VISION_RADIUS + 1):
            for start_dir in Direction:
                dx, dy = start_dir.vector
                start_pos = (head_x + dx, head_y + dy)
                
                if (0 <= start_pos[0] < snapshot.columns) and (0 <= start_pos[1] < snapshot.rows):
                    if start_pos not in body_set:
                        initial_visited = { (head_x, head_y), start_pos }
                        
                        if dfs_explore(start_pos, bfs_radius - 1, initial_visited):
                            return start_dir
                            
        return rng.choice(legal)


@register_strategy("Breadth First Search")
class BFSStrategy(MoveStrategy):
    def choose_move(self, snapshot: GameSnapshot, snake_id: str, rng: Random) -> Direction:
        # validasi: ambil langkah legal
        snake = snapshot.snake(snake_id)
        legal = snapshot.legal_moves_for(snake_id)
        if not legal or not snapshot.apples:
            return snake.direction

        # inisialisasi titik awal, titik badan
        head_x, head_y = snake.body[0] # Posisi kepala ular (x, y) 
        body_set = set(snake.body) # Set koordinat badan sebagai rintangan
    

        # Queue FIFO dan set lokasi visited
        queue = deque()
        visited = set()
        visited.add((head_x, head_y))

        # jelajah langkah UDLR awal
        for move in Direction:
            dx, dy = move.vector
            new_pos = head_x + dx, head_y + dy
            if (0 <= new_pos[0] < snapshot.columns) and (0 <= new_pos[1] < snapshot.rows):
                if new_pos not in body_set:
                    queue.append((new_pos, move)) # simpan tuple 
                    visited.add(new_pos)
        # proses BFS (FIFO)
        while queue:
            curr_pos, first_dir = queue.popleft() 

            if curr_pos in snapshot.apples:
                return first_dir    
        
            # eksplore 4 arah
            curr_x, curr_y = curr_pos
            for move in Direction:
                dx, dy = move.vector
                next_pos = (curr_x + dx, curr_y + dy)
                # cek papan dan rintangan
                if (0 <= next_pos[0] < snapshot.columns) and (0 <= next_pos[1] < snapshot.rows):
                    if next_pos not in body_set and next_pos not in visited:
                        visited.add(next_pos)
                        queue.append((next_pos, first_dir))

        return rng.choice(legal)

@register_strategy("Depth First Search")
class DFSStrategy(MoveStrategy):
    def choose_move(self, snapshot: GameSnapshot, snake_id: str, rng: Random) -> Direction:
        legal = snapshot.legal_moves_for(snake_id)
        snake = snapshot.snake(snake_id)
        if not legal or not snapshot.apples:
            return snake.direction

        head_x, head_y = snake.body[0]
        start = (head_x, head_y)
        obstacles = set(snake.body)
        
        open_list = [start]
        closed_set = set()
        parent_map = {}

        while open_list:
            current = open_list.pop()
            
            if current in snapshot.apples:
                path = []
                curr = current
                while curr in parent_map:
                    path.append(curr)
                    curr = parent_map[curr]
                path.reverse()
                
                if path:
                    next_x, next_y = path[0]
                    dx, dy = next_x - head_x, next_y - head_y
                    for move in legal:
                        if move.vector == (dx, dy):
                            return move
                            
            closed_set.add(current)

            for d in Direction:
                dx, dy = d.vector
                new_x, new_y = current[0] + dx, current[1] + dy
                new_node = (new_x, new_y)
                
                if not (0 <= new_x < snapshot.columns and 0 <= new_y < snapshot.rows):
                    continue
                if new_node in obstacles or new_node in closed_set:
                    continue
                
                if new_node not in open_list:
                    parent_map[new_node] = current
                    open_list.append(new_node)
                    
        return rng.choice(legal)
