from __future__ import annotations
from dataclasses import dataclass, field, InitVar
from functools import partial
import itertools
from typing import Any, Iterator, Optional, Literal
import random
from matplotlib import pyplot as plt
from enum import Enum
import numpy as np
import math
import random
from utils.spaces import FiniteSpace, TensorSpace, CompositeSpace


cell_states = Enum(
    "cell", {"AGENT": -1, "FREE": 0, "WALL": 1, "DOOR": 2, "FRUIT": 3, "TRACE": 4}
)


@dataclass(unsafe_hash=True)
class Position:
    room_idx: int | None = None
    rel_y: int = 0
    rel_x: int = 0
    room_shape: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, room_shape: tuple[int, int] | None) -> None:
        if room_shape is None and (self.rel_y < 0 or self.rel_x < 0):
            raise ValueError("Invalid coordinates: unknown room shape")
        if room_shape is not None:
            self.rel_y = self.rel_y % room_shape[0]
            self.rel_x = self.rel_x % room_shape[1]

    def __iter__(self) -> Iterator:
        return iter((self.room_idx, self.rel_y, self.rel_x))

    def one_hot(self, n_rooms: int | tuple[int, ...]) -> np.ndarray:
        one_hot_pos = np.zeros(n_rooms)
        one_hot_pos[self.room_idx] = 1
        return one_hot_pos


@dataclass
class Room:
    idx: int
    shape: tuple[int, int]
    board: np.ndarray = field(init=False)
    passages: dict[Position, Position] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.board = np.ones(self.shape) * cell_states.WALL.value
        self.board[1:-1, 1:-1] = (
            np.ones(tuple(s - 2 for s in self.shape)) * cell_states.FREE.value
        )

    def matrix_direction_to(
        self, other_room: Room, grid_shape: tuple[int, int]
    ) -> tuple[int, int]:
        y1, x1 = np.unravel_index(self.idx, grid_shape)
        y2, x2 = np.unravel_index(other_room.idx, grid_shape)
        return int(y2 - y1), int(x2 - x1)

    def add_passage(self, start: Position, dest: Position) -> None:
        if start.room_idx != self.idx and dest.room_idx != self.idx:
            raise ValueError("Invalid passage: passage is not in the current room")
        if start.room_idx == self.idx:
            self.board[start.rel_y, start.rel_x] = cell_states.DOOR.value
            self.passages[start] = dest
        if dest.room_idx == self.idx:
            self.board[dest.rel_y, dest.rel_x] = cell_states.DOOR.value

    def get_position_from(
        self, orthogonal_pos: int, paralel_pos: int, orthogonal_axis: int
    ) -> Position:
        if orthogonal_axis == 0:
            return Position(self.idx, orthogonal_pos, paralel_pos, self.shape)
        elif orthogonal_axis == 1:
            return Position(self.idx, paralel_pos, orthogonal_pos, self.shape)
        else:
            raise ValueError("Invalid axis: must be 0 or 1")

    def move_to(self, target: Position) -> Position:
        if target.room_idx != self.idx:
            raise ValueError("Invalid move: room is not the current room")
        if self.board[target.rel_y, target.rel_x] == cell_states.WALL.value:
            raise ValueError("Invalid move: cannot move to a wall")
        if target in self.passages:
            return self.passages[target]
        return target


@dataclass(eq=False)
class NRoomsZanin:
    grid_shape: tuple[int, int] = (2, 3)
    room_shapes: InitVar[tuple[int, int] | list[tuple[int, int]]] = (8, 8)

    start_agent_pos: tuple[int | None, int | None, int | None] = None, None, None
    start_fruits_pos: list[tuple[int | None, int | None, int | None]] = field(
        default_factory=lambda: [(None, None, None)]
    )

    n_doors: int = 1
    door_position_distr: Literal["UNIFORM"] | Literal["RANDOM"] = "UNIFORM"

    wall_hit_penalty: float = -0.1

    action_space: FiniteSpace = field(
        init=False, default_factory=lambda: FiniteSpace(["L", "R", "U", "D"])
    )

    def __post_init__(self, room_shapes: tuple[int, int] | list[tuple[int, int]]):
        if not isinstance(room_shapes, list):
            room_shapes = [room_shapes]
        if len(room_shapes) == 1:
            room_shapes = room_shapes * math.prod(self.grid_shape)
        self.rooms = [Room(idx, shape) for idx, shape in enumerate(room_shapes)]
        self.reset()

    def random_position(
        self,
        room_number: Optional[int] = None,
        rel_x: Optional[int] = None,
        rel_y: Optional[int] = None,
    ) -> Position:
        if room_number is None:
            room_number = random.choice(range(len(self.rooms)))
        if rel_y is None:
            rel_y = random.randint(1, self.rooms[room_number].shape[0] - 2)
        if rel_x is None:
            rel_x = random.randint(1, self.rooms[room_number].shape[1] - 2)
        return Position(room_number, rel_y, rel_x)

    def reset_trace(self) -> None:
        self.trace = [self.agent_pos]

    def reset(self, how: str = "base") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.rooms = [Room(idx, room.shape) for idx, room in enumerate(self.rooms)]
        self.create_passages()
        if how == "random":
            self.fruits_pos = [
                self.random_position() for _ in range(len(self.start_fruits_pos))
            ]
        elif how == "base":
            self.fruits_pos = [
                self.random_position(*start_pos) for start_pos in self.start_fruits_pos
            ]
        self.agent_pos = self.random_position(*self.start_agent_pos)
        self.reset_trace()
        return self.current_state, {}

    def create_passages(self) -> None:
        for room1, room2 in itertools.combinations(self.rooms, 2):
            direction = room1.matrix_direction_to(room2, self.grid_shape)
            if sum(abs(d) for d in direction) != 1:
                continue

            if sum(d for d in direction) == -1:
                room1, room2 = room2, room1

            ortogonal_axis = 0 if direction[1] == 1 else 1
            if self.door_position_distr == "RANDOM":
                axis_ortogonal_positions = random.sample(
                    range(1, room1.shape[ortogonal_axis] - 1), self.n_doors
                )
            elif self.door_position_distr == "UNIFORM":
                axis_ortogonal_positions = range(
                    (room1.shape[ortogonal_axis] - 1) // (self.n_doors + 1),
                    room1.shape[ortogonal_axis] - 1,
                    (room1.shape[ortogonal_axis] - 1) // (self.n_doors + 1) + 1,
                )
            else:
                raise ValueError(f"Invalid strategy: {self.door_position_distr}")

            for ort_pos in axis_ortogonal_positions:
                start_pos = room1.get_position_from(ort_pos, -1, ortogonal_axis)
                dest_pos = room2.get_position_from(ort_pos, 1, ortogonal_axis)
                room1.add_passage(start_pos, dest_pos)

                start_pos = room2.get_position_from(ort_pos, 0, ortogonal_axis)
                dest_pos = room1.get_position_from(ort_pos, -2, ortogonal_axis)
                room2.add_passage(start_pos, dest_pos)

    @property
    def observation_space(self) -> CompositeSpace:
        rooms = CompositeSpace(
            {idx: TensorSpace(room.shape) for idx, room in enumerate(self.rooms)}
        )
        fruit_pos = TensorSpace(shape=(len(self.rooms),))
        agent_pos = FiniteSpace(len(self.rooms))
        return CompositeSpace(
            {"rooms": rooms, "agent_room": agent_pos, "fruit_room": fruit_pos}
        )

    @property
    def current_state(self) -> dict[str, Any]:
        rooms = {idx: np.array(room.board) for idx, room in enumerate(self.rooms)}

        for room_idx, rel_y, rel_x in self.fruits_pos:
            if room_idx is not None:
                rooms[room_idx][rel_y, rel_x] = cell_states.FRUIT.value

        for room_idx, rel_y, rel_x in [self.agent_pos]:
            if room_idx is not None:
                rooms[room_idx][rel_y, rel_x] = cell_states.AGENT.value

        fruit_pos = np.zeros(len(self.rooms))
        for pos in self.fruits_pos:
            if pos.room_idx is not None:
                fruit_pos[pos.room_idx] += 1
        agent_pos = self.agent_pos.room_idx
        return {"rooms": rooms, "agent_room": agent_pos, "fruit_room": fruit_pos}

    def move_agent(self, action: Literal["L", "R", "U", "D"]) -> float:
        if action not in ["L", "R", "U", "D"]:
            raise ValueError(f"Invalid action {action}")

        (target_y, target_x) = {
            "L": (self.agent_pos.rel_y, self.agent_pos.rel_x - 1),
            "R": (self.agent_pos.rel_y, self.agent_pos.rel_x + 1),
            "U": (self.agent_pos.rel_y + 1, self.agent_pos.rel_x),
            "D": (self.agent_pos.rel_y - 1, self.agent_pos.rel_x),
        }[action]

        current_room = self.rooms[self.agent_pos.room_idx]
        target_position = Position(
            self.agent_pos.room_idx, target_y, target_x, current_room.shape
        )
        try:
            self.agent_pos = current_room.move_to(target_position)
        except ValueError:
            return self.wall_hit_penalty

        self.trace.append(self.agent_pos)
        return 0

    def step(
        self, action: Literal["L", "R", "U", "D"]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:

        penalty = self.move_agent(action)
        reward = 1.0 if self.agent_pos == self.fruits_pos else 0.0

        for idx, pos in enumerate(self.fruits_pos):
            if pos.room_idx is None:
                self.fruits_pos[idx] = self.random_position(*self.start_fruits_pos[idx])
            elif self.agent_pos == pos:
                self.reset_trace()
                self.fruits_pos[idx] = Position(None)
        return self.current_state, reward + penalty, False, False, {}

    def position2boardCoord(self, position: Position) -> None | tuple[int, int]:
        if position.room_idx is None:
            return None
        room_y, room_x = np.unravel_index(position.room_idx, self.grid_shape)
        abs_y = sum(room.shape[0] for room in self.rooms[:room_y]) + position.rel_y
        abs_x = sum(room.shape[1] for room in self.rooms[:room_x]) + position.rel_x
        return abs_y, abs_x

    def empty_board(self) -> np.ndarray:
        room_grid = np.array(self.rooms, dtype=object).reshape(self.grid_shape)
        board_grid = [[np.array(room.board) for room in row] for row in room_grid]
        tensor_board = np.block(board_grid)
        return tensor_board

    def render(self) -> None:
        tensor_board = self.empty_board()
        plt.figure(figsize=(tensor_board.shape[1] // 5, tensor_board.shape[0] // 5))
        plt.imshow(tensor_board, cmap="gray_r", origin="lower")

        for trace_pos in self.trace:
            if (abs_trace_pos := self.position2boardCoord(trace_pos)) is not None:
                plt.scatter(*abs_trace_pos[::-1], marker=".", c="blue")  # type: ignore
        if (abs_agent_pos := self.position2boardCoord(self.agent_pos)) is not None:
            plt.scatter(*abs_agent_pos[::-1], marker="o", c="green")  # type: ignore
        for pos in self.fruits_pos:
            if (abs_fruit_pos := self.position2boardCoord(pos)) is not None:
                plt.scatter(*abs_fruit_pos[::-1], marker="*", c="red")  # type: ignore

    def play(self):
        import pygame

        pygame.init()
        board = self.empty_board()
        scale = 50
        screen = pygame.display.set_mode(tuple(s * scale for s in board.shape[::-1]))
        clock = pygame.time.Clock()

        def draw_rect(y, x, rgb):
            verteces = (j * scale, i * scale, scale, scale)
            pygame.draw.rect(screen, pygame.Color(*rgb), verteces)

        def draw_circle(y, x, rgb):
            center = (x * scale + scale // 2, y * scale + scale // 2)
            pygame.draw.circle(screen, rgb, center, scale // 2)

        def draw_dot(y, x, rgb):
            center = (x * scale + scale // 2, y * scale + scale // 2)
            pygame.draw.circle(screen, rgb, center, scale // 10)

        draw_sprite = {
            cell_states.FREE.value: partial(draw_rect, rgb=(20, 20, 20)),
            cell_states.WALL.value: partial(draw_rect, rgb=(50, 50, 50)),
            cell_states.DOOR.value: partial(draw_rect, rgb=(0, 100, 100)),
            cell_states.FRUIT.value: partial(draw_circle, rgb=(255, 0, 0)),
            cell_states.AGENT.value: partial(draw_circle, rgb=(0, 255, 0)),
            cell_states.TRACE.value: partial(draw_dot, rgb=(0, 0, 255)),
        }

        running = True
        while running == True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill("black")
            board = self.empty_board()
            for i, row in enumerate(board):
                for j, cell in enumerate(row):
                    draw_sprite[cell](i, j)
            for trace_pos in self.trace:
                if (abs_trace_pos := self.position2boardCoord(trace_pos)) is not None:
                    draw_sprite[cell_states.TRACE.value](*abs_trace_pos)
            for pos in self.fruits_pos:
                if (fruit_pos := self.position2boardCoord(pos)) is not None:
                    draw_sprite[cell_states.FRUIT.value](*fruit_pos)
            if (agent_pos := self.position2boardCoord(self.agent_pos)) is not None:
                draw_sprite[cell_states.AGENT.value](*agent_pos)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.step("U")
            elif keys[pygame.K_s]:
                self.step("D")
            elif keys[pygame.K_a]:
                self.step("L")
            elif keys[pygame.K_d]:
                self.step("R")

            pygame.display.flip()

            dt = clock.tick(10)  # limits FPS

        pygame.quit()
