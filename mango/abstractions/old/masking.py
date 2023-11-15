from typing import Any, Mapping, Protocol, Sequence, TypeVar
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

ObsType = TypeVar("ObsType")



def add_frame_3D(matrix, n_1, n_2):
    # Get the original matrix dimensions
    original_rows, original_cols, nchannels = matrix.shape

    # Create a larger matrix with the frame
    framed_matrix = np.zeros((original_rows + 2 * n_1, original_cols + 2 * n_2, nchannels))

    # Place the original matrix inside the frame for each channel
    framed_matrix[n_1:n_1 + original_rows, n_2:n_2 + original_cols, :] = matrix

    return framed_matrix

class Window(Protocol[ObsType]):
    def abstract(self, observation: ObsType, concept:ObsType) -> npt.NDArray:
        ...
        



@dataclass(frozen=True, eq=False)
class CounterCondensationWindow(Window[int]):
    global_shape: tuple[int, int, int]
    condensation_window: tuple[int, int] = (2, 2)

    def abstract(self, observation: np.ndarray, concept: np.ndarray) -> np.ndarray:
        # Add frames around the matrices
        observation = observation.reshape(self.global_shape)
        observation_framed = self._add_frame(observation, *self.condensation_window)
        concept_framed = self._add_frame(concept, 1, 1)
        
        # Obtain the mask based on the decondensed concept
        mask = self._decondensing_concept(concept_framed)
        observation_framed = observation_framed*mask
        # Extract and return the relevant slices from the framed observation
        return observation_framed[self._nonzero_slices(mask)].flatten()

    def _add_frame(self, matrix: np.ndarray, n_1: int, n_2: int) -> np.ndarray:
        """Add a frame of n_1 elements vertically and n_2 elements horizontally around the matrix."""
        return add_frame_3D(matrix, n_1, n_2)

    def _decondensing_concept(self, concept: np.ndarray) -> np.ndarray:
        """Generate a mask based on the decondensed concept."""
        x, y = np.unravel_index(np.argmax(concept[:, :, -1]), concept.shape[:2])
        base = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        x_init, x_end = max(x - 1, 0), min(x + 2, concept.shape[0])
        y_init, y_end = max(y - 1, 0), min(y + 2, concept.shape[1])
        
        # Adjust the shape of the base matrix based on the position within the concept
        base = self._adjust_base_shape(base, x_init, x_end, y_init, y_end, concept.shape)
        
        # Create a new concept with the adjusted base shape
        new_concept = np.zeros_like(concept)
        new_concept[x_init:x_end, y_init:y_end, -1] = base
        
        # Repeat the decondensed concept to match the condensation window
        mask = new_concept.repeat(self.condensation_window[0], axis=0).repeat(self.condensation_window[1], axis=1)
        return np.tile(mask[:,:,-1][:,:,np.newaxis],mask.shape[2])

    def _adjust_base_shape(self, base: np.ndarray, x_init: int, x_end: int, y_init: int, y_end: int, shape: tuple[int, int]) -> np.ndarray:
        """Adjust the shape of the base matrix based on its position within the concept."""
        if x_init == 0 and x_end == 2:
            base = base[1:, :]
        elif x_init == shape[0] - 2 and x_end == shape[0]:
            base = base[:-1, :]
        if y_init == 0 and y_end == 2:
            base = base[:, 1:]
        elif y_init == shape[1] - 2 and y_end == shape[1]:
            base = base[:, :-1]
        return base

    def _nonzero_slices(self, mask: np.ndarray) -> tuple[slice, slice, slice]:
        """Obtain slices corresponding to the non-zero values in the mask."""
        return (
            slice(np.max(mask[:, :, 0], axis=1).nonzero()[0].min(), np.max(mask[:, :, 0], axis=1).nonzero()[0].max() + 1),
            slice(np.max(mask[:, :, 0], axis=0).nonzero()[0].min(), np.max(mask[:, :, 0], axis=0).nonzero()[0].max() + 1),
            slice(None)
        )