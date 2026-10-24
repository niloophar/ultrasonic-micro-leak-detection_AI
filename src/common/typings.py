"""
Type definitions for preprocessing pipeline.

Contains dataclasses for:
- Block: A block of aggregated measurements
- Relaxation: A thermal relaxation segment
- EquilibriumSegment: A thermally stable segment
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Block:
    """
    A block of aggregated measurements.

    Represents 64 consecutive valid measurements aggregated into a single block.

    Attributes:
        block_id: Unique block identifier within the file.
        block_mean: Mean value of measurements in this block.
        block_std: Standard deviation within this block.
        block_min: Minimum value in this block.
        block_max: Maximum value in this block.
        block_time: Center timestamp of this block.
        start_idx: Start index in the original DataFrame.
        end_idx: End index in the original DataFrame.
        n_points: Number of points in this block (typically 64).
        relaxation_id: ID of the relaxation this block belongs to (-1 if none).
    """
    block_id: int
    block_mean: float
    block_std: float
    block_min: float
    block_max: float
    block_time: datetime
    start_idx: int
    end_idx: int
    n_points: int = 64
    relaxation_id: int = -1


@dataclass
class Relaxation:
    """
    A thermal relaxation segment.

    Represents a continuous segment of measurements between jumps.

    Attributes:
        relaxation_id: Unique relaxation identifier within the file.
        start_block: Start block index in the file.
        end_block: End block index in the file.
        n_blocks: Total number of blocks in this relaxation.
        n_points: Total number of measurement points.
        start_time: Start timestamp.
        end_time: End timestamp.
        duration_seconds: Duration in seconds.
        mean_value: Mean block value across all blocks.
        std_value: Standard deviation of block values.
        slope: Overall slope (units/second).
        min_value: Minimum block value.
        max_value: Maximum block value.
        file_name: Source file name (optional).
        flow_type: Flow type ('flow', 'noflow', or 'undefined'; optional).
    """
    relaxation_id: int
    start_block: int
    end_block: int
    n_blocks: int
    n_points: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    mean_value: float
    std_value: float
    slope: float
    min_value: float
    max_value: float
    file_name: Optional[str] = None
    flow_type: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'Relaxation':
        """Create a Relaxation from a metadata dictionary."""
        return cls(
            relaxation_id=d['relaxation_id'],
            start_block=d['start_block'],
            end_block=d['end_block'],
            n_blocks=d['n_blocks'],
            n_points=d['n_points'],
            start_time=d['start_time'],
            end_time=d['end_time'],
            duration_seconds=d['duration_seconds'],
            mean_value=d['mean_value'],
            std_value=d['std_value'],
            slope=d['slope'],
            min_value=d['min_value'],
            max_value=d['max_value'],
            file_name=d.get('file_name'),
            flow_type=d.get('flow_type'),
        )

    def to_dict(self) -> dict:
        """Convert Relaxation to a dictionary."""
        return {
            'relaxation_id': self.relaxation_id,
            'start_block': self.start_block,
            'end_block': self.end_block,
            'n_blocks': self.n_blocks,
            'n_points': self.n_points,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.duration_seconds,
            'mean_value': self.mean_value,
            'std_value': self.std_value,
            'slope': self.slope,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'file_name': self.file_name,
            'flow_type': self.flow_type,
        }


@dataclass
class EquilibriumSegment:
    """
    A thermally stable (equilibrium) segment.

    Represents a contiguous region where the slope is approximately zero,
    indicating thermal equilibrium. Used for training leak-detection models.

    Attributes:
        segment_id: Unique segment identifier within the file.
        start_block: Start block index.
        end_block: End block index (exclusive).
        n_blocks: Number of blocks in this segment.
        start_time: Start timestamp.
        end_time: End timestamp.
        duration_seconds: Duration in seconds.
        mean_value: Mean block value across all blocks.
        std_value: Standard deviation of block values.
        mean_slope: Mean slope across windows (ps/s).
        source: 'leakage' or 'non_leakage'.
        relaxation_id: ID of the source relaxation (for non-leakage segments).
        file_name: Source file name.
    """
    segment_id: int
    start_block: int
    end_block: int
    n_blocks: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    mean_value: float
    std_value: float
    mean_slope: float
    source: str
    relaxation_id: Optional[int] = None
    file_name: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'EquilibriumSegment':
        """Create an EquilibriumSegment from a metadata dictionary."""
        return cls(
            segment_id=d['segment_id'],
            start_block=d['start_block'],
            end_block=d['end_block'],
            n_blocks=d['n_blocks'],
            start_time=d['start_time'],
            end_time=d['end_time'],
            duration_seconds=d['duration_seconds'],
            mean_value=d['mean_value'],
            std_value=d['std_value'],
            mean_slope=d['mean_slope'],
            source=d['source'],
            relaxation_id=d.get('relaxation_id'),
            file_name=d.get('file_name'),
        )

    def to_dict(self) -> dict:
        """Convert EquilibriumSegment to a dictionary."""
        return {
            'segment_id': self.segment_id,
            'start_block': self.start_block,
            'end_block': self.end_block,
            'n_blocks': self.n_blocks,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.duration_seconds,
            'mean_value': self.mean_value,
            'std_value': self.std_value,
            'mean_slope': self.mean_slope,
            'source': self.source,
            'relaxation_id': self.relaxation_id,
            'file_name': self.file_name,
        }
