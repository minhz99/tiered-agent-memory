"""TAM Control Planes — 7 lớp điều phối ngang."""
from tam.planes.query_plane import QueryPlane
from tam.planes.competition import CompetitionPlane
from tam.planes.abstraction import AbstractionPlane
from tam.planes.reasoning import ReasoningPlane
from tam.planes.state_plane import StatePlane
from tam.planes.scaling import ScalingPlane
from tam.planes.evolution import EvolutionPlane

__all__ = [
    "QueryPlane",
    "CompetitionPlane",
    "AbstractionPlane",
    "ReasoningPlane",
    "StatePlane",
    "ScalingPlane",
    "EvolutionPlane",
]
