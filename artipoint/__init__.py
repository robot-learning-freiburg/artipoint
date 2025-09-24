"""ArtiPoint: Articulated Object Tracking and Motion Estimation from RGB-D Data.

This package provides tools for tracking articulated objects from RGB-D data,
including hand segmentation, query point tracking, 3D projection, and motion
estimation using factor graphs.
"""

__version__ = "0.1.0"
__author__ = "ArtiPoint Team"

from .track.artipoint import ArtiPoint

__all__ = [
    "ArtiPoint",
]
