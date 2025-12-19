"""
Version information for duckreg.

This module provides version tracking for reproducibility of regression results.
When bugs are discovered, results computed with affected versions can be identified
and recomputed.

Version history:
- 0.2.0: Initial versioning system
"""

__version__ = "0.2.0"

# Git commit hash (populated during build/install if available)
__git_commit__ = None


def get_version() -> str:
    """Get the current version string."""
    return __version__


def get_version_info() -> dict:
    """Get comprehensive version information for result tracking.
    
    Returns:
        Dictionary containing:
        - version: The semantic version string
        - git_commit: Git commit hash if available
    """
    return {
        "version": __version__,
        "git_commit": __git_commit__,
    }
