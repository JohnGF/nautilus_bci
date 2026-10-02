"""Canonical repository path resolution for the Nautilus BCI toolkit.

Layout (post-reorg):
    <repo>/               repo root (this file lives in <repo>/scripts/utils/)
    <repo>/scripts/       Python project root (pyproject.toml, .venv)
    <repo>/assets/        static runtime assets (music_tracks, sounds, videos)
    <repo>/results/       generated analysis outputs (figures, JSON reports)
    <repo>/docs/          guides, science notes, vendor manuals
    <repo>/games/          interactive/game tasks (incl. submodules)
    <repo>/integrations/  bridges to external systems

Import this instead of hardcoding ``../music_tracks`` style relatives so the
code keeps working no matter where the repo is cloned or which cwd it is
launched from. Stdlib only — safe to import from anywhere.
"""

import os

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))

ASSETS_DIR = os.path.join(REPO_ROOT, "assets")
ASSETS_MUSIC = os.path.join(ASSETS_DIR, "music_tracks")
ASSETS_SOUNDS = os.path.join(ASSETS_DIR, "sounds")
ASSETS_VIDEOS = os.path.join(ASSETS_DIR, "videos")
ASSETS_VIDEO_AUDIO = os.path.join(ASSETS_VIDEOS, "audio")

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
GAMES_DIR = os.path.join(REPO_ROOT, "games")
INTEGRATIONS_DIR = os.path.join(REPO_ROOT, "integrations")


def results_subdir(*parts):
    """Return <repo>/results/<parts...> creating it if needed."""
    path = os.path.join(RESULTS_DIR, *parts)
    os.makedirs(path, exist_ok=True)
    return path
