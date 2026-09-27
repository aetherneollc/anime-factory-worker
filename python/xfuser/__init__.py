"""Minimal stand-in so SkyReels can import on a single GPU.

The official package imports talking-avatar modules that require xFuser (USP).
Reference-to-video on one card does not enable ``--use_usp``, so those symbols
are never called. This package only satisfies the import.
"""
