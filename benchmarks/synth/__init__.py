"""Synthetic edge-probing suite for dots.mocr.

Generates LaTeX documents (tables, formulas, algorithms, code listings) with
known ground truth, compiles them to PDF, renders pages to images, runs the
model, and scores the output to locate where the model's accuracy degrades.
"""

from .docs import SynthCase  # noqa: F401
