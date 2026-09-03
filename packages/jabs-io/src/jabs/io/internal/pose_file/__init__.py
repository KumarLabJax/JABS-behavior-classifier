"""The JABS pose file format, revision 1 (ADR 0002).

A single self-describing HDF5 file per video, whose contents are declared in a
JSON manifest rather than implied by a version number. The public surface is
assembled as the reader, writer and validator land; this module currently
exposes only the schema constants.
"""
