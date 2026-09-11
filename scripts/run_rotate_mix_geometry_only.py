#!/usr/bin/env python3
"""Geometry-only variant of the paired rotate-mix experiment."""
import os
from pathlib import Path
import runpy

os.environ['BREPPREDIFF_GEOMETRY_ONLY'] = '1'
runpy.run_path(str(Path(__file__).with_name('run_rotate_mix_discrete_off.py')), run_name='__main__')
