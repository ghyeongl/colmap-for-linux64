#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
script/__init__.py

이 패키지에서 ColmapPipeline을 import & re-export한다.
이렇게 하면, `from script import ColmapPipeline` 로 가져올 수 있다.
"""

from .colmap_pipeline import ColmapPipeline

__all__ = ["ColmapPipeline"]
