The `python/mkfig_*.py` scripts write their PDF figures to this folder.

`rig_render.png`, `archer_render.png` and the matching `*_anchors.json` files are written by
`python/render_rig_blender.py` and `python/render_archer_blender.py` (Blender's Python module) and
read by `python/mkfig_rig.py` and `python/mkfig_target.py`; they are kept here so the labelled
figures can be redrawn without Blender.
