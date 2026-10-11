"""render_archer_blender.py -- clay-mannequin archer for Fig. 1.5 (Section 1.2),
rendered with Blender's Python module (pip install bpy; tested bpy 5.2.2).
Not a listing.

The body is a metaball figure: each limb is a chain of spheres whose radius
tapers from joint to joint, and the trunk is a set of ellipsoids, so the
surfaces blend into one smooth mannequin. Stance: a right-handed archer side-on
to the target (+x), chest toward the camera, feet apart along the line of
shot; the left (bow) arm level toward the target, the right (draw) hand
anchored under the chin with the elbow at shoulder height behind.

Orthographic camera along +y, so image coordinates are a linear map of world
(x, z). Writes ../figures/archer_render.png (RGBA) and
../figures/archer_anchors.json (the map, the bow grip and the nock point),
which mkfig_target.py uses to place the vector bow, arrow and trajectories.

Usage:  python3 render_archer_blender.py [samples] [width_px]
"""
import sys, os, json, math
import bpy
from mathutils import Vector

SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 128
WIDTH = int(sys.argv[2]) if len(sys.argv) > 2 else 900
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# ---------------------------------------------------------------- skeleton
# Joints in metres: x toward the target, y toward the archer's back, z up.
# Each joint carries a skin radius (rx, ry); the Skin modifier wraps the
# skeleton in a mesh and a subdivision surface smooths it into one body.
NOCK = Vector((0.075, -0.02, 1.555))      # draw hand / nock point under the chin
GRIP = Vector((0.80, 0.0, 1.505))          # bow hand

J, R, E = {}, {}, []


GROW = 1.32                    # subdivision shrinks a skin by ~25 %; compensate


def joint(name, p, rx, ry=None):
    ry = ry if ry is not None else rx
    J[name] = Vector(p); R[name] = (rx * GROW, ry * GROW)


def bone(a, b):
    E.append((a, b))


joint("pelvis", (0.0, 0.0, 0.95), 0.150, 0.105)
joint("waist", (0.0, 0.0, 1.10), 0.125, 0.090)
joint("chest", (0.0, 0.0, 1.30), 0.165, 0.110)
joint("neckb", (0.005, 0.0, 1.47), 0.055, 0.052)
joint("headb", (0.02, 0.0, 1.575), 0.060, 0.062)
joint("headm", (0.035, 0.0, 1.655), 0.092, 0.090)
joint("headt", (0.035, 0.0, 1.725), 0.062, 0.062)
joint("chin", (0.095, -0.005, 1.600), 0.030, 0.035)
for side, sx in (("L", 1), ("R", -1)):
    joint("hip" + side, (0.095 * sx, 0.0, 0.92), 0.085)
    joint("thigh" + side, (0.125 * sx, -0.005, 0.70), 0.070)
    joint("knee" + side, (0.155 * sx, -0.01, 0.50), 0.052)
    joint("calf" + side, (0.180 * sx, 0.012, 0.33), 0.050)
    joint("ankle" + side, (0.205 * sx, 0.0, 0.09), 0.036)
    joint("heel" + side, (0.205 * sx, 0.03, 0.045), 0.035)
    joint("toe" + side, (0.205 * sx, -0.15, 0.035), 0.030, 0.035)
    joint("sh" + side, (0.200 * sx, 0.0, 1.43), 0.058)
    for a, b in (("pelvis", "hip"), ("hip", "thigh"), ("thigh", "knee"), ("knee", "calf"),
                 ("calf", "ankle"), ("ankle", "heel"), ("heel", "toe"), ("chest", "sh")):
        bone(a if a in ("pelvis", "chest") else a + side, b + side)
joint("elL", (0.50, 0.0, 1.47), 0.040)
joint("wrL", (0.74, 0.0, 1.495), 0.030)
joint("fistL", tuple(GRIP), 0.045, 0.040)
joint("elR", (-0.43, 0.03, 1.53), 0.040)
joint("wrR", (0.02, -0.015, 1.55), 0.031)
joint("handR", tuple(NOCK), 0.042, 0.036)
for a, b in (("pelvis", "waist"), ("waist", "chest"), ("chest", "neckb"), ("neckb", "headb"),
             ("headb", "headm"), ("headm", "headt"), ("headb", "chin"),
             ("shL", "elL"), ("elL", "wrL"), ("wrL", "fistL"),
             ("shR", "elR"), ("elR", "wrR"), ("wrR", "handR")):
    bone(a, b)

names = list(J)
me = bpy.data.meshes.new("body")
me.from_pydata([tuple(J[n]) for n in names], [(names.index(a), names.index(b)) for a, b in E], [])
body = bpy.data.objects.new("body", me); scene.collection.objects.link(body)
sk = body.modifiers.new("skin", "SKIN"); sk.use_smooth_shade = True; sk.branch_smoothing = 0.6
for i_, n in enumerate(names):
    v = me.skin_vertices[0].data[i_]
    v.radius = R[n]
    v.use_root = (n == "pelvis")
sub = body.modifiers.new("sub", "SUBSURF"); sub.levels = 2; sub.render_levels = 3
mb = me

# ---------------------------------------------------------------- material
clay = bpy.data.materials.new("clay")
clay.use_nodes = True
b = clay.node_tree.nodes["Principled BSDF"]
b.inputs["Base Color"].default_value = (0.30, 0.31, 0.33, 1)
b.inputs["Roughness"].default_value = 0.55
b.inputs["Coat Weight"].default_value = 0.15
mb.materials.append(clay)

bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
floor = bpy.context.object; floor.is_shadow_catcher = True

# ---------------------------------------------------------------- camera
XC, ZC, OS = 0.25, 0.93, 2.10              # view centre and ortho width (m)
cam_data = bpy.data.cameras.new("cam"); cam_data.type = "ORTHO"
cam_data.ortho_scale = OS
cam = bpy.data.objects.new("cam", cam_data); scene.collection.objects.link(cam)
cam.location = (XC, -8.0, ZC)
cam.rotation_euler = (math.pi / 2, 0, 0)    # look along +y
scene.camera = cam
H = int(WIDTH * 1.0)
scene.render.resolution_x, scene.render.resolution_y = WIDTH, H


def area(name, loc, size, energy):
    ld = bpy.data.lights.new(name, "AREA"); ld.size = size; ld.energy = energy
    lo = bpy.data.objects.new(name, ld); scene.collection.objects.link(lo)
    lo.location = loc
    lo.rotation_euler = (Vector((0, 0, 1.1)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()


area("key", (-2.2, -3.0, 3.2), 2.0, 420)
area("fill", (3.0, -2.5, 1.5), 3.0, 50)
area("rim", (0.5, 3.0, 2.5), 2.0, 160)
world = bpy.data.worlds.new("w"); scene.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.22

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "AgX"
scene.view_settings.exposure = -0.6
scene.render.filepath = os.path.join(OUT, "archer_render.png")
bpy.ops.render.render(write_still=True)

# world (x, z) of the image corners: ortho_scale spans the wider image side
half_w = OS / 2; half_h = OS / 2 * H / WIDTH
json.dump({"extent": [XC - half_w, XC + half_w, ZC - half_h, ZC + half_h],
           "grip": [GRIP.x, GRIP.z], "nock": [NOCK.x, NOCK.z]},
          open(os.path.join(OUT, "archer_anchors.json"), "w"), indent=1)
print("wrote archer_render.png and archer_anchors.json")
