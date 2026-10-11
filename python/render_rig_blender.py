"""render_rig_blender.py -- photoreal render of the two-magnet fatigue rig of
Section 17.4 (Fig. 'expt'), modelled after D. Chelidze's sketch. Not a listing.

Needs Blender's Python module (pip install bpy; tested with bpy 5.2.2, Python
3.13). Writes ../figures/rig_render.png (RGBA) and ../figures/rig_anchors.json
(normalized image coordinates of the label anchor points), which
mkfig_rig.py turns into the labelled vector-framed PDF.

Model (schematic proportions, not to scale): an inverted-L aluminium frame on
a base plate that is shaken along x (the bending direction); a steel
cantilever clamped under the top arm, hanging down, with two stiffener
strips bolted along it and a V-notch between the stiffeners and the clamp;
two electromagnets (copper coils on steel cores) on the base under the free
end, wired to a bench supply; two laser vibrometer heads (CH1, CH2) aimed at
the beam near the clamp. Axes: x bending/drive, y beam width, z up.

Usage:  python3 render_rig_blender.py [samples] [width_px]
"""
import sys, os, json, math
import bpy, bmesh
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 160
WIDTH = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


# ------------------------------------------------------------- materials
def mat(name, base, metallic=0.0, rough=0.5, aniso=0.0, emission=None, estr=0.0,
        coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base, 1)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    if aniso:
        b.inputs["Anisotropic"].default_value = aniso
    if coat:
        b.inputs["Coat Weight"].default_value = coat
    if emission:
        b.inputs["Emission Color"].default_value = (*emission, 1)
        b.inputs["Emission Strength"].default_value = estr
    return m


ALU = mat("alu", (0.80, 0.81, 0.83), metallic=1.0, rough=0.32, aniso=0.5)
ALU_DK = mat("alu_dark", (0.62, 0.63, 0.66), metallic=1.0, rough=0.38)
STEEL = mat("steel", (0.56, 0.57, 0.60), metallic=1.0, rough=0.25)
STIFF = mat("stiff", (0.22, 0.23, 0.26), metallic=0.9, rough=0.40)
BOLT = mat("bolt", (0.20, 0.20, 0.22), metallic=1.0, rough=0.35)
COPPER = mat("copper", (0.80, 0.42, 0.22), metallic=1.0, rough=0.30)
CORE = mat("core", (0.18, 0.18, 0.19), metallic=1.0, rough=0.45)
ANOD = mat("anodized", (0.03, 0.18, 0.40), metallic=0.6, rough=0.35, coat=0.4)
GLASS = mat("lens", (0.02, 0.02, 0.03), metallic=0.0, rough=0.05, coat=1.0)
LASER = mat("laser", (1.0, 0.05, 0.05), emission=(1.0, 0.03, 0.02), estr=40.0)
PLASTIC = mat("plastic", (0.86, 0.86, 0.84), rough=0.55)
PANEL = mat("panel", (0.10, 0.10, 0.11), rough=0.4)
DISPLAY = mat("display", (0.02, 0.06, 0.04), emission=(0.10, 0.85, 0.35), estr=1.2)
RED = mat("red", (0.70, 0.04, 0.04), rough=0.3, coat=0.6)
BLACK = mat("black", (0.02, 0.02, 0.02), rough=0.3, coat=0.6)
CABLE_R = mat("cable_r", (0.55, 0.05, 0.05), rough=0.45)
CABLE_K = mat("cable_k", (0.04, 0.04, 0.04), rough=0.45)
FLOOR = mat("floor", (0.85, 0.85, 0.85), rough=0.9)


# ------------------------------------------------------------- helpers
def box(name, lo, hi, m, bevel=0.006):
    c = [(a + b) / 2 for a, b in zip(lo, hi)]
    d = [(b - a) for a, b in zip(lo, hi)]
    bpy.ops.mesh.primitive_cube_add(location=c)
    o = bpy.context.object; o.name = name
    o.scale = [x / 2 for x in d]
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        bv = o.modifiers.new("bev", "BEVEL"); bv.width = bevel; bv.segments = 3
    o.data.materials.append(m)
    return o


def cyl(name, base, r, h, m, axis="z", verts=64, bevel=0.003):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, vertices=verts)
    o = bpy.context.object; o.name = name
    if axis == "z":
        o.location = (base[0], base[1], base[2] + h / 2)
    elif axis == "x":
        o.rotation_euler = (0, math.pi / 2, 0)
        o.location = (base[0] + h / 2, base[1], base[2])
    bpy.ops.object.shade_smooth()
    if bevel:
        bv = o.modifiers.new("bev", "BEVEL"); bv.width = bevel; bv.segments = 2
        bv.limit_method = "ANGLE"
    o.data.materials.append(m)
    return o


def swept_strip(name, zs, xoff, half_w, half_t, m, xshift=0.0):
    """Rectangular strip (2*half_t in x, 2*half_w in y) swept along z with
    lateral offset xoff(z) -- the deflected beam and its stiffeners."""
    bm = bmesh.new()
    rings = []
    for z in zs:
        x0 = xoff(z) + xshift
        rings.append([bm.verts.new((x0 + sx * half_t, sy * half_w, z))
                      for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    for r0, r1 in zip(rings[:-1], rings[1:]):
        for k in range(4):
            bm.faces.new((r0[k], r0[(k + 1) % 4], r1[(k + 1) % 4], r1[k]))
    bm.faces.new(rings[0][::-1]); bm.faces.new(rings[-1])
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me); scene.collection.objects.link(o)
    o.data.materials.append(m)
    bv = o.modifiers.new("bev", "BEVEL"); bv.width = min(half_t, half_w) * 0.4
    bv.segments = 2; bv.limit_method = "ANGLE"
    return o


def cable(name, pts, m, r=0.007, poly=False):
    cu = bpy.data.curves.new(name, "CURVE"); cu.dimensions = "3D"
    if poly:                     # exact path through the points (no overshoot)
        sp = cu.splines.new("POLY"); sp.points.add(len(pts) - 1)
        for pt, p in zip(sp.points, pts):
            pt.co = (p[0], p[1], p[2], 1.0)
    else:
        sp = cu.splines.new("BEZIER"); sp.bezier_points.add(len(pts) - 1)
        for bp, p in zip(sp.bezier_points, pts):
            bp.co = p; bp.handle_left_type = bp.handle_right_type = "AUTO"
    cu.bevel_depth = r; cu.bevel_resolution = 4
    o = bpy.data.objects.new(name, cu); scene.collection.objects.link(o)
    o.data.materials.append(m)
    return o


# ------------------------------------------------------------- geometry
box("base", (-0.70, -0.38, 0.0), (0.50, 0.38, 0.09), ALU, bevel=0.01)
box("post", (-0.70, -0.30, 0.09), (-0.60, 0.30, 1.98), ALU, bevel=0.008)
box("arm", (-0.70, -0.08, 1.86), (0.10, 0.08, 1.98), ALU, bevel=0.008)
box("clamp", (-0.07, -0.09, 1.68), (0.09, 0.09, 1.86), ALU_DK, bevel=0.006)
for y in (-0.05, 0.05):                                    # clamp bolts
    cyl("cbolt", (0.09, y, 1.78), 0.016, 0.014, BOLT, axis="x", verts=6, bevel=0.002)

z_top, z_tip = 1.72, 0.30
L, delta = z_top - z_tip, 0.10
W, T = 0.075, 0.006


z_s = 1.46                       # top of the stiffeners
Lf = z_top - z_s                 # flexible span between clamp and stiffeners
a_ = delta / (Lf**2 + 2 * Lf * (L - Lf))


def u(z):
    """All bending is in the flexible span (where the notch is); the stiffened
    part below it is rigid, so it continues as a straight line."""
    s = min(max(z_top - z, 0.0), L)
    if s <= Lf:
        return a_ * s * s
    return a_ * Lf**2 + 2 * a_ * Lf * (s - Lf)


zs = sorted(set([z_tip + (z_s - 0.01 - z_tip) * k / 6 for k in range(7)] +
               [z_s - 0.01 + (z_top - z_s + 0.01) * k / 40 for k in range(41)]))
beam = swept_strip("beam", zs, u, W, T, STEEL)
# V-notch on both edges: a square prism (axis along x, the thickness) turned
# 45 degrees about x and centred on the edge, so it cuts a V of depth
# NOTCH_A*sqrt(2) into the beam width
z_n, NOTCH_A = 1.56, 0.016
cutters = []
for sy in (-1, 1):
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    c = bpy.context.object; c.name = "notch_cutter"
    c.scale = (0.03, NOTCH_A, NOTCH_A)
    c.rotation_euler = (math.pi / 4, 0, 0)
    c.location = (u(z_n), sy * W, z_n)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    cutters.append(c)
cutter, cutter2 = cutters
for c in (cutter, cutter2):
    bo = beam.modifiers.new("notch", "BOOLEAN"); bo.object = c; bo.operation = "DIFFERENCE"
    c.hide_render = True; c.hide_viewport = True

zs2 = [0.42, z_s]                                            # straight, rigid
Ws, Ts = 0.045, 0.008
for sx in (-1, 1):
    swept_strip("stiffener", zs2, u, Ws, Ts, STIFF, xshift=sx * (T + Ts))
for z in (0.50, 0.75, 1.00, 1.25, 1.40):                    # bolts through both
    cyl("sbolt", (u(z) + T + 2 * Ts, 0.0, z), 0.012, 0.010, BOLT, axis="x", verts=6,
        bevel=0.002)
    cyl("snut", (u(z) - T - 2 * Ts - 0.010, 0.0, z), 0.012, 0.010, BOLT, axis="x", verts=6,
        bevel=0.002)

for xm in (-0.17, 0.17):                                    # electromagnets
    cyl("flange", (xm, 0.0, 0.09), 0.088, 0.012, CORE)
    coil = cyl("coil", (xm, 0.0, 0.102), 0.075, 0.085, COPPER, verts=96)
    # winding texture: fine wave bump
    nt = COPPER.node_tree
    if "Wave Texture" not in nt.nodes:
        wv = nt.nodes.new("ShaderNodeTexWave"); wv.wave_type = "BANDS"
        wv.bands_direction = "Z"; wv.inputs["Scale"].default_value = 260
        bm_ = nt.nodes.new("ShaderNodeBump"); bm_.inputs["Strength"].default_value = 0.25
        nt.links.new(wv.outputs["Fac"], bm_.inputs["Height"])
        nt.links.new(bm_.outputs["Normal"], nt.nodes["Principled BSDF"].inputs["Normal"])
    cyl("cap", (xm, 0.0, 0.187), 0.080, 0.010, CORE)
    cyl("pole", (xm, 0.0, 0.197), 0.030, 0.014, STEEL)

lasers = {}
# CH2 (top) reads the frame motion at the clamp block; CH1 reads the beam
# just below the clamp, above the notch. Their difference is the beam position.
z_ch1 = 1.625                     # on the beam, between the notch and the clamp
for name, zl, hit in (("CH2", 1.77, 0.09), ("CH1", z_ch1, u(z_ch1) + T)):
    box("head_" + name, (0.80, -0.075, zl - 0.055), (1.25, 0.075, zl + 0.055), ANOD, bevel=0.015)
    cyl("ring_" + name, (0.765, 0.0, zl), 0.045, 0.035, ALU_DK, axis="x")
    cyl("lens_" + name, (0.755, 0.0, zl), 0.032, 0.012, GLASS, axis="x")
    ln = 0.755 - hit
    cyl("beam_" + name, (hit, 0.0, zl), 0.004, ln, LASER, axis="x", verts=12, bevel=0)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.007, location=(hit + 0.001, 0, zl))
    dot = bpy.context.object; dot.data.materials.append(LASER)
    lasers[name] = (1.25, 0.0, zl)

X0, X1, Y0, Y1, H = 0.68, 1.14, -0.32, 0.06, 0.30              # bench supply
box("supply", (X0, Y0, 0.0), (X1, Y1, H), PLASTIC, bevel=0.02)
box("panel", (X0 + 0.02, Y0 - 0.004, 0.03), (X1 - 0.02, Y0 + 0.01, H - 0.03), PANEL, bevel=0.004)
box("display", (X0 + 0.05, Y0 - 0.008, 0.17), (X0 + 0.25, Y0, 0.25), DISPLAY, bevel=0.002)
for xk in (X0 + 0.30, X0 + 0.36):
    k = cyl("knob", (xk, Y0 - 0.006, 0.21), 0.022, 0.03, BLACK, axis="z")
    k.rotation_euler = (math.pi / 2, 0, 0); k.location = (xk, Y0 - 0.02, 0.21)
terms = [(X0 + 0.30, Y0 - 0.02, 0.09), (X0 + 0.37, Y0 - 0.02, 0.09)]
for p, m in zip(terms, (RED, BLACK)):
    k = cyl("post_t", p, 0.014, 0.03, m)
    k.rotation_euler = (math.pi / 2, 0, 0); k.location = p
for (xm, p, m) in ((0.17, terms[0], CABLE_R), (-0.17, terms[1], CABLE_K)):
    # floor -> up the front face of the base -> over its top edge -> coil lug;
    # a polyline with rounded corners, so the lead never passes through the base
    r_c, yf = 0.007, -0.38 - 0.008              # base front face at y = -0.38
    xe = xm + 0.11
    pts = [Vector(p) + Vector((0, -0.012, 0)),
           Vector((p[0] - 0.02, Y0 - 0.08, 0.05)),
           Vector((p[0] - 0.08, Y0 - 0.16, r_c)),
           Vector((xe + 0.10, yf - 0.08, r_c)),
           Vector((xe + 0.02, yf - 0.03, r_c)),
           Vector((xe, yf, 0.02)),
           Vector((xe, yf, 0.08)),
           Vector((xe, yf + 0.004, 0.09 + r_c)),
           Vector((xe, yf + 0.03, 0.09 + r_c)),
           Vector((xm + 0.09, -0.15, 0.09 + r_c)),
           Vector((xm + 0.07, -0.06, 0.13))]
    cable("lead", pts, m, r=r_c, poly=True)

bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
bpy.context.object.data.materials.append(FLOOR)
bpy.context.object.is_shadow_catcher = True   # shadows only, on transparent film

# ------------------------------------------------------------- camera
target = Vector((0.20, 0.0, 0.95))
az, el, dist = math.radians(38), math.radians(20), 5.2
loc = target + dist * Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el),
                              math.sin(el)))
cam_data = bpy.data.cameras.new("cam"); cam_data.lens = 50; cam_data.sensor_width = 36
cam = bpy.data.objects.new("cam", cam_data); scene.collection.objects.link(cam)
cam.location = loc
cam.rotation_euler = (target - loc).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam

# ------------------------------------------------------------- lights, world
def area(name, loc_, size, energy, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA"); ld.size = size; ld.energy = energy
    ld.color = color
    lo = bpy.data.objects.new(name, ld); scene.collection.objects.link(lo)
    lo.location = loc_
    lo.rotation_euler = (target - Vector(loc_)).to_track_quat("-Z", "Y").to_euler()


area("key", (-1.8, -3.0, 4.0), 3.0, 380)
area("fill", (3.5, -1.5, 2.0), 4.0, 110, (0.95, 0.97, 1.0))
area("rim", (-1.0, 3.0, 3.0), 2.5, 160)
# studio world: bright overhead, dark toward the horizon, so metals read as metal
world = bpy.data.worlds.new("w"); scene.world = world; world.use_nodes = True
wn = world.node_tree.nodes; wl = world.node_tree.links
tc = wn.new("ShaderNodeTexCoord"); sep = wn.new("ShaderNodeSeparateXYZ")
ramp = wn.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].position = 0.45; ramp.color_ramp.elements[0].color = (0.05, 0.05, 0.055, 1)
ramp.color_ramp.elements[1].position = 0.95; ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1)
mp = wn.new("ShaderNodeMapRange"); mp.inputs["From Min"].default_value = -1.0
wl.new(tc.outputs["Generated"], sep.inputs[0])
wl.new(sep.outputs["Z"], mp.inputs["Value"]); wl.new(mp.outputs["Result"], ramp.inputs["Fac"])
wl.new(ramp.outputs["Color"], wn["Background"].inputs["Color"])
wn["Background"].inputs["Strength"].default_value = 0.55
# camera rays see a plain light-grey backdrop instead of the gradient
lp = wn.new("ShaderNodeLightPath"); bg2 = wn.new("ShaderNodeBackground")
bg2.inputs["Color"].default_value = (1, 1, 1, 1); bg2.inputs["Strength"].default_value = 1.0
mix = wn.new("ShaderNodeMixShader")
wl.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
wl.new(wn["Background"].outputs["Background"], mix.inputs[1]); wl.new(bg2.outputs["Background"], mix.inputs[2])
wl.new(mix.outputs["Shader"], wn["World Output"].inputs["Surface"])

# ------------------------------------------------------------- render
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.render.resolution_x = WIDTH
scene.render.resolution_y = int(WIDTH * 0.86)
scene.render.film_transparent = True
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.view_settings.exposure = -0.3
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = os.path.join(OUT, "rig_render.png")
bpy.ops.render.render(write_still=True)

# anchor points for the labels (normalized image coords, origin bottom-left)
anchors = {
    "clamp": (0.09, -0.09, 1.80), "notch": (u(z_n), -W, z_n),
    "stiffeners": (u(1.05) + T + 2 * Ts, -Ws, 1.05), "magnet": (-0.17, -0.075, 0.15),
    "CH1": lasers["CH1"], "CH2": lasers["CH2"], "head2": (1.02, -0.075, 1.825),
    "supply": (X0 + 0.25, Y0, H), "drive_a": (-0.55, -0.50, 0.0), "drive_b": (0.30, -0.50, 0.0),
}
out = {k: tuple(world_to_camera_view(scene, cam, Vector(v)))[:2] for k, v in anchors.items()}
out["aspect"] = scene.render.resolution_y / scene.render.resolution_x
json.dump(out, open(os.path.join(OUT, "rig_anchors.json"), "w"), indent=1)
print("wrote rig_render.png and rig_anchors.json")
