"""humans.pdf -- schematic of two measured-kinematics studies (Chapter 8,
Fig. 'humans'), drawn as 3D stick figures in perspective. Replaces the
photographs of the earlier figure.

(a) Cycling on an ergometer, after Song et al. (2009): reflective markers on
    the joints for motion capture and a surface EMG electrode on the thigh.
(b) Load-carriage walking on a treadmill, after Schiffman et al. (2009):
    joint markers, a loaded pack, and a mask for respiratory gas analysis.

Geometry is built in 3D (x forward, y lateral, z up, metres, by hand) and
projected through a pinhole camera. Limb poses come from two-link inverse
kinematics in the sagittal plane. Everything is drawn back to front
(painter's algorithm); the far side of the body is lighter, and line widths
and markers shrink with distance.
"""
from _figstyle import *
from _scene3d import Camera, Scene, ik, label, use_cm_fonts

use_cm_fonts()

NEAR, FAR = "0.12", "0.55"          # limb shades, near and far side
MARK, MARK_FAR = COL[1], "#EBA585"  # reflective markers (orange), far side
EMG = "#A2142F"                     # EMG electrode (dark red)
GAS = COL[0]                        # mask and gas line (blue)
EQUIP = np.array([0.80, 0.80, 0.80])
PACK = np.array([0.56, 0.56, 0.44])


def neck(chest, head, r):
    return chest, head + r * (chest - head) / np.linalg.norm(chest - head)


fig, axs = plt.subplots(1, 2, figsize=(5.4, 3.0))
THIGH, SHANK, UARM, FARM = 0.45, 0.44, 0.30, 0.29
AZ, EL, DIST = -30, 16, 3.8

# ================================================================= cycling
cam = Camera(target=(0.18, 0.0, 0.85), az=AZ, el=EL, dist=DIST)
S = Scene(cam)
crank = np.array([0.18, 0.0, 0.42])
S.box((-0.22, -0.16, 0.0), (0.46, 0.16, 0.30), EQUIP, bias=0.3)    # base
S.disc_y(crank + [0.06, 0.0, 0.04], 0.21, (0.74, 0.74, 0.74))      # flywheel
S.line([(-0.03, 0, 0.30), (-0.03, 0, 1.03)], "0.45", 2.4)          # seat post
S.line([(-0.12, 0, 1.05), (0.07, 0, 1.05)], "0.25", 4.0)           # saddle
S.line([(0.40, 0, 0.30), (0.50, 0, 1.00), (0.60, 0, 1.06)], "0.45", 2.4)
S.line([(0.60, -0.22, 1.06), (0.60, 0.22, 1.06)], "0.30", 2.6)     # handlebar
pelvis, chest = np.array([0.0, 0, 1.10]), np.array([0.36, 0, 1.58])
head = np.array([0.48, 0, 1.79])
S.line([pelvis, chest], NEAR, 2.6)
S.line(neck(chest, head, 0.10), NEAR, 2.6)
S.head(head, 0.10, NEAR, 2.4)
for side, phase, shade, mk in ((-1, -35, NEAR, MARK), (+1, 145, FAR, MARK_FAR)):
    hip = pelvis + [0, 0.14 * side, 0]
    th = np.deg2rad(phase)
    ankle = crank + [0.17 * np.cos(th), 0.12 * side, 0.17 * np.sin(th)]
    knee = ik(hip, ankle, THIGH, SHANK, bend=+1)
    toe = ankle + [0.13, 0, -0.02]
    sh = chest + [0, 0.18 * side, 0]
    hand = np.array([0.60, 0.20 * side, 1.06])
    elbow = ik(sh, hand, UARM, FARM, bend=-1)
    S.line([pelvis, hip, knee, ankle, toe], shade, 2.4)
    S.line([crank + [0, 0.12 * side, 0], ankle], "0.45", 1.2)      # crank arm
    S.line([chest, sh, elbow, hand], shade, 2.0)
    for p in (sh, elbow, hand, hip, knee, ankle, toe):
        S.ball(p, 0.024, mk)
    if side < 0:
        emg_p = 0.5 * (hip + knee) + [0, -0.03, 0.03]
        S.ball(emg_p, 0.03, EMG)
        near_knee = knee
S.draw(axs[0])
label(axs[0], cam, "joint markers", near_knee, (0.55, -0.25))
label(axs[0], cam, "surface EMG", emg_p, (-1.0, 0.05))
label(axs[0], cam, "cycle ergometer", (0.30, -0.16, 0.12), (0.30, -0.90))
axs[0].set_title("(a) cycling: muscle fatigue", fontsize=9, loc="left")

# =========================================================== load carriage
cam = Camera(target=(0.0, 0.0, 0.85), az=AZ, el=EL, dist=DIST)
S = Scene(cam)
S.box((-0.62, -0.30, 0.0), (0.58, 0.30, 0.09), EQUIP, bias=0.6)    # treadmill
for side in (-1, 1):                                                 # rails
    y = 0.30 * side
    S.line([(0.50, y, 0.09), (0.50, y, 1.16), (0.12, y, 1.16)], "0.45", 1.8)
pelvis, chest = np.array([0.0, 0, 0.98]), np.array([0.05, 0, 1.58])
head = np.array([0.09, 0, 1.79])
S.box((-0.31, -0.16, 1.10), (-0.06, 0.16, 1.63), PACK)              # pack
S.line([pelvis, chest], NEAR, 2.6)
S.line(neck(chest, head, 0.10), NEAR, 2.6)
S.head(head, 0.10, NEAR, 2.4)
mask = head + np.array([0.10, -0.02, -0.04])
S.ball(mask, 0.045, GAS, ec=GAS)
S.line([mask, mask + [0.20, -0.02, 0.05], (0.62, -0.05, 1.86)], GAS, 1.2)
for side, ax_, az_, tdx, shade, mk in ((-1, 0.14, 0.15, 0.15, NEAR, MARK),
                                       (+1, -0.27, 0.20, 0.13, FAR, MARK_FAR)):
    hip = pelvis + [0, 0.13 * side, 0]
    ankle = np.array([ax_, 0.13 * side, az_])
    knee = ik(hip, ankle, THIGH, SHANK, bend=+1)
    toe = ankle + [tdx, 0, -0.06]
    sh = chest + [0, 0.18 * side, 0]
    hand = np.array([0.36, 0.28 * side, 1.16])
    elbow = ik(sh, hand, UARM, FARM, bend=-1)
    S.line([pelvis, hip, knee, ankle, toe], shade, 2.4)
    S.line([chest, sh, elbow, hand], shade, 2.0)
    for p in (sh, elbow, hand, hip, knee, ankle, toe):
        S.ball(p, 0.024, mk)
    if side < 0:
        near_knee = knee
S.draw(axs[1])
label(axs[1], cam, "gas analysis", mask, (0.30, 0.95))
label(axs[1], cam, "loaded pack", (-0.31, -0.16, 1.40), (-1.05, 0.80))
label(axs[1], cam, "joint markers", near_knee, (0.45, -0.25))
label(axs[1], cam, "treadmill", (0.30, -0.30, 0.05), (0.35, -0.95))
axs[1].set_title("(b) load carriage: O$_2$ consumption", fontsize=9, loc="left")

for ax in axs:
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.05, 1.0); ax.set_ylim(-1.05, 1.05)
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(FIGDIR + "humans.pdf")
print("wrote humans.pdf")
