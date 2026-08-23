"""The deployed control loop, drawn around the connection that is absent.

Experiment 2's Methods make a claim that is hard to see in prose: the
controller's internal state is a function of the echoes it has heard and of
nothing else. The argument has three parts, and the figure exists to make all
three visible at once.

  1. The only non-sonar input is the rotation the network itself commanded
     (`prev_rot = rotate_cmd` in SCRIPT_RunPolicy.py, and the trainer matches).
     Drawn as the efference copy arc, which starts and ends inside the
     controller -- it carries nothing the network did not just compute.
  2. The motors do not execute that rotation exactly, and the executed value
     never reaches the network: no odometry, no inertial sensor, no pose. Drawn
     as the dead-ended arrow, which is the point of the whole figure and is why
     a generic RNN box diagram would not do.
  3. The loop therefore closes through the WORLD: the robot moves, the geometry
     around it changes, and the next echo carries the consequence. That is the
     only path by which the controller learns that it moved.

Schematic, so no data is read and nothing here can go stale against a rerun.
Keep it in step with SCRIPT_RunPolicy.py's step order if that changes.

    Control_code/.venv/bin/python3 Paper/images/scripts/fig_control_loop.py
"""

import style

NAME = "fig_control_loop"

C_BOX = "#3E6B8A"        # the sonar chain
C_CTRL = "#8172B2"       # the controller
C_WORLD = "#7A5C3E"      # the path through the environment
C_DEAD = "#C44E52"       # the connection that does not exist

# Boxes: (x0, x1, y_centre, label, edge colour)
ROW_Y = 0.74
BOX_H = 0.20
BOXES = [
    (0.025, 0.165, ROW_Y, "Echoes\n(two ears)", C_BOX),
    (0.205, 0.355, ROW_Y, "Inverse\nmodel", C_BOX),
    (0.395, 0.590, ROW_Y, "Features", C_BOX),
    (0.630, 0.815, ROW_Y, "Recurrent network\n(32 units)", C_CTRL),
]
MOTOR = (0.835, 0.975, 0.30, "Motors")
X_OUT = 0.905          # where the commanded rotation turns down to the motors
Y_WORLD = 0.075        # the return path through the environment


def box(ax, x0, x1, yc, label, colour, fs=7.0, lw=1.1, fc="#FFFFFF"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x0, yc - BOX_H / 2), x1 - x0, BOX_H,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fc, ec=colour, lw=lw, zorder=3))
    ax.text((x0 + x1) / 2, yc, label, ha="center", va="center", fontsize=fs,
            zorder=4, linespacing=1.3)


def arrow(ax, p0, p1, colour, rad=0.0, ls="-", lw=1.1, head=True, z=2):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={rad}", color=colour, lw=lw,
        linestyle=ls, zorder=z, shrinkA=0, shrinkB=0, mutation_scale=9,
        arrowstyle="-|>" if head else "-"))


def main():
    style.setup()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(style.WIDTH_2COL, 2.35))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_facecolor("none")

    for b in BOXES:
        box(ax, *b)

    # ---- the forward chain -------------------------------------------------
    for (_, x1, _, _, _), (x0n, _, _, _, _) in zip(BOXES, BOXES[1:]):
        arrow(ax, (x1, ROW_Y), (x0n, ROW_Y), "0.35")

    # What the feature vector holds. Named rather than left as "features",
    # because the claim is about the channels: every one of them is sonar.
    ax.text(0.4925, ROW_Y - BOX_H / 2 - 0.03,
            "wall depth $\\times3$  ·  $p$(wall), $p$(pole)\n"
            "pole azimuth, range  ·  nearest range",
            ha="center", va="top", fontsize=5.4, color="0.35", linespacing=1.35)

    # ---- commanded rotation, out of the controller and down to the motors ---
    arrow(ax, (0.815, ROW_Y), (X_OUT, ROW_Y), "0.35", head=False)
    arrow(ax, (X_OUT, ROW_Y), (X_OUT, MOTOR[2] + BOX_H / 2), "0.35")
    ax.text(X_OUT + 0.012, (ROW_Y + MOTOR[2] + BOX_H / 2) / 2,
            "commanded\nrotation", ha="left", va="center", fontsize=5.8,
            color="0.3", linespacing=1.3)
    box(ax, *MOTOR, "#555555", fs=7.0)

    # ---- 1. the efference copy: out of the controller and back into it ------
    # Branches off the commanded-rotation line rather than starting in empty
    # space, so it reads as a copy of that signal and not as a second output.
    arrow(ax, (0.865, ROW_Y), (0.7225, ROW_Y + BOX_H / 2), C_CTRL, rad=0.75,
          lw=1.2)
    ax.text(0.44, 0.99,
            "efference copy: the rotation the network itself commanded",
            ha="center", va="top", fontsize=6.0, color=C_CTRL)

    # ---- 2. the connection that does not exist -----------------------------
    # Drawn dead-ended rather than omitted: an absent arrow reads as an
    # oversight, a stopped one reads as a statement.
    y_dead = MOTOR[2]
    arrow(ax, (MOTOR[0], y_dead), (0.665, y_dead), C_DEAD, ls=(0, (3, 2)),
          lw=1.1, head=False)
    ax.plot([0.665, 0.665], [y_dead - 0.055, y_dead + 0.055], color=C_DEAD,
            lw=2.0, solid_capstyle="butt", zorder=4)
    ax.text(0.655, y_dead + 0.075,
            "executed rotation: no odometry, no inertial sensor,\n"
            "no access to its own position",
            ha="right", va="bottom", fontsize=6.0, color=C_DEAD,
            linespacing=1.35)

    # ---- 3. the loop that does exist, through the world ---------------------
    x_echo = (BOXES[0][0] + BOXES[0][1]) / 2
    arrow(ax, (MOTOR[0] + 0.07, MOTOR[2] - BOX_H / 2), (x_echo, Y_WORLD),
          C_WORLD, rad=0.10, lw=1.2, head=False)
    arrow(ax, (x_echo, Y_WORLD), (x_echo, ROW_Y - BOX_H / 2), C_WORLD, lw=1.2)
    ax.text(0.50, Y_WORLD - 0.012,
            "the robot moves, and the next echo comes from the new position",
            ha="center", va="bottom", fontsize=6.4, color=C_WORLD)

    style.save(fig, NAME)
    print(f"[fig] {NAME}: 4 chain boxes + motors, "
          "efference copy in, executed rotation dead-ended")


if __name__ == "__main__":
    main()
