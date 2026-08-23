"""The deployed control loop for Experiment 2.

What the figure has to show is that the motor command the network feeds itself
is the same signal it sends to the motors -- its own output, not a report of
what the robot did. Everything else on the page exists to make that legible:

  * The network has two inputs, drawn as two arrows arriving side by side on
    its left face: the features the inverse model recovered, and the motor
    command. The motor command arrow leaves the network's output, so a reader
    can follow it round and see where it came from.
  * The SAME label goes on the arrow to the motors, because it is the same
    number. That is the whole argument.
  * Execution is noisy, so what the motors do is not what was commanded. The
    new pose is a box of its own, and nothing leads from it back to the
    network except the next emission.

An earlier version drew the missing odometry path as a dead-ended arrow and the
efference copy as a self-loop. Both failed: a blocked arrow needs something to
be blocked AT, and a loop from a network's output back into itself is the
standard picture of hidden-state recurrence, which is a different thing and is
not drawn here. Absence is shown by there being no arrow, not by drawing one
and stopping it.

Schematic, so no data is read and nothing here can go stale against a rerun.
Keep it in step with SCRIPT_RunPolicy.py's step order if that changes.

    Control_code/.venv/bin/python3 Paper/images/scripts/fig_control_loop.py
"""

import style

NAME = "fig_control_loop"

C_SONAR = "#3E6B8A"      # the sensory chain
C_CTRL = "#8172B2"       # the controller and the motor command it emits
C_WORLD = "#7A5C3E"      # the path that closes through the environment

BOX_H = 0.16
ROW_Y = 0.78             # the sense-to-command row
LOW_Y = 0.36             # what the command does in the world
Y_FEAT = ROW_Y - 0.045    # the features arrive low on the network's face
Y_CMD = ROW_Y + 0.045     # the motor command arrives high, alongside them
Y_TOP = 0.95             # the motor command's route back round
Y_RET = 0.08             # the next emission's route back round

SONAR = (0.030, 0.150, ROW_Y, "Sonar")
INV = (0.240, 0.400, ROW_Y, "Inverse\nmodel")
RNN = (0.565, 0.755, ROW_Y, "Recurrent network\n(32 units)")
MOTORS = (0.805, 0.950, LOW_Y, "Motors")
POSE = (0.335, 0.525, LOW_Y, "New pose")
X_SPLIT = 0.878          # where the motor command turns down to the motors
X_BACK = 0.455           # where it comes back down to the network's face


def box(ax, x0, x1, yc, label, colour, fs=7.2, lw=1.1):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x0, yc - BOX_H / 2), x1 - x0, BOX_H,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc="#FFFFFF", ec=colour, lw=lw, zorder=3))
    ax.text((x0 + x1) / 2, yc, label, ha="center", va="center", fontsize=fs,
            zorder=4, linespacing=1.3)


def seg(ax, pts, colour, ls="-", lw=1.1, head=True, z=2):
    """A polyline in axes coordinates, arrowhead on the final segment only."""
    from matplotlib.patches import FancyArrowPatch
    for i, (p0, p1) in enumerate(zip(pts, pts[1:])):
        last = i == len(pts) - 2
        ax.add_patch(FancyArrowPatch(
            p0, p1, color=colour, lw=lw, linestyle=ls, zorder=z, shrinkA=0,
            shrinkB=0, mutation_scale=11,
            arrowstyle="-|>" if (head and last) else "-"))


def main():
    style.setup()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(style.WIDTH_2COL, 2.45))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_facecolor("none")

    for b, c in ((SONAR, C_SONAR), (INV, C_SONAR), (RNN, C_CTRL),
                 (MOTORS, "#555555"), (POSE, C_WORLD)):
        box(ax, *b, c)

    # ---- sonar to inverse model --------------------------------------------
    seg(ax, [(SONAR[1], ROW_Y), (INV[0], ROW_Y)], "0.35")
    ax.text((SONAR[1] + INV[0]) / 2, ROW_Y + 0.028, "echoes", ha="center",
            va="bottom", fontsize=6.0, color="0.3")

    # ---- input 1: the features, low on the network's face ------------------
    seg(ax, [(INV[1], Y_FEAT), (RNN[0], Y_FEAT)], C_SONAR)
    ax.text((INV[1] + RNN[0]) / 2, Y_FEAT - 0.030, "features", ha="center",
            va="top", fontsize=6.6, color=C_SONAR)
    ax.text((INV[1] + RNN[0]) / 2, Y_FEAT - 0.088,
            "wall depth $\\times3$\n$p$(wall), $p$(pole)\n"
            "pole azimuth, range\nnearest range",
            ha="center", va="top", fontsize=5.2, color="0.4", linespacing=1.4)

    # ---- the motor command: out of the network, and straight back in -------
    # Drawn as one signal reaching two places. The label is repeated verbatim
    # on both arrows because it is the same number, which is the point.
    seg(ax, [(RNN[1], ROW_Y), (X_SPLIT, ROW_Y),
             (X_SPLIT, MOTORS[2] + BOX_H / 2)], C_CTRL)
    ax.text(X_SPLIT + 0.014, (ROW_Y + MOTORS[2] + BOX_H / 2) / 2,
            "motor\ncommand", ha="left", va="center", fontsize=6.6,
            color=C_CTRL, linespacing=1.3)
    seg(ax, [(0.815, ROW_Y), (0.815, Y_TOP), (X_BACK, Y_TOP),
             (X_BACK, Y_CMD), (RNN[0], Y_CMD)], C_CTRL)
    ax.text((X_BACK + 0.815) / 2, Y_TOP + 0.022, "motor command", ha="center",
            va="bottom", fontsize=6.6, color=C_CTRL)

    # ---- what the command does, and the only way back ----------------------
    seg(ax, [(MOTORS[0], LOW_Y), (POSE[1], LOW_Y)], C_WORLD)
    ax.text((MOTORS[0] + POSE[1]) / 2, LOW_Y + 0.028,
            "executed with error", ha="center", va="bottom", fontsize=6.4,
            color=C_WORLD)
    x_sonar = (SONAR[0] + SONAR[1]) / 2
    seg(ax, [(POSE[0], LOW_Y), (x_sonar, LOW_Y),
             (x_sonar, ROW_Y - BOX_H / 2)], C_WORLD, ls=(0, (3.5, 2)))
    ax.text(x_sonar + 0.022, Y_RET + 0.055, "the next emission is made\n"
            "from the new pose", ha="left", va="bottom", fontsize=6.4,
            color=C_WORLD, linespacing=1.35)

    style.save(fig, NAME)
    print(f"[fig] {NAME}: features + motor command as parallel inputs, "
          "command repeated on the arrow to the motors")


if __name__ == "__main__":
    main()
