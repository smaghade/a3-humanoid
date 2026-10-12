"""Draw the robot's path from the newest log (or a given CSV file) and print a summary."""
import csv
import glob
import math
import sys

import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("logs/run_*.csv"))[-1]
rows = list(csv.DictReader(open(path)))
t = [float(r["time_s"]) for r in rows]
x = [float(r["x_m"]) for r in rows]
y = [float(r["y_m"]) for r in rows]
yaw = [math.radians(float(r["facing_deg"])) for r in rows]
estop = [i for i, r in enumerate(rows) if r["status"] == "E-STOP"]

distance = sum(math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) for i in range(1, len(x)))
print(f"File:      {path}")
print(f"Duration:  {t[-1]:.1f} s")
print(f"Distance:  {distance:.2f} m walked")
print(f"Start:     x={x[0]:+.2f} y={y[0]:+.2f}")
print(f"End:       x={x[-1]:+.2f} y={y[-1]:+.2f} facing {float(rows[-1]['facing_deg']):+.0f} deg")

plt.figure(figsize=(7, 7))
plt.plot(x, y, "-", color="tab:blue", label="path")
step = max(1, len(x) // 25)                               # direction arrows along the path
plt.quiver(x[::step], y[::step], [math.cos(a) for a in yaw[::step]],
           [math.sin(a) for a in yaw[::step]], color="gray", scale=25, width=0.004)
plt.plot(x[0], y[0], "o", color="green", markersize=12, label="start")
plt.plot(x[-1], y[-1], "s", color="red", markersize=12, label="end")
if estop:
    plt.plot([x[i] for i in estop], [y[i] for i in estop], "x", color="black", label="emergency stop")
plt.axis("equal")
plt.grid(True)
plt.xlabel("x (m)  - forward at start")
plt.ylabel("y (m)  - left at start")
plt.title(f"Robot path ({distance:.1f} m in {t[-1]:.0f} s)")
plt.legend()
png = path.replace(".csv", ".png")
plt.savefig(png, dpi=120)
print(f"Picture:   {png}")
plt.show()
