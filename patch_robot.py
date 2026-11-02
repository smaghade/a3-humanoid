"""One-time patch: lets the preview robot walk up and down hills and steps smoothly.
Run from the project folder:  python patch_robot.py"""
import re

p = "robot/kinematic_sim_robot.py"
s = open(p).read()
if "self.ground = None" in s:
    print("kinematic_sim_robot.py already supports hills and steps - nothing to do")
else:
    a = "        self.animate_arms = True        # set False while the arms are used for grasping\n"
    line = re.compile(r"^        q\[b \+ 2\] -= self\._lowest_point_z\(\)[^\n]*\n", re.M)
    if a not in s or not line.search(s):
        print("Could not patch - paste this message to Claude")
    else:
        s = s.replace(a, a + "        self.ground = None              # optional ground-height function ground(x, y) for hills/steps\n"
                             "        self._gz = 0.0                  # smoothed ground height under the robot\n")
        s = line.sub('''        gz = 0.0
        if self.ground is not None:                    # walk up/down hills and steps smoothly
            target = float(self.ground(self.pose.x, self.pose.y))
            change = target - self._gz
            self._gz += change if dt <= 0 else max(-0.6 * dt, min(0.6 * dt, change))
            gz = self._gz
        q[b + 2] -= self._lowest_point_z() - gz                        # feet touch the floor
''', s, count=1)
        open(p, "w").write(s)
        print("kinematic_sim_robot.py patched - hills and steps supported")
