"""Read joint limits from the A3 MuJoCo model and write config/robot.yaml."""
import math
import mujoco
import yaml

MODEL = "a3_t3d0"
MJCF = f"A3-A3U-robot-model/{MODEL}/mjcf/{MODEL}.xml"

m = mujoco.MjModel.from_xml_path(MJCF)
limits = {}
for i in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    if m.jnt_limited[i] and m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
        lo, hi = (round(math.degrees(x), 1) for x in m.jnt_range[i])
        limits[name] = [lo, hi]

config = {
    "backend": "simulation",            # simulation | physical | mock
    "robot": {"model": MODEL, "mjcf": MJCF},
    "safety": {                         # starting values - confirm with AGIBOT specs
        "max_linear_velocity_mps": 0.3,
        "max_angular_velocity_radps": 0.5,
        "joint_limit_margin_deg": 5.0,  # stay this far inside each limit
        "command_timeout_s": 0.5,       # stop if no new command arrives
    },
    "joint_limits_deg": limits,
}

with open("config/robot.yaml", "w") as f:
    yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None)
print(f"Saved {len(limits)} joint limits to config/robot.yaml")
