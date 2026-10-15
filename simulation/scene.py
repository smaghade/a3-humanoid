"""Adds a working camera to the A3's head (and optionally a table, bottles and a cup).
AGIBOT's original files are not changed: everything is assembled in memory."""
import math

import mujoco
import numpy as np

HEAD_BODY = "head_pitch_Link"
CAMERA_NAME = "head_camera"
CAMERA_POS = [0.09, 0.0, -0.02]          # where the stereo-camera shape sits on the head
OBJECTS = ["red_bottle", "blue_bottle", "yellow_bottle", "cup"]
TABLE_HEIGHT = 0.75
# Known tables (the robot's "map"). front = direction the robot faces when standing at it.
TABLES = {
    "table_1": {"center": (1.20, 0.00), "half": (0.35, 0.60), "front": (1.0, 0.0)},   # with the bottles
    "table_2": {"center": (0.00, 1.60), "half": (0.60, 0.35), "front": (0.0, 1.0)},   # empty, on the left
}


def _find_body(spec, name):
    finder = getattr(spec, "body", None) or getattr(spec, "find_body")
    return finder(name)


def _camera_quat(tilt_down_deg):
    """Camera looking forward from the head, image-up = up, tilted down."""
    base = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
    t = math.radians(tilt_down_deg)
    tilt = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (tilt @ base).flatten())
    return quat.tolist()


def _add_object(world, name, pos, radius, half_height, rgba):
    body = world.add_body(name=name, pos=pos)
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, half_height, 0],
                  rgba=rgba, mass=0.3)


def build_scene(mjcf_path, with_objects=True, camera_fovy=70.0, camera_tilt_deg=30.0, obstacles=None):
    """Return (MjModel, spec): the robot + head camera (+ table and objects if with_objects)."""
    spec = mujoco.MjSpec.from_file(mjcf_path)
    head = _find_body(spec, HEAD_BODY)
    head.add_camera(name=CAMERA_NAME, pos=CAMERA_POS,
                    quat=_camera_quat(camera_tilt_deg), fovy=camera_fovy)
    if not with_objects:
        return spec.compile(), spec

    world = spec.worldbody
    t2 = TABLES["table_2"]                                    # the second, empty table
    table2 = world.add_body(name="table_2", pos=[*t2["center"], TABLE_HEIGHT / 2])
    table2.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[*t2["half"], TABLE_HEIGHT / 2],
                    rgba=[0.45, 0.45, 0.50, 1])
    table = world.add_body(name="table", pos=[1.2, 0.0, 0.375])
    table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.35, 0.6, 0.375], rgba=[0.55, 0.38, 0.22, 1])
    _add_object(world, "red_bottle", [1.05, 0.18, 0.87], 0.035, 0.12, [0.85, 0.1, 0.1, 1])
    _add_object(world, "blue_bottle", [1.05, -0.18, 0.87], 0.035, 0.12, [0.1, 0.25, 0.85, 1])
    _add_object(world, "yellow_bottle", [1.05, 0.38, 0.87], 0.035, 0.12, [0.95, 0.85, 0.1, 1])
    _add_object(world, "cup", [1.0, 0.0, 0.80], 0.045, 0.05, [0.95, 0.95, 0.95, 1])
    for ob in obstacles or []:                                # stones / boxes on the floor
        body = world.add_body(name=ob["name"], pos=ob["pos"])
        shape = mujoco.mjtGeom.mjGEOM_ELLIPSOID if ob["shape"] == "stone" else mujoco.mjtGeom.mjGEOM_BOX
        body.add_geom(type=shape, size=ob["size"], rgba=ob["rgba"])
    model = spec.compile()

    for key in spec.keys:                 # keep the objects on the table in the standing pose
        qpos = list(key.qpos)
        for name in OBJECTS:
            jid = model.body_jntadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
            adr = model.jnt_qposadr[jid]
            qpos[adr:adr + 7] = model.qpos0[adr:adr + 7]
        key.qpos = qpos
    if spec.keys:
        model = spec.compile()
    return model, spec


def camera_intrinsics(model, width, height, camera=CAMERA_NAME):
    """fx, fy, cx, cy in pixels (the camera_info from the Day 11 theory)."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    fovy = math.radians(model.cam_fovy[cam_id])
    fy = (height / 2) / math.tan(fovy / 2)
    return {"fx": fy, "fy": fy, "cx": width / 2, "cy": height / 2,
            "width": width, "height": height, "fovy_deg": float(model.cam_fovy[cam_id])}
