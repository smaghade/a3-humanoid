"""Day 19-20 (preview): move an arm so the hand reaches a point, and 'magic grasp'.

Arm IK = inverse kinematics: "put the hand HERE" -> the joint angles that do it.
Method: damped least squares with MuJoCo's Jacobian (how each joint moves the hand).
Joint limits are always respected. The waist can bend forward to help reach.

Magic grasp (plan Day 20 fallback): the A3 model has no finger joints, so once the
hand is close enough, the object is attached to the hand and moves with it.
"""
import mujoco
import numpy as np

PALM_OFFSET = np.array([0.12, 0.0, 0.0])      # palm centre in the wrist_yaw frame (metres)


class Arm:
    def __init__(self, model, data, side="left", use_waist=True):
        self.m, self.d, self.side = model, data, side
        names = [f"{side}_shoulder_pitch_joint", f"{side}_shoulder_roll_joint",
                 f"{side}_shoulder_yaw_joint", f"{side}_elbow_joint",
                 f"{side}_wrist_roll_joint", f"{side}_wrist_pitch_joint", f"{side}_wrist_yaw_joint"]
        if use_waist:
            names = ["waist_pitch_joint"] + names
        self.joints = [model.joint(n) for n in names]
        self.qadr = np.array([j.qposadr[0] for j in self.joints])
        self.dadr = np.array([j.dofadr[0] for j in self.joints])
        self.lo = np.array([j.range[0] for j in self.joints])
        self.hi = np.array([j.range[1] for j in self.joints])
        self.hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_Link")
        self.home = data.qpos[self.qadr].copy()

    def point(self, local=PALM_OFFSET):
        """World position of a point fixed to the hand (given in the wrist frame)."""
        return self.d.xpos[self.hand] + self.d.xmat[self.hand].reshape(3, 3) @ np.asarray(local)

    def palm(self):
        """World position of the palm centre."""
        return self.point(PALM_OFFSET)

    def step_towards(self, target, max_step=0.03, damping=0.05, tool=PALM_OFFSET):
        """One small IK step that moves the hand point `tool` (default: palm) towards the target.
        With tool = a held object's centre, this places the OBJECT at the target.
        Returns the remaining distance (m)."""
        mujoco.mj_forward(self.m, self.d)
        palm = self.point(tool)
        err = np.asarray(target) - palm
        jacp = np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, jacp, None, palm, self.hand)
        J = jacp[:, self.dadr]
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), err)
        dq = np.clip(dq, -max_step, max_step)                        # small, smooth moves
        q = np.clip(self.d.qpos[self.qadr] + dq, self.lo, self.hi)   # never past a joint limit
        self.d.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, self.d)
        return float(np.linalg.norm(np.asarray(target) - self.point(tool)))

    def step_home(self, max_step=0.03):
        """One small step back to the starting arm pose. Returns remaining joint error."""
        q = self.d.qpos[self.qadr]
        dq = np.clip(self.home - q, -max_step, max_step)
        self.d.qpos[self.qadr] = q + dq
        mujoco.mj_forward(self.m, self.d)
        return float(np.abs(self.home - self.d.qpos[self.qadr]).max())


class MagicGrasp:
    """Attach an object to the hand (kinematically) and release it again.

    The object's position is stored relative to the hand, so it stays in the hand when
    the arm moves AND when the robot turns or walks while carrying it.
    """

    def __init__(self, model, data, obj_name):
        self.m, self.d = model, data
        self.name = obj_name
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        self.qadr = model.jnt_qposadr[model.body_jntadr[self.body]]
        self.attached_to = None
        self.local = None              # object centre in the wrist frame

    def position(self):
        return self.d.xpos[self.body].copy()

    def attach(self, arm):
        self.attached_to = arm
        rot = self.d.xmat[arm.hand].reshape(3, 3)
        self.local = rot.T @ (self.position() - self.d.xpos[arm.hand])

    def update(self):
        """Call every step: an attached object follows the hand (position only; stays upright)."""
        if self.attached_to is not None:
            self.d.qpos[self.qadr:self.qadr + 3] = self.attached_to.point(self.local)
            self.d.qpos[self.qadr + 3:self.qadr + 7] = (1, 0, 0, 0)
            mujoco.mj_forward(self.m, self.d)

    def release(self, put_at=None):
        """Let go. put_at = exact spot to leave the object at (upright)."""
        self.attached_to = None
        if put_at is not None:
            self.d.qpos[self.qadr:self.qadr + 7] = [*put_at, 1, 0, 0, 0]
            mujoco.mj_forward(self.m, self.d)
