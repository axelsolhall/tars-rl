"""A third physics engine for sim-to-sim: Bullet (PyBullet), behind the same Engine interface.

Why: on 2026-11-04 Unreal's Chaos broke rigid-body physics in plain probes of TARS (a lone slab
tipped at the wrong force until its inertia conditioning was turned off; the jointed body slid at
~15 N where MuJoCo and a hand calculation give 23.5 N). With two engines that disagree, a third
says which one is the outlier. Bullet is a standard rigid-body engine in sim-to-real work, with
hard contacts and a reduced-coordinate (Featherstone) multibody for TARS's joints.

TARS is built from the BodySpec, like MuJoCo's MJCF: the central slab is the base; each arm hangs
on a hinge (about the body y axis, through the arm's centre) then a slide (along the arm), with a
tiny contact-free carriage between them, as in MuJoCo's two joints in one body. The servos are
MuJoCo's position actuator, clip(kp (target - q) - kv qdot, +-limit), computed before every 2 ms
step and applied as joint torques / forces (Bullet's own motors are switched off).

Friction: Bullet multiplies the two surfaces' coefficients, MuJoCo takes the larger. TARS's slabs
get 1.0 and every floor or box max(its own, the body's), so each pair's product is MuJoCo's max.

PyBullet is the `sim2sim` extra, Linux only (it does not build on macOS 26): run with
`uv run --extra sim2sim python -m tars_rl.unreal.verify --engines mujoco bullet ...` (the box).
"""

from __future__ import annotations

import numpy as np

from tars_rl.body.spec import BodySpec
from tars_rl.env.levels import Level
from tars_rl.unreal.bodystate import BodyState, joint_state

CARRIAGE_MASS_KG = 1e-3  # MuJoCo has none; Featherstone handles the mass ratio exactly
CARRIAGE_RADIUS_M = 0.005


def _wxyz(q_xyzw) -> np.ndarray:
    x, y, z, w = q_xyzw
    return np.array([w, x, y, z])


def _xyzw(q_wxyz) -> list[float]:
    w, x, y, z = q_wxyz
    return [x, y, z, w]


class BulletEngine:
    name = "bullet"

    def __init__(self, spec: BodySpec, ctrl_dt: float, solver_iterations: int = 50) -> None:
        import pybullet

        self.p = pybullet
        self.spec, self.ctrl_dt = spec, ctrl_dt
        self.n_sub = round(ctrl_dt / spec.timestep)
        self.client = pybullet.connect(pybullet.DIRECT)
        self.solver_iterations = solver_iterations
        self.level: Level | None = None
        self.tars: int | None = None
        self.push = (np.zeros(3), np.zeros(3))  # probes: (force N, point in the central's frame)

    # ---- world ---------------------------------------------------------------------------------
    def _world(self) -> None:
        p, c = self.p, self.client
        p.resetSimulation(physicsClientId=c)
        p.setGravity(0, 0, -9.81, physicsClientId=c)
        p.setPhysicsEngineParameter(fixedTimeStep=self.spec.timestep, numSubSteps=0,
                                    numSolverIterations=self.solver_iterations, physicsClientId=c)
        plane = p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE, physicsClientId=c),
                                  physicsClientId=c)
        p.changeDynamics(plane, -1, lateralFriction=self.spec.friction, restitution=0.0,
                         physicsClientId=c)
        assert self.level is not None
        for b in self.level.boxes:
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s / 2 for s in b.size],
                                           physicsClientId=c)
            body = p.createMultiBody(0, shape, basePosition=list(b.centre),
                                     baseOrientation=_xyzw(b.quat), physicsClientId=c)
            p.changeDynamics(body, -1, lateralFriction=max(b.friction, self.spec.friction),
                             restitution=0.0, physicsClientId=c)

    def _build_tars(self, at: np.ndarray) -> int:
        p, c, s = self.p, self.client, self.spec
        def box(size):
            half = [size.depth / 2, size.width / 2, size.height / 2]
            return p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=c)

        carriage = p.createCollisionShape(p.GEOM_SPHERE, radius=CARRIAGE_RADIUS_M,
                                          physicsClientId=c)
        arm = box(s.arm)
        arm_y = s.central.width / 2 + s.arm_gap + s.arm.width / 2
        n = [0.0, 0.0, 0.0]
        ident = [0.0, 0.0, 0.0, 1.0]
        tars = p.createMultiBody(
            baseMass=s.central.mass, baseCollisionShapeIndex=box(s.central), basePosition=list(at),
            linkMasses=[CARRIAGE_MASS_KG, s.arm.mass, CARRIAGE_MASS_KG, s.arm.mass],
            linkCollisionShapeIndices=[carriage, arm, carriage, arm],
            linkVisualShapeIndices=[-1] * 4,
            linkPositions=[[0, arm_y, 0], n, [0, -arm_y, 0], n],  # left arm at +y (MuJoCo)
            linkOrientations=[ident] * 4,
            linkInertialFramePositions=[n] * 4, linkInertialFrameOrientations=[ident] * 4,
            linkParentIndices=[0, 1, 0, 3],
            linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_PRISMATIC] * 2,
            linkJointAxis=[[0, 1, 0], [0, 0, 1]] * 2, physicsClientId=c)
        self.joints = [0, 2, 1, 3]  # BodySpec actuator order: left/right hinge, left/right slide
        for j, name in zip(self.joints, s.joint_names, strict=True):
            lo, hi = s.joint_range(name)
            p.changeDynamics(tars, j, jointLowerLimit=lo, jointUpperLimit=hi, physicsClientId=c)
        for link in (-1, 0, 1, 2, 3):
            p.changeDynamics(tars, link, lateralFriction=1.0, restitution=0.0, linearDamping=0.0,
                             angularDamping=0.0, physicsClientId=c)
        for link in (0, 2):  # carriages touch nothing
            p.setCollisionFilterGroupMask(tars, link, 0, 0, physicsClientId=c)
        for link in (1, 3):  # MuJoCo's <exclude>: arms never touch the central slab
            p.setCollisionFilterPair(tars, tars, -1, link, 0, physicsClientId=c)
        p.setJointMotorControlArray(tars, self.joints, p.VELOCITY_CONTROL, forces=[0.0] * 4,
                                    physicsClientId=c)
        return tars

    # ---- Engine interface ----------------------------------------------------------------------
    def load(self, level: Level) -> None:
        self.level = level

    def reset(self, spawn_xy: np.ndarray, flag: np.ndarray) -> BodyState:
        assert self.level is not None, "load a level first"
        self._world()  # fresh every episode: nothing carries over
        z = self.level.ground_z(spawn_xy) + self.spec.central.height / 2
        self.tars = self._build_tars(np.array([spawn_xy[0], spawn_xy[1], z]))
        self.t = 0.0
        self.targets = np.zeros(4)
        self.push = (np.zeros(3), np.zeros(3))
        return self._state()

    def set_flag(self, pos: np.ndarray) -> None:
        pass  # nothing is drawn

    def set_push(self, force: np.ndarray, at_local: np.ndarray) -> None:
        self.push = (np.asarray(force, float), np.asarray(at_local, float))

    def set_state(self, st: BodyState) -> BodyState:
        """One-step probes: the base from the central slab, the joints from the slabs' poses."""
        p, c = self.p, self.client
        q, qd = joint_state(self.spec, st)
        p.resetBasePositionAndOrientation(self.tars, st.pos[0].tolist(), _xyzw(st.quat[0]),
                                          physicsClientId=c)
        p.resetBaseVelocity(self.tars, st.linvel[0].tolist(), st.angvel[0].tolist(),
                            physicsClientId=c)
        for j, qi, qdi in zip(self.joints, q, qd, strict=True):
            p.resetJointState(self.tars, j, float(qi), float(qdi), physicsClientId=c)
        self.t = st.t
        return self._state()

    def step(self, ctrl: np.ndarray, hud: str | None = None) -> BodyState:
        p, c, s = self.p, self.client, self.spec
        servos = [s.hinge_servo, s.hinge_servo, s.slide_servo, s.slide_servo]
        for _ in range(self.n_sub):
            js = p.getJointStates(self.tars, self.joints, physicsClientId=c)
            u = [float(np.clip(sv.kp * (t - q[0]) - sv.kv * q[1], -sv.force_limit, sv.force_limit))
                 for sv, t, q in zip(servos, ctrl, js, strict=True)]
            p.setJointMotorControlArray(self.tars, self.joints, p.TORQUE_CONTROL, forces=u,
                                        physicsClientId=c)
            force, at = self.push
            if force.any():
                pos, quat = p.getBasePositionAndOrientation(self.tars, physicsClientId=c)
                r = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
                point = (np.array(pos) + r @ at).tolist()
                p.applyExternalForce(self.tars, -1, force.tolist(), point, p.WORLD_FRAME,
                                     physicsClientId=c)
            p.stepSimulation(physicsClientId=c)
        self.t += self.ctrl_dt
        return self._state()

    def _state(self) -> BodyState:
        p, c = self.p, self.client
        pos, quat = p.getBasePositionAndOrientation(self.tars, physicsClientId=c)
        v, w = p.getBaseVelocity(self.tars, physicsClientId=c)
        P, Q, V, W = [np.array(pos)], [_wxyz(quat)], [np.array(v)], [np.array(w)]
        for link in (1, 3):  # left arm, right arm; a box's centre of mass is its body origin
            ls = p.getLinkState(self.tars, link, computeLinkVelocity=1,
                                computeForwardKinematics=1, physicsClientId=c)
            P.append(np.array(ls[0]))
            Q.append(_wxyz(ls[1]))
            V.append(np.array(ls[6]))
            W.append(np.array(ls[7]))
        return BodyState(self.t, np.array(P), np.array(Q), np.array(V), np.array(W))

    def close(self) -> None:
        self.p.disconnect(physicsClientId=self.client)
