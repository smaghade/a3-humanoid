"""One-time patch: lets simulation/scene.py place stones and boxes on the floor.
Run from the project folder:  python patch_scene.py"""
p = "simulation/scene.py"
s = open(p).read()
if "obstacles=None" in s:
    print("scene.py already supports obstacles - nothing to do")
else:
    s = s.replace("def build_scene(mjcf_path, with_objects=True, camera_fovy=70.0, camera_tilt_deg=30.0):",
                  "def build_scene(mjcf_path, with_objects=True, camera_fovy=70.0, camera_tilt_deg=30.0, obstacles=None):")
    s = s.replace("    model = spec.compile()\n", '''    for ob in obstacles or []:                                # stones / boxes on the floor
        body = world.add_body(name=ob["name"], pos=ob["pos"])
        shape = mujoco.mjtGeom.mjGEOM_ELLIPSOID if ob["shape"] == "stone" else mujoco.mjtGeom.mjGEOM_BOX
        body.add_geom(type=shape, size=ob["size"], rgba=ob["rgba"])
    model = spec.compile()
''', 1)
    open(p, "w").write(s)
    ok = "obstacles=None" in s and "for ob in obstacles" in s
    print("scene.py patched - stones supported" if ok else "Could not patch - paste this message to Claude")
