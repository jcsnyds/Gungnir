#!/usr/bin/env python3
"""Make the package self-contained for one CRX model: copy its xacro macro and meshes into the package.

  adopt_robot.py MODEL [--macro FILE] [--meshes DIR]        e.g.  adopt_robot.py crx30ia

Without options it searches this package (config/, legacy/), ~/ws_fanuc/src and /tmp/fanuc_description for
<MODEL>_urdf_macro.xacro and for a <MODEL>/ directory that contains visual/ and collision/ meshes.
Installs config/robots/<MODEL>_urdf_macro.xacro and config/meshes/<MODEL>/, and rewrites the mesh URIs in the
macro to package://<this package>/config/meshes/<MODEL>/... Source files are copied, never moved."""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
PKG_NAME = re.search(r"<name>(.*?)</name>", (PKG / "package.xml").read_text()).group(1)
SKIP = {"build", "install", "log", ".git", "__pycache__"}
ROOTS = [PKG / "config", PKG / "legacy", Path.home() / "ws_fanuc" / "src", Path("/tmp/fanuc_description")]


def walk(root):
    if root.exists():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP)
            yield Path(dirpath), filenames


def find_macro(model):
    want = f"{model}_urdf_macro.xacro"
    for root in ROOTS:
        for d, files in walk(root):
            f = d / want
            if want in files and f.is_file() and f.stat().st_size > 200:      # is_file() is False for dangling links
                return f
    return None


def find_meshes(model):
    for root in ROOTS:
        for d, _ in walk(root):
            if d.name == model and (d / "visual").is_dir() and (d / "collision").is_dir():
                if any(p.is_file() for p in d.rglob("*")):
                    return d
    return None


def lfs_pointers(directory):
    bad = []
    for f in directory.rglob("*"):
        if f.is_file() and f.stat().st_size < 300:
            try:
                if f.read_bytes().startswith(b"version https://git-lfs"):
                    bad.append(f)
            except OSError:
                pass
    return bad


def number(s):
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--macro")
    ap.add_argument("--meshes")
    a = ap.parse_args()
    model = a.model

    macro = Path(a.macro).expanduser() if a.macro else find_macro(model)
    meshes = Path(a.meshes).expanduser() if a.meshes else find_meshes(model)
    problems = []
    if not macro or not macro.is_file():
        problems.append(f"no {model}_urdf_macro.xacro found")
    if not meshes or not meshes.is_dir():
        problems.append(f"no {model}/ mesh folder with visual/ and collision/ found")
    if problems:
        print("ERROR: " + "; ".join(problems))
        print("\nIf you have the files somewhere else:  adopt_robot.py %s --macro FILE --meshes DIR" % model)
        print("Otherwise get a fresh copy of FANUC's description repo (outside the workspace) and re-run:")
        print("  sudo apt install -y git-lfs && git lfs install")
        print("  git clone --depth 1 https://github.com/FANUC-CORPORATION/fanuc_description.git /tmp/fanuc_description")
        sys.exit(1)
    print(f"macro : {macro}\nmeshes: {meshes}")

    bad = lfs_pointers(meshes)
    if bad:
        print(f"\nERROR: {len(bad)} mesh files are git-LFS placeholders, not real meshes (e.g. {bad[0]}).")
        print("If they came from a git clone:  cd <that repo> && git lfs install && git lfs pull")
        sys.exit(1)

    text = macro.read_text()
    m = re.search(r'<xacro:macro\s+name="([^"]+)"\s+params="([^"]*)"', text, re.S)
    if not m:
        sys.exit("ERROR: no <xacro:macro name=... params=...> found in the macro file")
    params = " ".join(m.group(2).split())
    print(f"\nxacro macro: name={m.group(1)}  params: {params}")
    if m.group(1) != model:
        print(f"WARNING: the macro is called '{m.group(1)}', not '{model}'; robot_model:= must use that name")
    if not all(k in params for k in ("prefix", "parent", "*origin", "child")):
        print("WARNING: expected parameters prefix, parent, *origin and child (gungnir_geometry.urdf.xacro relies on them)")
    props = dict(re.findall(r'<xacro:property\s+name="(l_\w+|j[1-6]_(?:lower|upper|velocity)_limit)"\s+value="([^"]*)"', text))
    print("link lengths (m): " + "  ".join(f"{k}={v}" for k, v in sorted(props.items()) if k.startswith("l_")))
    for j in range(1, 7):
        lo, up, vel = (number(props.get(f"j{j}_{k}_limit", "?")) for k in ("lower", "upper", "velocity"))
        print(f"  J{j}: {lo:>7} .. {up:<7} deg   {vel} deg/s")
    i = text.find("J6-flange")
    o = re.search(r"<origin[^>]*>", text[i:]) if i >= 0 else None
    print("flange joint: " + (o.group(0) if o else "not found") + "   (5iA: xyz=\"${l_6} 0 0\", tool axis along +x)")

    pattern = re.compile(r'package://[A-Za-z0-9_]+/(?:config/)?meshes/(' + re.escape(model) + r'/[^"\']+)')
    text, n = pattern.subn(f"package://{PKG_NAME}/config/meshes/\\1", text)
    dest_macro = PKG / "config" / "robots" / f"{model}_urdf_macro.xacro"
    dest_meshes = PKG / "config" / "meshes" / model
    dest_macro.parent.mkdir(parents=True, exist_ok=True)
    dest_macro.write_text(text)
    if meshes.resolve() != dest_meshes.resolve():
        shutil.copytree(meshes, dest_meshes, dirs_exist_ok=True)

    refs = set(re.findall(r"package://" + re.escape(PKG_NAME) + r"/([^\"']+)", text))
    missing = sorted(r for r in refs if not (PKG / r).is_file())
    others = sorted(set(re.findall(r"package://(?!" + re.escape(PKG_NAME) + r"/)([^\"']+)", text)))
    print(f"\nwrote {dest_macro.relative_to(PKG)} ({n} mesh paths rewritten) and config/meshes/{model}/ "
          f"({sum(1 for p in dest_meshes.rglob('*') if p.is_file())} files)")
    if others:
        print("WARNING: URIs still pointing at other packages:", *others[:5])
    if missing:
        print(f"ERROR: {len(missing)} meshes referenced by the macro are missing, e.g. {missing[0]}")
        sys.exit(1)
    print("all referenced meshes exist")


if __name__ == "__main__":
    main()
