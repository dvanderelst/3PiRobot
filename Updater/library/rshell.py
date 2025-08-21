import shutil, os
import subprocess
from pathlib import Path

excludes = ['__pycache__', 'pololu_3pi_2040_robot', 'umsgpack']

def make_staging_copy(source_dir, dest_dir='staging'):
    """
    Copy source_dir into dest_dir, excluding any files/folders listed in excludes.
    Excludes are matched against directory or file names (not globs).
    """
    src = Path(source_dir).resolve()
    dst = Path(dest_dir).resolve()
    # Clear staging dir
    if dst.exists(): shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for root, dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        # filter directories in-place so os.walk skips them
        dirs[:] = [d for d in dirs if d not in excludes]
        # make current directory in dest
        dst_root = dst / rel_root
        dst_root.mkdir(parents=True, exist_ok=True)
        # copy files except excluded
        for f in files:
            if f in excludes: continue
            shutil.copy2(Path(root) / f, dst_root / f)
    return dst


def remove_same(port, staged_root, show=True):
    batch_size = 50
    stream = True
    base = "/pyboard"
    print(f"\n=== Removing from {port} ===")
    staged_root = Path(staged_root)
    files = []
    for f in staged_root.rglob("*"):
        if f.is_file():
            rel = f.relative_to(staged_root).as_posix()
            files.append(f"{base}/{rel}")
    if not files:
        if show: print(f"[{port}] nothing to remove")
        return 0, 0
    if show: print(f"[{port}] removing {len(files)} files …")
    removed, failed = 0, 0
    for i in range(0, len(files), batch_size):
        chunk = files[i:i+batch_size]
        cmd = ["rshell", "-p", port, "rm"] + chunk
        if show: print(f"[{port}] rm batch {i//batch_size+1} ({len(chunk)} files)")
        try:
            if stream:
                subprocess.run(cmd, check=True, text=True)
            else:
                res = subprocess.run(cmd, check=True, text=True, capture_output=True)
                if show and res.stdout.strip(): print(res.stdout, end="")
            removed += 1
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"[{port}] rm batch failed (rc={e.returncode})")
            if not stream and e.stdout: print("stdout:\n", e.stdout, sep="")
            if not stream and e.stderr: print("stderr:\n", e.stderr, sep="")
    if show: print(f"[{port}] remove finished: {removed} batch(es) ok, {failed} failed")
    return removed, failed


def upload(port, local_dir, baud=115200, mirror=True):
    local_dir = Path(local_dir).resolve().as_posix() + "/"
    rsync_cmd = ["rshell", "-p", port, "-b", str(baud), "rsync"]
    if mirror: rsync_cmd.append("-m")  # delete extras on board
    rsync_cmd.extend([local_dir, "/pyboard/"])
    print(f"\n=== Uploading to {port} ===")
    print(" ".join(rsync_cmd))
    subprocess.run(rsync_cmd, check=True)
    ls_cmd = ["rshell", "-p", port, "-b", str(baud), "ls", "/pyboard"]
    print("\n=== Contents of /pyboard on", port, "===")
    subprocess.run(ls_cmd, check=True)
