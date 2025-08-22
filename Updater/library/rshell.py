import shutil, os
import subprocess
from library import settings
from pathlib import Path
import time

def make_staging_copy(source_dir, dest_dir='staging', full=False):
    clock_skew_sec = 5
    ts = time.time() + max(0, clock_skew_sec)
    excluded_folders = settings.excluded_folders
    fixed_library_folders = settings.fixed_library_folders
    excludes = set(excluded_folders)
    if not full: excludes.update(fixed_library_folders)
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
        for fname in files:
            if fname in excludes: continue
            src_file = Path(root) / fname
            dst_file = dst_root / fname
            shutil.copy2(src_file, dst_file)
            os.utime(dst_file, (ts, ts))
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
