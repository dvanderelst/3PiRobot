import os
import zipfile
from datetime import datetime


def log_code(output_path, folders, label=None):
    """
    Archive all .py files found in the given folders into a zip file.

    Parameters
    ----------
    output_path : str
        Directory where the zip file will be written.
    folders : list[str]
        List of folder paths (relative or absolute) to collect .py files from.
        Files in each folder are added non-recursively unless the folder ends with '/**'.
    label : str, optional
        Extra label inserted into the zip filename, e.g. a run ID or script name.
        Filename format: code_<label>_<timestamp>.zip  (or code_<timestamp>.zip)
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'code_{label}_{timestamp}.zip' if label else f'code_{timestamp}.zip'
    zip_path = os.path.join(output_path, filename)

    os.makedirs(output_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in folders:
            recursive = folder.endswith('/**')
            folder = folder.removesuffix('/**')
            _add_folder(zf, folder, recursive)

    print(f'[CodeLogger] Code archived to {zip_path}')
    return zip_path


def _add_folder(zf, folder, recursive):
    if not os.path.isdir(folder):
        print(f'[CodeLogger] Warning: folder not found, skipping: {folder}')
        return
    if recursive:
        for dirpath, _, filenames in os.walk(folder):
            for filename in filenames:
                if filename.endswith('.py'):
                    full_path = os.path.join(dirpath, filename)
                    zf.write(full_path, arcname=full_path)
    else:
        for filename in os.listdir(folder):
            if filename.endswith('.py'):
                full_path = os.path.join(folder, filename)
                if os.path.isfile(full_path):
                    zf.write(full_path, arcname=full_path)
