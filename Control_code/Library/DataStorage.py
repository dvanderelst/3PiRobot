import os
import dill
import shutil
from datetime import datetime
from typing import Dict, Any, List
from os import path
from natsort import natsorted
from Library import Settings

class DataWriter:
    """
    A class for storing dictionary data in sequentially numbered files within a dedicated folder.
    
    Features:
    - Creates a folder for data storage
    - Automatically increments file numbers (data00001, data00002, etc.)
    - Uses dill serialization for arbitrary Python data
    - Stores named payload entries under a data dict
    - Provides metadata tracking (timestamps, file counts)
    """
    
    def __init__(self, base_folder: str = "data",
                 prefix: str = "data", padding: int = 5, autoclear: bool = False,
                 verbose: bool = True):
        """
        Initialize the DataWriter object.
        
        Parameters
        ----------
        base_folder : str
            The base folder where data will be stored
        prefix : str
            The prefix for data files (default: 'data')
        padding : int
            The number of digits for file numbering with zero-padding
        autoclear : bool
            If True, clear existing folder if it exists. If False, raise exception if folder exists.
        verbose : bool
            If True, print status messages. If False, silence output.
        """
        self.base_folder = path.join(Settings.data_folder, base_folder)
        self.prefix = prefix
        self.padding = padding
        self.file_counter = 0
        self.created_at = None
        self.autoclear = autoclear
        self.verbose = verbose
        
        # Create the base folder
        self._create_folder()
    
    def _create_folder(self):
        """Create the base folder if it doesn't exist."""
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)
            self.created_at = datetime.now()
            if self.verbose:
                print(f"Created data storage folder: {self.base_folder}")
            # Count existing files to determine next file number
            self._update_file_counter()
        else:
            if self.autoclear:
                # Clear existing folder
                import shutil
                shutil.rmtree(self.base_folder)
                os.makedirs(self.base_folder)
                self.created_at = datetime.now()
                if self.verbose:
                    print(f"Cleared and recreated data storage folder: {self.base_folder}")
                # Start fresh with file counter
                self.file_counter = 0
            else:
                # Raise exception if folder exists and autoclear is False
                raise FileExistsError(
                    f"Data storage folder already exists: {self.base_folder}. "
                    f"Set autoclear=True to clear existing folder, or use a different folder name."
                )
    
    def _update_file_counter(self):
        """Update the file counter based on existing files in the folder."""
        existing_files = []
        
        # List all files that match our naming pattern
        for filename in os.listdir(self.base_folder):
            if filename.startswith(self.prefix):
                try:
                    # Extract the number part
                    number_str = filename[len(self.prefix):].split('.')[0]
                    if number_str.isdigit():
                        existing_files.append(int(number_str))
                except (IndexError, ValueError):
                    continue
        
        if existing_files:
            self.file_counter = max(existing_files)
        else:
            self.file_counter = 0
    
    def _get_next_filename(self) -> str:
        """Generate the next filename with proper zero-padding."""
        self.file_counter += 1
        number_str = f"{self.file_counter:0{self.padding}d}"
        
        return f"{self.prefix}{number_str}.dill"
    
    def _get_filepath(self, filename: str) -> str:
        """Get the full filepath for a given filename."""
        return os.path.join(self.base_folder, filename)


    def files_folder(self, make=False):
        folder = os.path.join(self.base_folder, "files")
        if make: os.makedirs(folder, exist_ok=True)
        return folder


    def add_file(self, filepath: str) -> str:
        """
        Copy an external file into the data storage folder under a files/ subfolder.

        Parameters
        ----------
        filepath : str
            Path to the file to copy

        Returns
        -------
        str
            Destination filepath for the copied file
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        files_folder = self.files_folder()
        os.makedirs(files_folder, exist_ok=True)

        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        if ext == ".py":
            filename = f"{name}.copy"

        dest_path = os.path.join(files_folder, filename)
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            counter = 1
            while True:
                candidate = f"{base}_{counter}{ext}"
                dest_path = os.path.join(files_folder, candidate)
                if not os.path.exists(dest_path):
                    break
                counter += 1
        shutil.copy2(filepath, dest_path)

        if self.verbose:
            print(f"Copied file to: {dest_path}")

        return dest_path
    
    def save_data(self, *args: Dict[str, Any],
                  metadata: Dict[str, Any] = None,
                  **named_data: Any) -> str:
        """
        Save data to a new file with incrementing number.
        
        Parameters
        ----------
        *args : dict
            Optional single dict containing named data to be saved
        **named_data : Any
            Named data entries to be saved
        metadata : dict, optional
            Additional metadata to include with the data
            
        Returns
        -------
        str
            The filename where data was saved
        """
        data_dict: Dict[str, Any] = {}
        if len(args) > 1:
            raise ValueError("Only a single positional dict is supported")
        if len(args) == 1:
            if not isinstance(args[0], dict):
                raise TypeError("Positional argument must be a dict of named data")
            data_dict.update(args[0])

        overlap = set(data_dict.keys()) & set(named_data.keys())
        if overlap:
            overlap_str = ", ".join(sorted(overlap))
            raise ValueError(f"Duplicate data keys provided: {overlap_str}")

        data_dict.update(named_data)
        if 'meta' in data_dict:
            raise ValueError("The key 'meta' is reserved for metadata")
        if not data_dict:
            raise ValueError("No data provided to save")

        filename = self._get_next_filename()
        meta = {
            'timestamp': datetime.now().isoformat(),
            'file_number': self.file_counter,
            'filename': filename
        }
        
        # Add metadata if provided
        if metadata:
            meta.update(metadata)
        
        # Prepare the data structure
        save_dict = {
            'data': data_dict,
            'meta': meta
        }
        filepath = self._get_filepath(filename)
        
        # Save the data
        with open(filepath, 'wb') as f:
            dill.dump(save_dict, f)
        
        if self.verbose:
            print(f"Saved data to: {filepath}")
        return filename
    
    def save_multiple_dicts(self, *dicts: Dict[str, Any],
                            metadata: Dict[str, Any] = None,
                            **named_data: Any) -> str:
        """
        Save multiple named entries in a single file.
        
        Parameters
        ----------
        *dicts : variable number of dict arguments
            Dictionaries containing named data to be saved
        **named_data : Any
            Named data entries to be saved
        metadata : dict, optional
            Additional metadata to include with the data
            
        Returns
        -------
        str
            The filename where data was saved
        """
        return self.save_data(*dicts, metadata=metadata, **named_data)
    
    def get_latest_filename(self) -> str:
        """
        Get the filename of the most recently saved data file.
        
        Returns
        -------
        str or None
            The latest filename, or None if no files exist
        """
        if self.file_counter == 0:
            return None
        
        number_str = f"{self.file_counter:0{self.padding}d}"
        return f"{self.prefix}{number_str}.dill"
    
    def get_file_count(self) -> int:
        """
        Get the total number of data files saved.
        
        Returns
        -------
        int
            The number of data files
        """
        return self.file_counter
    
    def load_data(self, filename: str = None) -> Dict[str, Any]:
        """
        Load data from a specific file.
        
        Parameters
        ----------
        filename : str, optional
            The filename to load. If None, loads the latest file.
            
        Returns
        -------
        dict
            The loaded data with a 'meta' entry
        """
        if filename is None:
            filename = self.get_latest_filename()
            if filename is None:
                raise FileNotFoundError("No data files found")
        
        filepath = self._get_filepath(filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            loaded = dill.load(f)

        data = self._extract_payload(loaded)
        meta: Dict[str, Any] = {}
        if isinstance(loaded, dict):
            if isinstance(loaded.get('meta'), dict):
                meta = loaded.get('meta', {})
            else:
                for key in ('timestamp', 'file_number', 'filename', 'metadata'):
                    if key in loaded:
                        meta[key] = loaded[key]
        if isinstance(data, dict):
            result = dict(data)
            result['meta'] = meta
            return result

        return {'data': data, 'meta': meta}
    
    def get_all_filenames(self) -> List[str]:
        """
        Get all data filenames in the storage folder.
        
        Returns
        -------
        list of str
            List of all data filenames
        """
        filenames = []
        for filename in os.listdir(self.base_folder):
            if filename.startswith(self.prefix):
                if filename.endswith('.dill'):
                    filenames.append(filename)
        
        return natsorted(filenames)

    def iter_data(self, filenames: List[str] = None):
        """
        Iterate over stored data in natural filename order.

        Parameters
        ----------
        filenames : list of str, optional
            Specific filenames to load. If None, all files are loaded.
        """
        if filenames is None:
            filenames = self.get_all_filenames()
        else:
            filenames = natsorted(filenames)

        for filename in filenames:
            yield self.load_data(filename)

    def get_data_keys(self, filenames: List[str] = None) -> Dict[str, List[Any]]:
        """
        Get the keys present in stored data for each file.

        Parameters
        ----------
        filenames : list of str, optional
            Specific filenames to inspect. If None, all files are inspected.

        Returns
        -------
        dict
            Mapping of filename to list of keys found in the data payload.
        """
        if filenames is None:
            filenames = self.get_all_filenames()
        else:
            filenames = natsorted(filenames)

        results: Dict[str, List[Any]] = {}
        for filename in filenames:
            loaded = self.load_data(filename)
            data = self._payload_from_loaded(loaded)
            keys: List[Any] = []

            if isinstance(data, dict):
                keys = list(data.keys())
            elif isinstance(data, list):
                seen = set()
                for item in data:
                    if isinstance(item, dict):
                        for key in item.keys():
                            if key not in seen:
                                seen.add(key)
                                keys.append(key)

            results[filename] = keys

        return results

    @staticmethod
    def _extract_payload(loaded: Dict[str, Any]) -> Any:
        """Extract the data payload from a raw loaded file."""
        if isinstance(loaded, dict) and 'data' in loaded:
            return loaded.get('data')
        if isinstance(loaded, dict):
            return {key: val for key, val in loaded.items() if key != 'meta'}
        return loaded

    @staticmethod
    def _payload_from_loaded(loaded: Dict[str, Any]) -> Any:
        """Extract the payload from a load_data() result."""
        if 'data' in loaded and set(loaded.keys()) <= {'data', 'meta'}:
            return loaded.get('data')
        return {key: val for key, val in loaded.items() if key != 'meta'}

    def reset_counter(self):
        """Reset the file counter to start numbering from 0 again."""
        self.file_counter = 0
        if self.verbose:
            print("File counter reset to 0")

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the current storage state.

        Returns
        -------
        dict
            Information about the storage
        """
        return {
            'base_folder': self.base_folder,
            'prefix': self.prefix,
            'padding': self.padding,
            'file_count': self.file_counter,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'latest_file': self.get_latest_filename()
        }


class DataReader:
    """
    A read-only helper for loading data files without modifying storage.

    Features:
    - Never creates or deletes folders
    - Reads dill-serialized files in natural order
    - Supports iteration and key inspection
    """

    def __init__(self, base_folder: str = "data", prefix: str = "data", padding: int = 5):
        """
        Initialize the DataReader object.

        Parameters
        ----------
        base_folder : str
            The base folder where data is stored
        prefix : str
            The prefix for data files (default: 'data')
        padding : int
            The number of digits for file numbering with zero-padding
        """
        self.base_folder = path.join(Settings.data_folder, base_folder)
        self.prefix = prefix
        self.padding = padding

        if not os.path.exists(self.base_folder):
            raise FileNotFoundError(f"Data storage folder not found: {self.base_folder}")

    def _get_filepath(self, filename: str) -> str:
        """Get the full filepath for a given filename."""
        return os.path.join(self.base_folder, filename)

    def get_all_filenames(self) -> List[str]:
        """
        Get all data filenames in the storage folder.

        Returns
        -------
        list of str
            List of all data filenames
        """
        filenames = []
        for filename in os.listdir(self.base_folder):
            if filename.startswith(self.prefix) and filename.endswith('.dill'):
                filenames.append(filename)

        return natsorted(filenames)

    def load_data(self, filename: str) -> Dict[str, Any]:
        """
        Load data from a specific file.

        Parameters
        ----------
        filename : str
            The filename to load.

        Returns
        -------
        dict
            The loaded data with a 'meta' entry
        """
        filepath = self._get_filepath(filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'rb') as f:
            loaded = dill.load(f)

        data = self._extract_payload(loaded)
        meta: Dict[str, Any] = {}
        if isinstance(loaded, dict):
            if isinstance(loaded.get('meta'), dict):
                meta = loaded.get('meta', {})
            else:
                for key in ('timestamp', 'file_number', 'filename', 'metadata'):
                    if key in loaded:
                        meta[key] = loaded[key]
        if isinstance(data, dict):
            result = dict(data)
            result['meta'] = meta
            return result

        return {'data': data, 'meta': meta}

    def iter_data(self, filenames: List[str] = None):
        """
        Iterate over stored data in natural filename order.

        Parameters
        ----------
        filenames : list of str, optional
            Specific filenames to load. If None, all files are loaded.
        """
        if filenames is None:
            filenames = self.get_all_filenames()
        else:
            filenames = natsorted(filenames)

        for filename in filenames:
            yield self.load_data(filename)

    def get_data_at(self, index: int):
        """
        Load data by index in natural filename order.

        Parameters
        ----------
        index : int
            Zero-based index into the natural-sorted filenames list

        Returns
        -------
        dict
            The loaded data with a 'meta' entry
        """
        filenames = self.get_all_filenames()
        return self.load_data(filenames[index])

    def get_data_keys(self, filenames: List[str] = None) -> Dict[str, List[Any]]:
        """
        Get the keys present in stored data for each file.

        Parameters
        ----------
        filenames : list of str, optional
            Specific filenames to inspect. If None, all files are inspected.

        Returns
        -------
        dict
            Mapping of filename to list of keys found in the data payload.
        """
        if filenames is None:
            filenames = self.get_all_filenames()
        else:
            filenames = natsorted(filenames)

        results: Dict[str, List[Any]] = {}
        for filename in filenames:
            loaded = self.load_data(filename)
            data = self._payload_from_loaded(loaded)
            keys: List[Any] = []

            if isinstance(data, dict):
                keys = list(data.keys())
            elif isinstance(data, list):
                seen = set()
                for item in data:
                    if isinstance(item, dict):
                        for key in item.keys():
                            if key not in seen:
                                seen.add(key)
                                keys.append(key)

            results[filename] = keys

        return results

    def get_data_overview(self) -> Dict[str, Any]:
        """
        Get a structural overview of the data payload based on the first file.

        Returns
        -------
        dict
            A structure summary of the data in the first file.
        """
        filenames = self.get_all_filenames()
        if not filenames:
            return {}

        first = self.load_data(filenames[0])
        payload = self._payload_from_loaded(first)
        return self._describe_structure(payload)

    def get_field(self, *names: str, filenames: List[str] = None) -> List[Any]:
        """
        Get a nested field value from each stored file.

        Parameters
        ----------
        *names : str
            Path of keys to traverse (e.g., name1, name2, ...)
        filenames : list of str, optional
            Specific filenames to load. If None, all files are loaded.

        Returns
        -------
        list
            Values for the requested field from each file
        """
        if not names:
            raise ValueError("At least one field name is required")

        if filenames is None:
            filenames = self.get_all_filenames()
        else:
            filenames = natsorted(filenames)

        values: List[Any] = []
        for filename in filenames:
            current = self.load_data(filename)
            for name in names:
                if not isinstance(current, dict):
                    raise TypeError(f"Field '{name}' expects a dict, got {type(current).__name__}")
                current = current[name]
            values.append(current)

        return values

    @staticmethod
    def _extract_payload(loaded: Dict[str, Any]) -> Any:
        """Extract the data payload from a raw loaded file."""
        if isinstance(loaded, dict) and 'data' in loaded:
            return loaded.get('data')
        if isinstance(loaded, dict):
            return {key: val for key, val in loaded.items() if key != 'meta'}
        return loaded

    @staticmethod
    def _payload_from_loaded(loaded: Dict[str, Any]) -> Any:
        """Extract the payload from a load_data() result."""
        if 'data' in loaded and set(loaded.keys()) <= {'data', 'meta'}:
            return loaded.get('data')
        return {key: val for key, val in loaded.items() if key != 'meta'}

    def print_data_overview(self):
        """
        Print a human-readable overview of the data payload structure.
        """
        overview = self.get_data_overview()
        if not overview:
            print("No data files found.")
            return

        print("Data overview (from first file):")
        self._print_structure(overview)

    @staticmethod
    def _describe_structure(value: Any) -> Any:
        """Describe nested structures using dict keys and type names."""
        if isinstance(value, list):
            if not value:
                return []
            if all(isinstance(item, dict) for item in value):
                merged: Dict[Any, Any] = {}
                for item in value:
                    for key, val in item.items():
                        described = DataReader._describe_structure(val)
                        if key in merged:
                            merged[key] = DataReader._merge_structures(merged[key], described)
                        else:
                            merged[key] = described
                return [merged]
            if all(not isinstance(item, dict) for item in value):
                described_items = [DataReader._describe_structure(item) for item in value]
                return [DataReader._merge_many(described_items)]
            return [DataReader._merge_many(
                [DataReader._describe_structure(item) for item in value]
            )]
        if isinstance(value, dict):
            return {key: DataReader._describe_structure(val) for key, val in value.items()}
        return type(value).__name__

    @staticmethod
    def _merge_structures(left: Any, right: Any) -> Any:
        """Merge two structure descriptions into a single summary."""
        if left == right:
            return left
        if isinstance(left, dict) and isinstance(right, dict):
            merged = dict(left)
            for key, val in right.items():
                if key in merged:
                    merged[key] = DataReader._merge_structures(merged[key], val)
                else:
                    merged[key] = val
            return merged
        if isinstance(left, list) and isinstance(right, list):
            if not left:
                return right
            if not right:
                return left
            return [DataReader._merge_structures(left[0], right[0])]
        return DataReader._merge_many([left, right])

    @staticmethod
    def _merge_many(items: List[Any]) -> Any:
        """Merge many structure descriptions into a single summary."""
        unique = []
        for item in items:
            if item not in unique:
                unique.append(item)
        if len(unique) == 1:
            return unique[0]
        if all(isinstance(item, str) for item in unique):
            return " | ".join(sorted(unique))
        return unique

    @staticmethod
    def _print_structure(value: Any, indent: int = 0):
        """Print a structure summary with indentation."""
        prefix = "  " * indent
        if isinstance(value, dict):
            for key, val in value.items():
                if isinstance(val, (dict, list)):
                    print(f"{prefix}{key}:")
                    DataReader._print_structure(val, indent + 1)
                else:
                    print(f"{prefix}{key}: {val}")
        elif isinstance(value, list):
            print(f"{prefix}- list")
            if value:
                DataReader._print_structure(value[0], indent + 1)
        else:
            print(f"{prefix}{value}")
    
