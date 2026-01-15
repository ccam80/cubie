"""Disk-based caching for parsed CellML objects.

This module manages serialization and deserialization of parsed CellML
data structures using pickle. Cache files are stored in the generated/
directory alongside compiled code, with hash-based invalidation.
"""

from pathlib import Path
from typing import Optional
import pickle
from hashlib import sha256

from cubie.odesystems.symbolic.odefile import GENERATED_DIR
from cubie.odesystems.symbolic.parsing.parser import ParsedEquations
from cubie.odesystems.symbolic.indexedbasemaps import IndexedBases
from cubie._utils import PrecisionDType
from cubie.time_logger import default_timelogger


class CellMLCache:
    """Manage disk-based caching of parsed CellML objects.
    
    Cache files are stored at generated/<model_name>/cellml_cache.pkl
    with the first line containing a SHA256 hash of the source CellML
    file for validation. The remainder of the file is a pickled
    dictionary containing ParsedEquations, IndexedBases, and related
    metadata needed to reconstruct SymbolicODE without re-parsing.
    """
    
    def __init__(self, model_name: str, cellml_path: str) -> None:
        """Initialize cache manager for a CellML model.
        
        Parameters
        ----------
        model_name : str
            Name used for cache directory (typically cellml filename stem)
        cellml_path : str
            Path to source CellML file
        
        Raises
        ------
        TypeError
            If model_name or cellml_path are not strings
        ValueError
            If model_name is empty string
        FileNotFoundError
            If cellml_path does not exist
        """
        # Validate model_name type
        if not isinstance(model_name, str):
            raise TypeError(
                f"model_name must be str, got {type(model_name).__name__}"
            )
        
        # Validate cellml_path type
        if not isinstance(cellml_path, str):
            raise TypeError(
                f"cellml_path must be str, got "
                f"{type(cellml_path).__name__}"
            )
        
        # Validate model_name is not empty
        if not model_name:
            raise ValueError("model_name cannot be empty string")
        
        # Validate cellml_path exists
        cellml_path_obj = Path(cellml_path)
        if not cellml_path_obj.exists():
            raise FileNotFoundError(
                f"CellML file not found: {cellml_path}"
            )
        
        self.model_name = model_name
        self.cellml_path = cellml_path
        self.cache_dir = GENERATED_DIR / model_name
        self.cache_file = self.cache_dir / "cellml_cache.pkl"
    
    def get_cellml_hash(self) -> str:
        """Compute SHA256 hash of CellML file content.
        
        Reads entire file content and computes hash for cache validation.
        Whitespace changes will change the hash.
        
        Returns
        -------
        str
            Hexadecimal hash string (64 characters)
        
        Raises
        ------
        FileNotFoundError
            If CellML file has been deleted since initialization
        IOError
            If file cannot be read
        """
        # Read file content in binary mode for hashing
        with open(self.cellml_path, 'rb') as f:
            content = f.read()
        
        # Compute SHA256 hash
        hash_obj = sha256(content)
        return hash_obj.hexdigest()
    
    def cache_valid(self) -> bool:
        """Check if cache file exists and hash matches current file.
        
        Reads first line of cache file and compares against current
        CellML file hash. Returns False if cache doesn't exist or
        hash mismatch.
        
        Returns
        -------
        bool
            True if cache exists and is current, False otherwise
        """
        # Check cache file exists
        if not self.cache_file.exists():
            return False
        
        try:
            # Read first line (hash comment)
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            
            # Extract hash (remove # prefix if present)
            stored_hash = first_line.lstrip('#')
            
            # Compute current hash
            current_hash = self.get_cellml_hash()
            
            # Compare hashes
            return stored_hash == current_hash
        
        except Exception:
            # Any error reading cache = invalid cache
            return False
    
    def load_from_cache(self) -> Optional[dict]:
        """Load cached parse results from disk.
        
        Reads pickled data from cache file. First line is hash comment,
        remainder is pickled dictionary. Returns None if cache invalid
        or unpickling fails.
        
        Returns
        -------
        dict or None
            Dictionary with keys: 'cellml_hash', 'parsed_equations',
            'indexed_bases', 'all_symbols', 'user_functions', 'fn_hash',
            'precision', 'name'. Returns None if load fails.
        """
        # Validate cache before attempting load
        if not self.cache_valid():
            return None
        
        try:
            # Read file, skip first line (hash comment)
            with open(self.cache_file, 'rb') as f:
                # Read and discard first line
                _ = f.readline()
                # Unpickle remaining content
                cached_data = pickle.load(f)
            
            # Verify expected keys present
            required_keys = {
                'cellml_hash', 'parsed_equations', 'indexed_bases',
                'all_symbols', 'user_functions', 'fn_hash',
                'precision', 'name'
            }
            if not all(key in cached_data for key in required_keys):
                default_timelogger.print_message(
                    "Cache file missing required keys, will re-parse"
                )
                return None
            
            return cached_data
        
        except pickle.UnpicklingError as e:
            default_timelogger.print_message(
                f"Cache unpickling failed: {e}, will re-parse"
            )
            return None
        except Exception as e:
            default_timelogger.print_message(
                f"Cache load error: {e}, will re-parse"
            )
            return None
    
    def save_to_cache(
        self,
        parsed_equations: ParsedEquations,
        indexed_bases: IndexedBases,
        all_symbols: dict,
        user_functions: Optional[dict],
        fn_hash: str,
        precision: PrecisionDType,
        name: str,
    ) -> None:
        """Save parse results to cache file.
        
        Creates cache directory if needed, writes hash comment as first
        line, then pickles data dictionary. Silently continues if write
        fails (caching is opportunistic).
        
        Parameters
        ----------
        parsed_equations : ParsedEquations
            Equation container from parse_input
        indexed_bases : IndexedBases
            Index maps from parse_input
        all_symbols : dict
            Symbol mapping from parse_input
        user_functions : dict or None
            User-provided functions (may be None)
        fn_hash : str
            System hash from parse_input
        precision : PrecisionDType
            Floating-point precision
        name : str
            Model name
        """
        try:
            # Create cache directory if needed
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Compute current file hash
            cellml_hash = self.get_cellml_hash()
            
            # Build cache dictionary
            cache_data = {
                'cellml_hash': cellml_hash,
                'parsed_equations': parsed_equations,
                'indexed_bases': indexed_bases,
                'all_symbols': all_symbols,
                'user_functions': user_functions,
                'fn_hash': fn_hash,
                'precision': precision,
                'name': name,
            }
            
            # Write cache file: hash comment + pickled data
            with open(self.cache_file, 'wb') as f:
                # Write hash as first line (text mode for comment)
                f.write(f"#{cellml_hash}\n".encode('utf-8'))
                # Pickle data
                pickle.dump(
                    cache_data, f, protocol=pickle.HIGHEST_PROTOCOL
                )
        
        except PermissionError as e:
            default_timelogger.print_message(
                f"Cannot write cache (permission denied): {e}"
            )
        except Exception as e:
            default_timelogger.print_message(
                f"Cache save failed: {e}"
            )
