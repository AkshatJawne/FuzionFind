import os
import threading
import torch
import math
import numpy as np
from numba import cuda
from collections import defaultdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sklearn.ensemble import GradientBoostingRegressor
from torch import nn

# --- CUDA Kernel for Levenshtein Distance ---
# This function runs on the GPU to compute edit distances in parallel
@cuda.jit
def levenshtein_kernel(query, filenames, results):
    """
    CUDA kernel that calculates Levenshtein distance between query and filenames.
    This runs in parallel on the GPU for massive performance improvement.
    
    Args:
        query: The search query string
        filenames: Array of filenames to compare against
        results: Output array to store the calculated distances
    """
    # Get the current thread ID
    i = cuda.grid(1)
    
    # Only process if thread ID is within range of filenames
    if i < len(filenames):
        q_len = len(query)
        f_len = len(filenames[i])
        
        # Create a matrix for dynamic programming
        d = np.zeros((q_len + 1, f_len + 1), dtype=np.int32)
        
        # Initialize the matrix
        for x in range(q_len + 1):
            for y in range(f_len + 1):
                if x == 0:
                    # If query is empty, cost is insertions
                    d[x, y] = y
                elif y == 0:
                    # If filename is empty, cost is deletions
                    d[x, y] = x
                else:
                    # Calculate cost - 0 if characters match, 1 if they don't
                    cost = 0 if query[x - 1] == filenames[i][y - 1] else 1
                    
                    # Choose minimum of delete, insert, or substitute operations
                    d[x, y] = min(d[x - 1, y] + 1,        # Deletion
                                 d[x, y - 1] + 1,         # Insertion
                                 d[x - 1, y - 1] + cost)  # Substitution
        
        # Store the final distance in the results array
        results[i] = d[q_len, f_len]

# --- Machine Learning Model for Query Ranking ---
class SearchRanker(nn.Module):
    """
    Neural network model that ranks search results using multiple features.
    Takes distance score, TF-IDF score, and filename length as inputs.
    """
    def __init__(self):
        """Initialize the neural network with three layers"""
        super(SearchRanker, self).__init__()
        self.layer1 = nn.Linear(3, 64)   # Input layer: 3 features
        self.layer2 = nn.Linear(64, 32)  # Hidden layer
        self.output = nn.Linear(32, 1)   # Output layer: single score

    def forward(self, x):
        """Forward pass through the network"""
        x = torch.relu(self.layer1(x))  # Apply ReLU activation
        x = torch.relu(self.layer2(x))  # Apply ReLU activation
        return self.output(x)           # Return final score

# Initialize the ranking model
ranker = SearchRanker()

# --- File Indexing ---
# Set to store all indexed file paths
file_index = set()
# Dictionary to store TF-IDF scores for each file
tf_idf_scores = defaultdict(float)
# Thread lock for concurrent access to the file index
lock = threading.Lock()

def build_file_index(root):
    """
    Build an index of all files on the system and calculate TF-IDF scores.
    
    Args:
        root: The root directory to start indexing from
    """
    global file_index, tf_idf_scores
    file_index.clear()
    tf_idf_scores = defaultdict(float)
    file_count = defaultdict(int)  # Count occurrences of each token

    def scan_directory(directory):
        """
        Recursively scan a directory for files.
        
        Args:
            directory: Directory path to scan
            
        Returns:
            List of file paths
        """
        local_files = []
        try:
            # Walk through directory tree
            for root, _, files in os.walk(directory, followlinks=False):
                for file in files:
                    full_path = os.path.join(root, file)
                    local_files.append(full_path)
        except (PermissionError, FileNotFoundError):
            # Skip directories we can't access
            pass
        return local_files

    # Choose starting directories based on OS
    directories = ["/"] if os.name != "nt" else ["C:\\", "D:\\"]
    
    # Scan all directories and collect files
    for files in [scan_directory(d) for d in directories]:
        for file in files:
            file_index.add(file)
            # Count token occurrences across all files
            for token in os.path.basename(file).lower().split():
                file_count[token] += 1

    # Calculate TF-IDF scores
    total_files = len(file_index)
    for file in file_index:
        for token in os.path.basename(file).lower().split():
            # Term frequency - how often the token appears
            tf = file_count[token] / total_files
            # Inverse document frequency - rarity of the token across all files
            idf = math.log(total_files / (1 + file_count[token]))
            # Add the TF-IDF score for this token to the file's total score
            tf_idf_scores[file] += tf * idf

class FileChangeHandler(FileSystemEventHandler):
    """
    Handler for file system events to keep the index updated.
    Inherits from watchdog's FileSystemEventHandler.
    """
    def on_created(self, event):
        """Add newly created files to the index"""
        if not event.is_directory:
            with lock:
                file_index.add(event.src_path)

    def on_deleted(self, event):
        """Remove deleted files from the index"""
        if not event.is_directory:
            with lock:
                file_index.discard(event.src_path)

def start_watcher():
    """
    Start monitoring the file system for changes.
    
    Returns:
        Observer object that's monitoring the file system
    """
    observer = Observer()
    handler = FileChangeHandler()
    # Schedule the observer to watch root directory recursively
    observer.schedule(handler, "/", recursive=True)
    observer.start()
    return observer

# --- CUDA + ML-Enhanced Fuzzy Matching ---
def fuzzy_match(query):
    """
    Perform fuzzy matching against the file index using CUDA acceleration
    and machine learning ranking.
    
    Args:
        query: Search string
        
    Returns:
        List of top 10 matching file paths
    """
    with lock:
        # If query is empty, return the first 10 files
        if not query:
            return list(file_index)[:10]

        # Convert file index to list for CUDA processing
        filenames = list(file_index)
        # Initialize results array
        results = np.zeros(len(filenames), dtype=np.int32)
        
        # Transfer data to GPU
        d_query = cuda.to_device(query)
        d_filenames = cuda.to_device(np.array(filenames, dtype=np.str_))
        d_results = cuda.to_device(results)

        # Configure CUDA grid dimensions
        threads_per_block = 256
        blocks_per_grid = (len(filenames) + (threads_per_block - 1)) // threads_per_block
        
        # Launch CUDA kernel
        levenshtein_kernel[blocks_per_grid, threads_per_block](d_query, d_filenames, d_results)

        # Wait for GPU to finish
        cuda.synchronize()
        
        # Copy results back from GPU
        distances = d_results.copy_to_host()

        # Calculate scores using multiple factors
        scores = []
        for i, file in enumerate(filenames):
            # Lower distance means better match, so we negate it
            distance_score = -distances[i]
            # Get TF-IDF score for this file
            tfidf_score = tf_idf_scores.get(file, 0)

            # Create feature vector for ML model
            feature_vector = torch.tensor([[distance_score, tfidf_score, len(file)]], dtype=torch.float32)
            # Get ranking score from neural network
            ml_score = ranker(feature_vector).item()

            scores.append((file, ml_score))

        # Sort by score (descending) and take top 10
        return [x[0] for x in sorted(scores, key=lambda x: -x[1])][:10]

# --- Search CLI ---
def fuzzy_search_cli():
    """
    Command line interface for fuzzy search.
    Loops indefinitely until user quits.
    """
    print("Welcome to CUDA-Accelerated Fuzzy File Search with ML!")
    while True:
        # Get user input
        query = input("Enter search query (or 'q' to quit): ").strip()
        if query.lower() == "q":
            break
            
        # Perform search
        results = fuzzy_match(query)
        
        # Display results
        print("\nTop Matches:")
        for idx, res in enumerate(results):
            print(f"{idx + 1}. {res}")
        print("\n")

if __name__ == "__main__":
    # Entry point of the program
    print("Indexing files, please wait...")
    build_file_index("/")
    print("Indexing complete. Starting search interface...")
    start_watcher()
    fuzzy_search_cli()