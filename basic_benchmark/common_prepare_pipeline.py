import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(sys.path)

from controller.prepare_database import clear_db
from controller.initialize_main_tables import initialize_database_deduplication
from services.read_dataset_function import read_and_store_dataset_parallel
if __name__ == '__main__':
    # init database
    clear_db()
    initialize_database_deduplication(enable_index=True)
    start_time = time.time()
    print(os.cpu_count())
    read_and_store_dataset_parallel(load_number=1000000, start_row=0, num_threads=os.cpu_count(), dataset="wikipedia-22-12")
    # read_and_store_dataset_parallel(load_number=26151830, start_row=0, num_threads=os.cpu_count(), dataset="arxiv")
    elapsed_time = time.time() - start_time
    print(f"Time taken to execute read_dataset_deduplication_rbac: {elapsed_time:.2f} seconds")