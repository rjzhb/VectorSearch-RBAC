import os
import sys
import json
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(sys.path)
from services.config import get_db_connection
from controller.initialize_main_tables import create_indexes, drop_indexes
from basic_benchmark.common_function import drop_extra_tables, prepare_query_dataset, run_search_experiment, run_test, \
    get_index_type
from basic_benchmark.space_calculate import calculate_postfilter, calculate_rls
from controller.baseline.pg_row_security.row_level_security import disable_row_level_security, drop_database_users, \
    create_database_users, enable_row_level_security, search_documents_rls


def test_row_level_security(iterations=3, enable_index=False, index_type="ivfflat", statistics_type="sql",
                            generator_type="tree-based", warm_up=True):
    current_index_type = get_index_type("documentblocks")
    if enable_index:
        if current_index_type is not None and current_index_type != index_type:
            # If the existing index type doesn't match, drop and recreate the index
            print(f"Index type {current_index_type} does not match requested type {index_type}, recreating index.")
            drop_indexes()
        start_time = time.time()
        create_indexes(index_type)
        time_taken = time.time() - start_time
        print(f"Time taken to build index: {time_taken:.2f} seconds")
        with open("non-indexed_time.json", "w") as json_file:
            json.dump(time_taken, json_file, indent=4)

    else:
        drop_indexes()

    disable_row_level_security()
    drop_database_users()

    start_time = time.time()

    create_database_users()
    enable_row_level_security()

    init_time = time.time() - start_time
    print(f"Time taken to init: {init_time:.2f} seconds")

    start_time = time.time()
    # Generate queries
    queries = prepare_query_dataset(regenerate=False, num_queries=1000)
    elapsed_time = time.time() - start_time
    print(f"Time taken to generate query: {elapsed_time:.2f} seconds")

    start_time = time.time()

    run_test(queries, iterations=iterations, condition="row_level_security", statistics_type=statistics_type,
             generator_type=generator_type, enable_index=enable_index, index_type=index_type, warm_up=warm_up)

    elapsed_time = time.time() - start_time
    print(f"Time taken to query: {elapsed_time:.2f} seconds")

    # disable_row_level_security()
    # drop_database_users()


if __name__ == '__main__':
    test_row_level_security()
