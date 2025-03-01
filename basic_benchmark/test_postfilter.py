import os
import sys
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(sys.path)

from controller.initialize_main_tables import drop_indexes, create_indexes
from services.config import get_db_connection
from basic_benchmark.common_function import prepare_query_dataset, run_search_experiment, drop_extra_tables, run_test, \
    get_index_type


def test_postfilter(iterations=3, enable_index=False, index_type="ivfflat", statistics_type="sql",
                    generator_type="tree-based"):
    current_index_type = get_index_type("documentblocks")

    if enable_index:
        if current_index_type != index_type:
            # If the existing index type doesn't match, drop and recreate the index
            print(f"Index type {current_index_type} does not match requested type {index_type}, recreating index.")
            drop_indexes()

        create_indexes()
    else:
        drop_indexes()

    # Generate queries
    queries = prepare_query_dataset(regenerate=False, num_queries=1000)

    run_test(queries, iterations=iterations, condition=f"postfilter_multiple_round_alg1",
             statistics_type=statistics_type, generator_type=generator_type, enable_index=enable_index,
             index_type=index_type)
    run_test(queries, iterations=iterations, condition=f"postfilter_multiple_round_alg2",
             statistics_type=statistics_type, generator_type=generator_type, enable_index=enable_index,
             index_type=index_type)


if __name__ == '__main__':
    test_postfilter()
