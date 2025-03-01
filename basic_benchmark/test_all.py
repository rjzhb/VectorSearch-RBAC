
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(sys.path)
from basic_benchmark.test_partition_prefilter_by_combination_role import test_partition_prefilter_by_combination_role
from basic_benchmark.test_partition_prefilter_by_role import test_partition_prefilter_role

from basic_benchmark.test_dynamic_partition import test_dynamic_partition_search
from basic_benchmark.test_row_level_security import test_row_level_security
import efconfig


#python script.py --test RLS --ef_values 350 400 450
if __name__ == '__main__':
    import argparse

    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Run benchmark tests with partition and index strategies.")

    # Add arguments for test selection and EF values
    parser.add_argument('--algorithm', choices=['RLS', 'ROLE', 'USER', 'HONEYBEE'], required=True,
                        help="Select which test to run: RLS, ROLE, USER, or HONEYBEE")
    parser.add_argument('--efs', type=int, nargs='+', required=True,
                        help="List of EF search values to use (space-separated integers)")
    # Parse the arguments
    args = parser.parse_args()
    # Set fixed values
    enable_index = True
    index_type = 'hnsw'
    generator_type = ''

    # Get test type and EF values from arguments
    test_type = args.algorithm
    ef_search_values = args.efs

    # Inform the user of the chosen settings
    print(f"Test Type: {test_type}")
    print(f"EF Search Values: {ef_search_values}")
    print(f"Index Type: {index_type}")
    print(f"Enable Index: {enable_index}")
    import os

    if test_type == 'RLS':
        for ef in ef_search_values:
            efconfig.ef_search = ef
            print(f"Running RLS test with ef_search = {ef}")
            test_row_level_security(iterations=1,
                                    enable_index=enable_index,
                                    statistics_type="sql",
                                    index_type=index_type,
                                    generator_type=generator_type,
                                    warm_up=True)

    elif test_type == 'ROLE':
        for ef in ef_search_values:
            efconfig.ef_search = ef
            print(f"Running ROLE test with ef_search = {ef}")
            test_partition_prefilter_role(
                iterations=1,
                enable_index=enable_index,
                index_type=index_type,
                statistics_type="sql",
                generator_type=generator_type,
                record_recall=True,
                warm_up=True
            )

    elif test_type == 'USER':
        for ef in ef_search_values:
            efconfig.ef_search = ef
            print(f"Running USER test with ef_search = {ef}")
            test_partition_prefilter_by_combination_role(
                iterations=1,
                enable_index=enable_index,
                index_type=index_type,
                statistics_type="sql",
                generator_type=generator_type,
                record_recall=True,
                warm_up=True
            )

    elif test_type == 'HONEYBEE':
        for ef in ef_search_values:
            efconfig.ef_search = ef
            print(f"Running HONEYBEE test with ef_search = {ef}")
            test_dynamic_partition_search(iterations=1,
                                          enable_index=enable_index,
                                          statistics_type="sql",
                                          index_type=index_type,
                                          generator_type=generator_type,
                                          record_recall=True,
                                          warm_up=True)
