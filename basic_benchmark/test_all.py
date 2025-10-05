
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from services.logger import get_logger

logger = get_logger(__name__)
logger.debug("sys.path=%s", sys.path)
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
    parser.add_argument('--algorithm', choices=['RLS', 'ROLE', 'USER', 'AnonySys'], required=True,
                        help="Select which test to run: RLS, ROLE, USER, or AnonySys")
    parser.add_argument('--efs', type=int, nargs='+', required=True,
                        help="List of EF search values to use (space-separated integers)")
    parser.add_argument('--shared-vector-table', choices=['on', 'off'], dest='shared_vector_table',
                        help="Override hnsw.shared_vector_table for this run")
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
    logger.info("Test Type: %s", test_type)
    logger.info("EF Search Values: %s", ef_search_values)
    logger.info("Index Type: %s", index_type)
    logger.info("Enable Index: %s", enable_index)

    if args.shared_vector_table:
        pgoptions = os.environ.get("PGOPTIONS", "").strip()
        override = f"-c hnsw.shared_vector_table={args.shared_vector_table}"
        if pgoptions:
            if override not in pgoptions:
                pgoptions = f"{pgoptions} {override}"
        else:
            pgoptions = override
        os.environ["PGOPTIONS"] = pgoptions
        logger.info("Applied PGOPTIONS override: %s", os.environ["PGOPTIONS"])

    if test_type == 'RLS':
        for ef in ef_search_values:
            efconfig.ef_search = ef
            logger.info("Running RLS test with ef_search=%s", ef)
            test_row_level_security(iterations=1,
                                    enable_index=enable_index,
                                    statistics_type="sql",
                                    index_type=index_type,
                                    generator_type=generator_type,
                                    warm_up=True)

    elif test_type == 'ROLE':
        for ef in ef_search_values:
            efconfig.ef_search = ef
            logger.info("Running ROLE test with ef_search=%s", ef)
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
            logger.info("Running USER test with ef_search=%s", ef)
            test_partition_prefilter_by_combination_role(
                iterations=1,
                enable_index=enable_index,
                index_type=index_type,
                statistics_type="sql",
                generator_type=generator_type,
                record_recall=True,
                warm_up=True
            )

    elif test_type == 'AnonySys':
        for ef in ef_search_values:
            efconfig.ef_search = ef
            logger.info("Running AnonySys test with ef_search=%s", ef)
            test_dynamic_partition_search(iterations=1,
                                          enable_index=enable_index,
                                          statistics_type="sql",
                                          index_type=index_type,
                                          generator_type=generator_type,
                                          record_recall=True,
                                          warm_up=True)
