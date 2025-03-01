import os
import sys
import json
import time
from psycopg2 import sql
import importlib
from typing import Callable

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(sys.path)

from basic_benchmark.condition_config import CONDITION_CONFIG
from controller.baseline.postfilter.postfilter import search_documents_with_postfilter_alg1
from controller.baseline.prefilter.initialize_partitions import initialize_user_partitions, initialize_role_partitions, \
    initialize_combination_partitions, drop_prefilter_partition_tables
from services.config import get_db_connection
from services.read_dataset_function import generate_query_dataset, load_queries_from_dataset


def get_nprobe_value(config_file="config_params.json"):
    # Open and read the configuration file
    benchmark_folder = os.path.join(project_root, "basic_benchmark")
    file = os.path.join(benchmark_folder, config_file)
    with open(file, "r") as file:
        config = json.load(file)

    # Return only the nprobe value, defaulting to 1 if not found
    return config.get("nprobe", 1)


def get_index_type(table_name):
    # Connect to your PostgreSQL database
    conn = get_db_connection()
    cursor = conn.cursor()

    # SQL query to fetch index information for the table
    query = f"""
    SELECT indexdef FROM pg_indexes WHERE tablename = '{table_name}';
    """
    cursor.execute(query)
    indexes = cursor.fetchall()

    # Example of finding the index type based on the index definition
    for index in indexes:
        index_definition = index[0].lower()
        if 'ivfflat' in index_definition:
            return 'ivfflat'
        elif 'hnsw' in index_definition:
            return 'hnsw'

    # If no matching index type is found, return None or a default value
    return None


def drop_extra_tables():
    drop_prefilter_partition_tables()


def predicate_prefilter(user_id, query_vector, topk=5, statistics_type="sql"):
    if statistics_type == "sql":
        return predicate_prefilter_statistics_sql(user_id, query_vector, topk)
    elif statistics_type == "system": \
            return predicate_prefilter_statistics_system(user_id, query_vector, topk)


def predicate_postfilter(user_id, query_vector, topk=5, statistics_type="sql"):
    if statistics_type == "sql":
        return predicate_postfilter_statistics_sql(user_id, query_vector, topk)
    elif statistics_type == "system": \
            return predicate_postfilter_statistics_system(user_id, query_vector, topk)


def predicate_prefilter_statistics_sql(user_id, query_vector, topk=5):
    probes = get_nprobe_value()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"SET ivfflat.probes = {probes};")

    total_query_time = 0  # Variable to accumulate the total SQL query time
    total_blocks_accessed = 0  # Variable to accumulate the total blocks accessed

    vector_str = query_vector

    # Step 1: Retrieve all document_ids the user has permission to access using EXPLAIN ANALYZE
    explain_query = """
        EXPLAIN ANALYZE
        SELECT DISTINCT pa.document_id 
        FROM PermissionAssignment pa
        JOIN UserRoles ur ON pa.role_id = ur.role_id
        WHERE ur.user_id = %s
    """
    cur.execute(explain_query, [user_id])
    explain_plan = cur.fetchall()

    # ----------------------save query plan-----------start
    import inspect
    current_function_name = inspect.currentframe().f_code.co_name

    # save_query_plan(explain_plan, current_function_name)
    # ----------------------save query plan-------------end

    # Parse the execution time from EXPLAIN ANALYZE
    for row in explain_plan:
        if "Execution Time" in row[0]:
            query_time = float(row[0].split()[-2]) / 1000  # Convert ms to seconds
            total_query_time += query_time

    # Fetch accessible document_ids without EXPLAIN ANALYZE
    cur.execute(
        """
        SELECT DISTINCT pa.document_id 
        FROM PermissionAssignment pa
        JOIN UserRoles ur ON pa.role_id = ur.role_id
        WHERE ur.user_id = %s
        """,
        [user_id]
    )
    accessible_document_ids = cur.fetchall()
    accessible_document_ids = [doc_id for (doc_id,) in accessible_document_ids]

    if not accessible_document_ids:
        cur.close()
        conn.close()
        print(f"Total SQL query time: {total_query_time} seconds")
        return []  # No accessible documents for this user

    # Step 2: Query the closest vectors among the accessible document blocks using EXPLAIN ANALYZE
    explain_query = sql.SQL(
        """
        EXPLAIN ANALYZE
        SELECT db.block_id, db.block_content, db.vector <-> %s AS distance
        FROM documentblocks db
        WHERE db.document_id = ANY(%s)
        ORDER BY distance
        LIMIT %s
        """
    )
    cur.execute(explain_query, [vector_str, accessible_document_ids, topk])
    explain_plan = cur.fetchall()

    # ----------------------save query plan-----------start
    import inspect
    current_function_name = inspect.currentframe().f_code.co_name

    # save_query_plan(explain_plan, current_function_name)
    # ----------------------save query plan-------------end

    # Parse the execution time from EXPLAIN ANALYZE
    for row in explain_plan:
        if "Execution Time" in row[0]:
            query_time = float(row[0].split()[-2]) / 1000  # Convert ms to seconds
            total_query_time += query_time
        elif "rows=" in row[0]:
            # This line typically appears in the EXPLAIN output indicating rows processed
            # Depending on the exact format, you might need to adjust the parsing
            try:
                blocks_accessed = int(row[0].split("rows=")[1].split(" ")[0])
                total_blocks_accessed += blocks_accessed
            except (IndexError, ValueError):
                pass  # Handle cases where parsing fails

    # Perform the actual query without EXPLAIN ANALYZE to fetch results
    query = sql.SQL(
        """
        SELECT db.block_id, db.block_content, db.vector <-> %s AS distance
        FROM documentblocks db
        WHERE db.document_id = ANY(%s)
        ORDER BY distance
        LIMIT %s
        """
    )
    cur.execute(query, [vector_str, accessible_document_ids, topk])

    results = cur.fetchall()

    cur.execute("SELECT COUNT(block_id) FROM documentblocks;")
    total_blocks = cur.fetchone()[0]

    # Calculate selectivity
    block_selectivity = total_blocks_accessed / total_blocks if total_blocks else 0

    cur.close()
    conn.close()

    return results, total_query_time, block_selectivity


def predicate_prefilter_statistics_system(user_id, query_vector, topk=5):
    """
    Perform pre-filtered vector similarity search based on user permissions.
    """

    probes = get_nprobe_value()
    start_time = time.time()  # Start system time tracking

    # Establish database connection
    conn = get_db_connection()
    cur = conn.cursor()

    # Set the IVF flat probes parameter
    cur.execute(f"SET ivfflat.probes = {probes};")

    # Step 1: Retrieve all accessible document block IDs for the user
    cur.execute(
        """
        SELECT DISTINCT db.block_id
        FROM documentblocks db
        JOIN PermissionAssignment pa ON db.document_id = pa.document_id
        JOIN UserRoles ur ON pa.role_id = ur.role_id
        WHERE ur.user_id = %s
        """,
        [user_id]
    )
    accessible_block_ids = cur.fetchall()
    accessible_block_ids = [block_id for (block_id,) in accessible_block_ids]

    if not accessible_block_ids:
        return [], time.time() - start_time  # No accessible blocks

    # Step 2: Perform vector similarity search within the filtered blocks
    query = sql.SQL(
        """
        SELECT block_id, block_content, vector <-> %s AS distance
        FROM (
            SELECT block_id, block_content, vector 
            FROM documentblocks 
            WHERE block_id = ANY(%s)
        ) AS filtered_blocks
        ORDER BY distance ASC
        LIMIT %s
        """
    )

    cur.execute(query, [query_vector, accessible_block_ids, topk])

    results = cur.fetchall()

    # Close the database connection
    cur.close()
    conn.close()

    return results, time.time() - start_time


#
# def predicate_prefilter_statistics_system(user_id, query_vector, topk=5):
#     probes = get_nprobe_value()
#     import time
#     start_time = time.time()  # Start system time tracking
#     conn = get_db_connection()
#     cur = conn.cursor()
#     cur.execute(f"SET ivfflat.probes = {probes};")
#     # Convert query vector to string for the SQL query
#     vector_str = query_vector
#
#     # Step 1: Retrieve all document_ids that the user has permission to access
#     cur.execute(
#         """
#         SELECT DISTINCT pa.document_id
#         FROM PermissionAssignment pa
#         JOIN UserRoles ur ON pa.role_id = ur.role_id
#         WHERE ur.user_id = %s
#         """,
#         [user_id]
#     )
#     accessible_document_ids = cur.fetchall()
#     accessible_document_ids = [doc_id for (doc_id,) in accessible_document_ids]
#
#     if not accessible_document_ids:
#         return []  # No accessible documents for this user
#
#     # Step 2: Query the closest vectors among the accessible document blocks
#     query = sql.SQL(
#         """
#         SELECT db.block_id, db.block_content, db.vector <-> %s AS distance
#         FROM documentblocks db
#         WHERE db.document_id = ANY(%s)
#         ORDER BY distance
#         LIMIT %s
#         """
#     )
#
#     cur.execute(query, [vector_str, accessible_document_ids, topk])
#
#     results = cur.fetchall()
#     cur.close()
#     conn.close()
#
#     return results, time.time() - start_time


def predicate_postfilter_statistics_sql(user_id, query_vector, topk=5):
    probes = get_nprobe_value()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"SET ivfflat.probes = {probes};")

    total_query_time = 0  # Variable to accumulate the total SQL query time
    total_blocks_accessed = 0

    vector_str = query_vector

    # Combined query using EXPLAIN ANALYZE to retrieve closest vectors among accessible document blocks
    explain_query = """
        EXPLAIN ANALYZE
        SELECT db.block_id, db.block_content, db.vector <-> %s AS distance
        FROM documentblocks db
        WHERE db.document_id IN (
            SELECT DISTINCT pa.document_id 
            FROM PermissionAssignment pa
            JOIN UserRoles ur ON pa.role_id = ur.role_id
            WHERE ur.user_id = %s
        )
        ORDER BY distance
        LIMIT %s
    """
    cur.execute(explain_query, [vector_str, user_id, topk])
    explain_plan = cur.fetchall()

    # ----------------------save query plan-----------start
    import inspect
    current_function_name = inspect.currentframe().f_code.co_name

    # save_query_plan(explain_plan, current_function_name)
    # ----------------------save query plan-------------end

    # Parse the execution time from EXPLAIN ANALYZE
    for row in explain_plan:
        line = row[0]
        if "Execution Time" in line:
            # Extract the execution time in milliseconds and convert to seconds
            try:
                query_time = float(line.split()[-2]) / 1000
                total_query_time += query_time
            except (IndexError, ValueError):
                pass  # Handle unexpected format gracefully
        elif "rows=" in line:
            # Extract the number of rows (blocks) accessed
            try:
                blocks_accessed = int(line.split("rows=")[1].split(" ")[0])
                total_blocks_accessed += blocks_accessed
            except (IndexError, ValueError):
                pass  # Handle unexpected format gracefully

    # Perform the actual query without EXPLAIN ANALYZE to fetch results
    query = """
        SELECT db.block_id, db.block_content, db.vector <-> %s AS distance
        FROM documentblocks db
        WHERE db.document_id IN (
            SELECT DISTINCT pa.document_id 
            FROM PermissionAssignment pa
            JOIN UserRoles ur ON pa.role_id = ur.role_id
            WHERE ur.user_id = %s
        )
        ORDER BY distance
        LIMIT %s
    """
    cur.execute(query, [vector_str, user_id, topk])

    results = cur.fetchall()

    # Step 3: Calculate the total number of blocks to compute selectivity
    total_blocks_query = "SELECT COUNT(block_id) FROM documentblocks;"
    cur.execute(total_blocks_query)
    total_blocks = cur.fetchone()[0]

    # Step 4: Calculate selectivity
    block_selectivity = (total_blocks_accessed / total_blocks) if total_blocks else 0

    cur.close()
    conn.close()

    return results, total_query_time, block_selectivity


def predicate_postfilter_statistics_system(user_id, query_vector, topk=5):
    probes = get_nprobe_value()
    import time
    start_time = time.time()  # Start system time tracking
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"SET ivfflat.probes = {probes};")
    # Convert query vector to string for the SQL query
    vector_str = query_vector

    # Combined query to retrieve closest vectors among accessible document blocks
    query = """
        SELECT db.block_id, db.block_content, db.vector <-> %s AS distance
        FROM documentblocks db
        WHERE db.document_id IN (
            SELECT DISTINCT pa.document_id 
            FROM PermissionAssignment pa
            JOIN UserRoles ur ON pa.role_id = ur.role_id
            WHERE ur.user_id = %s
        )
        ORDER BY distance
        LIMIT %s
    """

    cur.execute(query, [vector_str, user_id, topk])

    results = cur.fetchall()
    cur.close()
    conn.close()

    return results, time.time() - start_time


def ground_truth_func(user_id, query_vector, topk=5):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SET enable_indexscan = off;")
    cur.execute("SET enable_bitmapscan = off;")
    cur.execute("SET enable_indexonlyscan = off;")

    # Convert query vector to string for the SQL query
    vector_str = query_vector

    # SQL query to perform vector search with conditional filtering
    query = """
        SELECT db.block_id, db.document_id, db.block_content, db.vector <-> %s AS distance
        FROM documentblocks db
        JOIN PermissionAssignment pa ON db.document_id = pa.document_id
        JOIN UserRoles ur ON pa.role_id = ur.role_id
        WHERE ur.user_id = %s
        ORDER BY distance
        LIMIT %s
    """

    cur.execute(query, [vector_str, user_id, topk])

    results = cur.fetchall()

    cur.execute("RESET enable_indexscan;")
    cur.execute("RESET enable_bitmapscan;")
    cur.execute("RESET enable_indexonlyscan;")

    cur.close()
    conn.close()

    return results


def prepare_query_dataset(regenerate=True, num_queries=1000):
    query_dataset_file = "query_dataset.json"
    # Check if the dataset file exists
    if not os.path.exists(query_dataset_file) or regenerate:
        generate_query_dataset(num_queries=num_queries, topk=5, output_file="query_dataset.json")

    # Load queries from the dataset
    queries = load_queries_from_dataset(query_dataset_file)

    # Print the loaded queries for inspection
    for query in queries:
        print(f"User {query['user_id']} with query vector: {query['query_vector']} and topk: {query.get('topk', 5)}")

    return queries


def compute_recall(true_results, predicted_results):
    true_set = set(true_results)
    predicted_set = set(predicted_results)
    intersection_size = len(true_set & predicted_set)

    recall = intersection_size / len(true_set)
    return recall


def load_function(import_path: str) -> Callable:
    """
    Dynamically loads a function from a given import path.

    Args:
        import_path (str): The full import path of the function (e.g., 'module.submodule.function').

    Returns:
        Callable: The imported function.
    """
    module_path, function_name = import_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


def save_query_plan(explain_plan, current_function_name):
    # ----------------------save query plan-----------start
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = os.path.join(os.getcwd(), f"{current_function_name}.txt")

    with open(file_path, 'a') as f:
        f.write(f"\n--- Query Plan at {current_time} ---\n")
        for row in explain_plan:
            f.write(row[0] + "\n")

    print(f"EXPLAIN ANALYZE output saved to {file_path}")

    # ----------------------save query plan-------------end


def run_test(
        queries,
        condition,
        iterations=1,
        output_file=None,
        enable_index=False,
        index_type="ivfflat",
        statistics_type="sql",
        generator_type="tree-based",
        record_recall=True,
        warm_up=True,
):
    """
    Runs an experiment with the provided search function and computes recall, query time, and optionally block selectivity.

    Args:
        queries (list): List of queries to execute. Each query should be a dictionary with keys:
                       - "user_id": The ID of the user.
                       - "query_vector": The vector to search against.
                       - "topk" (optional): The number of top results to return (default is 5).
        condition (str): The condition under which to run the search. Possible values:
                         - "user", "role", "combination", "alg1", "alg2",
                         - "postfilter", "prefilter", "rls", "partition_proposal".
        iterations (int): Number of iterations to run (default is 1).
        output_file (str, optional): Path to the output JSON file (default is based on condition).
        enable_index (bool): Whether to enable indexing (default is False).
        statistics_type (str): Type of statistics to collect ("sql" or "system").
        strategy (optional): Partitioning strategy to use for the search function, if applicable.

    Returns:
        None: Results are saved to the specified output_file.
    """

    # Validate condition
    if condition not in CONDITION_CONFIG:
        raise ValueError(f"Invalid condition specified: {condition}")

    # Retrieve configuration based on condition
    config = CONDITION_CONFIG[condition]
    search_func = load_function(config["search_func_path"])
    space_calc_func = load_function(config["space_calc_func_path"])
    extra_params = config["extra_params"].copy()

    # Set default output_file if not provided
    if output_file is None:
        import efconfig
        if enable_index:
            output_file = f"{condition}_{index_type}_{statistics_type}_{generator_type}_{efconfig.ef_search}_avg_results.json"
        else:
            output_file = f"{condition}_{index_type}_{statistics_type}_{generator_type}_avg_results.json"

    # If queries_num is specified in config and not overridden, set it
    if 'queries_num' in config["extra_params"] and 'queries_num' not in extra_params:
        extra_params["queries_num"] = config["extra_params"]["queries_num"]

    # Initialize aggregation variables
    all_results = []
    total_recall = 0
    total_time = 0
    total_selectivity = 0  # Only relevant for 'sql' statistics_type

    # Run the experiment
    experiment_result = run_search_experiment(
        queries=queries,
        search_func=search_func,
        statistics_type=statistics_type,
        queries_num=extra_params.get("queries_num"),
        generator_type=generator_type,
        enable_index=enable_index,
        iterations=iterations,
        index_type=index_type,
        record_recall=record_recall,
        warm_up=warm_up,
    )

    # Extract average recall and query time from run_search_experiment
    # Assuming run_search_experiment returns a dict with these keys
    avg_recall = experiment_result.get("avg_recall", 0)
    avg_query_time = experiment_result.get("avg_query_time", 0)
    avg_block_selectivity = experiment_result.get("avg_block_selectivity", 0) if statistics_type == "sql" else None

    # Aggregate the results
    all_results.extend(experiment_result.get("all_results", []))
    total_recall += avg_recall
    total_time += avg_query_time
    if statistics_type == "sql":
        total_selectivity += avg_block_selectivity

    # Calculate final averages
    avg_recall_final = total_recall
    avg_query_time_final = total_time
    avg_block_selectivity_final = total_selectivity

    print("The time unit is seconds")
    print(f"Average Recall: {avg_recall_final:.4f}")
    print(f"Average Query Time: {avg_query_time_final:.4f} seconds")

    # Calculate space used based on condition
    space_used_mb = space_calc_func(condition)

    print(f"Space used: {space_used_mb:.2f} MB")

    # Aggregate results for JSON output
    results_data = {
        "condition": condition,
        "iterations": iterations,
        "enable_index": enable_index,
        "average_results": {
            "avg_recall": avg_recall_final,
            "avg_query_time": avg_query_time_final,
        },
        "space_used_mb": space_used_mb,
        "index_type": index_type,
        "statistics_type": statistics_type,
        "generator_type": generator_type,
    }

    if statistics_type == "sql":
        results_data["average_results"]["avg_block_selectivity"] = avg_block_selectivity_final

    # Write results to JSON file
    with open(output_file, "w") as json_file:
        json.dump(results_data, json_file, indent=4)

    print(f"Results saved to {output_file}")


def run_search_experiment(queries, search_func, queries_num=None, statistics_type="sql",
                          generator_type="tree-based", enable_index=False, index_type="ivfflat", iterations=1,
                          record_recall=True, plot=False, warm_up=True):
    """
    Runs the search experiment and returns aggregated results.

    Args:
        queries (list): List of query dictionaries.
        search_func (function): The search function to execute.
        queries_num (int, optional): Number of queries to process.
        statistics_type (str): Type of statistics to collect ("sql" or "system").

    Returns:
        dict: Contains 'all_results', 'avg_recall', 'avg_query_time', and optionally 'avg_block_selectivity'.
    """
    all_results = []
    total_recall = 0
    total_query_time = 0
    processed_queries = 0

    actual_queries = queries_num if queries_num else len(queries)

    recalls = []
    for query in queries[:actual_queries]:
        user_id = query["user_id"]
        query_vector = query["query_vector"]
        topk = query.get("topk", 5)

        query_total_recall = 0
        query_total_time = 0
        ground_truth_results = None

        # Get ground truth
        if record_recall:
            ground_truth_results = ground_truth_func(user_id=user_id, query_vector=query_vector, topk=topk)
        for _ in range(iterations):
            # Run search function
            if warm_up:
                for _ in range(2):
                    search_results = search_func(
                        user_id=user_id,
                        query_vector=query_vector,
                        topk=topk,
                        statistics_type=statistics_type
                    )

            search_results = search_func(
                user_id=user_id,
                query_vector=query_vector,
                topk=topk,
                statistics_type=statistics_type
            )

            # Unpack search results based on statistics_type
            results, query_time = search_results

            if record_recall:
                # Compute recall
                predicted_results = set((res[1], res[0]) for res in results)  # (document_id, block_id)
                true_results = set((gt[1], gt[0]) for gt in ground_truth_results)  # (document_id, block_id)
                recall = compute_recall(true_results, predicted_results)
                if recall == 0:
                    debug = 1
                recalls.append(recall)
                query_total_recall += recall
                # print(f"{user_id}:{recall}")
            query_total_time += query_time

        # Calculate averages for this query after iterations
        avg_recall = query_total_recall / iterations if iterations else 0
        avg_query_time = query_total_time / iterations if iterations else 0

        # Aggregate results
        all_results.append({
            "user_id": user_id,
            "query_vector": query_vector,
            "recall": avg_recall,
            "query_time": avg_query_time,
            "qps": 1 / avg_query_time if avg_query_time > 0 else 0,
        })

        total_recall += avg_recall
        total_query_time += avg_query_time

        processed_queries += 1


    # Calculate averages
    avg_recall = total_recall / processed_queries if processed_queries else 0
    avg_query_time = total_query_time / processed_queries if processed_queries else 0

    output_file = f"{search_func.__name__}_{generator_type}_{enable_index}_{index_type}_results.json"
    with open(output_file, 'w') as json_file:
        json.dump(all_results, json_file, indent=4)

    return {
        "avg_recall": avg_recall,
        "avg_query_time": avg_query_time,
    }
