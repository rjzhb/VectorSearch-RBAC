import random

import numpy as np
import os
import sys

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)
print(project_root)
from controller.dynamic_partition.hnsw.analysis.analysis_hnsw_recall import search_documents_rls_for_join_time_analysis, \
    search_documents_rls_for_analysis_with_execution_time

from psycopg2 import sql
import sys
import os

from services.config import get_db_connection


def search_documents_role_partition_analysis(user_id, query_vector, topk=5, ef_searchs=None):
    """
    Search documents with role partition statistics for multiple ef_search values.
    Computes SQL run time for each ef_search value.
    :param user_id: User ID for which to execute the query.
    :param query_vector: Query vector for similarity search.
    :param topk: Number of top results to retrieve.
    :param ef_searchs: List of ef_search values to evaluate.
    :return: A dictionary mapping each ef_search to its (query time, total rows).
    """
    results = {}
    conn = get_db_connection()  # Get the database connection
    cur = conn.cursor()
    cur.execute(f"SET max_parallel_workers_per_gather = 0;")

    # Query 1: Get the role IDs for the user
    cur.execute("SELECT role_id FROM userroles WHERE user_id = %s", [user_id])
    role_ids = cur.fetchall()

    if not role_ids:
        print(f"No roles found for user_id {user_id}.")
        cur.close()
        conn.close()
        return {ef_search: (0, 0) for ef_search in ef_searchs}

    # Randomly select one role_id
    first_role_id  = random.choice([row[0] for row in role_ids])

    # Query for the role's document blocks
    table_name = sql.Identifier(f"documentblocks_role_{first_role_id}")

    # Count total rows in the table
    cur.execute(sql.SQL("SELECT COUNT(*) FROM {};").format(table_name))
    n_total_rows = cur.fetchone()[0]  # Total rows for the role

    # Process each ef_search value
    for ef_search in ef_searchs:
        cur.execute(f"SET hnsw.ef_search = {ef_search};")  # Dynamically set ef_search

        # Execute EXPLAIN ANALYZE to time the query
        explain_query = sql.SQL(
            """
            EXPLAIN (ANALYZE, VERBOSE)
            SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
            FROM {}
            ORDER BY distance
            LIMIT %s
            """
        ).format(table_name)

        cur.execute(explain_query, [query_vector, topk])
        explain_plan = cur.fetchall()

        # Parse query time from EXPLAIN ANALYZE
        total_query_time = 0
        for row in explain_plan:
            if "Execution Time" in row[0]:
                query_time = float(row[0].split()[-2]) * 1000 * 1000  # Convert to nanoseconds
                total_query_time += query_time

        results[ef_search] = (total_query_time, n_total_rows)

    cur.close()
    conn.close()

    return results


def search_documents_brute_force_for_analysis_with_execution_time(user_id, query_vector, topk, ef_search_values):
    """
    Search documents with role-level security for multiple ef_search values using EXPLAIN ANALYZE.
    Computes SQL execution time for each ef_search value.

    :param user_id: User ID for which to execute the query.
    :param query_vector: Query vector for similarity search.
    :param topk: Number of top results to retrieve.
    :param ef_search_values: List of ef_search values to evaluate.
    :return: A dictionary mapping each ef_search to its (query time, total rows).
    """
    results = {}
    conn = get_db_connection()  # Reuse the same connection for this user
    cur = conn.cursor()

    try:
        # Disable parallelism for consistent timing
        cur.execute(f"SET max_parallel_workers_per_gather = 0;")
        # Query for the role's document blocks
        table_name = sql.Identifier("documentblocks")

        # Count total rows in the table
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {};").format(table_name))
        n_total_rows = cur.fetchone()[0]  # Total rows for the first role

        # Process each ef_search value
        for ef_search in ef_search_values:
            cur.execute(f"SET LOCAL hnsw.ef_search = {ef_search};")  # Dynamically set ef_search

            # Execute EXPLAIN ANALYZE to time the query
            explain_query = sql.SQL(
                """
                EXPLAIN (ANALYZE, VERBOSE)
                SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                FROM {}
                ORDER BY distance
                LIMIT %s
                """
            ).format(table_name)

            cur.execute(explain_query, [query_vector, topk])
            explain_plan = cur.fetchall()

            # Parse query time and join time from EXPLAIN ANALYZE
            total_adjusted_time = 0  # Initialize cumulative adjusted query time

            for row in explain_plan:
                line = row[0].strip()
                # Parse overall execution time
                if "Execution Time" in line:
                    query_time = float(line.split()[-2]) * 1000 * 1000  # Convert ms to ns
                    total_adjusted_time += query_time

            # Store results for the current ef_search
            results[ef_search] = (total_adjusted_time, n_total_rows)

    finally:
        cur.close()
        conn.close()

    return results


def run_experiment_on_ef_search(queries, ef_search_values=[40, 80, 120, 160, 200, 240, 280, 320, 360]):
    """
    Run experiments for a given query dataset.
    Only include the third repetition's time in calculations.
    Calculate the average k value for all queries.
    """
    ef_search_results = {}  # Store k values grouped by ef_search
    from controller.dynamic_partition.hnsw.validate.modelqps_vs_realqps import dynamic_partition_search_analysis
    actual_query_times = {ef_search: [] for ef_search in ef_search_values}
    total_rows_by_ef_search = {ef_search: [] for ef_search in ef_search_values}  # Store total rows per ef_search
    # disable_row_level_security()
    # drop_database_users()
    # create_database_users()
    # enable_row_level_security()

    for query in queries:
        user_id = query["user_id"]
        query_vector = query["query_vector"]

        # Execute three repetitions
        for repetition in range(3):
            # Perform the search and return results for all ef_search values
            # query_results = dynamic_partition_search_analysis(
            #     user_id, query_vector, topk=5, ef_searchs=ef_search_values
            # )
            query_results = search_documents_role_partition_analysis(user_id, query_vector, topk=5,
                                                                     ef_searchs=ef_search_values)
            # query_results = search_documents_rls_for_analysis_with_execution_time(user_id, query_vector, topk=5,
            #                                                                       ef_search_values=ef_search_values)
            # query_results = search_documents_brute_force_for_analysis_with_execution_time(user_id, query_vector, topk=5,
            #                                                                               ef_search_values=ef_search_values)
            # Only use the third repetition's results
            if repetition == 2:
                for ef_search, (query_time, n_total_rows) in query_results.items():
                    if n_total_rows > 0:
                        actual_query_times[ef_search].append(query_time)
                        total_rows_by_ef_search[ef_search].append(n_total_rows)
                        k = query_time / np.log(n_total_rows)  # Modified formula
                        if ef_search not in ef_search_results:
                            ef_search_results[ef_search] = []
                        ef_search_results[ef_search].append(k)
                break

    # Step 2: Compute the average k value for each ef_search after all queries
    results = []
    for ef_search in ef_search_values:
        avg_k = np.mean(ef_search_results[ef_search]) if ef_search_results[ef_search] else 0
        avg_query_time = np.mean(actual_query_times[ef_search]) if actual_query_times[ef_search] else 0
        avg_total_rows = np.mean(total_rows_by_ef_search[ef_search]) if total_rows_by_ef_search[ef_search] else 0
        results.append({
            "ef_search": ef_search,
            "avg_k": avg_k,
            "k_values": ef_search_results[ef_search],
            "avg_query_time": avg_query_time,
            "query_times": actual_query_times[ef_search],
            "avg_total_rows": avg_total_rows,
            "total_rows": total_rows_by_ef_search[ef_search]
        })
    return results


def fit_query_time_function_with_log(results):
    """
    Fit f(ef_search) to the average query_time values using a linear function,
    incorporating log(n) into the calculation.

    Parameters:
    - results: List of dictionaries containing "ef_search" and "avg_query_time".
    - logn_values: List of log(n) values corresponding to the query dataset.

    Returns:
    - params: Fitted parameters for the linear model (a, b).
    """
    ef_search_values = np.array([result["ef_search"] for result in results])
    avg_query_times = np.array([result["avg_query_time"] for result in results])
    avg_total_rows = np.array([result["avg_total_rows"] for result in results])

    # Incorporate log(n) into the average query time
    logn_values = np.log(avg_total_rows)
    normalized_query_times = avg_query_times / logn_values

    # Define the linear function: f(x) = a * x + b
    def func(x, a, b):
        return a * x + b

    # Fit the curve
    initial_params = [1, 1]  # Initial guesses for a, b
    params, _ = curve_fit(func, ef_search_values, normalized_query_times, p0=initial_params)

    # Generate fitted values
    fitted_query_times = func(ef_search_values, *params) * logn_values

    # Plot the original data and fitted curve
    plt.figure(figsize=(10, 6))
    plt.scatter(ef_search_values, avg_query_times, label="Data (Average Query Times)", color="blue")
    plt.plot(ef_search_values, fitted_query_times, label=f"Fitted Curve: a * x + b\na={params[0]:.2f}, b={params[1]:.2f}",
             color="red")
    plt.xlabel("ef_search")
    plt.ylabel("Query Time")
    plt.title("Fitting Query Time as a Function of ef_search (Linear Model)")
    plt.legend()
    plt.grid(True)
    plot_filename = f'query_time_analysis.png'
    plt.savefig(plot_filename)
    plt.show()

    print(f"Fitted parameters: a={params[0]:.2f}, b={params[1]:.2f}")
    return params

def fit_ef_search_function_linear(results):
    """
    Fit f(ef_search) to the average k values using a linear function.
    """
    ef_search_values = np.array([result["ef_search"] for result in results])
    avg_k_values = np.array([result["avg_k"] for result in results])

    # Define the linear function: f(x) = a * x + b
    def func(x, a, b):
        return a * x + b

    # Fit the curve
    initial_params = [1, 1]  # Initial guesses for a, b
    params, _ = curve_fit(func, ef_search_values, avg_k_values, p0=initial_params)

    # Generate fitted values
    fitted_k_values = func(ef_search_values, *params)

    # Plot the original data and fitted curve
    plt.figure(figsize=(10, 6))
    plt.scatter(ef_search_values, avg_k_values, label="Data", color="blue")
    plt.plot(ef_search_values, fitted_k_values, label=f"Fitted Curve: a * x + b\na={params[0]:.2f}, b={params[1]:.2f}",
             color="red")
    plt.xlabel("ef_search")
    plt.ylabel("Average k Value")
    plt.title("Fitting Average k as a Function of ef_search (Linear Model)")
    plt.legend()
    plt.grid(True)
    plot_filename = f'qps_analysis.png'
    plt.savefig(plot_filename)
    plt.show()

    print(f"Fitted parameters: a={params[0]:.2f}, b={params[1]:.2f}")
    return params


def run_experiment_on_join_time(queries):
    """
    Run experiments across the entire query dataset to calculate the average join time.
    Use the third repetition's join time in calculations.
    """
    results = []  # To store results for each query's join time

    for query in queries:
        user_id = query["user_id"]
        query_vector = query["query_vector"]

        join_times = []  # Store join times for three repetitions

        # Execute three repetitions
        for repetition in range(3):
            join_time = search_documents_rls_for_join_time_analysis(
                user_id, query_vector
            )
            join_times.append(join_time)

        # Only use the third repetition's join time
        if len(join_times) >= 3:
            third_join_time = join_times[2]  # Take the third repetition
            results.append(third_join_time)

    # Calculate the average join time across all queries
    avg_join_time = np.mean(results) if results else 0

    return avg_join_time


def get_hnsw_qps_parameters():
    import json

    # Load generated queries
    benchmark_folder = os.path.join(project_root, "basic_benchmark")
    query_dataset_path = os.path.join(benchmark_folder, "query_dataset.json")
    with open(query_dataset_path, "r") as infile:
        queries = json.load(infile)

    ef_search_values = [20, 40, 80, 120, 200, 300, 400]

    # Run experiments
    results = run_experiment_on_ef_search(queries, ef_search_values)

    # Fit the function to the results using a linear model
    # fitted_params = fit_ef_search_function_linear(results)
    fitted_params = fit_query_time_function_with_log(results)
    join_times = run_experiment_on_join_time(queries)
    # Print fitted parameters
    print(f"Fitted Function Parameters: a={fitted_params[0]:.2f}, b={fitted_params[1]:.2f}")
    return fitted_params, join_times


if __name__ == "__main__":
    get_hnsw_qps_parameters()
