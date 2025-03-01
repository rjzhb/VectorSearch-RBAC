import json
import sys
import os
from psycopg2 import sql
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from controller.clear_database import clear_tables
from controller.prepare_database import create_database_if_not_exists, create_pgvector_extension, read_file
from services.embedding_service import generate_embedding
from services.config import get_db_connection


def search_documents_with_postfilter_alg1_statistics_sql(user_id, query_vector, topk=5):
    """
    Searches document blocks based on vector similarity using a post-filtering algorithm (Algorithm 1).
    Utilizes EXPLAIN ANALYZE to track query execution time and block access statistics
    to calculate selectivity.

    Parameters:
        user_id (int): The ID of the user performing the search.
        query_vector (list or tuple): The vector to search against.
        topk (int): The number of top results to return.

    Returns:
        tuple: A tuple containing:
            - filtered_results (list of tuples): The fetched and filtered query results.
            - total_query_time (float): The accumulated SQL query time in seconds.
            - block_selectivity (float): The selectivity based on blocks accessed.
    """
    from basic_benchmark.common_function import get_nprobe_value  # Ensure this import is necessary

    probes = get_nprobe_value()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"SET ivfflat.probes = {probes};")

    total_query_time = 0  # Accumulate total SQL query time
    total_blocks_accessed = 0  # Accumulate total blocks accessed
    m = 2
    vector_str = query_vector
    filtered_results = []
    remaining_topk = topk
    fetched_blocks = set()  # Track fetched (block_id, document_id) pairs

    try:
        # Step 1: Retrieve the user's roles using EXPLAIN ANALYZE
        explain_query = """
            EXPLAIN ANALYZE
            SELECT role_id FROM UserRoles WHERE user_id = %s
        """
        cur.execute(explain_query, [user_id])
        explain_plan = cur.fetchall()

        # Parse the execution time from EXPLAIN ANALYZE
        for row in explain_plan:
            line = row[0]
            if "Execution Time" in line:
                try:
                    query_time = float(line.split()[-2]) / 1000  # ms to seconds
                    total_query_time += query_time
                except (IndexError, ValueError):
                    pass  # Handle unexpected format

        # Fetch user roles without EXPLAIN ANALYZE for actual processing
        cur.execute("SELECT role_id FROM UserRoles WHERE user_id = %s", [user_id])
        user_roles = cur.fetchall()
        user_role_ids = [role_id for (role_id,) in user_roles]

        # Step 2: Iterate to fetch and filter results until topk is reached
        while remaining_topk > 0:
            # Step 2a: Construct EXPLAIN ANALYZE query
            if fetched_blocks:
                # Prepare the NOT IN clause safely
                not_in_clause = sql.SQL("NOT IN ({})").format(
                    sql.SQL(', ').join(sql.Placeholder() * len(fetched_blocks))
                )
                explain_query = sql.SQL("""
                    EXPLAIN (ANALYZE, VERBOSE)
                    SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                    FROM documentblocks
                    WHERE (block_id, document_id) {}
                    ORDER BY distance
                    LIMIT %s
                """).format(not_in_clause)
                # Flatten the fetched_blocks set for parameters
                params = [vector_str] + list(fetched_blocks) + [remaining_topk * m]
                cur.execute(explain_query, params)
            else:
                explain_query = """
                    EXPLAIN (ANALYZE, VERBOSE)
                    SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                    FROM documentblocks
                    ORDER BY distance
                    LIMIT %s
                """
                cur.execute(explain_query, [vector_str, remaining_topk * m])

            explain_plan = cur.fetchall()

            # Parse the execution time from EXPLAIN ANALYZE
            for row in explain_plan:
                line = row[0]
                if "Execution Time" in line:
                    try:
                        query_time = float(line.split()[-2]) / 1000  # ms to seconds
                        total_query_time += query_time
                    except (IndexError, ValueError):
                        pass  # Handle unexpected format
                elif "rows=" in line:
                    blocks_accessed = int(line.split("rows=")[1].split(" ")[0])
                    total_blocks_accessed += blocks_accessed
            # Step 2b: Perform the actual query without EXPLAIN ANALYZE
            if fetched_blocks:
                query = sql.SQL("""
                    SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                    FROM documentblocks
                    WHERE (block_id, document_id) {}
                    ORDER BY distance
                    LIMIT %s
                """).format(not_in_clause)
                cur.execute(query, params)
            else:
                query = """
                    SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                    FROM documentblocks
                    ORDER BY distance
                    LIMIT %s
                """
                cur.execute(query, [vector_str, remaining_topk * m])

            results = cur.fetchall()

            if not results:
                break  # No more results to fetch

            # Update fetched_blocks
            fetched_blocks.update((result[0], result[1]) for result in results)

            # Step 3: Postfilter results based on the user's roles and permissions using EXPLAIN ANALYZE
            for result in results:
                if len(filtered_results) >= topk:
                    break  # Desired number of results reached

                block_id = result[0]
                document_id = result[1]

                # EXPLAIN ANALYZE for permission check
                explain_permission_query = """
                    EXPLAIN ANALYZE
                    SELECT 1
                    FROM PermissionAssignment pa
                    WHERE pa.role_id = ANY(%s) AND pa.document_id = %s
                """
                cur.execute(explain_permission_query, [user_role_ids, document_id])
                permission_explain = cur.fetchall()

                # Parse the execution time from EXPLAIN ANALYZE
                for row in permission_explain:
                    line = row[0]
                    if "Execution Time" in line:
                        try:
                            query_time = float(line.split()[-2]) / 1000  # ms to seconds
                            total_query_time += query_time
                        except (IndexError, ValueError):
                            pass  # Handle unexpected format

                # Actual permission check
                cur.execute(
                    """
                    SELECT 1
                    FROM PermissionAssignment pa
                    WHERE pa.role_id = ANY(%s) AND pa.document_id = %s
                    """, [user_role_ids, document_id]
                )
                permission_check = cur.fetchone()

                if permission_check:
                    filtered_results.append(result)

            remaining_topk = topk - len(filtered_results)

        # Step 4: Calculate the total number of blocks to compute selectivity
        cur.execute("SELECT COUNT(block_id) FROM documentblocks;")
        total_blocks = cur.fetchone()[0]

        # Calculate selectivity
        block_selectivity = (total_blocks_accessed / total_blocks) if total_blocks else 0

    finally:
        # Ensure resources are cleaned up
        cur.close()
        conn.close()

    return filtered_results[:topk], total_query_time, block_selectivity


def search_documents_with_postfilter_alg1_statistics_system(user_id, query_vector, topk=5):
    from basic_benchmark.common_function import get_nprobe_value
    probes = get_nprobe_value()
    import time
    start_time = time.time()  # Start system time tracking
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"SET ivfflat.probes = {probes};")

    m = 2
    vector_str = query_vector
    filtered_results = []
    remaining_topk = topk
    fetched_blocks = set()  # To keep track of already fetched (block_id, document_id) pairs

    # Step 1: Retrieve the user's roles (do this once outside the loop)
    cur.execute(
        """
        SELECT role_id FROM UserRoles WHERE user_id = %s
        """, [user_id]
    )
    user_roles = cur.fetchall()
    user_role_ids = [role_id for (role_id,) in user_roles]

    while remaining_topk > 0:
        # Step 2: Query to retrieve more document blocks based on vector similarity
        if fetched_blocks:
            query = sql.SQL(
                """
                SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                FROM documentblocks
                WHERE (block_id, document_id) NOT IN %s
                ORDER BY distance
                LIMIT %s
                """
            )
            cur.execute(query, [vector_str, tuple(fetched_blocks), remaining_topk * m])
        else:
            query = sql.SQL(
                """
                SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                FROM documentblocks
                ORDER BY distance
                LIMIT %s
                """
            )
            cur.execute(query, [vector_str, remaining_topk * m])

        results = cur.fetchall()

        if not results:
            break  # No more results to fetch

        # Record fetched (block_id, document_id) pairs to avoid fetching them again
        fetched_blocks.update((result[0], result[1]) for result in results)

        # Step 3: Postfilter results based on the user's roles and permissions
        filtered_batch = []
        for result in results:
            block_id = result[0]
            document_id = result[1]
            # Check if the document is accessible by any of the user's roles
            cur.execute(
                """
                SELECT 1
                FROM PermissionAssignment pa
                WHERE pa.role_id = ANY(%s) AND pa.document_id = %s
                """, [user_role_ids, document_id]
            )
            permission_check = cur.fetchone()
            if permission_check:
                filtered_batch.append(result)
                if len(filtered_results) + len(filtered_batch) == topk:
                    break

        # Add filtered results to the main result set
        filtered_results.extend(filtered_batch)

        # Update remaining_topk after filtering
        remaining_topk = topk - len(filtered_results)

    cur.close()
    conn.close()

    # Step 4: Return the topk filtered results
    return filtered_results[:topk], time.time() - start_time


def search_documents_with_postfilter_alg1(user_id, query_vector, topk=5, statistics_type="sql"):
    if statistics_type == "sql":
        return search_documents_with_postfilter_alg1_statistics_sql(user_id, query_vector, topk)
    elif statistics_type == "system": \
            return search_documents_with_postfilter_alg1_statistics_system(user_id, query_vector, topk)


def search_documents_with_postfilter_alg2_statistics_sql(user_id, query_vector, topk=5):
    """
    Searches document blocks based on vector similarity using a post-filtering algorithm (Algorithm 2).
    Utilizes EXPLAIN ANALYZE to track query execution time and block access statistics
    to calculate selectivity.

    Parameters:
        user_id (int): The ID of the user performing the search.
        query_vector (list or tuple): The vector to search against.
        topk (int): The number of top results to return.

    Returns:
        tuple: A tuple containing:
            - filtered_results (list of tuples): The fetched and filtered query results.
            - total_query_time (float): The accumulated SQL query time in seconds.
            - block_selectivity (float): The selectivity based on blocks accessed.
    """
    from basic_benchmark.common_function import get_nprobe_value  # Ensure this import is necessary

    probes = get_nprobe_value()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"SET ivfflat.probes = {probes};")

    total_query_time = 0  # Accumulate total SQL query time
    total_blocks_accessed = 0  # Accumulate total blocks accessed
    m = 2
    query_multiplier = 1
    vector_str = query_vector
    filtered_results = []
    fetched_blocks = set()  # Optional: To avoid re-fetching blocks

    try:
        # Step 1: Retrieve the user's roles using EXPLAIN ANALYZE
        explain_query = """
            EXPLAIN ANALYZE
            SELECT role_id FROM UserRoles WHERE user_id = %s
        """
        cur.execute(explain_query, [user_id])
        explain_plan = cur.fetchall()

        # Parse the execution time from EXPLAIN ANALYZE
        for row in explain_plan:
            line = row[0]
            if "Execution Time" in line:
                try:
                    query_time = float(line.split()[-2]) / 1000  # ms to seconds
                    total_query_time += query_time
                except (IndexError, ValueError):
                    pass  # Handle unexpected format

        # Fetch user roles without EXPLAIN ANALYZE for actual processing
        cur.execute("SELECT role_id FROM UserRoles WHERE user_id = %s", [user_id])
        user_roles = cur.fetchall()
        user_role_ids = [role_id for (role_id,) in user_roles]

        # Step 2: Iterate to fetch and filter results until topk is reached
        while len(filtered_results) < topk:
            blocks_to_fetch = query_multiplier * m * topk

            # Step 2a: Construct EXPLAIN ANALYZE query
            if fetched_blocks:
                # Prepare the NOT IN clause safely
                not_in_clause = sql.SQL("NOT IN ({})").format(
                    sql.SQL(', ').join(sql.Placeholder() * len(fetched_blocks))
                )
                explain_query = sql.SQL("""
                    EXPLAIN (ANALYZE, VERBOSE)
                    SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                    FROM documentblocks
                    WHERE (block_id, document_id) {}
                    ORDER BY distance
                    LIMIT %s
                """).format(not_in_clause)
                # Flatten the fetched_blocks set for parameters
                params = [vector_str] + list(fetched_blocks) + [blocks_to_fetch]
                cur.execute(explain_query, params)
            else:
                explain_query = """
                    EXPLAIN (ANALYZE, VERBOSE)
                    SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                    FROM documentblocks
                    ORDER BY distance
                    LIMIT %s
                """
                cur.execute(explain_query, [vector_str, blocks_to_fetch])

            explain_plan = cur.fetchall()

            # Parse the execution time from EXPLAIN ANALYZE
            for row in explain_plan:
                line = row[0]
                if "Execution Time" in line:
                    try:
                        query_time = float(line.split()[-2]) / 1000  # ms to seconds
                        total_query_time += query_time
                    except (IndexError, ValueError):
                        pass  # Handle unexpected format
                elif "rows=" in line:
                    blocks_accessed = int(line.split("rows=")[1].split(" ")[0])
                    total_blocks_accessed += blocks_accessed

            # Step 2b: Perform the actual query without EXPLAIN ANALYZE
            if fetched_blocks:
                query = sql.SQL("""
                    SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                    FROM documentblocks
                    WHERE (block_id, document_id) {}
                    ORDER BY distance
                    LIMIT %s
                """).format(not_in_clause)
                cur.execute(query, params)
            else:
                query = """
                    SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                    FROM documentblocks
                    ORDER BY distance
                    LIMIT %s
                """
                cur.execute(query, [vector_str, blocks_to_fetch])

            results = cur.fetchall()

            if not results:
                break  # No more results to fetch

            # Update fetched_blocks
            fetched_blocks.update((result[0], result[1]) for result in results)

            # Step 3: Postfilter results based on the user's roles and permissions using EXPLAIN ANALYZE
            for result in results:
                if len(filtered_results) >= topk:
                    break  # Desired number of results reached

                block_id = result[0]
                document_id = result[1]

                # EXPLAIN ANALYZE for permission check
                explain_permission_query = """
                    EXPLAIN ANALYZE
                    SELECT 1
                    FROM PermissionAssignment pa
                    WHERE pa.role_id = ANY(%s) AND pa.document_id = %s
                """
                cur.execute(explain_permission_query, [user_role_ids, document_id])
                permission_explain = cur.fetchall()

                # Parse the execution time from EXPLAIN ANALYZE
                for row in permission_explain:
                    line = row[0]
                    if "Execution Time" in line:
                        try:
                            query_time = float(line.split()[-2]) / 1000  # ms to seconds
                            total_query_time += query_time
                        except (IndexError, ValueError):
                            pass  # Handle unexpected format

                # Actual permission check
                cur.execute(
                    """
                    SELECT 1
                    FROM PermissionAssignment pa
                    WHERE pa.role_id = ANY(%s) AND pa.document_id = %s
                    """, [user_role_ids, document_id]
                )
                permission_check = cur.fetchone()

                if permission_check:
                    filtered_results.append(result)

            query_multiplier += 1  # Increment multiplier for next iteration

        # Step 4: Calculate the total number of blocks to compute selectivity
        cur.execute("SELECT COUNT(block_id) FROM documentblocks;")
        total_blocks = cur.fetchone()[0]

        # Calculate selectivity
        block_selectivity = (total_blocks_accessed / total_blocks) if total_blocks else 0

    finally:
        # Ensure resources are cleaned up
        cur.close()
        conn.close()

    return filtered_results[:topk], total_query_time, block_selectivity


def search_documents_with_postfilter_alg2_statistics_system(user_id, query_vector, topk=5):
    from basic_benchmark.common_function import get_nprobe_value
    probes = get_nprobe_value()
    import time
    start_time = time.time()  # Start system time tracking
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"SET ivfflat.probes = {probes};")
    m = 2
    query_multiplier = 1
    vector_str = query_vector
    filtered_results = []

    # Step 1: Retrieve the user's roles
    cur.execute(
        """
        SELECT role_id FROM UserRoles WHERE user_id = %s
        """, [user_id]
    )
    user_roles = cur.fetchall()
    user_role_ids = [role_id for (role_id,) in user_roles]

    while len(filtered_results) < topk:
        filtered_results = []
        # Step 2: Calculate the number of blocks to fetch in this iteration
        blocks_to_fetch = query_multiplier * m * topk

        # Query to retrieve more document blocks based on vector similarity
        query = sql.SQL(
            """
            SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
            FROM documentblocks
            ORDER BY distance
            LIMIT %s
            """
        )
        cur.execute(query, [vector_str, blocks_to_fetch])

        results = cur.fetchall()

        if not results:
            break  # No more results to fetch

        # Step 3: Postfilter results based on the user's roles and permissions
        for result in results:
            block_id = result[0]
            document_id = result[1]
            # Check if the document is accessible by any of the user's roles
            cur.execute(
                """
                SELECT 1
                FROM PermissionAssignment pa
                WHERE pa.role_id = ANY(%s) AND pa.document_id = %s
                """, [user_role_ids, document_id]
            )
            permission_check = cur.fetchone()
            if permission_check:
                filtered_results.append(result)
                if len(filtered_results) >= topk:
                    break

        # Update query_multiplier for the next iteration
        query_multiplier += 1

    cur.close()
    conn.close()

    # Step 4: Return the topk filtered results
    return filtered_results[:topk], time.time() - start_time


def search_documents_with_postfilter_alg2(user_id, query_vector, topk=5, statistics_type="sql"):
    if statistics_type == "sql":
        return search_documents_with_postfilter_alg2_statistics_sql(user_id, query_vector, topk)
    elif statistics_type == "system": \
            return search_documents_with_postfilter_alg2_statistics_system(user_id, query_vector, topk)
