import os
import sys

import psycopg2
from psycopg2 import sql
from wasabi import table

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(sys.path)
from services.config import get_db_connection


def calculate_size_in_mb(table_names):
    """
    Calculate the total storage space used by the specified tables.

    Args:
        table_names (list): List of table names to calculate size for.

    Returns:
        float: The total size of the specified tables in MB.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    total_size = 0
    for table in table_names:
        cur.execute(sql.SQL("SELECT pg_total_relation_size(%s);"), [table])
        table_size = cur.fetchone()[0]
        total_size += table_size

    cur.close()
    conn.close()

    # Convert the total size from bytes to MB
    return total_size / (1024 * 1024)


def calculate_prefilter(condition):
    """
    Calculate the total storage space used by the tables associated with the specified condition.

    Args:
        condition (str): The partitioning condition ("role", "user", or "combination_role").

    Returns:
        float: The total size of the relevant tables in MB.
    """
    # List of common tables to calculate size for
    tables = [
        'Documents',
        'PermissionAssignment',
        'Roles',
        'UserRoles',
        'Users'
    ]

    conn = get_db_connection()
    cur = conn.cursor()

    # Determine the specific partition tables to include based on the condition
    if condition == "prefilter_partition_role":
        cur.execute("SELECT tablename FROM pg_tables WHERE tablename LIKE 'documentblocks_role_%';")
    elif condition == "prefilter_partition_combination":
        cur.execute("SELECT tablename FROM pg_tables WHERE tablename LIKE 'documentblocks_comb_%';")
    else:
        raise ValueError("Invalid condition specified")

    partition_tables = cur.fetchall()

    # Add partition table names to the list of tables to calculate size for
    for table_name in partition_tables:
        tables.append(table_name[0])

    # Calculate and return the total size in MB
    return calculate_size_in_mb(tables)


def calculate_postfilter(condition=None):
    """
    Calculate the total storage space used by the postfilter scenario.

    Returns:
        float: The total size of the relevant tables in MB.
    """
    tables = [
        'Documents',
        'PermissionAssignment',
        'Roles',
        'UserRoles',
        'Users',
        'documentblocks'
    ]

    # Calculate and return the total size in MB
    return calculate_size_in_mb(tables)


def calculate_rls(condition=None):
    """
    Calculate the total storage space used by the postfilter scenario, including RLS policies.

    Returns:
        float: The total size of the relevant tables and RLS policies in MB.
    """
    tables = [
        'Documents',
        'PermissionAssignment',
        'Roles',
        'UserRoles',
        'Users',
        'DocumentBlocks',
    ]

    # Calculate the total size of the tables
    total_size_mb = calculate_size_in_mb(tables)

    # Calculate the space used by RLS policies
    rls_policy_size_mb = float(calculate_rls_policy_size())  # Convert decimal to float

    # Return the sum of table sizes and RLS policy size
    return total_size_mb + rls_policy_size_mb


def calculate_rls_policy_size(dynamic_partition=False):
    """
    Calculate the total storage space used by RLS policies and optionally include materialized views.

    Args:
        dynamic_partition (bool): If True, include the size of the materialized view.

    Returns:
        float: The total size of RLS policies and materialized views in MB.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Calculate the size of RLS policies
    cur.execute("""
        SELECT pg_total_relation_size('pg_policy') / 1024.0 / 1024.0;
    """)
    rls_policy_size_mb = cur.fetchone()[0]

    # If dynamic_partition is True, include the size of the materialized view
    if dynamic_partition:
        cur.execute("""
            SELECT pg_total_relation_size('user_accessible_documents') / 1024.0 / 1024.0;
        """)
        materialized_view_size_mb = cur.fetchone()[0]
        total_size_mb = rls_policy_size_mb + materialized_view_size_mb
    else:
        total_size_mb = rls_policy_size_mb

    cur.close()
    conn.close()

    return total_size_mb


def calculate_partition_proposal(condition):
    """
    Calculate the total storage space used by the partition proposal scenario.

    Args:
        condition (str): The partitioning strategy ("lsh", "role_partition", "uniform").

    Returns:
        float: The total size of the relevant tables in MB.
    """
    tables = [
        'Documents',
        'PermissionAssignment',
        'Roles',
        'UserRoles',
        'Users'
    ]

    conn = get_db_connection()
    cur = conn.cursor()

    # Determine the specific partition tables to include based on the condition
    if condition == "lsh":
        cur.execute("SELECT tablename FROM pg_tables WHERE tablename LIKE 'lsh_partition_%';")
    elif condition == "role":
        cur.execute("SELECT tablename FROM pg_tables WHERE tablename LIKE 'role_document_partition_%';")
    elif condition == "uniform_disjoint_partition":
        cur.execute("SELECT tablename FROM pg_tables WHERE tablename LIKE 'documentblocks_partition_%';")
    else:
        raise ValueError("Invalid condition specified")

    partition_tables = cur.fetchall()

    # Add partition table names to the list of tables to calculate size for
    for table_name in partition_tables:
        tables.append(table_name[0])

    cur.close()
    conn.close()

    # Calculate and return the total size in MB
    return calculate_size_in_mb(tables)


def calculate_dynamic_partition(condition=None):
    """
    Calculate the total storage space used by dynamic partition tables, including RolePartitions.

    Returns:
        float: The total size of the dynamic partition tables in MB.
    """
    tables = [
        'Documents', 'PermissionAssignment', 'Roles',
        'UserRoles', 'Users', 'CombRolePartitions'
    ]

    conn = get_db_connection()
    cur = conn.cursor()

    # Retrieve dynamic partition tables
    cur.execute("SELECT tablename FROM pg_tables WHERE tablename LIKE 'documentblocks_partition_%';")
    partition_tables = cur.fetchall()

    # Add partition table names to the list of tables to calculate size for
    tables.extend([table_name[0] for table_name in partition_tables])

    cur.close()
    conn.close()
    # Calculate the space used by RLS policies
    rls_policy_size_mb = float(calculate_rls_policy_size(dynamic_partition=True))  # Convert decimal to float
    # Calculate and return the total size in MB
    return calculate_size_in_mb(tables) + rls_policy_size_mb
