#!/bin/bash

# Exit immediately if any command fails
set -e

# Check if generator type is provided
if [ -z "$1" ]; then
    echo "Please provide a generator type: random, treebased, erbac"
    exit 1
fi

# Check if index type is provided
index_type=$2
if [ -n "$index_type" ]; then
    if [[ "$index_type" != "ivfflat" && "$index_type" != "hnsw" ]]; then
        echo "Invalid index type. Please use: ivfflat or hnsw"
        exit 1
    fi
else
    echo "No index_type provided, proceeding without index_type."
fi

# Check if skip initialization is provided as a third argument
skip_init=false
if [ "$3" == "skip_init" ]; then
    skip_init=true
fi

# Store the provided generator argument
generator=$1

# Use a case statement to select the appropriate generator based on the input
case $generator in
    "random")
        echo "Running store_random_rbac_generate_data.py"
        python3 services/rbac_generator/store_random_rbac_generate_data.py
        ;;
    "treebased")
        echo "Running store_tree_based_rbac_generate_data.py"
        python3 services/rbac_generator/store_tree_based_rbac_generate_data.py
        ;;
    "erbac")
        echo "Running store_erbac_generate_data.py"
        python3 services/rbac_generator/store_erbac_generate_data.py
        ;;
    *)
        # If the argument doesn't match any valid option, show an error
        echo "Unknown generator type: $generator"
        echo "Please use: random, treebased, erbac"
        exit 1
        ;;
esac

# Move to the benchmark directory
cd benchmark

# Conditionally skip initialization if skip_init is false
if [ "$skip_init" = false ]; then
    echo "Running initialization scripts"

    # Run generate_queries.py
    python3 generate_queries.py

    # Only pass index_type if it is provided
    python3 initialize_role_partition_tables.py
#    python3 initialize_uniform_disjoint_tables.py --num_partitions 100
else
    echo "Skipping initialization as per user request"
fi

# Always run the test script, only pass index_type if it is provided
if [ -n "$index_type" ]; then
    python3 test_all.py --index_type $index_type --generator_type $generator
else
    python3 test_all.py --generator_type $generator
fi