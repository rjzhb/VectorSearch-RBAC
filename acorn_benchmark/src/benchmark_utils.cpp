#include "benchmark_utils.h"
#include <faiss/IndexACORN.h>
#include <pqxx/pqxx>
#include <chrono>
#include <iostream>
#include <sstream>
#include <set>
#include <unordered_map>
#include <limits>
#include <thread>
#include <faiss/IndexFlat.h> // Include FAISS header for IndexFlatL2
#include <pqxx/pqxx>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>

// Helper to parse vector string
std::vector<float> parse_vector(const std::string &vector_str) {
    std::vector<float> vec;
    std::istringstream ss(vector_str.substr(1, vector_str.size() - 2)); // Remove brackets
    std::string token;
    while (std::getline(ss, token, ',')) {
        vec.push_back(std::stof(token));
    }
    return vec;
}

double compute_recall(
    const std::vector<std::pair<int, int>> &ground_truth,
    const std::vector<std::pair<int, int>> &retrieved,
    size_t topk
) {
    // Check for empty ground truth or retrieved vectors to avoid division by zero
    if (ground_truth.empty() || retrieved.empty()) {
        return 0.0;
    }

    // Ensure both vectors are divisible by topk
    if (ground_truth.size() % topk != 0 || retrieved.size() % topk != 0) {
        throw std::runtime_error("Ground truth and retrieved sizes are not aligned with topk.");
    }

    // Ensure both vectors are of the same size
    if (ground_truth.size() != retrieved.size()) {
        throw std::runtime_error("Ground truth and retrieved sizes do not match.");
    }

    size_t num_queries = ground_truth.size() / topk;
    double total_recall = 0.0;

    // Pre-split ground truth and retrieved results into per-query sets
    std::vector<std::set<std::pair<int, int>>> ground_truth_sets(num_queries);
    std::vector<std::set<std::pair<int, int>>> retrieved_sets(num_queries);

    for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
        for (size_t i = 0; i < topk; ++i) {
            size_t idx = query_idx * topk + i;
            ground_truth_sets[query_idx].insert(ground_truth[idx]);
            retrieved_sets[query_idx].insert(retrieved[idx]);
        }
    }

    // Compute recall for each query and record mismatches
    for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
        const auto &ground_truth_set = ground_truth_sets[query_idx];
        const auto &retrieved_set = retrieved_sets[query_idx];

        // Calculate the intersection size
        std::vector<std::pair<int, int>> intersection;
        std::set_intersection(
            ground_truth_set.begin(), ground_truth_set.end(),
            retrieved_set.begin(), retrieved_set.end(),
            std::back_inserter(intersection)
        );

        // Accumulate recall for this query
        double query_recall = static_cast<double>(intersection.size()) / ground_truth_set.size();
        // if (query_recall != 1) {
        //     std::cout << query_recall << std::endl;
        // }
        total_recall += query_recall;
    }

    // Calculate average recall over all queries
    return total_recall / num_queries;
}
// Helper function to fetch roles, documents, and role-to-document mappings
std::tuple<std::vector<int>, std::vector<int>, std::unordered_map<int, std::set<int> > >
fetch_role_to_documents(const std::string &conn_info) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    // Fetch roles
    pqxx::result role_res = txn.exec("SELECT role_id FROM Roles;");
    std::vector<int> roles;
    for (const auto &row: role_res) {
        roles.push_back(row[0].as<int>());
    }

    // Fetch document IDs from PermissionAssignment
    pqxx::result doc_res = txn.exec(
        "SELECT DISTINCT document_id FROM PermissionAssignment ORDER BY document_id;"
    );
    std::vector<int> documents;
    for (const auto &row: doc_res) {
        documents.push_back(row[0].as<int>());
    }

    // Fetch document IDs from documentblocks
    pqxx::result block_doc_res = txn.exec(
        "SELECT DISTINCT document_id FROM documentblocks ORDER BY document_id;"
    );
    std::vector<int> documentblocks_documents;
    for (const auto &row: block_doc_res) {
        documentblocks_documents.push_back(row[0].as<int>());
    }

    // Validate consistency between PermissionAssignment and documentblocks
    if (documents == documentblocks_documents) {
        std::cout << "The document_id lists from PermissionAssignment and documentblocks are identical."
                << std::endl;
    } else {
        std::cout << "The document_id lists are not identical." << std::endl;
    }

    // Fetch permissions (role_id -> document_id)
    pqxx::result permissions_res = txn.exec("SELECT role_id, document_id FROM PermissionAssignment;");
    std::unordered_map<int, std::set<int> > role_to_documents;
    for (const auto &row: permissions_res) {
        int role_id = row[0].as<int>();
        int document_id = row[1].as<int>();
        role_to_documents[role_id].insert(document_id);
    }


    txn.commit();
    return {roles, documents, role_to_documents};
}

std::vector<std::pair<int, int> > compute_groundtruth(
    const std::vector<Query> &queries,
    const std::string &conn_info
) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    // Disable index scans for sequential scan
    txn.exec("SET enable_indexscan = off;");
    txn.exec("SET enable_bitmapscan = off;");
    txn.exec("SET enable_indexonlyscan = off;");

    std::vector<std::pair<int, int> > ground_truth_doc_blocks;

    // Loop through each query
    for (const auto &query_obj: queries) {
        int user_id = query_obj.user_id;
        int topk = query_obj.topk;

        // Combine vector into PostgreSQL-compatible format directly in the query
        std::ostringstream query;
        query << "SELECT db.document_id, db.block_id, "
                << "db.vector <-> '[";
        for (size_t i = 0; i < query_obj.query_vector.size(); ++i) {
            query << query_obj.query_vector[i];
            if (i < query_obj.query_vector.size() - 1) {
                query << ",";
            }
        }
        query << "]' AS distance "
                << "FROM documentblocks db "
                << "JOIN PermissionAssignment pa ON db.document_id = pa.document_id "
                << "JOIN UserRoles ur ON pa.role_id = ur.role_id "
                << "WHERE ur.user_id = " << user_id << " "
                << "ORDER BY distance "
                << "LIMIT " << topk << ";";

        // Execute the query
        pqxx::result result = txn.exec(query.str());
        // Process results and append to ground truth
        for (const auto &row: result) {
            int document_id = row[0].as<int>();
            int block_id = row[1].as<int>();
            ground_truth_doc_blocks.emplace_back(document_id, block_id);
        }
    }

    // Reset index scanning options
    txn.exec("RESET enable_indexscan;");
    txn.exec("RESET enable_bitmapscan;");
    txn.exec("RESET enable_indexonlyscan;");

    txn.commit();

    return ground_truth_doc_blocks;
}


// Fetch metadata for filtering based on user_id
std::unordered_map<int, std::vector<int> > fetch_metadata(
    const std::vector<Query> &queries,
    const std::string &conn_info
) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    // Initialize metadata map
    std::unordered_map<int, std::vector<int> > metadata_map;

    for (const auto &query: queries) {
        int user_id = query.user_id;

        // Fetch document IDs accessible to this user
        pqxx::result metadata_res = txn.exec_params(
            "SELECT DISTINCT pa.document_id "
            "FROM PermissionAssignment pa "
            "JOIN UserRoles ur ON pa.role_id = ur.role_id "
            "WHERE ur.user_id = $1;",
            user_id
        );

        std::vector<int> document_ids;
        for (const auto &row: metadata_res) {
            document_ids.push_back(row[0].as<int>());
        }

        metadata_map[user_id] = std::move(document_ids);
    }

    txn.commit();
    return metadata_map;
}

std::vector<char> build_acorn_filter_ids_map(
    const std::vector<Query> &queries,
    const std::string &conn_info
) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    // Fetch all document IDs and block IDs from documentblocks
    pqxx::result vector_res = txn.exec(
        "SELECT document_id, block_id "
        "FROM documentblocks "
        "ORDER BY document_id, block_id;"
    );

    // Map block index to its corresponding document_id
    std::vector<int> document_ids; // Document ID for each block
    for (const auto &row: vector_res) {
        document_ids.push_back(row[0].as<int>()); // document_id
    }

    size_t nq = queries.size(); // Number of queries
    size_t N = document_ids.size(); // Number of blocks (vectors)

    // Initialize filter_ids_map
    std::vector<char> filter_ids_map(nq * N, 0);

    for (size_t xq = 0; xq < nq; ++xq) {
        int user_id = queries[xq].user_id;

        // Fetch document IDs accessible to this user
        pqxx::result metadata_res = txn.exec_params(
            "SELECT DISTINCT pa.document_id "
            "FROM PermissionAssignment pa "
            "JOIN UserRoles ur ON pa.role_id = ur.role_id "
            "WHERE ur.user_id = $1;",
            user_id
        );

        // Convert result to a set for quick lookup
        std::set<int> allowed_document_ids;
        for (const auto &row: metadata_res) {
            allowed_document_ids.insert(row[0].as<int>());
        }

        // Populate filter_ids_map for this query
        for (size_t xb = 0; xb < N; ++xb) {
            if (allowed_document_ids.count(document_ids[xb]) > 0) {
                filter_ids_map[xq * N + xb] = 1; // Mark block as valid
            }
        }
    }

    txn.commit();
    return filter_ids_map;
}

std::vector<char> generate_fields_map(
    const std::string &conn_info,
    const std::string &table_name,
    int user_id
) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    // Fetch valid document IDs for this user
    std::string doc_query = R"(
        SELECT DISTINCT pa.document_id
        FROM PermissionAssignment pa
        JOIN UserRoles ur ON pa.role_id = ur.role_id
        WHERE ur.user_id = $1
    )";
    pqxx::result valid_docs_res = txn.exec_params(doc_query, user_id);

    // Create a set of allowed document IDs
    std::set<int> allowed_doc_ids;
    for (const auto &row: valid_docs_res) {
        allowed_doc_ids.insert(row[0].as<int>());
    }

    // Fetch all (document_id, block_id) pairs in this partition
    std::string block_query = R"(
        SELECT document_id, block_id
        FROM )" + table_name + R"(
        ORDER BY document_id, block_id
    )";
    pqxx::result block_res = txn.exec(block_query);

    // Build fields_map
    std::vector<char> fields_map(block_res.size(), 0);
    size_t idx = 0;
    for (const auto &row: block_res) {
        int document_id = row[0].as<int>();
        if (allowed_doc_ids.count(document_id)) {
            fields_map[idx] = 1; // Mark as accessible
        }
        idx++;
    }

    txn.commit();
    return fields_map;
}

std::unordered_map<std::string, std::vector<std::pair<int, int>>> fetch_all_document_block_maps(
    const std::string &conn_info
) {
    pqxx::connection conn(conn_info);
    pqxx::work txn(conn);

    // Retrieve all partition tables
    pqxx::result partition_res = txn.exec(
        "SELECT tablename FROM pg_tables WHERE tablename LIKE 'documentblocks_partition_%';"
    );

    std::unordered_map<std::string, std::vector<std::pair<int, int>>> document_block_maps;

    for (const auto &row : partition_res) {
        std::string table_name = row[0].c_str();

        // Fetch document-block mapping for the current partition
        pqxx::result doc_res = txn.exec(
            "SELECT document_id, block_id FROM " + table_name + " ORDER BY document_id, block_id;"
        );

        std::vector<std::pair<int, int>> document_block_map;
        for (const auto &doc_row : doc_res) {
            document_block_map.emplace_back(doc_row[0].as<int>(), doc_row[1].as<int>());
        }

        if (document_block_map.empty()) {
            throw std::runtime_error("No document-block mappings found in partition: " + table_name);
        }

        document_block_maps[table_name] = std::move(document_block_map);
    }

    return document_block_maps;
}

// Function to get the size of a table in MB
double get_table_size_in_mb(const std::string &table_name, pqxx::connection &conn) {
    pqxx::work txn(conn);
    std::string query = "SELECT pg_table_size('" + table_name + "');"; // Use pg_table_size to exclude index size
    pqxx::result res = txn.exec(query);
    txn.commit();

    if (res.empty()) {
        return 0.0;
    }

    return res[0][0].as<double>() / (1024 * 1024);
}

// Function to calculate the total size of all specified tables
double calculate_table_sizes(const std::vector<std::string> &tables, pqxx::connection &conn) {
    double total_size = 0.0;
    for (const auto &table_name: tables) {
        total_size += get_table_size_in_mb(table_name, conn);
    }
    return total_size;
}

// Function to calculate index file sizes in a specific directory
double calculate_index_file_sizes(const std::string &directory_path) {
    double total_size = 0.0;

    // Open the directory
    DIR *dir = opendir(directory_path.c_str());
    if (dir == nullptr) {
        std::cerr << "Error: Could not open directory " << directory_path << std::endl;
        return 0.0;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        // Check if it's a regular file and has ".faiss" extension
        std::string file_name = entry->d_name;
        if (file_name.substr(file_name.find_last_of(".") + 1) == "faiss") {
            std::string file_path = directory_path + "/" + file_name;

            // Use stat to get file size
            struct stat file_stat;
            if (stat(file_path.c_str(), &file_stat) == 0) {
                total_size += file_stat.st_size / (1024.0 * 1024.0); // Convert bytes to MB
            } else {
                std::cerr << "Error: Could not get file size for " << file_path << std::endl;
            }
        }
    }

    // Close the directory
    closedir(dir);
    return total_size;
}

std::pair<double, double> print_database_and_index_statistics(const std::string &conn_info, const std::string &project_root) {
    try {
        pqxx::connection conn(conn_info);

        // Static tables
        std::vector<std::string> static_tables = {
            "Documents", "PermissionAssignment", "Roles", "UserRoles", "Users", "CombRolePartitions"
        };

        // Add dynamic partition tables
        pqxx::work txn(conn);
        pqxx::result partition_res = txn.exec("SELECT tablename FROM pg_tables WHERE tablename LIKE 'documentblocks_partition_%';");
        txn.commit();

        // Separate partition tables and documentblocks table
        std::vector<std::string> partition_tables;
        for (const auto &row : partition_res) {
            partition_tables.push_back(row[0].c_str());
        }

        // Add documentblocks to static tables for the first calculation
        std::vector<std::string> documentblocks_tables = static_tables;
        documentblocks_tables.push_back("documentblocks");

        // Add partition tables to static tables for the second calculation
        std::vector<std::string> dynamic_partition_tables = static_tables;
        dynamic_partition_tables.insert(dynamic_partition_tables.end(), partition_tables.begin(), partition_tables.end());

        // Calculate sizes
        double documentblocks_size_mb = calculate_table_sizes(documentblocks_tables, conn);
        double dynamic_partition_size_mb = calculate_table_sizes(dynamic_partition_tables, conn);

        // Index file paths
        std::string acorn_index_dir = project_root + "/acorn_benchmark/index_file/";
        std::string partition_index_dir = project_root + "/acorn_benchmark/index_file/dynamic_partition/";

        // Calculate index sizes
        double acorn_index_size_mb = calculate_index_file_sizes(acorn_index_dir);
        double partition_index_size_mb = calculate_index_file_sizes(partition_index_dir);

        // Return total sizes for ACORN and dynamic partition solutions
        return {
            documentblocks_size_mb + acorn_index_size_mb,
            dynamic_partition_size_mb + partition_index_size_mb
        };
    } catch (const std::exception &e) {
        std::cerr << "Error in statistics calculation: " << e.what() << std::endl;
        return {0.0, 0.0}; // Return zero in case of error
    } catch (...) {
        std::cerr << "Unknown error occurred in statistics calculation." << std::endl;
        return {0.0, 0.0}; // Return zero in case of unknown error
    }
}
