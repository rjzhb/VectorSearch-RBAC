#include "json_utils.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

std::string get_project_root() {
    std::filesystem::path file_path = std::filesystem::absolute(__FILE__);

    return (file_path.parent_path().parent_path().parent_path()).string();
}


// Parse database configuration from a JSON file
std::string parse_db_config(const std::string& config_path) {
    // Open the JSON file
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }

    // Parse the JSON file
    nlohmann::json config;
    config_file >> config;

    // Build the connection string
    std::string conn_info = "dbname=" + config["dbname"].get<std::string>() +
                            " user=" + config["user"].get<std::string>() +
                            " password=" + config["password"].get<std::string>() +
                            " host=" + config["host"].get<std::string>() +
                            " port=" + config["port"].get<std::string>();

    return conn_info;
}

std::vector<Query> read_queries(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open query dataset file.");
    }

    nlohmann::json json_data;
    file >> json_data;

    std::vector<Query> queries;
    for (const auto& query : json_data) {
        Query q;
        q.user_id = query["user_id"];
        std::string vec_str = query["query_vector"];

        // Parse vector from JSON string
        std::stringstream ss(vec_str.substr(1, vec_str.size() - 2));
        std::vector<float> vector;
        float value;
        while (ss >> value) {
            vector.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }

        q.query_vector = std::move(vector);
        q.topk = query["topk"];
        q.query_block_selectivity = query["query_block_selectivity"];
        queries.push_back(q);
    }

    return queries;
}