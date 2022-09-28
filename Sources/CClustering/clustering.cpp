//
//  clustering.cpp
//
//  Created by Julien Plu on 07/07/2022.
//

#include "clustering.hpp"


Model::Model(std::string model_path, uint16_t hidden_size) {
    std::string str_model_path(model_path);
    this->env = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "clustering");
    Ort::SessionOptions sessionOptions;
    
    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    this->session = std::make_unique<Ort::Session>(*this->env, model_path.data(), sessionOptions);
    this->hidden_size = hidden_size;
}

std::tuple<std::vector<float>, float> Model::predict(std::vector<int32_t> input_ids) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = this->session->GetInputCount();
    size_t num_output_nodes = this->session->GetOutputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<const char*> output_node_names(num_output_nodes);
    std::vector<int32_t> attention_mask(input_ids.size(), 1);
    std::vector<int32_t> token_type_ids(input_ids.size(), 0);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_node_dims = {1, static_cast<int64_t>(input_ids.size())};
    std::vector<Ort::Value> ort_inputs;
    
    ort_inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(memory_info, const_cast<int32_t*>(input_ids.data()), input_ids.size(), input_node_dims.data(), input_node_dims.size()));
    ort_inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(memory_info, const_cast<int32_t*>(attention_mask.data()), attention_mask.size(), input_node_dims.data(), input_node_dims.size()));
    ort_inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(memory_info, const_cast<int32_t*>(token_type_ids.data()), token_type_ids.size(), input_node_dims.data(), input_node_dims.size()));
    
    for (int i = 0; i < num_input_nodes; i++) {
        char* input_name = this->session->GetInputName(i, allocator);
        input_node_names[i] = input_name;
    }
    
    for (int i = 0; i < num_output_nodes; i++) {
        char* output_name = this->session->GetOutputName(i, allocator);
        output_node_names[i] = output_name;
    }
    
    std::vector<Ort::Value> output_tensors = this->session->Run(Ort::RunOptions{}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), output_node_names.size());
    float* output = output_tensors.front().GetTensorMutableData<float>();
    float ms = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
    std::vector<float> sentence_emdedding(output, output + this->hidden_size);
    
    return std::tuple<std::vector<float>, float>(sentence_emdedding, ms / 1000000);
}

Tokenizer::Tokenizer(std::string tokenizer_path, uint16_t max_seq_length) {
    std::string str_tokenizer_model_path(tokenizer_path);

    const auto status = this->tokenizer.Load(str_tokenizer_model_path);
    
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
    }
    
    this->max_seq_length = max_seq_length;
}

std::tuple<std::vector<int32_t>, float> Tokenizer::tokenize(std::string content) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    
    if (!this->tokenizer.status().ok()) {
        float ms = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
        
        return std::tuple<std::vector<int32_t>, float>(std::vector<int32_t>(), ms / 1000000);
    }
    
    std::vector<int32_t> input_ids;
    
    this->tokenizer.Encode(content, &input_ids);
    
    std::transform(input_ids.begin(), input_ids.end(), input_ids.begin(), [](int id){return id+1;});
    
    if (input_ids.size() > this->max_seq_length - 2) {
        input_ids.resize(this->max_seq_length - 2);
    }
    
    input_ids.push_back(2);
    input_ids.push_back(0);
    
    std::rotate(input_ids.rbegin(), input_ids.rbegin() + 1, input_ids.rend());
    
    float ms = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
    
    return std::tuple<std::vector<int32_t>, float>(input_ids, ms / 1000000);
}

Clustering::Clustering(const float threshold, const char* model_path, uint16_t hidden_size, const char* tokenizer_model_path, uint16_t max_seq_length) : model(model_path, hidden_size), tokenizer(tokenizer_model_path, max_seq_length) {
    if (threshold > 0) {
        this->threshold = threshold;
    }
}

inline float Clustering::norm(const std::vector<float> &vector) {
    float sum = std::inner_product(vector.begin(), vector.end(), vector.begin(), 0.0);
    
    return std::sqrt(sum);
}

inline std::vector<float> Clustering::normalize(const std::vector<float> &vector) {
    double norm_value = this->norm(vector);
    
    if (norm_value > 0) {
        std::vector<float> normalized_vector;
        
        for (int i = 0; i < vector.size(); i++) {
            normalized_vector.push_back(vector[i] / norm_value);
        }
        
        return normalized_vector;
    }
    
    return vector;
}

inline float Clustering::cosine_similarity(const std::vector<float> &vector1, const std::vector<float> &vector2) {
    std::vector<float> vector1_norm = this->normalize(vector1);
    std::vector<float> vector2_norm = this->normalize(vector2);
    std::vector<float> zeros(vector1.size(), 0);
    
    if (vector1_norm == zeros || vector2_norm == zeros) {
        return 0.0;
    }
    
    double vector1_norm_vector2_norm_dot_product = std::inner_product(vector1_norm.begin(), vector1_norm.end(), vector2_norm.begin(), 0.0);
    double similarity = vector1_norm_vector2_norm_dot_product / (this->norm(vector1_norm) * this->norm(vector2_norm));
    
    return similarity;
}

void Clustering::cosine_similarity_matrix() {
    this->similarities.clear();
    
    for (int i = 0;i < this->embeddings.size();i++) {
        std::vector<float> current_cosine_similarities;
        
        for (int j = 0;j < this->embeddings.size();j++) {
            current_cosine_similarities.push_back(this->cosine_similarity(this->embeddings[i], this->embeddings[j]));
        }
        
        this->similarities.push_back(current_cosine_similarities);
    }
}

inline std::vector<int> Clustering::argsort(const std::vector<float> &array) {
    std::vector<int> indices_vector(array.size());
    
    std::iota(indices_vector.begin(), indices_vector.end(), 0);
    std::sort(indices_vector.begin(), indices_vector.end(),
              [&array](int left, int right) -> bool {
                  return array[left] > array[right];
              });
    
    return indices_vector;
}

inline std::tuple<std::vector<float>, std::vector<int>> Clustering::topk(const uint16_t k, const std::vector<float> &array) {
    std::vector<int> indices_vector = this->argsort(array);
    std::vector<float> sorted_vector(array);
    
    std::sort(sorted_vector.begin(), sorted_vector.end(), std::greater<double>());
    
    return std::tuple<std::vector<float>, std::vector<int>>({sorted_vector.begin(), sorted_vector.begin() + k}, {indices_vector.begin(), indices_vector.begin() + k});
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<int>>> Clustering::topk_matrix(const uint16_t k) {
    std::vector<std::vector<float>> values;
    std::vector<std::vector<int>> indices;
    
    for (int i = 0;i < this->similarities.size();i++) {
        std::tuple<std::vector<float>, std::vector<int>> tmp_values_indices = this->topk(k, this->similarities[i]);
        
        values.push_back(std::get<0>(tmp_values_indices));
        indices.push_back(std::get<1>(tmp_values_indices));
    }
    
    return std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<int>>>(values, indices);
}

std::tuple<std::vector<uint16_t>, std::vector<uint16_t>> Clustering::compute_clusters() {
    std::vector<int> null_clusters;
    std::vector<std::vector<int>> extracted_clusters;
    std::vector<std::vector<float>> topk_values = std::get<0>(this->topk_matrix(1));

    for (int i = 0;i < topk_values.size();i++) {
        if (topk_values[i].back() == 0.0) {
            null_clusters.push_back(i);
        } else if (topk_values[i].back() >= this->threshold) {
            std::vector<int> new_cluster;
            std::tuple<std::vector<float>, std::vector<int>> topk_res = this->topk(this->embeddings.size(), this->similarities[i]);
            std::vector<float> top_val_large = std::get<0>(topk_res);
            std::vector<int> top_idx_large = std::get<1>(topk_res);
            
            if (top_val_large.back() < this->threshold) {
                for (int j = 0;j < top_idx_large.size();j++) {
                    if (top_val_large[j] <= this->threshold) {
                        break;
                    }
                    
                    new_cluster.push_back(top_idx_large[j]);
                }
            } else {
                for (int j = 0;j < this->similarities[i].size();j++) {
                    if (this->similarities[i][j] >= this->threshold) {
                        new_cluster.push_back(j);
                    }
                }
            }
            extracted_clusters.push_back(new_cluster);
        }
    }
    
    if (null_clusters.size() > 0) {
        extracted_clusters.push_back(null_clusters);
    }
    
    std::sort(extracted_clusters.begin(), extracted_clusters.end(), [](const std::vector<int> &a, const std::vector<int> &b){ return a.size() > b.size(); });
    std::vector<uint16_t> unique_clusters;
    std::vector<uint16_t> clusters_size;
    std::set<uint16_t> extracted_ids;

    for (int i = 0;i < extracted_clusters.size();i++) {
        std::vector<int> sorted_cluster(extracted_clusters[i]);
        std::vector<uint16_t> non_overlapped_cluster;
        
        std::sort(sorted_cluster.begin(), sorted_cluster.end());
        
        for (int j = 0;j < sorted_cluster.size();j++) {
            std::set<uint16_t>::iterator it = extracted_ids.find(sorted_cluster[j]);
            
            if (it == extracted_ids.end()) {
                non_overlapped_cluster.push_back(sorted_cluster[j]);
                extracted_ids.insert(sorted_cluster[j]);
            }
        }

        if (non_overlapped_cluster.size() >= 1) {
            unique_clusters.reserve(unique_clusters.size() + distance(non_overlapped_cluster.begin(), non_overlapped_cluster.end()));
            unique_clusters.insert(unique_clusters.end(), non_overlapped_cluster.begin(), non_overlapped_cluster.end());
            clusters_size.push_back(non_overlapped_cluster.size());
        }
    }
    
    return std::tuple<std::vector<uint16_t>, std::vector<uint16_t>>(unique_clusters, clusters_size);
}

int Clustering::add_textual_item(const char* text, const int idx, ClusteringResult* result) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::string content(text);
    
    if (content.size() == 0) {
        std::vector<float> zeros(this->model.hidden_size, 0);
        
        result->performance_tokenizer = 0;
        result->performance_inference = 0;
        
        auto it_pos = this->embeddings.begin() + idx;
        
        this->embeddings.insert(it_pos, zeros);
    } else {
        std::tuple<std::vector<int32_t>, float> tokenizer_output = this->tokenizer.tokenize(std::string(content));
        std::tuple<std::vector<float>, float> inference_output = this->model.predict(std::get<0>(tokenizer_output));
        
        result->performance_tokenizer = std::get<1>(tokenizer_output);
        result->performance_inference = std::get<1>(inference_output);
        
        auto it_pos = this->embeddings.begin() + idx;
        
        this->embeddings.insert(it_pos, std::get<0>(inference_output));
    }
    
    this->cosine_similarity_matrix();
    
    std::tuple<std::vector<uint16_t>, std::vector<uint16_t>> result_clusters = this->compute_clusters();
    
    assert(std::get<0>(result_clusters).size() == this->embeddings.size());
    
    this->format_clustering_result(result_clusters, result, start);
    
    return 0;
}

int Clustering::remove_textual_item(const int idx, const int from_add, ClusteringResult* result) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    this->embeddings.erase(this->embeddings.begin() + idx);
    
    for (int i = 0;i < this->similarities.size();i++) {
        this->similarities[i].erase(this->similarities[i].begin() + idx);
    }
    
    this->similarities.erase(this->similarities.begin() + idx);
    
    if (this->embeddings.size() > 0 && from_add == 0) {
        std::tuple<std::vector<uint16_t>, std::vector<uint16_t>> result_clusters = this->compute_clusters();
        
        assert(std::get<0>(result_clusters).size() == this->embeddings.size());
        
        result->performance_tokenizer = 0;
        result->performance_inference = 0;
        
        this->format_clustering_result(result_clusters, result, start);
    } else {
        result->cluster = new ClusterDefinition();
        result->cluster->indices = new uint16_t[0];
        result->cluster->clusters_split = new uint16_t[0];
    }
    
    return 0;
}

// Turn the C++ results into a Swift understandable structure.
void Clustering::format_clustering_result(std::tuple<std::vector<uint16_t>, std::vector<uint16_t>> result_clusters, ClusteringResult* result, std::chrono::high_resolution_clock::time_point start) {
    std::vector<uint16_t> unique_clusters = std::get<0>(result_clusters);
    std::vector<uint16_t> clusters_size = std::get<1>(result_clusters);
    std::vector<float> single_d_similarities;
    
    for (auto sim : this->similarities) {
        single_d_similarities.reserve(single_d_similarities.size() + distance(sim.begin(), sim.end()));
        single_d_similarities.insert(single_d_similarities.end(), sim.begin(), sim.end());
    }
    
    result->cluster = new ClusterDefinition();
    result->cluster->indices = new uint16_t[unique_clusters.size()];
    result->cluster->clusters_split = new uint16_t[clusters_size.size()];
    
    std::memcpy(result->cluster->indices, unique_clusters.data(), sizeof(uint16_t) * unique_clusters.size());
    std::memcpy(result->cluster->clusters_split, clusters_size.data(), sizeof(uint16_t) * clusters_size.size());
    
    result->cluster->indices_size = unique_clusters.size();
    result->cluster->clusters_split_size = clusters_size.size();
    
    float ms = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
    
    result->performance_clustering = ms / 1000000;
}

// Compute the optimal threshold for a given cluster.
int Clustering::recompute_clustering_threshold(const ClusterDefinition* expected_clusters, ClusteringResult* result) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    std::vector<uint16_t> converted_expected_clusters;
    
    for (int i = 0;i < expected_clusters->clusters_split_size;i++) {
        std::vector<uint16_t> cluster (expected_clusters->clusters_split[i], i);
        
        converted_expected_clusters.reserve(converted_expected_clusters.size() + distance(cluster.begin(), cluster.end()));
        converted_expected_clusters.insert(converted_expected_clusters.end(), cluster.begin(), cluster.end());
    }
    
    std::tuple<std::vector<uint16_t>, std::vector<uint16_t>> best_clusters;
    float best_acc = 0.0;
    float best_threshold = 0.0;
    
    result->performance_tokenizer = 0;
    result->performance_inference = 0;
    
    for (float i = 0.0001;i < 1.0;i+=0.0001) {
        this->threshold = i;
        
        std::tuple<std::vector<uint16_t>, std::vector<uint16_t>> result_clusters = this->compute_clusters();
        
        assert(std::get<0>(result_clusters).size() == this->similarities.size());
        
        std::vector<uint16_t> new_clusters;
        
        for (int j = 0;j < std::get<1>(result_clusters).size();j++) {
            std::vector<uint16_t> cluster (std::get<1>(result_clusters)[j], j);
            new_clusters.reserve(new_clusters.size() + distance(cluster.begin(), cluster.end()));
            new_clusters.insert(new_clusters.end(), cluster.begin(), cluster.end());
        }
        
        int diff = 0;
        
        for (int j = 0;j < new_clusters.size();j++) {
            if (new_clusters.at(j) != converted_expected_clusters.at(j)) {
                diff++;
            }
        }
        
        float acc = (new_clusters.size() - diff) / float(new_clusters.size());
        
        if (acc == 1) {
            best_clusters = result_clusters;
            
            break;
        } else {
            if (acc > best_acc) {
                best_threshold = i;
                best_acc = acc;
                best_clusters = result_clusters;
            }
        }
    }
    
    this->threshold = best_threshold;
    this->format_clustering_result(best_clusters, result, start);
    
    return 0;
}

float Clustering::get_threshold() {
    return this->threshold;
}
