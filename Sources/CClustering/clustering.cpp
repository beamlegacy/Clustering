//
//  model_inference_wrapper.cpp
//  sentencepiece-swift
//
//  Created by Julien Plu on 09/05/2022.
//

#include "clustering.hpp"


Model::Model(const char* model_path, int32_t hidden_size) {
    std::string str_model_path(model_path);
    this->env = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "clustering");
    Ort::SessionOptions sessionOptions;
    
    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    this->session = std::make_unique<Ort::Session>(*this->env, model_path, sessionOptions);
    this->hidden_size = hidden_size;
}

int Model::predict(const TokenizerResult* tokenizer_result, ModelResult* result) {
    std::vector<int32_t> input_ids_v(tokenizer_result->input_ids, tokenizer_result->input_ids + tokenizer_result->size);
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = this->session->GetInputCount();
    size_t num_output_nodes = this->session->GetOutputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<const char*> output_node_names(num_output_nodes);
    std::vector<int32_t> attention_mask_v(input_ids_v.size(), 1);
    std::vector<int32_t> token_type_ids_v(input_ids_v.size(), 0);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_node_dims = {1, static_cast<int64_t>(input_ids_v.size())};
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(memory_info, const_cast<int32_t*>(input_ids_v.data()), input_ids_v.size(), input_node_dims.data(), input_node_dims.size()));
    ort_inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(memory_info, const_cast<int32_t*>(attention_mask_v.data()), attention_mask_v.size(), input_node_dims.data(), input_node_dims.size()));
    ort_inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(memory_info, const_cast<int32_t*>(token_type_ids_v.data()), token_type_ids_v.size(), input_node_dims.data(), input_node_dims.size()));
    
    for (int i = 0; i < num_input_nodes; i++) {
        char* input_name = this->session->GetInputName(i, allocator);
        input_node_names[i] = input_name;
    }
    
    for (int i = 0; i < num_output_nodes; i++) {
        char* output_name = this->session->GetOutputName(i, allocator);
        output_node_names[i] = output_name;
    }
    
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::vector<Ort::Value> output_tensors = this->session->Run(Ort::RunOptions{}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), output_node_names.size());
    float ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    float* sentence_embedding = output_tensors.front().GetTensorMutableData<float>();
    result->weigths = new float[this->hidden_size];
    
    std::memcpy(result->weigths, sentence_embedding, sizeof(float) * this->hidden_size);
    
    result->size = this->hidden_size;
    result->performance = ms;
    
    return 0;
}

Tokenizer::Tokenizer(const char* tokenizer_path, int32_t max_seq_length) {
    std::string str_tokenizer_model_path(tokenizer_path);

    const auto status = this->tokenizer.Load(str_tokenizer_model_path);
    
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
    }
    
    this->max_seq_length = max_seq_length;
}

int Tokenizer::tokenize(const char* text, TokenizerResult* result) {
    if (!this->tokenizer.status().ok()) {
        return 1;
    }
    
    std::vector<int32_t> input_ids_v;
    std::string content(text);
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    this->tokenizer.Encode(content, &input_ids_v);
    float ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    
    std::transform(input_ids_v.begin(), input_ids_v.end(), input_ids_v.begin(), [](int id){return id+1;});
    
    if (input_ids_v.size() > this->max_seq_length - 2) {
        input_ids_v.resize(this->max_seq_length - 2);
    }
    
    input_ids_v.push_back(2);
    input_ids_v.push_back(0);
    
    std::rotate(input_ids_v.rbegin(), input_ids_v.rbegin() + 1, input_ids_v.rend());
    
    result->input_ids = new int32_t[input_ids_v.size()];
    
    std::memcpy(result->input_ids, input_ids_v.data(), sizeof(int32_t) * input_ids_v.size());
    
    result->size = input_ids_v.size();
    result->performance = ms;
    
    return 0;
}

Clustering::Clustering() {}

double Clustering::norm(const std::vector<double> &vector) {
    double sum = std::inner_product(vector.begin(), vector.end(), vector.begin(), 0.0);
    
    return std::sqrt(sum);
}

std::vector<double> Clustering::normalize(const std::vector<double> &vector) {
    double norm_value = this->norm(vector);
    
    if (norm_value > 0) {
        std::vector<double> normalized_vector;
        
        for (int i = 0; i < vector.size(); i++) {
            normalized_vector.push_back(vector[i] / norm_value);
        }
        
        return normalized_vector;
    }
    
    return vector;
}

double Clustering::cosine_similarity(const std::vector<double> &vector1, const std::vector<double> &vector2) {
    std::vector<double> vector1_norm = this->normalize(vector1);
    std::vector<double> vector2_norm = this->normalize(vector2);
    std::vector<double> zeros(vector1.size(), 0);
    
    if (vector1_norm == zeros || vector2_norm == zeros) {
        return 0.0;
    }
    
    double vector1_norm_vector2_norm_dot_product = std::inner_product(vector1_norm.begin(), vector1_norm.end(), vector2_norm.begin(), 0.0);
    double similarity = vector1_norm_vector2_norm_dot_product / (this->norm(vector1_norm) * this->norm(vector2_norm));
    
    return similarity;
}

void Clustering::cosine_similarity_matrix(const std::vector<std::vector<double>> &embeddings) {
    this->similarities.clear();
    for (int i = 0;i < embeddings.size();i++) {
        std::vector<double> current_cosine_similarities;
        
        for (int j = 0;j < embeddings.size();j++) {
            current_cosine_similarities.push_back(this->cosine_similarity(embeddings[i], embeddings[j]));
        }
        
        this->similarities.push_back(current_cosine_similarities);
    }
}

std::vector<int> Clustering::argsort(const std::vector<double> &array) {
    std::vector<int> indices(array.size());
    
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] > array[right];
              });

    return indices;
}

std::tuple<std::vector<double>, std::vector<int>> Clustering::topk(const int k, const std::vector<double> &array) {
    std::vector<int> indices_vector = this->argsort(array);
    
    /*std::cerr << "C++: indices vector: [";
    for (auto val: indices_vector) {
        std::cerr << std::setprecision(15) << val << ", ";
    }
    std::cerr << "]" << std::endl;*/
    
    std::vector<double> sorted_vector(array);
    
    std::sort(sorted_vector.begin(), sorted_vector.end(), std::greater<int>());
    
    return std::tuple<std::vector<double>, std::vector<int>>({sorted_vector.begin(), sorted_vector.begin() + k}, {indices_vector.begin(), indices_vector.begin() + k});
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<int>>> Clustering::topk_matrix(const int k) {
    std::vector<std::vector<double>> values;
    std::vector<std::vector<int>> indices;
    
    for (int i = 0;i < this->similarities.size();i++) {
        std::tuple<std::vector<double>, std::vector<int>> tmp_values_indices = this->topk(k, this->similarities[i]);
        
        values.push_back(std::get<0>(tmp_values_indices));
        indices.push_back(std::get<1>(tmp_values_indices));
    }
    
    return std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<int>>>(values, indices);
}

int Clustering::create_clusters(const double** embeddings, const int hidden_size, const int nb_pages, ClusteringResult* result) {
    std::vector<std::vector<double>> converted_embeddings;
    
    for (int i = 0;i < nb_pages;i++) {
        std::vector<double> emdedding (embeddings[i], embeddings[i] + hidden_size);
        
        converted_embeddings.push_back(emdedding);
    }
    
    std::vector<int> null_clusters;
    std::vector<std::vector<int>> extracted_clusters;
    
    this->cosine_similarity_matrix(converted_embeddings);
    
    std::vector<std::vector<double>> topk_values = std::get<0>(this->topk_matrix(1));
    
    for (int i = 0;i < topk_values.size();i++) {
        if (topk_values[i].back() == 0.0) {
            null_clusters.push_back(i);
        } else if (topk_values[i].back() >= 0.3105) {
            std::vector<int> new_clusters;
            std::tuple<std::vector<double>, std::vector<int>> topk_res = this->topk(converted_embeddings.size(), this->similarities[i]);
            std::vector<double> top_val_large = std::get<0>(topk_res);
            std::vector<int> top_idx_large = std::get<1>(topk_res);
            
            if (top_val_large.back() < 0.3105) {
                for (int j = 0;j < top_idx_large.size();j++) {
                    if (top_val_large[j] < 0.3105) {
                        break;
                    }
                    
                    new_clusters.push_back(top_idx_large[j]);
                }
            } else {
                for (int j = 0;j < this->similarities[i].size();j++) {
                    if (this->similarities[i][j] >= 0.3105) {
                        new_clusters.push_back(j);
                    }
                }
            }
            
            extracted_clusters.push_back(new_clusters);
        }
    }
    
    if (null_clusters.size() > 0) {
        extracted_clusters.push_back(null_clusters);
    }
    
    std::sort(extracted_clusters.begin(), extracted_clusters.end(), [](const std::vector<int> &a, const std::vector<int> &b){ return a.size() > b.size(); });
    
    std::vector<int> unique_clusters;
    std::vector<unsigned long> clusters_size;
    std::set<int> extracted_ids;

    for (auto val: extracted_clusters) {
        std::cerr << "[";
        for (auto val2: val) {
            std::cerr << val2 << ", ";
        }
        std::cerr << "], ";
    }
    std::cerr << "]" << std::endl;
    for (int i = 0;i < extracted_clusters.size();i++) {
        std::vector<int> sorted_cluster(extracted_clusters[i]);
        std::vector<int> non_overlapped_cluster;
        
        std::sort(sorted_cluster.begin(), sorted_cluster.end());
        
        for (int j = 0;j < sorted_cluster.size();j++) {
            std::set<int>::iterator it = extracted_ids.find(sorted_cluster[j]);
            
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
    
    assert(unique_clusters.size() == converted_embeddings.size());
    
    result->indices = new int[unique_clusters.size()];
    result->clusters_split = new unsigned long[clusters_size.size()];
    
    std::memcpy(result->indices, unique_clusters.data(), sizeof(int) * unique_clusters.size());
    std::memcpy(result->clusters_split, clusters_size.data(), sizeof(unsigned long) * clusters_size.size());
    
    result->indices_size = unique_clusters.size();
    result->clusters_split_size = clusters_size.size();
    
    return 0;
}
