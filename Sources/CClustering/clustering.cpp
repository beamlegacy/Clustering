//
//  model_inference_wrapper.cpp
//  sentencepiece-swift
//
//  Created by Julien Plu on 09/05/2022.
//

#include "clustering.hpp"
#include <iostream>
#include <vector>


ModelInferenceWrapper::ModelInferenceWrapper(const char* model_path, const char* tokenizer_model_path) {
    std::string str_tokenizer_model_path(tokenizer_model_path);

    const auto status = this->tokenizer.Load(str_tokenizer_model_path);
    
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
    }
    
    std::string str_model_path(model_path);
    this->env = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "clustering");
    Ort::SessionOptions sessionOptions;
    
    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    this->session = std::make_unique<Ort::Session>(*this->env, model_path, sessionOptions);
}

int ModelInferenceWrapper::infer(const char* text, ModelInferenceResult* result) {
    if (!this->tokenizer.status().ok()) {
        return 1;
    }
    
    std::vector<int32_t> input_ids_v;
    std::string content(text);
    
    this->tokenizer.Encode(content, &input_ids_v);
    
    std::transform(input_ids_v.begin(), input_ids_v.end(), input_ids_v.begin(), [](int id){return id+1;});
    
    if (input_ids_v.size() > 126) {
        input_ids_v.resize(126);
        input_ids_v.push_back(2);
    }
    
    input_ids_v.push_back(0);
    std::rotate(input_ids_v.rbegin(), input_ids_v.rbegin() + 1, input_ids_v.rend());
    
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
    result->weigths = new float[384];
    
    std::memcpy(result->weigths, sentence_embedding, sizeof(float)*384);
    
    result->size = 384;
    result->performance = ms;
    
    return 0;
}
