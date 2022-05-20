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
    
    this->model = tvm::runtime::Module::LoadFromFile(str_model_path);
    
}

int ModelInferenceWrapper::infer(const char* text, ModelInferenceResult* result) {
    if (!this->tokenizer.status().ok()) {
        return 1;
    }
    
    std::vector<int> ids;
    std::string content(text);
    
    this->tokenizer.Encode(content, &ids);
    
    std::transform(ids.begin(), ids.end(), ids.begin(), [](int id){return id+1;});
    
    ids.push_back(2);
    
    for (size_t i = ids.size() + 1;i < 512;i++) {
        ids.push_back(1);
    }
    
    ids.push_back(0);
    
    std::rotate(ids.rbegin(), ids.rbegin() + 1, ids.rend());
    
    DLDevice dev = {kDLMetal, 0};
    DLDevice cpu = {kDLCPU, 0};
    std::cerr << "oijoijo" << std::endl;
    try {
        tvm::runtime::Module gmod = this->model.GetFunction("default")(dev);
        std::cerr << "oijoijo" << std::endl;
        tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
        tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
        tvm::runtime::PackedFunc run = gmod.GetFunction("run");
        tvm::runtime::NDArray input_ids = tvm::runtime::NDArray::Empty({1, 512}, DLDataType{kDLInt, 64, 1}, cpu);
        tvm::runtime::NDArray attention_mask = tvm::runtime::NDArray::Empty({1, 512}, DLDataType{kDLFloat, 32, 1}, cpu);
        
        for (size_t i = 0;i < 512;i++) {
            static_cast<int64_t*>(input_ids->data)[i] = ids[i];
            if (ids[i] != 1) {
                static_cast<float*>(attention_mask->data)[i] = 1.0;
            } else {
                static_cast<float*>(attention_mask->data)[i] = 0.0;
            }
        }
        
        set_input(0, input_ids);
        set_input(1, attention_mask);
        
        TVMSynchronize(cpu.device_type, cpu.device_id, nullptr);
        TVMSynchronize(dev.device_type, dev.device_id, nullptr);
        
        auto start = std::chrono::steady_clock::now();
        
        run();
        
        auto end = std::chrono::steady_clock::now();
        float ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        tvm::runtime::NDArray last_hidden_state = tvm::runtime::NDArray::Empty({1, 512, 384}, DLDataType{kDLFloat, 32, 1}, cpu);
        
        get_output(0, last_hidden_state);
        
        Eigen::MatrixXf token_embeddings(ids.size(), 384);
        Eigen::MatrixXf input_mask_expanded(ids.size(), 384);
        Eigen::MatrixXf clamped_input_mask_expanded(1, 384);
        Eigen::VectorXf final_output(384);
        
        for (size_t i = 0;i < ids.size();i++) {
            for (size_t j = 0;j < 384;j++) {
                token_embeddings(i, j) = static_cast<float*>(last_hidden_state->data)[(i+1)*j];
                input_mask_expanded(i, j) = static_cast<float*>(attention_mask->data)[i];

            }
        }
        
        clamped_input_mask_expanded = input_mask_expanded.colwise().sum();
        clamped_input_mask_expanded = (clamped_input_mask_expanded.array() == 0.0).select(0.000000001, clamped_input_mask_expanded);
        final_output = ((token_embeddings.array() * input_mask_expanded.array()).matrix().colwise().sum()).array() / clamped_input_mask_expanded.array();
        result->weigths = new float[final_output.size()];
        
        std::memcpy(result->weigths, final_output.data(), sizeof(float)*final_output.size());
        
        result->size = final_output.size();
        result->performance = ms;
    
        return 0;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}
