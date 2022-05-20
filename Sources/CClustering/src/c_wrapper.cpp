//
//  c_wrapper.cpp
//  sentencepiece-swift
//
//  Created by Julien Plu on 09/05/2022.
//

#include "../clustering.hpp"
#include <iostream>
#include <algorithm>

extern "C" void* createModelInferenceWrapper(const char* model_path, const char* tokenizer_model_path) {
    ModelInferenceWrapper* model_inference_wrapper = new ModelInferenceWrapper(model_path, tokenizer_model_path);
    
    return (void*) model_inference_wrapper;
}

extern "C" int doModelInference(void* handle, const char* text, struct ModelInferenceResult* result) {
    ModelInferenceWrapper* model_inference_wrapper = (ModelInferenceWrapper*)handle;
    model_inference_wrapper->infer(text, result);
    
    return 0;
}

extern "C" void removeModelInferenceWrapper(void* handle) {
    ModelInferenceWrapper* model_inference_wrapper = (ModelInferenceWrapper*)handle;
    
    delete model_inference_wrapper;
}
