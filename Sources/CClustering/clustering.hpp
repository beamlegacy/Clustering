//
//  model_inference_wrapper.hpp
//  sentencepiece-swift
//
//  Created by Julien Plu on 09/05/2022.
//

#ifndef clustering_hpp
#define clustering_hpp

#include "sentencepiece_processor.hpp"
#include "tvm/runtime/module.h"
#include "tvm/runtime/packed_func.h"
#include "tvm/runtime/registry.h"
#include <iostream>
#include "Eigen/Dense"
#include <chrono>
#include <thread>


struct ModelInferenceResult {
    float* weigths;
    unsigned long size;
    float performance;
};

class ModelInferenceWrapper {
    private:
        sentencepiece::SentencePieceProcessor tokenizer;
        tvm::runtime::Module model;
    public:
        ModelInferenceWrapper(const char* model_path, const char* tokenizer_model_path);
        int infer(const char* text, ModelInferenceResult* result);
};

#endif /* clustering_hpp */
