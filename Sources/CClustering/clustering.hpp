//
//  clustering.hpp
//  clustering
//
//  Created by Julien Plu on 07/07/2022.
//

#ifndef clustering_hpp
#define clustering_hpp

#include "sentencepiece_processor.hpp"
#include "onnxruntime_cxx_api.h"
#include <iostream>
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
        std::unique_ptr<Ort::Session> session;
        std::unique_ptr<Ort::Env> env;
    public:
        ModelInferenceWrapper(const char* model_path, const char* tokenizer_model_path);
        int infer(const char* text, ModelInferenceResult* result);
};

#endif /* clustering_hpp */
