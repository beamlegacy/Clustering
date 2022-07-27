//
//  clustering.hpp
//
//  Created by Julien Plu on 07/07/2022.
//

#ifndef clustering_hpp
#define clustering_hpp

#include "sentencepiece_processor.hpp"
#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>


struct ModelResult {
    float* weigths;
    unsigned long size;
    float performance;
};

struct TokenizerResult {
    int32_t* input_ids;
    unsigned long size;
    float performance;
};

class Tokenizer {
    private:
        sentencepiece::SentencePieceProcessor tokenizer;
        int32_t max_seq_length;
    public:
        Tokenizer(const char* tokenizer_model_path, int32_t max_seq_length);
        int tokenize(const char* text, TokenizerResult* result);
};

class Model {
    private:
        std::unique_ptr<Ort::Session> session;
        std::unique_ptr<Ort::Env> env;
        int32_t hidden_size;
    public:
        Model(const char* model_path, int32_t hidden_size);
        int predict(const TokenizerResult* tokenizer_result, ModelResult* result);
};

#endif /* clustering_hpp */
