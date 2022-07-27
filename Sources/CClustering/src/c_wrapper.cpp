//
//  c_wrapper.cpp
//
//  Created by Julien Plu on 09/05/2022.
//

#include "../clustering.hpp"


extern "C" void* createModel(const char* model_path, int32_t hidden_size) {
    Model* model = new Model(model_path, hidden_size);
    
    return (void*) model;
}

extern "C" int predict(void* handle, const struct TokenizerResult* tokenizer_result, struct ModelResult* result) {
    Model* model = (Model*)handle;
    model->predict(tokenizer_result, result);
    
    return 0;
}

extern "C" void removeModel(void* handle) {
    Model* model = (Model*)handle;
    
    delete model;
}

extern "C" void* createTokenizer(const char* tokenizer_path, int32_t max_seq_length) {
    Tokenizer* tokenizer = new Tokenizer(tokenizer_path, max_seq_length);
    
    return (void*) tokenizer;
}

extern "C" int tokenize(void* handle, const char* text, struct TokenizerResult* result) {
    Tokenizer* tokenizer = (Tokenizer*)handle;
    tokenizer->tokenize(text, result);
    
    return 0;
}

extern "C" void removeTokenizer(void* handle) {
    Tokenizer* tokenizer = (Tokenizer*)handle;
    
    delete tokenizer;
}
