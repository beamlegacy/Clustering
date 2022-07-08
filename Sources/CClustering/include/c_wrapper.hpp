//
//  c_wrapper.hpp
//  clustering
//
//  Created by Julien Plu on 07/07/2022.
//

#ifndef c_wrapper_hpp
#define c_wrapper_hpp

struct ModelInferenceResult {
    float* weigths;
    unsigned long size;
    float performance;
};


void* createModelInferenceWrapper(const char* model_path, const char* tokenizer_model_path);
int doModelInference(void* handle, const char* text, struct ModelInferenceResult* result);
void removeModelInferenceWrapper(void* handle);

#endif /* c_wrapper_hpp */
