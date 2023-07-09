#pragma once
#include <torch/script.h>

class TorchModel {
public:
    TorchModel(const char* model_path){
        try{
            m_model = torch::jit::load(model_path);
            m_model.eval();
        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
        }
    };

    
    

private:
    torch::jit::script::Module m_model;
};