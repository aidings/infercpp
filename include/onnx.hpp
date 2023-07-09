#include <string>
#include <cmath>
#include <numeric>
#include <iostream>
#include "onnxruntime_cxx_api.h"
using namespace std;


class OnnxModel
{
public:
    OnnxModel()
    {
        m_session_options.SetIntraOpNumThreads(1);
        m_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    };

    void load(const string& model_path){
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        m_session = Ort::Session(env, model_path.c_str(), m_session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_input_nodes = m_session.GetInputCount();
        size_t num_output_nodes = m_session.GetOutputCount();
       
        std::vector<int64_t> tensor_shape;

        m_input_node_dims.clear();
        m_input_node_names.clear();

        for (int i = 0; i < num_input_nodes; i++) {
            const char* input_name = m_session.GetInputName(i, allocator);
            m_input_node_names.push_back(input_name);
            Ort::TypeInfo type_info = m_session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            tensor_shape = tensor_info.GetShape();
            
            m_input_node_dims.push_back(tensor_shape);
        }

        m_output_node_dims.clear();
        m_output_node_names.clear();
        m_output_node_types.clear();
        for (int i = 0; i < num_output_nodes; i++) {
            const char* output_name = m_session.GetOutputName(i, allocator);
            m_output_node_names.push_back(output_name);
            Ort::TypeInfo type_info = m_session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            m_output_node_types.push_back(tensor_info.GetElementType());
            tensor_shape = tensor_info.GetShape();
            m_output_node_dims.push_back(tensor_shape);
        }
        m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    }

    const string get_input_node_name(int index=0)
    {
        return m_input_node_names[index];
    };

    const string get_output_node_name(int index=0)
    {
        return m_output_node_names[index];
    };

    std::vector<Ort::Value> inference(std::vector<Ort::Value>& model_input)
    {
        auto output_tensors = m_session.Run(
                    Ort::RunOptions{ nullptr }, 
                    m_input_node_names.data(), 
                    &model_input, 
                    m_input_node_names.size(),
                    m_output_node_names.data(),
                    m_output_node_names.size());
        
        try{
            assert(output_tensors.size() == m_output_node_names.size());
        }
        catch(const std::exception& e){
            std::cerr << e.what() << '\n';
        }

        return output_tensors;
    };

    const int64_t* get_output_shape(int index=0)
    {
        return m_output_node_dims[index].data();
    };

protected:
    template <typename T>
    void create_tensor(std::vector<int64_t> tensor_shape, std::vector<Ort::Value>& model_tensor, const std::vector<T>& data=std::vector<T>()){
        int64_t size = shape2size(tensor_shape);
        if (data.size() == 0){
            data = std::vector<T>(size, 0);
        }
        else if (data.size() != size){
            throw std::runtime_error("data size does not match tensor size");
        }
        Ort::Value tensor = Ort::Value::CreateTensor<T>(m_memoryInfo, outputTensorValues.data(), size, outputDims.data(), outputDims.size()));
        model_tensor.push_back(tensor);
    };

    template <typename T>
    const T* get_tensor_ptr(Ort::Value& tensor){
        return tensor.GetTensorData<T>();
    };

    const void* get_output_ptr(std::vector<Ort::Value>& output_tensors, int index=0){
        int32_t type = m_output_node_types[index];
        const void* output_ptr = nullptr;
        switch (type)
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            output_ptr = output_tensors[index].GetTensorData<float>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            output_ptr = output_tensors[index].GetTensorData<uint8_t>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            output_ptr = output_tensors[index].GetTensorData<int8_t>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            output_ptr = output_tensors[index].GetTensorData<uint16_t>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            output_ptr = output_tensors[index].GetTensorData<int16_t>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            output_ptr = output_tensors[index].GetTensorData<int32_t>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            output_ptr = output_tensors[index].GetTensorData<int64_t>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            output_ptr = output_tensors[index].GetTensorData<double>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            output_ptr = output_tensors[index].GetTensorData<uint32_t>();
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            output_ptr = output_tensors[index].GetTensorData<uint64_t>();
            break;
        default:
            throw std::runtime_error("unsupported data type");
        }

        return output_ptr;
    };

    int64_t shape2size(std::vector<int64_t> v){
        return accumulate(v.begin(), v.end(), 1, std::multiplies<int64_t>());
    };
private:
    Ort::MemoryInfo m_memoryInfo;
    Ort::Session m_session;
    Ort::SessionOptions m_session_options;
    std::vector<const char*> m_input_node_names;
    std::vector<const char*> m_output_node_names;
    std::vector<std::vector<int64_t>> m_input_node_dims;
    std::vector<std::vector<int64_t>> m_output_node_dims;
    std::vector<int32_t> m_output_node_types;
};