#include "ort_pointpillars_infer.h"


bool OrtPointPillarsInfer::LoadONNXModel(const std::string& model_path)
{
    try{
        LoadModel(model_path, g_pointpillars_s_);
    }catch(Ort::Exception& e){
        LOGPF("ort exception: %s\n", e.what());
        return false;
    }
    return true;
}

/* img should be rgb format */
void OrtPointPillarsInfer::RunPointpillarsModel(POINTPILLARS_MODEL_INPUT_t& input, POINTPILLARS_MODEL_OUTPUT_t& output)
{
    HANG_STOPWATCH();
    /* prepare input data */
    std::vector<const char*>& input_node_names = g_pointpillars_s_.input_node_names;
    std::vector<std::vector<int64_t>>& input_node_dims = g_pointpillars_s_.input_node_dims;
    std::vector<ONNXTensorElementDataType>& input_types = g_pointpillars_s_.input_types;
    std::vector<OrtValue*>& input_tensors = g_pointpillars_s_.input_tensors;

    std::vector<const char*>& output_node_names = g_pointpillars_s_.output_node_names;
    std::vector<std::vector<int64_t>>& output_node_dims = g_pointpillars_s_.output_node_dims;
    std::vector<ONNXTensorElementDataType>& output_types = g_pointpillars_s_.output_types;
    std::vector<OrtValue*>& output_tensors = g_pointpillars_s_.output_tensors;

    /* move input vector into input_tensors */
    OrtMemoryInfo* memory_info;
    CheckStatus(g_ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    input_node_dims[0][0] = input.voxels.size()/(64*4);
    LOGPF("reset input_node_dims[0][0] to %ld", input_node_dims[0][0]);
    CheckStatus(g_ort_->CreateTensorWithDataAsOrtValue(
                    memory_info, reinterpret_cast<void*>(input.voxels.data()), sizeof(float)*input.voxels.size(),
                    input_node_dims[0].data(), input_node_dims[0].size(), input_types[0], &input_tensors[0]));
    g_ort_->ReleaseMemoryInfo(memory_info);
    LOGPF("CreateTensorWithDataAsOrtValue for voxels: %ld", sizeof(float)*input.voxels.size());

    CheckStatus(g_ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    input_node_dims[1][0] = input.num_points.size();
    LOGPF("reset input_node_dims[1][0] to %ld", input_node_dims[1][0]);
    CheckStatus(g_ort_->CreateTensorWithDataAsOrtValue(
                    memory_info, reinterpret_cast<void*>(input.num_points.data()), sizeof(int32_t)*input.num_points.size(),
                    input_node_dims[1].data(), input_node_dims[1].size(), input_types[1], &input_tensors[1]));
    g_ort_->ReleaseMemoryInfo(memory_info);
    LOGPF("CreateTensorWithDataAsOrtValue for num_points: %ld", sizeof(int32_t)*input.num_points.size());

    CheckStatus(g_ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    input_node_dims[2][0] = input.coors.size()/4;
    LOGPF("reset input_node_dims[2][0] to %ld", input_node_dims[2][0]);
    CheckStatus(g_ort_->CreateTensorWithDataAsOrtValue(
                    memory_info, reinterpret_cast<void*>(input.coors.data()), sizeof(int32_t)*input.coors.size(),
                    input_node_dims[2].data(), input_node_dims[2].size(), input_types[2], &input_tensors[2]));
    g_ort_->ReleaseMemoryInfo(memory_info);
    LOGPF("CreateTensorWithDataAsOrtValue for coors: %ld", sizeof(int32_t)*input.coors.size());

    /* do inference */
    CheckStatus(g_ort_->Run(g_pointpillars_s_.sess, nullptr, input_node_names.data(), (const OrtValue* const*)input_tensors.data(),
                    input_tensors.size(), output_node_names.data(), output_node_names.size(), output_tensors.data()));

    /* postprocess */
    // assert (output_node_names.size() == 1);
    // LOGPF("retrieve output[0]: %s\n", output_node_names[0]);
    float* cls_score;
    CheckStatus(g_ort_->GetTensorMutableData(output_tensors[0], reinterpret_cast<void**>(&cls_score)));
    size_t cls_score_size = 1;
    for(size_t k=0; k<output_node_dims[0].size(); k++)
    {
        cls_score_size *= output_node_dims[0][k];
    }
    output.cls_score.assign(cls_score, cls_score+cls_score_size);
    LOGPF("cls_score_size: %ld", cls_score_size);

    float* bbox_pred;
    CheckStatus(g_ort_->GetTensorMutableData(output_tensors[1], reinterpret_cast<void**>(&bbox_pred)));
    size_t bbox_pred_size = 1;
    for(size_t k=0; k<output_node_dims[1].size(); k++)
    {
        bbox_pred_size *= output_node_dims[1][k];
    }
    output.bbox_pred.assign(bbox_pred, bbox_pred+bbox_pred_size);
    LOGPF("bbox_pred_size: %ld", bbox_pred_size);

    float* dir_cls_pred;
    CheckStatus(g_ort_->GetTensorMutableData(output_tensors[2], reinterpret_cast<void**>(&dir_cls_pred)));
    size_t dir_cls_pred_size = 1;
    for(size_t k=0; k<output_node_dims[2].size(); k++)
    {
        dir_cls_pred_size *= output_node_dims[2][k];
    }
    output.dir_cls_pred.assign(dir_cls_pred, dir_cls_pred+dir_cls_pred_size);
    LOGPF("dir_cls_pred: %ld", dir_cls_pred_size);
}

void OrtPointPillarsInfer::TestPointpillarsModel()
{
    LOGPF("testing pointpillar onnx model\n");
    POINTPILLARS_MODEL_INPUT_t input;
    POINTPILLARS_MODEL_OUTPUT_t output;

    for(int i=1; i<10; i++)
    {
        const size_t N = MAX_NUM_VOXELS_ / i;
        input.voxels.resize(N*64*4);
        input.num_points.resize(N);
        input.coors.resize(N*4);
        RunPointpillarsModel(input, output);
    }
}

void OrtPointPillarsInfer::LoadModel(const std::string& model_path, ORT_S_t& model_s)
{
    LOGPF("loading onnx model: %s\n", model_path.c_str());

    CheckStatus(g_ort_->CreateSession(env_, model_path.c_str(), session_options_, &model_s.sess));
    LOGPF("CreateSession sucess.");

    OrtAllocator* allocator;
    CheckStatus(g_ort_->GetAllocatorWithDefaultOptions(&allocator));
    LOGPF("GetAllocatorWithDefaultOptions sucess.");

    size_t num_input_nodes;
    CheckStatus(g_ort_->SessionGetInputCount(model_s.sess, &num_input_nodes));
    LOGPF("SessionGetInputCount sucess: %ld", num_input_nodes);

    std::vector<const char*>& input_node_names = model_s.input_node_names;
    std::vector<std::vector<int64_t>>& input_node_dims = model_s.input_node_dims;
    std::vector<ONNXTensorElementDataType>& input_types = model_s.input_types;
    std::vector<OrtValue*>& input_tensors = model_s.input_tensors;

    input_node_names.resize(num_input_nodes);
    input_node_dims.resize(num_input_nodes);
    input_types.resize(num_input_nodes);
    input_tensors.resize(num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; i++) 
    {
        // Get input node names
        char* input_name;
        CheckStatus(g_ort_->SessionGetInputName(model_s.sess, i, allocator, &input_name));
        input_node_names[i] = input_name;
        LOGPF("input_node_names[%d]:  %s", i, input_name);

        // Get input node types
        OrtTypeInfo* typeinfo;
        CheckStatus(g_ort_->SessionGetInputTypeInfo(model_s.sess, i, &typeinfo));
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort_->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_ort_->GetTensorElementType(tensor_info, &type));
        input_types[i] = type;
        LOGPF("input_types[%d]: %d", i, (int)(type));

        // Get input shapes/dims
        size_t num_dims;
        CheckStatus(g_ort_->GetDimensionsCount(tensor_info, &num_dims));
        LOGPF("GetDimensionsCount[%d]: %d", i, (int)(num_dims));

        input_node_dims[i].resize(num_dims);
        CheckStatus(g_ort_->GetDimensions(tensor_info, input_node_dims[i].data(), num_dims));
        for(size_t j=0; j<num_dims; j++)
        {
            LOGPF("GetDimensions[%d][%d]: %ld", i, j, input_node_dims[i][j]);
        }

        // size_t tensor_size;
        // CheckStatus(g_ort_->GetTensorShapeElementCount(tensor_info, &tensor_size));
        // LOGPF("GetTensorShapeElementCount[%d]: %ld", i, tensor_size);

        std::string dimstr="(";
        for(int k=0; k<num_dims; ++k){
            dimstr += std::to_string(input_node_dims[i][k]);
            dimstr += ",";
        }
        dimstr += ")";

        /* print input tensor information */
        LOGPF("input[%ld]-%s, type: %d, dims: %s\n", i, input_name, type, dimstr.c_str());

        if (typeinfo) g_ort_->ReleaseTypeInfo(typeinfo);
    }

    size_t num_output_nodes;
    std::vector<const char*>& output_node_names = model_s.output_node_names;
    std::vector<std::vector<int64_t>>& output_node_dims = model_s.output_node_dims;
    std::vector<ONNXTensorElementDataType>& output_types = model_s.output_types;
    std::vector<OrtValue*>& output_tensors = model_s.output_tensors;

    CheckStatus(g_ort_->SessionGetOutputCount(model_s.sess, &num_output_nodes));
    LOGPF("num_output_nodes: %ld\n", num_output_nodes);
    output_node_names.resize(num_output_nodes);
    output_node_dims.resize(num_output_nodes);
    output_tensors.resize(num_output_nodes);
    output_types.resize(num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; i++) 
    {
        // Get output node names
        char* output_name;
        CheckStatus(g_ort_->SessionGetOutputName(model_s.sess, i, allocator, &output_name));
        output_node_names[i] = output_name;
        // LOGPF("%ld-output_name: %s\n", i, output_name);

        OrtTypeInfo* typeinfo;
        CheckStatus(g_ort_->SessionGetOutputTypeInfo(model_s.sess, i, &typeinfo));
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort_->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_ort_->GetTensorElementType(tensor_info, &type));
        output_types[i] = type;
        // LOGPF("%ld-type: %d\n", i, type);

        // Get output shapes/dims
        size_t num_dims;
        CheckStatus(g_ort_->GetDimensionsCount(tensor_info, &num_dims));
        output_node_dims[i].resize(num_dims);
        CheckStatus(g_ort_->GetDimensions(tensor_info, (int64_t*)output_node_dims[i].data(), num_dims));
        // LOGPF("%ld-num_dims: %ld\n", i, num_dims);

        /* when it's variable output, tensor_size could be negative, so tensor_size will overflow */
        // size_t tensor_size;
        // CheckStatus(g_ort_->GetTensorShapeElementCount(tensor_info, &tensor_size));
        // LOGPF("%ld-tensor_size: %ld\n", i, tensor_size);

        std::string dimstr="(";
        for(int k=0; k<num_dims; ++k){
            dimstr += std::to_string(output_node_dims[i][k]);
            dimstr += ",";
        }
        dimstr += ")";
        /* print output tensor information */
        LOGPF("output[%ld]-%s, type: %d, dims: %s\n", i, output_name, type, dimstr.c_str());

        if (typeinfo) g_ort_->ReleaseTypeInfo(typeinfo);
    }
}

bool OrtPointPillarsInfer::CheckStatus(OrtStatus* status) 
{
    if (status != nullptr) {
        const char* msg = g_ort_->GetErrorMessage(status);
        std::cerr << msg << std::endl;
        // g_ort_->ReleaseStatus(status);
        throw Ort::Exception(msg, OrtErrorCode::ORT_EP_FAIL);
    }
    return true;
}

bool OrtPointPillarsInfer::InitOrt()
{
    g_ort_base_ = OrtGetApiBase();
    if (!g_ort_base_){
        LOGPF("Failed to OrtGetApiBase.\n");
        return false;
    }

    LOGPF("ort version: %s\n", g_ort_base_->GetVersionString());

    g_ort_ = g_ort_base_->GetApi(ORT_API_VERSION);
    if (!g_ort_) {
        LOGPF("Failed to init ONNX Runtime engine.\n");
        return false;
    }

    CheckStatus(g_ort_->CreateEnv(ORT_LOGGING_LEVEL_INFO, "pointpillars", &env_));
    if (!env_) {
        LOGPF("Failed to CreateEnv.\n");
        return false;
    }

    /* use default session option is ok */
    CheckStatus(g_ort_->CreateSessionOptions(&session_options_));
    // if(EnableCuda(session_options_)<0){
    //     return false;
    // }
    // CheckStatus(g_ort_->SetIntraOpNumThreads(session_options_, 1));
    // CheckStatus(g_ort_->SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_ALL));
    // std::vector<const char*> options_keys = {"runtime", "buffer_type"};
    // std::vector<const char*> options_values = {backend.c_str(), "FLOAT"};  // set to TF8 if use quantized data
    // CheckStatus(g_ort_->SessionOptionsAppendExecutionProvider(session_options_, "SNPE", options_keys.data(), options_values.data(), options_keys.size()));

    return true;
}

void OrtPointPillarsInfer::DestroyOrt()
{
    if(session_options_) g_ort_->ReleaseSessionOptions(session_options_);
    if(g_pointpillars_s_.sess) g_ort_->ReleaseSession(g_pointpillars_s_.sess);
    if(env_) g_ort_->ReleaseEnv(env_);
}

void OrtPointPillarsInfer::VerifyInputOutputCount(OrtSession* sess) 
{
    size_t count;
    CheckStatus(g_ort_->SessionGetInputCount(sess, &count));
    if(count != 1)
    {
        LOGPF("SessionGetInputCount: %ld", count);
        abort();
    }
    CheckStatus(g_ort_->SessionGetOutputCount(sess, &count));
    if(count != 1)
    {
        LOGPF("SessionGetInputCount: %ld", count);
        abort();
    }
}

int OrtPointPillarsInfer::EnableCuda(OrtSessionOptions* session_options) 
{
    // OrtCUDAProviderOptions is a C struct. C programming language doesn't have constructors/destructors.
    OrtCUDAProviderOptions o;
    // Here we use memset to initialize every field of the above data struct to zero.
    memset(&o, 0, sizeof(o));
    // But is zero a valid value for every variable? Not quite. It is not guaranteed. In the other words: does every enum
    // type contain zero? The following line can be omitted because EXHAUSTIVE is mapped to zero in onnxruntime_c_api.h.
    o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    o.gpu_mem_limit = SIZE_MAX;
    OrtStatus* onnx_status = g_ort_->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
    if (onnx_status != NULL) {
        const char* msg = g_ort_->GetErrorMessage(onnx_status);
        LOGPF("%s\n", msg);
        g_ort_->ReleaseStatus(onnx_status);
        return -1;
    }
    return 0;
}

#if 0
void OrtPointPillarsInfer::DmprOnnxPreprocess(const cv::Mat& img, const size_t h, const size_t w, std::vector<float>& output)
{
    // HANG_STOPWATCH();
    const size_t c = img.channels();
    std::vector<float> vec_hwc(OpenclHelper::normalize(OpenclHelper::resize(img, w, h)).reshape(1, 1));
    OpenclHelper::hwc2chw<float, float>(vec_hwc, h, w, c, output, 1.0f);
}
#endif