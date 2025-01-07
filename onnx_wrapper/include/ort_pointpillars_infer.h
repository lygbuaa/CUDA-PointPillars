#ifndef __ORT_PILLAR_INFER_H__
#define __ORT_PILLAR_INFER_H__

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include "logging_utils.h"
// #include "apa_utils.h"
// #include "cl_helper.h"

typedef struct
{
    std::vector<float> voxels;          /** (-1, 64, 4) */
    std::vector<int32_t> num_points;    /** (-1) */
    std::vector<int32_t> coors;         /** (-1, 4) */
}POINTPILLARS_MODEL_INPUT_t;

typedef struct
{
    std::vector<float> cls_score;       /** (1, 140, 200, 200) */
    std::vector<float> bbox_pred;       /** (1, 126, 200, 200) */
    std::vector<float> dir_cls_pred;    /** (1, 28, 200, 200) */
}POINTPILLARS_MODEL_OUTPUT_t;

class OrtPointPillarsInfer
{
public:
/* replaced by CheckStatus() */
    #define ORT_ABORT_ON_ERROR(expr)                             \
    do {                                                       \
        OrtStatus* onnx_status = (expr);                         \
        if (onnx_status != NULL) {                               \
            const char* msg = g_ort_->GetErrorMessage(onnx_status); \
            fprintf(stderr, "%s\n", msg);                          \
            g_ort_->ReleaseStatus(onnx_status);                    \
            abort();                                               \
        }                                                        \
    } while (0);

    typedef struct
    {
        OrtSession* sess = nullptr;
        std::vector<const char*> input_node_names;
        std::vector<std::vector<int64_t>> input_node_dims;
        std::vector<ONNXTensorElementDataType> input_types;
        std::vector<OrtValue*> input_tensors;
        std::vector<const char*> output_node_names;
        std::vector<std::vector<int64_t>> output_node_dims;
        std::vector<ONNXTensorElementDataType> output_types;
        std::vector<OrtValue*> output_tensors;
    }ORT_S_t;

public:
    OrtPointPillarsInfer()
    {
        InitOrt();
    }

    ~OrtPointPillarsInfer()
    {
        DestroyOrt();
    }

    bool LoadONNXModel(const std::string& model_path);
    void RunPointpillarsModel(POINTPILLARS_MODEL_INPUT_t& input, POINTPILLARS_MODEL_OUTPUT_t& output);
    void TestPointpillarsModel();

private:
    void LoadModel(const std::string& model_path, ORT_S_t& model_s);
    bool CheckStatus(OrtStatus* status);
    bool InitOrt();
    void DestroyOrt();
    void VerifyInputOutputCount(OrtSession* sess);
    int EnableCuda(OrtSessionOptions* session_options);

private:
    const OrtApi* g_ort_ = nullptr;
    const OrtApiBase* g_ort_base_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtSessionOptions* session_options_ = nullptr;

    ORT_S_t g_pointpillars_s_;
    constexpr static size_t MAX_NUM_VOXELS_ = 40000;
};

#endif //__ORT_PILLAR_INFER_H__