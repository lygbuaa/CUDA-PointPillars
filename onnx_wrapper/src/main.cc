#include "signal_handlers.h"
#include "ort_pointpillars_infer.h"

int main(int argc, char **argv)
{
    for(int i = 0; i < argc; i++){
        LOGPF("argv[%d] = %s\n", i, argv[i]);
    }

    if(argc < 2)
    {
        LOGPF("missing onnx model path.");
        return -1;
    }

#if __LOGGING_BACKEND__ == 3
    // __setup_spdlog__(argv[1], spdlog::level::info);
    __setup_spdlog__(argv[1], static_cast<spdlog::level::level_enum>(atoi(argv[4])));
#endif

    std::vector<int> sigs;
    sigs.push_back(SIGABRT);
    sigs.push_back(SIGTERM);
    SignalHandlers::RegisterBackTraceSignals(sigs);
    SignalHandlers::RegisterBreakSignals(SIGINT);

    auto ppl = std::make_shared<OrtPointPillarsInfer>();
    ppl -> LoadONNXModel(argv[1]);
    ppl -> TestPointpillarsModel();

    uint32_t heartbeat = 0;
    while(!SignalHandlers::BreakByUser())
    {
        RLOGI("main heartbeat: %d", heartbeat++);
        sleep(5);
    }

    // ppl->Stop();

#if __LOGGING_BACKEND__ == 3
    __shutdown_spdlog__();
#endif
    return 0;
}