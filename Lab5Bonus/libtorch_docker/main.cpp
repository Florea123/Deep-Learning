#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>

int main(int argc, const char* argv[]) {
    std::string model_path = "checkpoints/best_resnet34_traced.pt";
    std::string device_str = "cpu";
    int iterations = 50;
    int batch_size = 200;

    if (argc > 1) model_path = argv[1];
    if (argc > 2) device_str = argv[2];
    if (argc > 3) iterations = std::stoi(argv[3]);
    if (argc > 4) batch_size = std::stoi(argv[4]);

    torch::DeviceType device_type = torch::kCPU;
    if (device_str == "cuda" || device_str == "gpu") {
        if (torch::cuda::is_available()) {
            device_type = torch::kCUDA;
                      << torch::cuda::device_count() << ")\n";
        } else {
            std::cout << "[Notice] CUDA requested but not available. Falling back to CPU.\n";
            device_type = torch::kCPU;
        }
    } else {
        std::cout << "[Device] Initialized Native CPU\n";
    }
    torch::Device device(device_type);

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path, device);
        module.eval();
    } catch (const c10::Error& e) {
        std::cerr << "[Error] Failed to load model: " << e.what() << "\n";
        return -1;
    }

    auto input_tensor = torch::randn({batch_size, 3, 32, 32}, device);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    {
        c10::InferenceMode guard(true);
        for (int i = 0; i < 5; ++i) {
            auto out = module.forward(inputs);
        }
    }

    if (device_type == torch::kCUDA) {
        torch::cuda::synchronize();
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    {
        c10::InferenceMode guard(true);
        for (int i = 0; i < iterations; ++i) {
            auto output = module.forward(inputs);
        }
    }

    if (device_type == torch::kCUDA) {
        torch::cuda::synchronize();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    double total_time_sec = elapsed.count();
    double avg_latency_batch_ms = (total_time_sec / iterations) * 1000.0;
    double avg_latency_img_ms = avg_latency_batch_ms / batch_size;
    double throughput_fps = (iterations * batch_size) / total_time_sec;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << " Total Elapsed Time     : " << total_time_sec << " seconds\n";
    std::cout << " Average Batch Latency  : " << avg_latency_batch_ms << " ms (" << batch_size << " images/batch)\n";
    std::cout << " Average Image Latency  : " << avg_latency_img_ms << " ms/image\n";
    std::cout << " Throughput             : " << std::setprecision(1) << throughput_fps << " images/sec\n";

    return 0;
}
