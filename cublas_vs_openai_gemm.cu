// Simple profiling program for cublas and openai-gemm matmul kernels
//
// how to compile:
// nvcc -o matmul --std=c++11 -arch=compute_60 -code=sm_60 cublas_vs_openai_gemm.cu openai-gemm/lib/c_interface.o -lcublas -lcuda -I .

#include <cublas_v2.h>
#include <cassert>
#include <cstdio>

#include "openai-gemm/include/c_interface.h"

typedef unsigned short     uint16;

// A simple GPU Timer taken from CUB
struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() { cudaEventRecord(start, 0); }

  void Stop() { cudaEventRecord(stop, 0); }

  float ElapsedMillis() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

__global__ void MemcpyFloat2Half(float *h_data, uint16 *d_data, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= size) return;
  float val = __ldg(h_data + idx);
  d_data[idx] = static_cast<uint16>(val);
}

int CeilOfRatio(int a, int b) { return (a + b - 1) / b; }

void PrintRedGreen(float val, float threshold) {
  if (val > threshold) {
    // print in green
    printf(
        "\x1B[32m"
        "%5.2f",
        val);
  } else {
    // print in red
    printf(
        "\x1B[31m"
        "%5.2f",
        val);
  }
  // back to white
  printf("\x1B[37m");
}

void RunTest(int m, int k, int n, int transa, int transb) {
  // Prepare on device matrix data
  const int A_rows = transa ? m : k;
  const int A_cols = transa ? k : m;
  const int B_rows = transb ? k : n;
  const int B_cols = transb ? n : k;
  const int C_rows = n;
  const int C_cols = m;

  GpuTimer gpu_timer;

  float elapsed_millis;
  float throughput;

  cublasHandle_t handle;
  cublasCreate(&handle);

  float *host_A =
      reinterpret_cast<float *>(malloc(A_rows * A_cols * sizeof(float)));
  float *host_B =
      reinterpret_cast<float *>(malloc(B_rows * B_cols * sizeof(float)));
  for (int i = 0; i < A_rows * A_cols; i++) host_A[i] = i;
  for (int i = 0; i < B_rows * B_cols; i++) host_B[i] = i;

  float *device_A_float, *device_B_float, *device_C_float;
  uint16 *device_A_half, *device_B_half, *device_C_half;
  cudaMalloc(&device_A_float, A_rows * A_cols * sizeof(float));
  cudaMalloc(&device_B_float, B_rows * B_cols * sizeof(float));
  cudaMalloc(&device_C_float, C_rows * C_cols * sizeof(float));

  cudaMalloc(&device_A_half, A_rows * A_cols * sizeof(float));
  cudaMalloc(&device_B_half, B_rows * B_cols * sizeof(float));
  cudaMalloc(&device_C_half, C_rows * C_cols * sizeof(float));

  cudaMemcpy(device_A_float, host_A, A_rows * A_cols * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_B_float, host_B, B_rows * B_cols * sizeof(float),
             cudaMemcpyHostToDevice);

  int thread_num = 1024;
  int size_A = A_rows * A_cols;
  int block_num = (size_A + thread_num - 1) / thread_num;
  MemcpyFloat2Half<<<block_num, thread_num>>>(device_A_float, device_A_half,
                                              size_A);
  int size_B = B_rows * B_cols;
  block_num = (size_B + thread_num - 1) / thread_num;
  MemcpyFloat2Half<<<block_num, thread_num>>>(device_B_float, device_B_half,
                                              size_B);

  float alpha = 1.0f;
  float beta = 0.0f;

  printf("%d %5d %5d %d %d,\t", m, k, n, transa, transb);

  float best_float_runtime = -1.0f;
  float best_float_throughput = -1.0f;
  float best_half_runtime = -1.0f;
  float best_half_throughput = -1.0f;
  float ratio_32, ratio_16;
  // Use OpenAI Gemm and output profiling results
  unsigned int grid;
  get_grid_limits('s', transa, transb, &grid);
  for (int g = 0; g < grid; ++g) {
      unsigned int shared;
      get_shared_limits('s', transa, transb, g, &shared);
      for (int s = 0; s < shared; ++s) {
          bool res;
          for (int warmup = 0; warmup < 2; ++warmup) {
              res = openai_sgemm(
                      device_A_float, device_B_float, device_C_float, transa, transb, m, n,
                      k, (transa ? A_cols : A_rows), (transb ? B_cols : B_rows), C_rows,
                      alpha, beta, nullptr, g, s);
              if (!res) break;
          }
          gpu_timer.Start();
          for (int iter = 0; iter < 10; ++iter) {
              res = openai_sgemm(
                      device_A_float, device_B_float, device_C_float, transa, transb, m, n,
                      k, (transa ? A_cols : A_rows), (transb ? B_cols : B_rows), C_rows,
                      alpha, beta, nullptr, g, s);
              if (!res) break;
          }
          gpu_timer.Stop();
          if (res) {
              elapsed_millis = gpu_timer.ElapsedMillis()/10;
              throughput = 1.0f / elapsed_millis / 1000000.0f * m * n * k * 2;
              if (best_float_runtime < 0 || elapsed_millis < best_float_runtime) {
                  best_float_runtime = elapsed_millis;
                  best_float_throughput = throughput;
              }
          }
          for (int warmup = 0; warmup < 2; ++warmup) {
              res = openai_hgemm(device_A_half, device_B_half, device_C_half, transa,
                  transb, m, n, k, (transa ? A_cols : A_rows),
                  (transb ? B_cols : B_rows), C_rows, alpha, beta,
                  nullptr, g, s);
              if (!res) break;
          }
          gpu_timer.Start();
          for (int iter = 0; iter < 10; ++iter) {
              res = openai_hgemm(device_A_half, device_B_half, device_C_half, transa,
                  transb, m, n, k, (transa ? A_cols : A_rows),
                  (transb ? B_cols : B_rows), C_rows, alpha, beta,
                  nullptr, g, s);
              if (!res) break;
          }
          gpu_timer.Stop();
          if (res) {
              elapsed_millis = gpu_timer.ElapsedMillis()/10;
              throughput = 1.0f / elapsed_millis / 1000000.0f * m * n * k * 2;
              if (best_half_runtime < 0 || elapsed_millis < best_half_runtime) {
                  best_half_runtime = elapsed_millis;
                  best_half_throughput = throughput;
              }
          }
      }
  }
  printf("%5.2f,\t", best_float_throughput);
  printf("%5.2f,\t", best_half_throughput);
  ratio_32 = best_float_throughput;
  ratio_16 = best_half_throughput;

  // Use cublasGemmEx
  best_float_runtime = -1.0f;
  best_float_throughput = -1.0f;
  best_half_runtime = -1.0f;
  best_half_throughput = -1.0f;

  // Use cublasGemmEx
  cublasGemmAlgo_t algos[] = {
    CUBLAS_GEMM_DFALT,
    CUBLAS_GEMM_ALGO0,
    CUBLAS_GEMM_ALGO1,
    CUBLAS_GEMM_ALGO2,
    CUBLAS_GEMM_ALGO3,
    CUBLAS_GEMM_ALGO4,
    CUBLAS_GEMM_ALGO5,
    CUBLAS_GEMM_ALGO6,
    CUBLAS_GEMM_ALGO7,
#if __CUDACC_VER_MAJOR__ >= 9
    CUBLAS_GEMM_ALGO8,
    CUBLAS_GEMM_ALGO9,
    CUBLAS_GEMM_ALGO10,
    CUBLAS_GEMM_ALGO11,
    CUBLAS_GEMM_ALGO12,
    CUBLAS_GEMM_ALGO13,
    CUBLAS_GEMM_ALGO14,
    CUBLAS_GEMM_ALGO15,
    CUBLAS_GEMM_ALGO16,
    CUBLAS_GEMM_ALGO17,
#endif
  };

  for (int i = 0; i < sizeof(algos)/sizeof(algos[0]); ++i) {
      for (int warmup = 0; warmup < 2; ++warmup) {
          auto result =
              cublasGemmEx(handle, (transb ? CUBLAS_OP_T : CUBLAS_OP_N),
                      (transa ? CUBLAS_OP_T : CUBLAS_OP_N), n, m, k, &alpha,
                      device_B_float, CUDA_R_32F,
                      /*ldb=*/(transb ? k : n), device_A_float, CUDA_R_32F,
                      /*ldb=*/(transa ? m : k), &beta, device_C_float,
                      CUDA_R_32F, /*ldc=*/n, CUDA_R_32F, algos[i]);
      }
    gpu_timer.Start();
    for (int iter = 0; iter < 9; ++iter) {
        auto result =
            cublasGemmEx(handle, (transb ? CUBLAS_OP_T : CUBLAS_OP_N),
                    (transa ? CUBLAS_OP_T : CUBLAS_OP_N), n, m, k, &alpha,
                    device_B_float, CUDA_R_32F,
                    /*ldb=*/(transb ? k : n), device_A_float, CUDA_R_32F,
                    /*ldb=*/(transa ? m : k), &beta, device_C_float,
                    CUDA_R_32F, /*ldc=*/n, CUDA_R_32F, algos[i]);
    }
    auto result =
            cublasGemmEx(handle, (transb ? CUBLAS_OP_T : CUBLAS_OP_N),
                    (transa ? CUBLAS_OP_T : CUBLAS_OP_N), n, m, k, &alpha,
                    device_B_float, CUDA_R_32F,
                    /*ldb=*/(transb ? k : n), device_A_float, CUDA_R_32F,
                    /*ldb=*/(transa ? m : k), &beta, device_C_float,
                    CUDA_R_32F, /*ldc=*/n, CUDA_R_32F, algos[i]);
    gpu_timer.Stop();
    if (result == 0) {
        elapsed_millis = gpu_timer.ElapsedMillis()/10;
        throughput = 1.0f / elapsed_millis / 1000000.0f * m * n * k * 2;
        if (best_float_runtime < 0 || elapsed_millis < best_float_runtime) {
            best_float_runtime = elapsed_millis;
            best_float_throughput = throughput;
        }
    }
  }

  for (int i = 0; i < sizeof(algos)/sizeof(algos[0]); ++i) {
      for (int warmup = 0; warmup < 2; ++warmup) {
          auto result =
              cublasGemmEx(handle, (transb ? CUBLAS_OP_T : CUBLAS_OP_N),
                      (transa ? CUBLAS_OP_T : CUBLAS_OP_N), n, m, k, &alpha,
                      device_B_half, CUDA_R_16F,
                      /*ldb=*/(transb ? k : n), device_A_half, CUDA_R_16F,
                      /*ldb=*/(transa ? m : k), &beta, device_C_half, CUDA_R_16F,
                      /*ldc=*/n, CUDA_R_32F, algos[i]);
      }
      gpu_timer.Start();
      for (int iter = 0; iter < 9; ++iter) {
          auto result =
              cublasGemmEx(handle, (transb ? CUBLAS_OP_T : CUBLAS_OP_N),
                      (transa ? CUBLAS_OP_T : CUBLAS_OP_N), n, m, k, &alpha,
                      device_B_half, CUDA_R_16F,
                      /*ldb=*/(transb ? k : n), device_A_half, CUDA_R_16F,
                      /*ldb=*/(transa ? m : k), &beta, device_C_half, CUDA_R_16F,
                      /*ldc=*/n, CUDA_R_32F, algos[i]);
      }

      auto result =
          cublasGemmEx(handle, (transb ? CUBLAS_OP_T : CUBLAS_OP_N),
                  (transa ? CUBLAS_OP_T : CUBLAS_OP_N), n, m, k, &alpha,
                  device_B_half, CUDA_R_16F,
                  /*ldb=*/(transb ? k : n), device_A_half, CUDA_R_16F,
                  /*ldb=*/(transa ? m : k), &beta, device_C_half, CUDA_R_16F,
                  /*ldc=*/n, CUDA_R_32F, algos[i]);
      gpu_timer.Stop();
      if (result == 0) {
          elapsed_millis = gpu_timer.ElapsedMillis()/10;
          throughput = 1.0f / elapsed_millis / 1000000.0f * m * n * k * 2;
          if (best_half_runtime < 0 || elapsed_millis < best_half_runtime) {
              best_half_runtime = elapsed_millis;
              best_half_throughput = throughput;
          }
      }
  }

  for (int i = 0; i < sizeof(algos)/sizeof(algos[0]); ++i) {
      for (int warmup = 0; warmup < 2; ++warmup) {
          auto result =
              cublasGemmEx(handle, (transb ? CUBLAS_OP_T : CUBLAS_OP_N),
                      (transa ? CUBLAS_OP_T : CUBLAS_OP_N), n, m, k, &alpha,
                      device_B_half, CUDA_R_16F,
                      /*ldb=*/(transb ? k : n), device_A_half, CUDA_R_16F,
                      /*ldb=*/(transa ? m : k), &beta, device_C_half, CUDA_R_16F,
                      /*ldc=*/n, CUDA_R_16F, algos[i]);
      }
      gpu_timer.Start();
      for (int iter = 0; iter < 9; ++iter) {
          auto result =
              cublasGemmEx(handle, (transb ? CUBLAS_OP_T : CUBLAS_OP_N),
                      (transa ? CUBLAS_OP_T : CUBLAS_OP_N), n, m, k, &alpha,
                      device_B_half, CUDA_R_16F,
                      /*ldb=*/(transb ? k : n), device_A_half, CUDA_R_16F,
                      /*ldb=*/(transa ? m : k), &beta, device_C_half, CUDA_R_16F,
                      /*ldc=*/n, CUDA_R_16F, algos[i]);
      }

      auto result =
          cublasGemmEx(handle, (transb ? CUBLAS_OP_T : CUBLAS_OP_N),
                  (transa ? CUBLAS_OP_T : CUBLAS_OP_N), n, m, k, &alpha,
                  device_B_half, CUDA_R_16F,
                  /*ldb=*/(transb ? k : n), device_A_half, CUDA_R_16F,
                  /*ldb=*/(transa ? m : k), &beta, device_C_half, CUDA_R_16F,
                  /*ldc=*/n, CUDA_R_16F, algos[i]);

      gpu_timer.Stop();
      elapsed_millis = gpu_timer.ElapsedMillis()/10;
      throughput = 1.0f / elapsed_millis / 1000000.0f * m * n * k * 2;
      if (result == 0) {
          if (best_half_runtime < 0 || elapsed_millis < best_half_runtime) {
              best_half_runtime = elapsed_millis;
              best_half_throughput = throughput;
          }
      }
  }
  printf("%5.2f,\t", best_float_throughput);
  printf("%5.2f,\t", best_half_throughput);
  ratio_32 /= best_float_throughput;
  ratio_16 /= best_half_throughput;
  PrintRedGreen(ratio_32, 1.0f);
  printf("\t");
  PrintRedGreen(ratio_16, 1.0f);
  printf("\n");

  cudaFree(device_A_float);
  cudaFree(device_B_float);
  cudaFree(device_C_float);
  cudaFree(device_A_half);
  cudaFree(device_B_half);
  cudaFree(device_C_half);
  cublasDestroy(handle);
  free(host_A);
  free(host_B);
}

int main(int argc, char *argv[]) {
  printf("m,k,n,ta,tb\t\toai_32\toai_16\tnv_32\tnv_16\tr_32\tr_16\n");
  // Typical fully connected layers
  RunTest(1, 512, 512, 0, 0);
  RunTest(8, 512, 512, 0, 0);
  RunTest(16, 512, 512, 0, 0);
  RunTest(128, 512, 512, 0, 0);

  RunTest(1, 1024, 1024, 0, 0);
  RunTest(8, 1024, 1024, 0, 0);
  RunTest(16, 1024, 1024, 0, 0);
  RunTest(128, 1024, 1024, 0, 0);
  RunTest(4096, 4096, 4096, 0, 0);

  // Backward for fully connected layers
  RunTest(1, 1024, 1024, 0, 1);
  RunTest(8, 1024, 1024, 0, 1);
  RunTest(16, 1024, 1024, 0, 1);
  RunTest(128, 1024, 1024, 0, 1);

  // Forward softmax with large output size
  RunTest(1, 200, 10000, 0, 0);
  RunTest(8, 200, 10000, 0, 0);
  RunTest(20, 200, 10000, 0, 0);
  RunTest(20, 200, 20000, 0, 0);

  // Backward softmax with large output size
  RunTest(1, 10000, 200, 0, 1);
  RunTest(1, 10000, 200, 0, 0);
  RunTest(8, 10000, 200, 0, 0);
  RunTest(20, 10000, 200, 0, 1);
  RunTest(20, 20000, 200, 0, 1);

  printf("\n\nBenchmarks from openai-gemm github repo.\n\n");

  // Benchmarks from openai-gemm github repo.
  RunTest(16, 1760, 1760, 0, 0);
  RunTest(32, 1760, 1760, 0, 0);
  RunTest(64, 1760, 1760, 0, 0);
  RunTest(128, 1760, 1760, 0, 0);
  RunTest(7000, 1760, 1760, 0, 0);

  RunTest(16, 2048, 2048, 0, 0);
  RunTest(32, 2048, 2048, 0, 0);
  RunTest(64, 2048, 2048, 0, 0);
  RunTest(128, 2048, 2048, 0, 0);
  RunTest(7000, 2048, 2048, 0, 0);

  RunTest(16, 2560, 2560, 0, 0);
  RunTest(32, 2560, 2560, 0, 0);
  RunTest(64, 2560, 2560, 0, 0);
  RunTest(128, 2560, 2560, 0, 0);
  RunTest(7000, 2560, 2560, 0, 0);

  RunTest(16, 4096, 4096, 0, 0);
  RunTest(32, 4096, 4096, 0, 0);
  RunTest(64, 4096, 4096, 0, 0);
  RunTest(128, 4096, 4096, 0, 0);
  RunTest(7000, 4096, 4096, 0, 0);

  RunTest(16, 1760, 1760, 0, 1);
  RunTest(32, 1760, 1760, 0, 1);
  RunTest(64, 1760, 1760, 0, 1);
  RunTest(128, 1760, 1760, 0, 1);
  RunTest(7000, 1760, 1760, 0, 1);

  RunTest(16, 2048, 2048, 0, 1);
  RunTest(32, 2048, 2048, 0, 1);
  RunTest(64, 2048, 2048, 0, 1);
  RunTest(128, 2048, 2048, 0, 1);
  RunTest(7000, 2048, 2048, 0, 1);

  RunTest(16, 2560, 2560, 0, 1);
  RunTest(32, 2560, 2560, 0, 1);
  RunTest(64, 2560, 2560, 0, 1);
  RunTest(128, 2560, 2560, 0, 1);
  RunTest(7000, 2560, 2560, 0, 1);

  RunTest(16, 4096, 4096, 0, 1);
  RunTest(32, 4096, 4096, 0, 1);
  RunTest(64, 4096, 4096, 0, 1);
  RunTest(128, 4096, 4096, 0, 1);
  RunTest(7000, 4096, 4096, 0, 1);

  RunTest(7133, 1760, 1760, 1, 0);
  RunTest(7133, 2048, 2048, 1, 0);
  RunTest(7133, 2560, 2560, 1, 0);
  RunTest(7133, 4096, 4096, 1, 0);
  RunTest(9124, 1760, 5124, 0, 0);
  RunTest(9124, 2048, 5124, 0, 0);
  RunTest(9124, 2560, 5124, 0, 0);
  RunTest(9124, 4096, 5124, 0, 0);
  RunTest(9124, 2048, 5124, 0, 1);
  RunTest(9124, 2560, 5124, 0, 1);
  RunTest(9124, 4096, 5124, 0, 1);
  RunTest(8457, 1760, 35, 0, 0);
  RunTest(8457, 2048, 35, 0, 0);
  RunTest(8457, 2560, 35, 0, 0);
  RunTest(8457, 4096, 35, 0, 0);
  RunTest(8457, 1760, 35, 0, 1);
  RunTest(8457, 2048, 35, 0, 1);
  RunTest(8457, 2560, 35, 0, 1);
  RunTest(8457, 4096, 35, 0, 1);
  RunTest(16, 2560, 7680, 0, 0);
  RunTest(32, 2560, 7680, 0, 0);
  RunTest(64, 2560, 7680, 0, 0);
  RunTest(128, 2560, 7680, 0, 0);

  RunTest(16, 1024, 3072, 0, 0);
  RunTest(32, 1024, 3072, 0, 0);
  RunTest(64, 1024, 3072, 0, 0);
  RunTest(128, 1024, 3072, 0, 0);

  RunTest(7435, 1024, 3072, 1, 0);
  RunTest(5481, 2560, 7680, 1, 0);

  // Following tests failed on openai-gemm
  // RunTest(9124, 1760, 5124, 0, 1);
  // RunTest(16, 2560, 7680, 0, 1);
  // RunTest(32, 2560, 7680, 0, 1);
  // RunTest(64, 2560, 7680, 0, 1);
  // RunTest(128, 2560, 7680, 0, 1);

  // RunTest(16, 1024, 3072, 0, 1);
  // RunTest(32, 1024, 3072, 0, 1);
  // RunTest(64, 1024, 3072, 0, 1);
  // RunTest(128, 1024, 3072, 0, 1);

  return 0;
}
