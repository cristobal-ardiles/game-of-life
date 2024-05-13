#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "kernel.cuh"

struct Times {
  long create_data;
  long copy_to_host;
  long execution;
  long copy_to_device;
  inline long total() {
    return create_data + copy_to_host + execution + copy_to_device; 
  }
};

Times t;

unsigned char get_one(float prob){
  // Obtains 1 with probability prob, and with (1-prob) zero
  float n = (float) std::rand() / ((float) RAND_MAX + 1);
  if (n < prob){
    return 1;
  }
  return 0; 
}

void swap(unsigned char* &a, unsigned char* &b){
  unsigned char* temp = a;
  a = b; 
  b = temp; 
  return; 
}

bool simulate(int n, int m, int T, int blockSize, int gridSize) {
  using std::chrono::microseconds;
  std::size_t size = n*m; 
  std::vector<unsigned char> initial(size, 0);

  // Create the memory buffers
  unsigned char *curr, *next;
  cudaMalloc(&curr, size);
  cudaMalloc(&next, size);

  // Assign values to host variables
  auto t_start = std::chrono::high_resolution_clock::now();
  for (int i=1; i<n-1; i++){
    int index = i*m + 2; 
    initial[index] = 1;
    // for (int j=1; j<m-1; j++){
    //   int index = i*m+j; 
      
    //   initial[index] = get_one(0.5);
    // }
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  // Copy values from host variables to device
  t_start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(curr, initial.data(), size, cudaMemcpyHostToDevice);
  cudaMemset(next,0,size);
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_device =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();


  // Execute the function on the device (using 32 threads here)
  std::cout << "Running kernel " << T << " times" << std::endl; 
  t_start = std::chrono::high_resolution_clock::now();
  for (int tm=0; tm<T; tm++){
    game_of_cuda<<<gridSize, blockSize>>>(curr, next, n, m);
    cudaDeviceSynchronize();
    swap(curr,next); 
  }
  t_end = std::chrono::high_resolution_clock::now();
  t.execution =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Copy the output variable from device to host
  t_start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(initial.data(), curr, size, cudaMemcpyDeviceToHost);
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_host =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Print the result
  std::cout << "Time to create data: " << t.create_data << " microseconds\n";
  std::cout << "Time to copy data to device: " << t.copy_to_device
            << " microseconds\n";
  std::cout << "Time to execute kernel: " << t.execution << " microseconds\n";
  std::cout << "Time to copy data to host: " << t.copy_to_host
            << " microseconds\n";
  std::cout << "Time to execute the whole program: " << t.total()
            << " microseconds\n";
  return true;

}

int main(int argc, char* argv[]) {
  if (argc != 7) {
    std::cerr << "Uso: " << argv[0] << " <rows> <columns> <periods> <block size> <grid size> <output file>"
              << std::endl;
    return 2;
  }
  int n = std::stoi(argv[1]);
  int m = std::stoi(argv[2]);
  int T = std::stoi(argv[3]);
  int bs = std::stoi(argv[4]);
  int gs = std::stoi(argv[5]);

  if (!simulate(n, m, T, bs, gs)) {
    std::cerr << "CUDA: Error while executing the simulation" << std::endl;
    return 3;
  }

  std::ofstream out;
  out.open(argv[6], std::ios::app | std::ios::out);
  if (!out.is_open()) {
    std::cerr << "Error while opening file: '" << argv[2] << "'" << std::endl;
    return 4;
  }
  // params
  out << n << ',' << m << ',' << T << "," << bs << "," << gs << ",";
  // times
  out << t.create_data << "," << t.copy_to_device << "," << t.execution << "," << t.copy_to_host << "," << t.total() << "\n";

  std::cout << "Data written to " << argv[6] << std::endl;
  return 0;
}
