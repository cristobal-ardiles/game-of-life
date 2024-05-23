#include <cstddef>
#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/opencl.hpp>
#endif  // DEBUG
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>

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
cl::Program prog;
cl::CommandQueue queue;

void print_matrix(std::vector<unsigned char> matrix, int n, int m){
  for (int i=0; i<n; i++){
    for (int j=0; j<m; j++){
      int index = i*m+j; 
      if (j<m-1){
        std::cout << (int) matrix[index] << ' ';
      } else {
        std::cout << (int) matrix[index] << std::endl;
      }
    }
  }
  std::cout << "===============" << std::endl; 
}

bool init() {
  auto error = 0; 
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;
  cl::Platform::get(&platforms);
  for (auto& p : platforms) {
    p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.size() > 0) break;
  }
  if (devices.size() == 0) {
    std::cerr << "Not GPU device found" << std::endl;
    return false;
  }

  std::cout << "GPU Used: " << devices.front().getInfo<CL_DEVICE_NAME>()
            << std::endl;

  cl::Context context(devices.front(), NULL, NULL, NULL, &error);
  std::cout << "Context creation: " << error << std::endl; 
  queue = cl::CommandQueue(context, devices.front(), 0, &error);
  std::cout << "Queue creation: " << error << std::endl; 
  std::ifstream sourceFile("src/cl/kernel.cl");
  std::stringstream sourceCode;
  sourceCode << sourceFile.rdbuf();
  // std::cout << "Read source code: " << sourceCode.str().substr(0,50) << '\n' << std::endl; 
  prog = cl::Program(context, sourceCode.str(), true, &error);
  std::cout << "Program creation error: " << error << std::endl; 

  return true;
}

bool simulate(int n, int m, int T, int localSize, int globalSize) {
  using std::chrono::microseconds;
  std::size_t size = n*m;
  std::size_t mem_size = size * sizeof(unsigned char); 
  std::vector<unsigned char> initial(size, 0);
  std::vector<unsigned char> base_next(size, 0); 

  // Create the memory buffers
  cl::Buffer b1(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, mem_size);
  cl::Buffer b2(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, mem_size);

  // Assign values to host variables
  auto t_start = std::chrono::high_resolution_clock::now();
  for (int i = 1; i < n-1; i++) {
    int index = i*m + 2;
    initial[index] = 1; 
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  print_matrix(initial,n,m);
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  // Copy values from host variables to device
  t_start = std::chrono::high_resolution_clock::now();
  // usar CL_FALSE para hacerlo asíncrono
  queue.enqueueWriteBuffer(b1, CL_TRUE, 0, mem_size, initial.data());
  queue.enqueueWriteBuffer(b2, CL_TRUE, 0, mem_size, base_next.data());
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_device =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  // Make kernel
  auto error = -1; 
  cl::Kernel kernel(prog, "game_of_cl", &error);
  std::cout << "Kernel creation: " << error << std::endl; 
  cl::Kernel print_mat_kernel(prog, "print_matrix");
  // Execute the function on the device (using 32 threads here)
  cl::NDRange gSize(globalSize);
  cl::NDRange lSize(localSize);

  t_start = std::chrono::high_resolution_clock::now();
  for (int t=0; t<T; t++){
    // Print Matrix
    // If t is even, then b1 is the current state, else it is b2
    if (t%2==0) {
      print_mat_kernel.setArg(0, b1);
    } else {
      print_mat_kernel.setArg(0, b2);
    }
    print_mat_kernel.setArg(1,n);
    print_mat_kernel.setArg(2,m);
    queue.enqueueNDRangeKernel(print_mat_kernel, cl::NullRange, gSize, lSize);
    auto error = queue.finish();
    // std::cout << "Error: " << error << std::endl; 
    // Launch iteration kernel. Same rule is applied (b1 is the state if t is even)
    if (t%2==0){
      kernel.setArg(0, b1);
      kernel.setArg(1, b2);
    } else {
      kernel.setArg(0, b2);
      kernel.setArg(1, b1); 
    }
    kernel.setArg(2, n);
    kernel.setArg(3, m);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, gSize, lSize);
    error = queue.finish();
    // std::cout << "It " << t << ": error" << error << std::endl; 
  }
  // std::cout << "It " << t << ": error" << error << std::endl; 
  t_end = std::chrono::high_resolution_clock::now();
  t.execution =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Copy the output variable from device to host
  t_start = std::chrono::high_resolution_clock::now();
  if (T%2==0){
    queue.enqueueReadBuffer(b1, CL_TRUE, 0, mem_size, initial.data());
  } else {
    queue.enqueueReadBuffer(b2, CL_TRUE, 0, mem_size, initial.data());
  }  
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_host =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Print the result
  std::cout << "FINAL STATE: " << std::endl;
  print_matrix(initial,n,m);
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
  if (!init()) return 1;

  if (argc != 7) {
    std::cerr << "Uso: " << argv[0]
              << " <rows> <columns> <periods> <local size> <global size> <output file>"
              << std::endl;
    return 2;
  }
  int n = std::stoi(argv[1]);
  int m = std::stoi(argv[2]);
  int T = std::stoi(argv[3]);
  int ls = std::stoi(argv[4]);
  int gs = std::stoi(argv[5]);

  if (!simulate(n,m,T,ls,gs)) {
    std::cerr << "CL: Error while executing the simulation" << std::endl;
    return 3;
  }

  std::ofstream out;
  out.open(argv[6], std::ios::app | std::ios::out);
  if (!out.is_open()) {
    std::cerr << "Error while opening file: '" << argv[6] << "'" << std::endl;
    return 4;
  }
  // params
  out << n << "," << m << ',' << T << ',' << ls << "," << gs << ",";
  // times
  out << t.create_data << "," << t.copy_to_device << "," << t.execution << ","
      << t.copy_to_host << "," << t.total() << "\n";

  std::cout << "Data written to " << argv[6] << std::endl;
  return 0;
}
