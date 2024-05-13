#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "kernel.h"


struct Times {
  long create_data;
  long execution;

  long total() { return create_data + execution; }
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

bool simulate(int n, int m, int T) {
  using std::chrono::microseconds;
  int size = n*m;
  std::vector<unsigned char> initial(size,0);

  auto t_start = std::chrono::high_resolution_clock::now();
  for (int i=1; i<n; i++){
    for (int j=1; j<m; j++){
      int index = i*m+j; 
      initial[index] = get_one(0.5);
    }
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  t_start = std::chrono::high_resolution_clock::now();
  game_of_cpu(initial.data(), n, m, T);
  t_end = std::chrono::high_resolution_clock::now();
  t.execution =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  std::cout << "Time to create data: " << t.create_data << " microseconds\n";
  std::cout << "Time to execute kernel: " << t.execution << " microseconds\n";
  std::cout << "Time to execute the whole program: " << t.total()
            << " microseconds\n";

  return true;
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr << "Uso: " << argv[0] << " <rows> <columns> <periods> <out_file>"
              << std::endl;
    return 2;
  }

  int n = std::stoi(argv[1]);
  int m = std::stoi(argv[2]);
  int T = std::stoi(argv[3]); 
  if (!simulate(n,m,T)) {
    std::cerr << "Error while executing the simulation" << std::endl;
    return 3;
  }

  std::ofstream out;
  out.open(argv[4], std::ios::app | std::ios::out);
  if (!out.is_open()) {
    std::cerr << "Error while opening file: '" << argv[2] << "'" << std::endl;
    return 4;
  }
  out << n << ',' << m << ',' << T << "," << t.create_data << "," << t.execution << "," << t.total()
      << "\n";

  std::cout << "Data written to " << argv[2] << std::endl;
  return 0;
}
