#include "kernel.h"
#include <vector>
#include <algorithm>
#include <iostream>

inline void check_neighbour(int i, int j, int rel_i, int rel_j, int n, int m, std::vector<unsigned char> state, int& counter){
  int neigh_i = (i+rel_i) % n; 
  int neigh_j = (j+rel_j) % m; 
  int neigh_index = neigh_i*m + neigh_j; 
  if (state[neigh_index]) counter++; 
}

void print_matrix(std::vector<unsigned char> mat, int n, int m){
  for (int i=0; i<n; i++){
    for (int j=0; j<m; j++){
      int index = i*m + j; 
      std::cout << (int) mat[index];
      if (j < m-1){
        std::cout << ' '; 
      }
    }
    std::cout << std::endl; 
  }
}

void game_of_cpu(unsigned char* initial, int n, int m, int T){
  int size = n*m;
  std::vector<unsigned char> curr(size); 
  std::copy(initial, initial+size, curr.data());
  std::vector<unsigned char> next(curr);
  // Simulate T periods
  for(int t=0; t<T; t++){
    // Simulate each cell
    for (int i=0; i<n; i++){
      for (int j=0; j<m; j++){
        int index = i*m+j; 
        int alive_neighs = 0; 
        //Check each neighbour
        check_neighbour(i,j,-1,-1,n,m,curr,alive_neighs); // North West neighbour 
        check_neighbour(i,j,-1, 0,n,m,curr,alive_neighs); // N 
        check_neighbour(i,j,-1, 1,n,m,curr,alive_neighs); // NE 
        check_neighbour(i,j, 0,-1,n,m,curr,alive_neighs); // W  
        check_neighbour(i,j, 0, 1,n,m,curr,alive_neighs); // E 
        check_neighbour(i,j, 1,-1,n,m,curr,alive_neighs); // SW  
        check_neighbour(i,j, 1, 0,n,m,curr,alive_neighs); // S  
        check_neighbour(i,j, 1, 1,n,m,curr,alive_neighs); // SE

        // Apply rules
        if (curr[index]){
          // If cell is alive
          if (alive_neighs < 2 || alive_neighs >= 4) {
            next[index] = 0;
          } else {
            next[index] = 1; 
          }
        } else {
          // If cell is dead
          if (alive_neighs == 3){
            next[index] = 1; 
          } else {
            next[index] = 0; 
          }
        }
      }
    }
    // print_matrix(curr, n, m); 
    curr.swap(next); 
    // std::cout << "==========" << std::endl; 
  }
}