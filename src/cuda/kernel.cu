#include "kernel.cuh"
#include "stdio.h"
__device__ void check_neighbour(int i, int j, int rel_i, int rel_j, int n, int m, unsigned char* state, int& counter){
  int neigh_i = (i+rel_i) % n; 
  int neigh_j = (j+rel_j) % m; 
  int neigh_index = neigh_i*m + neigh_j; 
  if (state[neigh_index]) counter++; 
}

__global__ void game_of_cuda(unsigned char *curr, unsigned char *next, 
                            int n, int m){
  // Simulates a single step on game of life. 
  int size = n*m;
  int idx = blockDim.x * blockIdx.x + threadIdx.x; 
  printf("I'm thread %d\n", idx); 
  for (int thread = idx; thread < size; thread += gridDim.x*blockDim.x){
    // Get row, column index 
    int i = thread / m; 
    int j = thread % m; 
    int alive = 0; 
    //Check each neighbour
    check_neighbour(i,j,-1,-1,n,m,curr,alive); // North West neighbour 
    check_neighbour(i,j,-1, 0,n,m,curr,alive); // N 
    check_neighbour(i,j,-1, 1,n,m,curr,alive); // NE 
    check_neighbour(i,j, 0,-1,n,m,curr,alive); // W  
    check_neighbour(i,j, 0, 1,n,m,curr,alive); // E 
    check_neighbour(i,j, 1,-1,n,m,curr,alive); // SW  
    check_neighbour(i,j, 1, 0,n,m,curr,alive); // S  
    check_neighbour(i,j, 1, 1,n,m,curr,alive); // SE

    // Apply rules. TODO: fix divergence. We have to figure out how to run this concurrently
    if (curr[thread]){
      // If cell is alive
      next[thread] = (alive < 2 || alive >= 4) ? 0 : 1; 
    } else {
      // If cell is dead
      next[thread] = alive == 3 ? 1 : 0;
    }
  }
}

