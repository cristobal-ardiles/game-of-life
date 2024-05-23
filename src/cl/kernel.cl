void check_neighbour(int i, int j, int rel_i, int rel_j, int n, int m, global unsigned char *state, int *counter){
  int neigh_i = (i+rel_i+n) % n; 
  int neigh_j = (j+rel_j+m) % m; 
  int neigh_index = neigh_i*m + neigh_j; 
  if (state[neigh_index]) {
    (*counter)++;
  }  
  return;
}

kernel void print_matrix(global unsigned char *matrix, int n, int m){
  int gindex = get_global_id(0);
  if (gindex==0){
    for (int i=0; i<n; i++){
      for (int j=0; j<m; j++){
        int index = i*m+j; 
        if (j<m-1){
          printf("%d ", matrix[index]);
        } else {
          printf("%d\n", matrix[index]); 
        }
      }
    }
    printf("=================\n"); 
  }
  return; 
}

kernel void game_of_cl(global unsigned char *curr, 
                       global unsigned char *next, 
                       int n, int m) {
  int size = n*m; 
  int gindex = get_global_id(0);
  int gsize = get_global_size(0);
  for (int thread = gindex; thread < size; thread += gsize){
    // Get row, column index 
    int i = thread / m; 
    int j = thread % m; 
    int alive = 0; 
    //Check each neighbour
    check_neighbour(i,j,-1,-1,n,m,curr,&alive); // North West neighbour 
    check_neighbour(i,j,-1, 0,n,m,curr,&alive); // N 
    check_neighbour(i,j,-1, 1,n,m,curr,&alive); // NE 
    check_neighbour(i,j, 0,-1,n,m,curr,&alive); // W  
    check_neighbour(i,j, 0, 1,n,m,curr,&alive); // E 
    check_neighbour(i,j, 1,-1,n,m,curr,&alive); // SW  
    check_neighbour(i,j, 1, 0,n,m,curr,&alive); // S  
    check_neighbour(i,j, 1, 1,n,m,curr,&alive); // SE
    // Apply rules. TODO: fix divergence. We have to figure out how to run this concurrently
    if (curr[thread]){
      // If cell is alive
      next[thread] = (alive < 2 || alive >= 4) ? 0 : 1; 
    } else {
      // If cell is dead
      next[thread] = alive == 3 ? 1 : 0;
    }
    return; 
  }
}
