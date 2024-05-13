
__global__ void game_of_cuda(unsigned char *curr, unsigned char *next, 
                            int n, int m);

__global__ void print_matrix(unsigned char* curr, int n, int m);