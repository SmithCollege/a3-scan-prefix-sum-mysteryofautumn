#include <iostream>
#include <math.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>

#define BLOCK_SIZE 256
#define N 100

double get_clock() {
    struct timeval tv; 
    int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok<0) { printf("gettimeofday error"); }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

__global__ void scan(int *in, int *out, int stride){
    int index = threadIdx.x + blockIdx.x * blockDim.x; 

    if (index >= stride && index < N) {
        out[index] = in[index] + in[index - stride];

    } else if (index < stride) {   
        out[index] = in[index];     //copy the exact value
    }

}

int main(void)
{
    // memory allocation
    int *in, *out, *temp;
    cudaMallocManaged(&in, N * sizeof(int));
    cudaMallocManaged(&out, N * sizeof(int));

    // array initialization
    for (int i = 0; i < N; i++) {
        in[i] = 1;
        out[i] = 0;
    }

    // start timer
    double t0 = get_clock();

    // Run kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for(int stride = 1; stride < N; stride*=2){
        scan<<<numBlocks, BLOCK_SIZE>>>(in, out, stride);
        temp = in;
        in = out;
        out = temp; //swap in and out array memory location
    }

    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // stop timer
    double t1 = get_clock();
    printf("%d\n", in[N-1]);
    printf("time per call: %f ns\n", (1000000000.0 * (t1 - t0) / N));

    // Free memory
    cudaFree(in);
    cudaFree(out);
    cudaFree(temp);


    return 0;
}
