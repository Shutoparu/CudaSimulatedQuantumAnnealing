#include <stdio.h>

texture<int, 1, cudaReadModeElementType> text;

__global__ void printTexture(int num, int* out){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<num){
        out[idx] = tex1Dfetch(text,idx);
    }
}

int main(){

    int * local;
    cudaMallocHost(&local, 5*sizeof(int));
    for(int i=0; i<5; i++){
        local[i] = i;
    }
    
    int* device;
    cudaMalloc(&device, 5*sizeof(int));
    cudaMemcpy(device, local, 5*sizeof(int), cudaMemcpyHostToDevice);

    cudaBindTexture(0, text, device, 5*sizeof(int));

    int *out;
    cudaMallocManaged(&out, 5*sizeof(int));

    printTexture<<<3,3>>>(5, out);
    cudaDeviceSynchronize();

    for(int i=0; i<5; i++){
        printf("%d\n",out[i]);
    }

}