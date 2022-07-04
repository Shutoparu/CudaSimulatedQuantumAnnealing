#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <random>

#define K 0.5

texture<float, 1, cudaReadModeElementType> Q_text;
texture<int, 1, cudaReadModeElementType> s_text;
texture<int, 1, cudaReadModeElementType> pre_s_text;

void cudaErr(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        printf("%s, %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

float energy(int *s, float *Q, int dim)
{
    double *temp;
    temp = (double *)malloc(sizeof(double) * dim);
    float sum = 0;
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            temp[i] += s[j] * Q[i * dim + j];
        }
    }
    for (int i = 0; i < dim; i++)
    {
        sum += temp[i] * s[i];
    }
    free(temp);
    return sum;
}

__global__ void calculate(int *trotterBlock, int spinIdx, int dim, int trotterNum, int beta, int seed)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    // if (index < dim)
    if (index == 0)
    {
        curandState state;
        curand_init(seed, index, 0, &state);

        // calculate<<<1, 1>>>(trotterBlock[j * dim + i - j]);
        int quboRowIdx = spinIdx % dim;
        float sum = 0;
        for (int i = 0; i < dim; i++)
        {
            sum += tex1Dfetch(s_text, spinIdx) * tex1Dfetch(Q_text, quboRowIdx * dim + i);
        }

        // calcPre<<<1, 1>>>(trotterBlock[(j - 1) * dim + i - j]);
        if (spinIdx - dim >= 0)
        {
            // sum -= tex1Dfetch(s_text, spinIdx - dim) * K;
        }
        else
        {
            // sum -= K;
        }

        // calcAfter<<<1, 1>>>(prevTrotterBlock[(j + 1) * dim + i - j]);
        if (spinIdx + dim < dim * trotterNum)
        {
            // sum += tex1Dfetch(pre_s_text, spinIdx + dim) * K;
        }
        else
        {
            // sum += K;
        }

        // check flip
        if (exp(-1 * sum / beta) > curand_uniform(&state))
        {
            trotterBlock[spinIdx] *= -1;
        }
    }
}

extern "C"
{
    float simulatedQA(int *s, float *Q, int dim, int trotterNum, int totalSweeps);
}

float simulatedQA(int *s, float *Q, int dim, int trotterNum, int totalSweeps)
{
    srand(1);

    int *trotterBlock;
    cudaMalloc(&trotterBlock, trotterNum * dim * sizeof(int));

    int *trotterBlockLocal;
    cudaMallocHost(&trotterBlockLocal, trotterNum * dim * sizeof(int));

    int *prevTrotterBlock;
    cudaMalloc(&prevTrotterBlock, trotterNum * dim * sizeof(int));

    for (int i = 0; i < trotterNum; i++)
    {
        cudaMemcpy(&trotterBlockLocal[i * dim], s, dim * sizeof(int), cudaMemcpyHostToHost);
    }

    cudaMemcpy(trotterBlock, trotterBlockLocal, dim * trotterNum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(prevTrotterBlock, trotterBlock, dim * trotterNum * sizeof(int), cudaMemcpyDeviceToDevice);

    float *Q_dev;
    cudaMalloc(&Q_dev, dim * dim * sizeof(float));
    cudaErr(cudaMemcpy(Q_dev, Q, dim * dim * sizeof(float), cudaMemcpyHostToDevice));

    cudaErr(cudaBindTexture(0, Q_text, Q_dev, dim * dim * sizeof(float)));
    cudaErr(cudaBindTexture(0, s_text, trotterBlock, dim * trotterNum * sizeof(int)));
    cudaErr(cudaBindTexture(0, pre_s_text, prevTrotterBlock, dim * trotterNum * sizeof(int)));

    for (int sweep = 1; sweep <= totalSweeps; sweep++)
    {
        for (int i = 0; i < dim + trotterNum - 1; i++)
        {
            for (int j = 0; j <= i && j < trotterNum; j++)
            {
                if (i - j < dim)
                {
                    cudaStream_t stream;
                    cudaStreamCreate(&stream);
                    // calculate in parallel psudocode //
                    calculate<<<32, 32, 0, stream>>>(trotterBlock, j * dim + i - j, dim, trotterNum, sweep, rand());
                    cudaStreamDestroy(stream);
                }
            }
            cudaDeviceSynchronize();
        }
        cudaMemcpy(prevTrotterBlock, trotterBlock, trotterNum * dim * sizeof(int), cudaMemcpyDeviceToDevice);

        cudaMemcpy(trotterBlockLocal, trotterBlock, trotterNum * dim * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(s, &trotterBlockLocal[dim * (trotterNum - 1)], dim * sizeof(int), cudaMemcpyHostToHost);
        printf("%.9f\n", energy(s, Q, dim));
    }

    cudaMemcpy(trotterBlockLocal, trotterBlock, trotterNum * dim * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(s, &trotterBlockLocal[dim * (trotterNum - 1)], dim * sizeof(int), cudaMemcpyHostToHost);

    cudaUnbindTexture(Q_text);
    cudaUnbindTexture(s_text);
    cudaUnbindTexture(pre_s_text);

    cudaFree(trotterBlock);
    cudaFree(prevTrotterBlock);
    cudaFree(Q_dev);
    cudaFreeHost(trotterBlockLocal);

    return energy(s, Q, dim);
}