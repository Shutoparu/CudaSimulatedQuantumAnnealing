#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <random>

#define K 0

texture<float, 1, cudaReadModeElementType> Q_text;
texture<int, 1, cudaReadModeElementType> s_text;
texture<int, 1, cudaReadModeElementType> pre_s_text;

/**
 * @brief used to check cuda error
 *
 * @param err cuda error to be checked
 */
void cudaErr (cudaError_t err) {
    if (err != cudaSuccess) {
        printf ("%s, %s\n", cudaGetErrorName (err), cudaGetErrorString (err));
        exit (1);
    }
}

/**
 * @brief function to calculate energy. currently running on cpu
 *
 * @param s binary array
 * @param Q qubo matrix
 * @param dim dimention of the array
 * @return calculated energy.
 */
float energy (int* s, float* Q, int dim) {
    float* temp;
    temp = (float*)malloc (sizeof (float) * dim);
    float sum = 0;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            temp[i] += s[j] * Q[i * dim + j];
        }
    }
    for (int i = 0; i < dim; i++) {
        sum += temp[i] * s[i];
        temp[i] = 0;
    }
    free (temp);
    return sum;
}

/**
 * @brief create the beta array
 *
 * @param betaStart starting value of beta
 * @param betaStop ending value of beta
 * @param beta the beta array to be returned
 * @param sweeps the length of beta array
 */
void getAnnealingBeta (int betaStart, int betaStop, float* beta, int sweeps) {

    float logBetaStart = log ((float)betaStart);
    float logBetaStop = log ((float)betaStop);
    float logBetaRange = (logBetaStop - logBetaStart) / (float)sweeps;
    for (int i = 0; i < sweeps; i++) {
        beta[i] = exp (logBetaStart + logBetaRange * i);
    }
}

/**
 * @brief used to determine whether to flip a bit or not
 *
 * @param trotterBlock all trotters in an array
 * @param spinIdx the index of the checked bit in the trotter block
 * @param dim dimetnion of the bit array
 * @param trotterNum numbers of trotters
 * @param beta used to determine passing threshold
 * @param seed used to generate random number
 */
__global__ void calculate (int* trotterBlock, int spinIdx, int dim, int trotterNum, int beta, int seed) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    // if (index < dim)
    if (index == 0) {
        int flipped = 0;
        float delta_E = 0;
        curandState state;
        curand_init (seed, index, 0, &state);

        // get energy change for flipping the bit [i] (check delta_E)

        // check flip
        if (tex1Dfetch (s_text, spinIdx) == 0) {
            flipped = 1;
        }

        for (int n = 0; n < dim; n++) {
            if (n == spinIdx % dim && flipped == 1) {
                delta_E += tex1Dfetch (Q_text, (spinIdx % dim) * dim + n); // time consuming
            } else {
                delta_E += tex1Dfetch (s_text, dim * (spinIdx % dim) + n) * tex1Dfetch (Q_text, (spinIdx % dim) * dim + n); // time consuming
            }
        }
        if (flipped == 0) {
            delta_E *= -1;
        }

        // calcPre<<<1, 1>>>(trotterBlock[(j - 1) * dim + i - j]);
        if (spinIdx - dim >= 0) {
            delta_E -= tex1Dfetch (s_text, spinIdx - dim) * K;
        } else {
            delta_E -= tex1Dfetch (s_text, dim * (trotterNum - 1) + spinIdx % dim) * K;
        }

        // calcAfter<<<1, 1>>>(prevTrotterBlock[(j + 1) * dim + i - j]);
        if (spinIdx + dim < dim * trotterNum) {
            delta_E += tex1Dfetch (s_text, spinIdx + dim) * K;
        } else {
            delta_E += tex1Dfetch (s_text, spinIdx % dim) * K;
        }

        if (flipped == 0) {
            delta_E *= -1;
        }

        if (delta_E < 0) {
            trotterBlock[spinIdx] *= -1;
            trotterBlock[spinIdx] += 1;
        } else if (exp (-1 * delta_E / beta) > curand_uniform (&state)) {
            trotterBlock[spinIdx] *= -1;
            trotterBlock[spinIdx] += 1;
        }
    }
}

extern "C"
{
    float simulatedQA (int* s, float* Q, int dim, int trotterNum, int totalSweeps);
}

/**
 * @brief the code to run a simulated quantum annealing algorithm
 *
 * @param s binary array
 * @param Q qubo matrix
 * @param dim size of binary array
 * @param trotterNum numbers of trotters needed
 * @param totalSweeps numbers of monte carlo steps
 * @return the final energy after the algorithm
 */
float simulatedQA (int* s, float* Q, int dim, int trotterNum, int totalSweeps) {
    srand (1);

    int* trotterBlock;
    cudaMalloc (&trotterBlock, trotterNum * dim * sizeof (int));

    int* prevTrotterBlock;
    cudaMalloc (&prevTrotterBlock, trotterNum * dim * sizeof (int));

    for (int i = 0; i < trotterNum; i++) {
        cudaErr (cudaMemcpy (&trotterBlock[i * dim], s, dim * sizeof (int), cudaMemcpyHostToDevice));
    }

    cudaErr (cudaMemcpy (prevTrotterBlock, trotterBlock, dim * trotterNum * sizeof (int), cudaMemcpyDeviceToDevice));

    float* beta;
    cudaMallocManaged (&beta, totalSweeps * sizeof (float));

    float betaStart = 1;
    float betaEnd = 100;

    getAnnealingBeta (betaStart, betaEnd, beta, totalSweeps);

    float* Q_dev;
    cudaErr (cudaMalloc (&Q_dev, dim * dim * sizeof (float)));
    cudaErr (cudaMemcpy (Q_dev, Q, dim * dim * sizeof (float), cudaMemcpyHostToDevice));

    cudaErr (cudaBindTexture (0, Q_text, Q_dev, dim * dim * sizeof (float)));
    cudaErr (cudaBindTexture (0, s_text, trotterBlock, dim * trotterNum * sizeof (int)));
    cudaErr (cudaBindTexture (0, pre_s_text, prevTrotterBlock, dim * trotterNum * sizeof (int)));

    for (int sweep = 0; sweep < totalSweeps; sweep++) {
        for (int i = 0; i < dim + trotterNum - 1; i++) {
            for (int j = 0; j < trotterNum && j <= i; j++) {
                if (i - j < dim) {
                    cudaStream_t stream;
                    cudaStreamCreate (&stream);
                    // calculate in parallel psudocode //
                    calculate << <1, 1, 0, stream >> > (trotterBlock, j * dim + i - j, dim, trotterNum, beta[sweep], rand ());
                    cudaStreamDestroy (stream);
                }
            }
            cudaDeviceSynchronize ();
        }
        cudaMemcpy (prevTrotterBlock, trotterBlock, trotterNum * dim * sizeof (int), cudaMemcpyDeviceToDevice);

        cudaMemcpy (s, &trotterBlock[dim * (trotterNum - 1)], dim * sizeof (int), cudaMemcpyDeviceToHost); // dim * (trotterNum - 1)
        for (int i = 0; i < dim; i++) {
            printf ("%d", s[i]);
        }
        printf ("\n");
        // printf("%.9f\n", energy(s, Q, dim));
    }

    cudaMemcpy (s, &trotterBlock[0], dim * sizeof (int), cudaMemcpyDeviceToHost); // dim * (trotterNum - 1)

    cudaUnbindTexture (Q_text);
    cudaUnbindTexture (s_text);
    cudaUnbindTexture (pre_s_text);

    cudaFree (trotterBlock);
    cudaFree (prevTrotterBlock);
    cudaFree (Q_dev);

    float en;
    en = energy (s, Q, dim);
    printf ("%f\n", en);
    return en;
}