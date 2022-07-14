#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define K 0.5

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
 * @brief used to determine whether to flip a bit or not. runs on cpu
 *
 * @param trotterBlock all trotters in an array
 * @param spinIdx the index of the checked bit in the trotter block
 * @param dim dimetnion of the bit array
 * @param trotterNum numbers of trotters
 * @param beta used to determine passing threshold
 * @param seed used to generate random number
 */
void calculate (int* trotterBlock, int* prevTrotterBlock, float* qubo, int spinIdx, int dim, int trotterNum, int beta, float random) {

    int flipped = 0;
    float delta_E = 0;

    // get energy change for flipping the bit [i] (check delta_E)

    // check flip
    if (trotterBlock[spinIdx] == 0) {
        flipped = 1;
    }

    for (int n = 0; n < dim; n++) {
        if (n == spinIdx % dim && flipped == 1) {
            delta_E += qubo[(spinIdx % dim) * dim + n]; // time consuming
        } else {
            delta_E += trotterBlock[dim * (spinIdx % dim) + n] * qubo[(spinIdx % dim) * dim + n]; // time consuming
        }
    }
    if (flipped == 0) {
        delta_E *= -1;
    }

    // calcPre<<<1, 1>>>(trotterBlock[(j - 1) * dim + i - j]);
    if (spinIdx - dim >= 0) {
        delta_E -= trotterBlock[spinIdx - dim] * K;
    } else {
        delta_E -= prevTrotterBlock[dim * (trotterNum - 1) + spinIdx % dim] * K;
    }

    // calcAfter<<<1, 1>>>(prevTrotterBlock[(j + 1) * dim + i - j]);
    if (spinIdx + dim < dim * trotterNum) {
        delta_E += prevTrotterBlock[spinIdx + dim] * K;
    } else {
        delta_E += prevTrotterBlock[spinIdx % dim] * K;
    }

    if (delta_E < 0) {
        trotterBlock[spinIdx] *= -1;
        trotterBlock[spinIdx] += 1;
    } else if (exp (-1 * delta_E / beta) > random) {
        trotterBlock[spinIdx] *= -1;
        trotterBlock[spinIdx] += 1;
    }
}

/**
 * @brief the code to run a simulated quantum annealing algorithm on cpu
 *
 * @param s binary array
 * @param Q qubo matrix
 * @param dim size of binary array
 * @param trotterNum numbers of trotters needed
 * @param totalSweeps numbers of monte carlo steps
 * @return the final energy after the algorithm
 */
extern float simulatedQA (int* s, float* Q, int dim, int trotterNum, int totalSweeps) {
    srand (1);

    int* trotterBlock;
    trotterBlock = (int*)malloc (trotterNum * dim * sizeof (int));

    int* prevTrotterBlock;
    prevTrotterBlock = (int*)malloc (trotterNum * dim * sizeof (int));

    for (int i = 0; i < trotterNum; i++) {
        memcpy (&trotterBlock[i * dim], s, dim * sizeof (int));
    }

    memcpy (prevTrotterBlock, trotterBlock, dim * trotterNum * sizeof (int));

    float* beta;
    beta = (float*)malloc (totalSweeps * sizeof (float));
    float betaStart = 1;
    float betaEnd = 100;
    getAnnealingBeta (betaStart, betaEnd, beta, totalSweeps);

    for (int sweep = 0; sweep < totalSweeps; sweep++) {
        for (int i = 0; i < dim + trotterNum - 1; i++) {
            for (int j = 0; j < trotterNum && j <= i; j++) {
                if (i - j < dim) {
                    // calculate in parallel psudocode //
                    calculate (trotterBlock, prevTrotterBlock, Q, j * dim + i - j, dim, trotterNum, beta[sweep], rand ());
                }
            }
        }
        memcpy (prevTrotterBlock, trotterBlock, trotterNum * dim * sizeof (int));

        memcpy (s, &trotterBlock[dim * (trotterNum - 1)], dim * sizeof (int));
        for (int i = 0; i < dim; i++) {
            printf ("%d", s[i]);
        }
        printf ("\n");
        // printf("%.5f\n", energy(s, Q, dim));
    }

    memcpy (s, &trotterBlock[dim * (trotterNum - 1)], dim * sizeof (int));

    free (trotterBlock);
    free (prevTrotterBlock);
    free (beta);

    float en;
    en = energy (s, Q, dim);
    // printf("\n%f\n\n", en);
    return en;
}

int main () {
    srand (1);
    int dim = 100;
    int trotterNum = 4;
    int totalSweeps = 200;
    int* s;
    s = (int*)malloc (dim * sizeof (int));
    for (int i = 0; i < dim; i++) {
        s[i] = 1;
    }
    float* Q;
    Q = (float*)malloc (dim * dim * sizeof (float));
    for (int i = 0; i < dim * dim; i++) {
        Q[i] = rand () / RAND_MAX;
    }

    simulatedQA (s, Q, dim, trotterNum, totalSweeps);
}