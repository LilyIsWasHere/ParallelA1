#include <iostream>
#include <mpi.h>
#include <cmath>
#define PARALLEL 1;

double monte_carlo_integral(long samples) {
#ifndef PARALLEL

    srand(time(nullptr));

    double sum = 0.0;
    for (long i = 0; i < samples; ++i) {
        // Simulation
        double x = (double)rand() / RAND_MAX;

        sum += sqrt(1 - x * x);
    }

#else


    int npes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double starttime, endtime;
    if (myrank == 0) {
        starttime = MPI_Wtime();
    }

    srand(time(nullptr) & myrank);

    long local_samples = samples / npes;
    if (myrank == 0) {
        local_samples += samples % npes;
    }

    double sum = 0.0;
    for (long i = 0; i < local_samples; ++i) {
        double x = (double)rand() / RAND_MAX;
        // Simulation
        sum += sqrt(1 - x * x);
    }

    double final_sum = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&sum, &final_sum,1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (myrank == 0) {
        endtime   = MPI_Wtime();
        printf("%f seconds\n",endtime-starttime);
    }
    if (myrank == 0) {
        printf("Pi = %f\n", final_sum / samples);
    }

    return final_sum / samples;


#endif

    return sum / samples;
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    monte_carlo_integral(1000000000);

    MPI_Finalize();
}
