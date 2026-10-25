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


void monte_carlo_task(void *inP, void *inoutP, int *len, MPI_Datatype *dptr) {
    int* in = (int*)inP;
    double* out = (double*)inoutP;

    int npes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);



    srand(time(nullptr) & myrank);
    double x = (double)rand() / RAND_MAX;
    *out = x;
}


typedef struct MonteCarloResult_S {
    int rank;
    double result;
} MonteCarloResult;

double monte_carlo_integral_pooled(long samples) {

    int allocated_tasks = 0;

    int npes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double starttime, endtime;
    if (myrank == 0) {
        starttime = MPI_Wtime();
    }
    srand(time(nullptr) & myrank);

    if (myrank == 0) {
        double starttime, endtime;
        if (myrank == 0) {
            starttime = MPI_Wtime();
        }

        double sum = 0;


        MPI_Request rqs[100];
        double results[100];

        for (int i = 1; i < npes; i++) {
            MPI_Send(&myrank, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Irecv(&results[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &rqs[i]);
        }

        int samples_taken = 0;
        while (samples_taken < samples) {

            for (int i = 1; i < npes; i++) {
                MPI_Request req = rqs[i];
                int flag;
                MPI_Status status;
                MPI_Test(&req, &flag, &status);
                if (flag) {

                    sum += results[i];

                    MPI_Send(&myrank, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

                    MPI_Irecv(&results[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &rqs[i]);
                    samples_taken++;

                    if (samples_taken >= samples) break;

                }
            }
        }

        for (int i = 1; i < npes; i++) {
            int shutdown = -1;
            MPI_Send(&shutdown, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

            int shutdown_rank;
            MPI_Status status;
            MPI_Recv(&shutdown_rank, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        }

        double sim_result = sum / double(samples_taken);

        endtime   = MPI_Wtime();
        printf("%f seconds\n",endtime-starttime);

        printf("result: %f\n", sim_result);
        return sim_result;

    }
    else {

        while (true) {
            int x;
            MPI_Status status;

            MPI_Recv(&x, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            double rand_val = (double)rand() / RAND_MAX;

            // shutdown signal
            if (x < 0) {
                MPI_Send(&myrank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
                return 0.0;
            }

            double result = sqrt(1 - rand_val * rand_val);

            MPI_Send(&result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    monte_carlo_integral(1000000000);

    MPI_Finalize();
}
