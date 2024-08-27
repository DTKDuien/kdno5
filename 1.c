#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <pthread.h>

// Function prototypes
void generate_random_matrix(int n, int m, double** A, double* b);
void solve_sequential(int n, int m, double** A, double* b, double* x);
void solve_openmp(int n, int m, double** A, double* b, double* x);
void* solve_pthreads(void* arg);

// Structure to hold arguments for pthreads
typedef struct {
    int n;
    int m;
    double** A;
    double* b;
    double* x;
    int thread_id;
    int num_threads;
} pthread_args;

int main() {
    int n = 1000;  // Number of equations
    int m = 1000;   // Number of unknowns (columns of matrix A)
    double** A;
    double* b;
    double* x_seq;
    double* x_omp;
    double* x_pth;
    
    // Allocate memory for the matrix and vectors
    A = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        A[i] = (double*)malloc(m * sizeof(double));  // Allocate m columns
    }
    b = (double*)malloc(n * sizeof(double));
    x_seq = (double*)malloc(n * sizeof(double));
    x_omp = (double*)malloc(n * sizeof(double));
    x_pth = (double*)malloc(n * sizeof(double));
    
    // Generate random matrix and vector
    generate_random_matrix(n, m, A, b);

    // Measure time for sequential solution
    clock_t start = clock();
    solve_sequential(n, m, A, b, x_seq);
    clock_t end = clock();
    double time_seq = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Sequential time: %f seconds\n", time_seq);

    // Measure time for OpenMP solution
    start = clock();
    solve_openmp(n, m, A, b, x_omp);
    end = clock();
    double time_omp = (double)(end - start) / CLOCKS_PER_SEC;
    printf("OpenMP time: %f seconds\n", time_omp);

    // Measure time for Pthreads solution
    int num_threads = 4;  // Number of threads
    pthread_t threads[num_threads];
    pthread_args args[num_threads];
    start = clock();
    for (int i = 0; i < num_threads; i++) {
        args[i].n = n;
        args[i].m = m;  // Pass m to pthread_args
        args[i].A = A;
        args[i].b = b;
        args[i].x = x_pth;
        args[i].thread_id = i;
        args[i].num_threads = num_threads;
        pthread_create(&threads[i], NULL, solve_pthreads, (void*)&args[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    end = clock();
    double time_pth = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Pthreads time: %f seconds\n", time_pth);

    // Verify correctness (compare x_seq, x_omp, x_pth if needed)

    // Free allocated memory
    for (int i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);
    free(b);
    free(x_seq);
    free(x_omp);
    free(x_pth);

    return 0;
}

// Function to generate a random matrix and vector
void generate_random_matrix(int n, int m, double** A, double* b) {
    srand(time(NULL));
    printf("Randomly generated matrix A (%d x %d) and vector b (%d):\n", n, m, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[i][j] = (double)(rand() % 100);
            printf("%f ", A[i][j]);
        }
        b[i] = (double)(rand() % 100);
        printf(" | %f\n", b[i]);
    }
    printf("\n");
}

// Function to solve the system sequentially
void solve_sequential(int n, int m, double** A, double* b, double* x) {
    // Implementation of a simple solver (e.g., Gaussian elimination)
    for (int i = 0; i < n; i++) {
        x[i] = b[i] / A[i][i];
        for (int j = 0; j < m; j++) {
            if (i != j) {
                x[i] -= A[i][j] * x[j] / A[i][i];
            }
        }
    }
}

// Function to solve the system using OpenMP
void solve_openmp(int n, int m, double** A, double* b, double* x) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] = b[i] / A[i][i];
        for (int j = 0; j < m; j++) {
            if (i != j) {
                x[i] -= A[i][j] * x[j] / A[i][i];
            }
        }
    }
}

// Function to solve the system using Pthreads
void* solve_pthreads(void* arg) {
    pthread_args* args = (pthread_args*)arg;
    int n = args->n;
    int m = args->m;  // Receive m
    double** A = args->A;
    double* b = args->b;
    double* x = args->x;
    int thread_id = args->thread_id;
    int num_threads = args->num_threads;

    for (int i = thread_id; i < n; i += num_threads) {
        x[i] = b[i] / A[i][i];
        for (int j = 0; j < m; j++) {
            if (i != j) {
                x[i] -= A[i][j] * x[j] / A[i][i];
            }
        }
    }
    return NULL;
}
