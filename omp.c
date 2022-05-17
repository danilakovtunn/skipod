/* Include benchmark-specific header. */
#include "jacobi-1d.h"
#include <omp.h>
double bench_t_start, bench_t_end;

static
double rtclock() {
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
        printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}


void bench_timer_start() {
    bench_t_start = rtclock ();
}


void bench_timer_stop() {
    bench_t_end = rtclock ();
}


void bench_timer_print() {
    printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}


static
void init_array (int n, double *A, double *B) {
    int i;

    for (i = 0; i < n; i++) {
        A[i] = ((double) i + 2) / n;
        B[i] = ((double) i + 3) / n;
    }
}


static
void print_array(int n, double *A) {
    int i;

    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stderr, "begin dump: %s", "A");
    for (i = 0; i < n; i++) {
        if (i % 20 == 0) 
            fprintf(stderr, "\n");
        fprintf(stderr, "%0.2lf ", A[i]);
    }
    fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}


static
void kernel_jacobi_1d(int tsteps, int n, double *A, double *B) {
    int t, i;
    for (t = 0; t < tsteps; t++) {
        #pragma omp parallel shared(B) private(i)
        {
            #pragma omp for
            for (i = 1; i < n - 1; i++)
                B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
        }
        #pragma omp barrier
        #pragma omp parallel shared(A) private(i)
        {
            #pragma omp for
            for (i = 1; i < n - 1; i++)
 	            A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
        }
        #pragma omp barrier
    }
}


int main(int argc, char *argv[]) {
    int n_s[] = {30, 120, 400, 2000, 4000};//, 8000, 16000, 32000, 64000};
    int step_s[] = {20, 40, 100, 500, 1000};//, 2000, 4000, 8000, 16000};

    double *A; 
    double *B;
    int max = 64;
    int k = 1;
    while (k <= 64) {
        omp_set_num_threads(k);
        printf("THREADS = %d\n", k);
        for (int i = 1; i < 2; i++) {
            int n = n_s[i];
            int tsteps = step_s[i];
            printf("n=%d tsteps=%d\n", n, tsteps);
            A = (double *) malloc (n * sizeof(double));
            B = (double *) malloc (n * sizeof(double));

            init_array (n, A, B);
            bench_timer_start();

            kernel_jacobi_1d(tsteps, n, A, B);

            bench_timer_stop();
            bench_timer_print();

            printf("A\n");
            print_array(n, A);
            printf("B\n");
            print_array(n, B);
            free(A);
            free(B);
        }

        printf("\n");
        k *= 2;
    }
    return 0;
}
