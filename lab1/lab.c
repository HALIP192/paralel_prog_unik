
#include <time.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#ifndef SIZE
# define SIZE 4096
#endif

#ifndef NR_THREADS
# define NR_THREADS 8
#endif

static void init_mtrx(float **array, int size);
static void memset_mtrx(float **array, int size, float val);
static void print_mtrx(float **array, int size);
static void st_mltpl(float **A, float **B, float **C, int size);
static void mt_mltpl(float **A, float **B, float **C, int size);
static void omp_mltpl(float **A, float **B, float **C, int size);
static float timer_diff(struct timespec *begin, struct timespec *end);
static void alloc_mtrx(float ***a, float ***b, float ***c, int size);

int main(int argc, char *argv[])
{
	float **A, **B, **C;
	struct timespec first, second;

	if (argc != 2) {
		return 1;
    }

	srand(123);

	alloc_mtrx(&A, &B, &C, SIZE);
	init_mtrx(A, SIZE);
	init_mtrx(B, SIZE);
	memset_mtrx(C, SIZE, 0);

	clock_gettime(CLOCK_MONOTONIC, &first);

    int swtch = atoi(argv[1]);

    switch(swtch) {
	case 1:
        printf("single thread\n");
		st_mltpl(A, B, C, SIZE);
		break;
	case 2:
        printf("omp\n");
		omp_mltpl(A, B, C, SIZE);
		break;
	case 3:
        printf("multithread\n");
		mt_mltpl(A, B, C, SIZE);
		break;
    default:
        printf("Wrong arguments:\n"
               "1 - single thread\n"
               "2 - omp\n"
               "3 - pthread\n");
        return 1;
	}

	clock_gettime(CLOCK_MONOTONIC, &second);

	if (SIZE < 20) {
		print_mtrx(C, SIZE);
    }

	printf("time spend into calc: %f sec\n", timer_diff(&first, &second));

	return 0;
}

static void init_mtrx(float **array, int size)
{
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			array[i][j] = (float)rand() / (float)RAND_MAX * 10;
		}
	}
}

static void memset_mtrx(float **array, int size, float val)
{
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			array[i][j] = val;
		}
	}
}

static void print_mtrx(float **array, int size)
{
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("\t%f", array[i][j]);
		}
		printf("\n");
	}
}
static void st_mltpl(float **A, float **B, float **C, int size)
{
	for (int i = 0; i < size; i++) {
		for (int k = 0; k < size; k++) {
			for (int j = 0; j < size; j++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

static float **pA;
static float **pB;
static float **pC;

static void *mltpl_task(unsigned long tid)
{
	for (int i = tid; i < SIZE; i += NR_THREADS) {
		for (int k = 0; k < SIZE; k++) {
			for (int j = 0; j < SIZE; j++) {
				pC[i][j] += pA[i][k] * pB[k][j];
			}
		}
	}

	return NULL;
}

static void mt_mltpl(float **A, float **B, float **C, int size)
{
	(void)size;
	pthread_t threads[NR_THREADS];
	pA = A;
	pB = B;
	pC = C;

	for (unsigned long i = 0; i < NR_THREADS; i++) {
		pthread_create(&threads[i], NULL, (void *)mltpl_task, (void *)i);
	}

	for (int i = 0; i < NR_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}
}

static void omp_mltpl(float **A, float **B, float **C, int size)
{
#pragma omp parallel for num_threads(NR_THREADS)
	for (int i = 0; i < size; i++) {
		for (int k = 0; k < size; k++) {
			for (int j = 0; j < size; j++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

static float timer_diff(struct timespec *begin, struct timespec *end)
{
	float res = 0;

	res += end->tv_sec - begin->tv_sec;
	res += (end->tv_nsec - begin->tv_nsec) * 1e-9f;

	return res;
}

static void alloc_mtrx(float ***a, float ***b, float ***c, int size)
{
	*a = malloc(sizeof(float *) * size);
	*b = malloc(sizeof(float *) * size);
	*c = malloc(sizeof(float *) * size);

    if (!(*a) || !(*b) || !(*c)) {
        perror("malloc\n");
        return ;
    }

	for (int i = 0; i < size; ++i) {
		(*a)[i] = malloc(sizeof(float) * size);
		(*b)[i] = malloc(sizeof(float) * size);
		(*c)[i] = malloc(sizeof(float) * size);
        
        if (!((*a)[i]) || !((*b)[i]) || !((*c)[i])) {
            perror("malloc\n");
            return ;
        }
	}
}


