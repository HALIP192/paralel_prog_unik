#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define SIZE 4096
#define NR_THREADS 20

static float timer_diff(struct timespec *begin, struct timespec *end);
static void alloc_mtrx(float ***mat, int size);
static void gen_mtrx(float **mat, int size);
static void print_mtrx(float **mat, int size);
static void st_gause(float **mat, int size);
static void cpy_mtrx(float **dst, float **src, int size);
static void omp_gause(float **mat, int size);
static void validate(float **orig, float **res, int size);
static void mt_join_gause(float **mat, int size);
static void *gause_task_backward_join(long tid);
static void *gause_task_forward_join(long tid);
static void mt_barrier_gause(float **mat, int size);
static void *gause_task_barrier(long tid);

int main(int argc, char **argv)
{
	float **mat = NULL;
	float **orig = NULL;
	struct timespec first, second;

	if (argc != 2) {
		return 1;
    }

	srand(123);
	alloc_mtrx(&mat, SIZE);
	gen_mtrx(mat, SIZE);

	if (SIZE < 20) {
		alloc_mtrx(&orig, SIZE);
		cpy_mtrx(orig, mat, SIZE);

		printf("initial matrix\n");
		print_mtrx(mat, SIZE);
	}

	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &first);
    int swtch = atoi(argv[1]);
	switch (swtch) {
	case 1:
        printf("single thread\n");
		st_gause(mat, SIZE);
		break;
	case 2:
        printf("pthread with barier\n");
		mt_barrier_gause(mat, SIZE);
		break;
    case 3:
        printf("pthread with join\n");
		mt_join_gause(mat, SIZE);
		break;
	case 4:
        printf("omp\n");
		omp_gause(mat, SIZE);
		break;
	default:
        printf("Wrong input:\n"
               "1 - single thread\n"
               "2 - pthread with barrier\n"
               "3 - pthread with join\n"
               "4 - omp\n");
		return 1;
	}
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &second);

	if (SIZE < 20)
		validate(orig, mat, SIZE);

	printf("time spend into calc: %f sec\n", timer_diff(&first, &second));
}
static float timer_diff(struct timespec *begin, struct timespec *end)
{
    float res = 0;

	res += end->tv_sec - begin->tv_sec;
	res += (end->tv_nsec -  begin->tv_nsec) * 1e-9f;

	return res;
}

static void alloc_mtrx(float ***mat, int size)
{
	*mat = malloc(sizeof(float *) * size);

	if (*mat == NULL)
		goto oom;

	for (int y = 0; y < size; y++) {
		(*mat)[y] = malloc(sizeof(float) * (size + 1));
		if ((*mat)[y] == NULL)
			goto oom;
	}
	return;

oom:
	perror("OOM\n");
	abort();
}

static void gen_mtrx(float **mat, int size)
{
	for (int y = 0; y < size; y++) {
		for (int x = 0; x <= size; x++) {
			int v = rand() % 18; // 0 .. 18
			v -= 9; // -9 .. 8
			if (v >= 0)
				v++; // -9 .. -1, 1 .. 9

			mat[y][x] = v;
		}
	}
}

static void cpy_mtrx(float **dst, float **src, int size)
{
	for (int y = 0; y < size; y++) {
		for (int x = 0; x <= size; x++) {
			dst[y][x] = src[y][x];
		}
	}
}

static void print_mtrx(float **mat, int size)
{
	for (int y = 0; y < size; y++) {
		for (int x = 0; x < size; x++) {
			printf("% 8.3f ", mat[y][x]);
		}

		printf(" | ");
		printf("% 8.3f\n", mat[y][size]);
	}
}

static void st_gause(float **mat, int size)
{
	for (int pass = 0; pass < size - 1; pass++) {
		for (int row = pass + 1; row < size; row++) {
			float frac = mat[row][pass] / mat[pass][pass];

			for (int col = pass + 1; col <= size; col++) {
				mat[row][col] -= frac * mat[pass][col];
			}
		}
	}

	for (int pass = 0; pass < size; pass++) {
		for (int row = size - 2 - pass; row >= 0; row--) {
			float frac = mat[row][size - 1 - pass] / mat[size - 1 - pass][size - 1 - pass];

			mat[row][size] -= frac * mat[size - 1 - pass][size];
		}

		mat[size - pass - 1][size] /= mat[size - 1 - pass][size - 1 - pass];
	}

}

static float **Mat;
static volatile int Pass;
static pthread_barrier_t barrier;

static void *gause_task_barrier(long tid)
{
	const int size = SIZE;
	int pass;

	for (pass = 0; pass < size - 1; pass++) {
		for (int row = pass + 1 + tid; row < size; row += NR_THREADS) {
			float frac = Mat[row][pass] / Mat[pass][pass];

			for (int col = pass + 1; col <= size; col++) {
				Mat[row][col] -= frac * Mat[pass][col];
			}
		}
		pthread_barrier_wait(&barrier);
	}

	for (pass = 0; pass < size; pass++) {
		for (int row = size - 2 - pass - tid; row >= 0; row -= NR_THREADS) {
			float frac = Mat[row][size - 1 - pass] / Mat[size - 1 - pass][size - 1 - pass];

			Mat[row][size] -= frac * Mat[size - 1 - pass][size];
		}

		pthread_barrier_wait(&barrier);
		if (tid == 0)
			Mat[size - pass - 1][size] /= Mat[size - 1 - pass][size - 1 - pass];
	}

	return NULL;
}

static void mt_barrier_gause(float **mat, int size)
{
	pthread_t threads[NR_THREADS];
	(void)size;

	Mat = mat;
	pthread_barrier_init(&barrier, NULL, NR_THREADS);

	for (long i = 0; i < NR_THREADS; i++) {
		pthread_create(&threads[i], NULL, (void *)gause_task_barrier, (void *)i);
	}

	for (int i = 0; i < NR_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	pthread_barrier_destroy(&barrier);
}

static void *gause_task_forward_join(long tid)
{
	const int size = SIZE;
	const int pass = Pass;

	for (int row = pass + 1 + tid; row < size; row += NR_THREADS) {
		float frac = Mat[row][pass] / Mat[pass][pass];

		for (int col = pass + 1; col <= size; col++) {
			Mat[row][col] -= frac * Mat[pass][col];
		}
	}

	return NULL;
}

static void *gause_task_backward_join(long tid)
{
	const int size = SIZE;
	const int pass = Pass;

	for (int row = size - 2 - pass - tid; row >= 0; row -= NR_THREADS) {
		float frac = Mat[row][size - 1 - pass] / Mat[size - 1 - pass][size - 1 - pass];

		Mat[row][size] -= frac * Mat[size - 1 - pass][size];
	}

	return NULL;
}

static void mt_join_gause(float **mat, int size)
{
	pthread_t threads[NR_THREADS];
	Mat = mat;

	for (int pass = 0; pass < size - 1; pass++) {
		Pass = pass;
		for (long t = 0; t < NR_THREADS; t++) {
			pthread_create(&threads[t], NULL, (void *)gause_task_forward_join, (void *)t);
		}
		for (long t = 0; t < NR_THREADS; t++) {
			pthread_join(threads[t], NULL);
		}
	}

	for (int pass = 0; pass < size; pass++) {
		Pass = pass;
		for (long t = 0; t < NR_THREADS; t++) {
			pthread_create(&threads[t], NULL, (void *)gause_task_backward_join, (void *)t);
		}
		for (long t = 0; t < NR_THREADS; t++) {
			pthread_join(threads[t], NULL);
		}
		mat[size - pass - 1][size] /= mat[size - 1 - pass][size - 1 - pass];
	}
}

static void omp_gause(float **mat, int size)
{
	for (int pass = 0; pass < size - 1; pass++) {
		#pragma omp parallel for
		for (int row = pass + 1; row < size; row++) {
			float frac = mat[row][pass] / mat[pass][pass];

			for (int col = pass + 1; col <= size; col++) {
				mat[row][col] -= frac * mat[pass][col];
			}
		}
	}

	for (int pass = 0; pass < size; pass++) {
		#pragma omp parallel for
		for (int row = size - 2 - pass; row >= 0; row--) {
			float frac = mat[row][size - 1 - pass] / mat[size - 1 - pass][size - 1 - pass];

			mat[row][size] -= frac * mat[size - 1 - pass][size];
		}

		mat[size - pass - 1][size] /= mat[size - 1 - pass][size - 1 - pass];
	}

}

static void validate(float **orig, float **res, int size)
{
	for (int row = 0; row < size; row++) {
		float r = 0;

		for (int col = 0; col < size; col++) {
			r += res[col][size] * orig[row][col];
		}
		r -= orig[row][size];
		printf("root %d (%f) error: %f\n", row, res[row][size], r);
	}
}

