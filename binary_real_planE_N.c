#include <stdio.h>
#include <stdlib.h>
#include "outer_void.c"

#define N_vec 256
#define N_bvec 10
#define VECTOR_SIZE 4096
#define VECTOR_DIFF 512*512
#define DIFF_1 4
#define DIFF_2 8
#define DIFF_3 12
#define DIFF_4 48
#define DIFF_5 64
#define DIFF_6 128
#define DIFF_7 256
#define DIFF_8 512
#define DIFF_9 1024
#define DIFF_10 2048
#define DIFF_11 8192
#define DIFF_12 34868
#define DIFF_13 6246532

void a_0() {printf("This is a_0\n");}

// shuffle vector	
/* void generate_handler_vector(void (**handlers)(), void (**dst)()) { */
/* 	for (int i = 0; i < N_vec; i++) { */
/* 		size_t handlerIdx = rand() % N_vec; */
/* 		dst[i] = handlers[handlerIdx]; */
/* 	} */
/* } */

void write_image(char *filename, void (**src)()) {
        FILE *fp;
	if ((fp = fopen(filename, "wb")) == NULL) {
		fprintf(stderr, "fopen error\n");
		exit(EXIT_FAILURE);
	}
	fwrite(src, sizeof(src), N_vec, fp);
}

int main(int argc, char *argv[])
{
	if (argc != 3) {
		printf("./binary <n_start> <n_images>\n");
	 	exit(EXIT_FAILURE);
	}
	int index = 1;
	int ri;
	int offset;
	int bi, bv;
	
	// make table similar to real IDT
	void (**handlers)() = malloc(VECTOR_SIZE);

	handlers[0] = a_0;
	handlers[index++] = handlers[index-1] - DIFF_10;
	handlers[index++] = handlers[index-1] + DIFF_8;
	handlers[index++] = handlers[index-1] - DIFF_8;
	for (int j = 0; j < 7; j++, index++) {
		handlers[index] = handlers[index-1] + DIFF_4;
	}
	handlers[index++] = handlers[index-1] - DIFF_10;
	for (int j = 0; j < 2; j++, index++) {
		handlers[index] = handlers[index-1] + DIFF_6;
	}
	handlers[index++] = handlers[index-1] + DIFF_10;
	for (int j = 0; j < 2; j++, index++) {
		handlers[index] = handlers[index-1] + DIFF_4;
	}
	handlers[index++] = handlers[index-1] - DIFF_10;
	handlers[index++] = handlers[index-1] + DIFF_10;
	handlers[index++] = handlers[index-1] + DIFF_11; // greatly
	for (int j = 0; j < 11; j++, index++) {
		handlers[index] = handlers[index-1] + DIFF_2;
	}
	handlers[index++] = handlers[index-1] - DIFF_11; // greatly
	handlers[index++] = handlers[index-1] - DIFF_8;
	for (int k = 0; k < 13; k++) {
		for (int j = 0; j < 6; j++, index++) {
			handlers[index] = handlers[index-1] + DIFF_1;
		}
		handlers[index++] = handlers[index-1] + DIFF_2;
	}
	for (int j = 0; j < 4; j++, index++) {
		handlers[index] = handlers[index-1] + DIFF_1;
	}
	handlers[index++] = handlers[index-1] + DIFF_9;
	handlers[index++] = handlers[index-1] - DIFF_9;
	for (int k = 0; k < 15; k++) {
		handlers[index++] = handlers[index-1] + DIFF_2;
		for (int j = 0; j < 6; j++, index++) {
			handlers[index] = handlers[index-1] + DIFF_1;
		}
	}
	handlers[index++] = handlers[index-1] + DIFF_2;
	for (int j = 0; j < 3; j++, index++) {
		handlers[index] = handlers[index-1] + DIFF_1;
	}
	handlers[index++] = handlers[index-1] + DIFF_7;
	handlers[index++] = handlers[index-1] - DIFF_7;
	handlers[index++] = handlers[index-1] + DIFF_1;
	handlers[index++] = handlers[index-1] + DIFF_8;
	handlers[index++] = handlers[index-1] - DIFF_8;
	for (int j = 0; j < 2; j++, index++) {
		handlers[index] = handlers[index-1] + DIFF_1;
	}
	handlers[index++] = handlers[index-1] + DIFF_9;
	handlers[index++] = handlers[index-1] - DIFF_9;
	handlers[index++] = handlers[index-1] - DIFF_7;
	handlers[index++] = handlers[index-1] + DIFF_8;
	for (int j = 0; j < 6; j++, index++) {
		handlers[index] = handlers[index-1] + DIFF_7;
	}
	

	/* for (int i = 0; i < index; i++) { */
	/* 	printf("handlers[%d] = %d\n", i, handlers[i]); */
	/* } */

	// handlers_b is mal-handlers table
	void (**handlers_b)() = malloc(VECTOR_SIZE);
	handlers_b[0] = b_0;
	handlers_b[1] = b_1;
	handlers_b[2] = b_2;
	handlers_b[3] = b_3;
	handlers_b[4] = b_4;
	handlers_b[5] = b_5;
	handlers_b[6] = b_6;
	handlers_b[7] = b_7;
	handlers_b[8] = b_8;
	handlers_b[9] = b_9;
   
	// n_images execution
	int n_start = atoi(argv[1]);
	int n_images = atoi(argv[2]);
	bi = 128; // syscall_interrupt : 0x80 
	for (int i = n_start; i < (n_start + n_images); i++) {
		// add handlers to offset by KASLR
		void (**moved_vectors)() = malloc(VECTOR_SIZE);
		ri = rand() % 100;
		offset = VECTOR_DIFF * ri;
		for (int j = 0; j < N_vec; j++) {
			moved_vectors[j] = handlers[j] + offset;
		}
		// write true binary to file
		char filename_ideal[20];
		sprintf(filename_ideal, "result_ideal_%d", i);
		write_image(filename_ideal, moved_vectors);
		// insert mal_handler into moved_vectors
        bv = rand() % N_bvec;
		moved_vectors[bi] = handlers_b[bv];
		
		// write mal binary to file
		char filename[20];
		sprintf(filename, "result_%d", i);
		write_image(filename, moved_vectors);
		free(moved_vectors);
	}
	// free memory
	free(handlers);
	free(handlers_b);

	return 0;
}
