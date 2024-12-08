CC = clang
CFLAGS = -Wall -Wextra -Werror -O3 -ffast-math -fopenmp

.PHONY: lab1 lab2 lab3

all: lab1 lab2 lab3

lab1:
	$(CC) $(CFLAGS) lab1/lab.c -o lab1.bin

lab2:
	$(CC) $(CFLAGS) lab2/lab.c -o lab2.bin

lab3:
	$(CC) $(CFLAGS) lab3/lab.c -lm -o lab3.bin
