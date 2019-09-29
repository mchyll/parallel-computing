##
# Game of Life Makefile
#
# @file
# @version 0.1

SRC := $(wildcard *.c)
SRC += $(wildcard libs/*.c)
SRC := $(filter-out serial_main.c compare.c bmptest.c, $(SRC))
OBJ := $(patsubst %.c,%.o,$(SRC))

CC := mpicc

DEBUG = 1
ifdef DEBUG
FLAGS := -g
else
FLAGS := -O3
endif

.PHONY: clean

main: $(OBJ)
	$(CC) $(FLAGS) $^ -o $@

serial: serial_main.c libs/bitmap.c
	gcc $(FLAGS) $^ -o $@

run: main compare
	mpirun main img/before.bmp after.bmp
	./compare after.bmp after_serial.bmp

runserial: serial
	./serial img/before.bmp after_serial.bmp

compare: compare.c libs/bitmap.c
	gcc -g $^ -o $@

$(OBJ) : %.o : %.c
	$(CC) $(FLAGS) -c $< -o $@

clean:
	rm -Rf $(OBJ)
	rm -Rf main

# end
