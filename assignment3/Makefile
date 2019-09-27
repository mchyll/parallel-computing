##
# Game of Life Makefile
#
# @file
# @version 0.1

SRC := $(wildcard *.c)
SRC += $(wildcard libs/*.c)
SRC := $(filter-out serial_main.c, $(SRC))
OBJ := $(patsubst %.c,%.o,$(SRC))

CC := mpicc

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

run: main
	mpirun main img/before.bmp aftermath.bmp

runserial: serial
	./serial img/before.bmp seriaftermath.bmp

$(OBJ) : %.o : %.c
	$(CC) $(FLAGS) -c $< -o $@

clean:
	rm -Rf $(OBJ)
	rm -Rf main

# end
