#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

typedef struct {
    char* str;
} Obj;

void swp(Obj **one, Obj **two) {
    Obj *helper = *two;
    *two = *one;
    *one = helper;
}

int main() {
    // Obj *o1 = malloc(sizeof(Obj));
    // Obj *o2 = malloc(sizeof(Obj));

    // *o1 = (Obj) { "Magnus 1" };
    // *o2 = (Obj) { "Conrad 2" };

    // printf("o1: %s\n", o1->str);
    // printf("o2: %s\nSwap\n", o2->str);

    // swp(&o1, &o2);

    // printf("o1: %s\n", o1->str);
    // printf("o2: %s\n", o2->str);

    // Obj **ptrptr = &o1;
    // printf("ptrptr->str: %s\n", (*ptrptr)->str);


    // struct timeval time_start;
    // gettimeofday(&time_start, NULL);
    // printf("Start: %d\n", time_start.tv_usec);

    struct timespec time_start;
    clock_gettime(CLOCK_REALTIME, &time_start);
    double t_start = time_start.tv_sec + (double)time_start.tv_nsec / 1e9;
    printf("Start: %f\n", t_start);

    sleep(3);

    // struct timeval time_end;
    // gettimeofday(&time_end, NULL);
    // printf("End: %d\n", time_end.tv_usec);

    struct timespec time_end;
    clock_gettime(CLOCK_REALTIME, &time_end);
    double t_end = time_end.tv_sec + (double)time_end.tv_nsec / 1e9;
    printf("End: %f\n", t_end);

    double elapsed = t_end - t_start;

    // double elapsed = (double) (time_end.tv_usec - time_start.tv_usec) / (double)1e6;
    printf("Elapsed: %f\n", elapsed);
}
