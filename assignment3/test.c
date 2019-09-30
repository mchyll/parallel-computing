#include <stdio.h>
#include <stdlib.h>

typedef struct {
    char* str;
} Obj;

void swp(Obj **one, Obj **two) {
    Obj *helper = *two;
    *two = *one;
    *one = helper;
}

int main() {
    Obj *o1 = malloc(sizeof(Obj));
    Obj *o2 = malloc(sizeof(Obj));

    *o1 = (Obj) { "Magnus 1" };
    *o2 = (Obj) { "Conrad 2" };

    printf("o1: %s\n", o1->str);
    printf("o2: %s\nSwap\n", o2->str);

    swp(&o1, &o2);

    printf("o1: %s\n", o1->str);
    printf("o2: %s\n", o2->str);

    Obj **ptrptr = &o1;
    printf("ptrptr->str: %s\n", (*ptrptr)->str);
}
