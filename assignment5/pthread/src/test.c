#include <pthread.h>
#include <stdio.h>

pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;

void* test(void* nope) {
    printf("Yas\n");
    pthread_mutex_lock(&mut);
    printf("Locked once (count=1)\n");
    pthread_mutex_lock(&mut);
    printf("Locked twice (count=2)\n");
    pthread_mutex_unlock(&mut);
    printf("Unlocked once (count=1)\n");
    pthread_mutex_unlock(&mut);
    printf("Unlocked twice (count=0)\n");
}

int _main(int argc, char *argv[]) {
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mut, &attr);

    pthread_t t;
    printf("Create thread: %d\n", pthread_create(&t, NULL, test, NULL));
    pthread_setname_np(t, "worker");
    pthread_join(t, NULL);
    return 0;
}
