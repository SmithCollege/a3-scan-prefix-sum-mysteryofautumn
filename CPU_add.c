#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 1000000

double get_clock() {
    //return local time 
    struct timeval tv; 
    int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok<0) { printf("get time ofday error"); }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main(){
  
    // allocate memory
    int* input = malloc(sizeof(int) * SIZE);
    int* output = malloc(sizeof(int) * SIZE);

    // array initialization
    for(int i=0; i< SIZE; i++){
        input[i] = 1;
    }

    // start timer
    double t0 = get_clock();

    // addition
    output[0] = input[0];
    for(int i=1; i< SIZE; i++){
        output[i] = input[i] + output[i-1];
    }
    
    // stop timer
    double t1 = get_clock();

    // result
    printf("%d\n", output[SIZE -1 ]);
    printf("time: %f ns\n", (1000000000.0*(t1-t0)) );

    return 0;
}