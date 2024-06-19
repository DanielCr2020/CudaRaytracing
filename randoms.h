#include <vector>
#include <string>
#include <time.h>

std::vector<float> getRandoms(long amount){
    srand(time(NULL));
    std::vector<float> randVect;
    for(long i=0;i<amount;i++){
        randVect.push_back(((double) rand()/(RAND_MAX)));
    }
    return randVect;
}