#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string.h>
#include <errno.h>
#include <fstream>

#include "randoms.h"

#define LOOP_BOUND 11

int main(int argc, char* argv[]){

    //get parameters:
    int imageWidth=300;
    int imageHeight=200;
    int samplesPerPixel=5;
    int loopBound=11;
    int tx=8;
    int ty=8;

    std::ifstream inputParameters("parameters.txt");
    std::string line;
    while(std::getline(inputParameters, line)){
        int pos=line.find("=");
        std::string key=line.substr(0,pos);
        int value=std::stoi(line.substr(pos+1));
        if(key=="imageWidth"){
            imageWidth=value;
        }
        else if(key=="imageHeight"){
            imageHeight=value;
        }
        else if(key=="samplesPerPixel"){
            samplesPerPixel=value;
        }
        else if(key=="loopBound"){
            loopBound=value;
        }
    }

    inputParameters.close();
    
    //Random objects:
    // (2*LOOP_BOUND*2*LOOP_BOUND)*(1)
    printf("Starting\n");
    std::vector<float> objectRandoms=getRandoms((2*loopBound*2*loopBound)*(1));
    std::ofstream objRandFile("object_randoms.txt");

    for(float &r: objectRandoms){
        objRandFile << r << "\n";
    }

    objRandFile.close();

    //Random raytracing:
    // (image height)*(image width)*(samples per pixel)*(2+2)
    //for now: 600*400*10*4
    //imageWidth*imageHeight*samplesPerPixel*4;
    int total=100+(samplesPerPixel*4);
    // unsigned long amount=(total);

    std::vector<float> raytraceRandoms=getRandoms(total);

    std::ofstream rtRandFile("raytrace_randoms.txt");

    for(float &r: raytraceRandoms){
        rtRandFile << r << "\n";
    }

    rtRandFile.close();

    printf("Done\n");
    return 0;
}