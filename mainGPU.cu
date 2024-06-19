#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3GPU.h"
#include "rayGPU.h"
#include "sphereGPU.h"
#include "hittable_listGPU.h"
#include "cameraGPU.h"
#include "materialGPU.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) checkCuda( (val), #val, __FILE__, __LINE__ )

void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error=" << static_cast<unsigned int>(result) << " " << cudaGetErrorString(result)
            << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hittable **world, float rand1, float rand2, float rand3) {
    ray curRay=r;
    vec3 curAttenuation=vec3(1.0,1.0,1.0);
    for(int i=0; i<50; i++) {
        hitRecord rec;

        if ((*world)->hit(curRay, 0.001, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;   //3 randoms
            
            if(rec.mat_ptr->scatter(curRay, rec, attenuation, scattered, rand1,rand2,rand3)) {
                curAttenuation *= attenuation;
                curRay=scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction=unitVector(curRay.direction());
            float t=0.5*(unit_direction.y() + 1.0);
            vec3 c=(1.0-t)*vec3(1.0,1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return curAttenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

// __global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    // int i=threadIdx.x + blockIdx.x * blockDim.x;
    // int j=threadIdx.y + blockIdx.y * blockDim.y;
    // if((i >= max_x) || (j >= max_y)) return;
    // int pixel_index=j*max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    // curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
// }

__global__ void renderNaive(vec3 *fb, int max_x, int max_y, int samplesPerPixel, camera **cam, hittable **world, float* rtRandoms) {
    // int i=threadIdx.x + blockIdx.x * blockDim.x;
    // int j=threadIdx.y + blockIdx.y * blockDim.y;
    if(1){
        for(int j=max_y-1;j>=0;j--){
            for(int i=0;i<max_x;i++){
                vec3 col(0, 0, 0);
                for(int s=0;s<samplesPerPixel;s++){
                    float u=float(i+rtRandoms[s]) / float(max_x);
                    float v=float(j+rtRandoms[s+1]) / float(max_y);
                    ray r=(*cam)->getRay(u, v, rtRandoms[s+2], rtRandoms[s+3]);
                    col += color(r, world, rtRandoms[s+4],rtRandoms[s+5],rtRandoms[s+6]);
                }
                col /= float(samplesPerPixel);
                col=vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );
                int ir=int(255.99*col[0]); 
                int ig=int(255.99*col[1]); 
                int ib=int(255.99*col[2]); 
                fb[(j*max_x)+i]=col;
            }
            // printf("Rendering\n");
        }
    }
}

__device__ __constant__ float dRaytraceRandomsConst[8150];

__global__ void render(vec3 *fb, int max_x, int max_y, int samplesPerPixel, camera **cam, hittable **world, float* dRaytraceRandoms) {
    int i=threadIdx.x + blockIdx.x * blockDim.x;
    int j=threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index=j*max_x + i;
    vec3 col(0,0,0);
    for(int s=0; s<samplesPerPixel; s++) {
        float u=float(i+dRaytraceRandoms[s]) / float(max_x);
        float v=float(j+dRaytraceRandoms[s+1]) / float(max_y);
        ray r=(*cam)->getRay(u, v, dRaytraceRandoms[s+2],dRaytraceRandoms[s+3]);
        col += color(r, world, dRaytraceRandoms[s+4],dRaytraceRandoms[s+5],dRaytraceRandoms[s+6]);
    }
    col /= float(samplesPerPixel);
    col[0]=sqrt(col[0]);
    col[1]=sqrt(col[1]);
    col[2]=sqrt(col[2]);
    fb[pixel_index]=col;
}


//max amount of floats that can be in constant memory (16366)
__device__ __constant__ float objRandConst[8150];

#define orc objRandConst
#define bix blockIdx.x
#define bdx blockDim.x
#define tix threadIdx.x

//single threaded - for adding ground and hardcoded spheres
__global__ void createWorldConstants(hittable** dList, hittable** dWorld, camera** dCamera, int imageWidth, int imageHeight, int numHittables){
        if(threadIdx.x==0 && blockIdx.x==0){
            //ground
            // dList[numHittables++]=new sphere(vec3(0,-1000,-1),1000,
            // new matte(vec3(0.03,0.05,0.03)), 1,1,1);

            dList[numHittables++]=new sphere(vec3(0,-1000,-1),1000,
            new metal(vec3(0.1,0.1,0.1),0.0),1,1,1);

            // (+towards -away, +up -down, +left -right)

            //hardcoded large shapes
            //glass
            dList[numHittables++]=new sphere(vec3(0,1.3,0),1.3,new glass(1.5),
            orc[numHittables],orc[numHittables+1],orc[numHittables+2]);

            dList[numHittables++]=new sphere(vec3(10,0.75,-1),0.75,new glass(1.5),
            orc[numHittables],orc[numHittables+1],orc[numHittables+2]);

            //matte
            dList[numHittables++]=new sphere(vec3(-4,1,-3),1.0,new matte(vec3(1,1,1)),
            orc[numHittables],orc[numHittables+1],orc[numHittables+2]);

            dList[numHittables++]=new sphere(vec3(-6,1,6),1.0,new matte(vec3(0.1,0.5,0.8)),
            orc[numHittables],orc[numHittables+1],orc[numHittables+2]);

            //metal
            dList[numHittables++]=new sphere(vec3(4,1,0),1.0,new metal(vec3(0.7,0.6,0.5),0.0),
            orc[numHittables],orc[numHittables+1],orc[numHittables+2]);

            dList[numHittables++]=new sphere(vec3(-7,1.5,-7),1.5,new metal(vec3(0.7,0.85,1),0.0),
            orc[numHittables],orc[numHittables+1],orc[numHittables+2]);

            dList[numHittables++]=new sphere(vec3(-3,2.5,7),2.5,new metal(vec3(0.8,0.85,1),0.0),
            orc[numHittables],orc[numHittables+1],orc[numHittables+2]);

            *dWorld=new hittable_list(dList,numHittables);
            vec3 lookfrom(21,4,-5);
            vec3 lookat(0,0,0);
            float distToFocus=10.0;
            float aperture=0.0;
            *dCamera=new camera(lookfrom,lookat,vec3(0,1,0),30.0,
                float(imageWidth)/float(imageHeight),
                aperture,
                distToFocus);
        }
}

__global__ void createWorldRandoms(hittable** dList, float* dObjRand){
    float thread=1.25*(threadIdx.x-(blockDim.x/2.0));
    float block=1.25*(blockIdx.x-(blockDim.x/2.0));
    int loc=blockDim.x*blockIdx.x+threadIdx.x;

    float materialChoice=dObjRand[loc];
    vec3 center(block+dObjRand[(loc+1)],0.45*dObjRand[loc+6],thread+dObjRand[(loc+2)]);
    if(materialChoice<0.25){
        dList[loc]=new sphere(center,0.4*dObjRand[loc+6],new matte(vec3(dObjRand[loc]*dObjRand[(loc+1)], dObjRand[(loc+2)]*dObjRand[(loc+3)], dObjRand[(loc+4)]*dObjRand[(loc+5)])),
        dObjRand[loc],dObjRand[loc+1],dObjRand[loc+2]);
        loc+=6;
    }
    else if(materialChoice<0.5){        //smooth metal
        dList[loc]=new sphere(center,0.4*dObjRand[loc+6],new metal(vec3(0.5*(0.5+dObjRand[loc]), 0.5*(0.5+dObjRand[loc+1]), 0.5*(0.5+dObjRand[loc+2])), 0.0),
        0,0,0);
        loc+=4;
    }
    else if(materialChoice<0.75){       //fuzzy metal
        dList[loc]=new sphere(center,0.4*dObjRand[loc+6],new metal(vec3(0.5*(0.5+dObjRand[loc]), 0.5*(0.5+dObjRand[loc+1]), 0.5*(0.5+dObjRand[loc+2])), 0.5*(dObjRand[loc+3])),
        0,0,0);
        loc+=4;
    }
    else{
        dList[loc]=new sphere(center,0.4*dObjRand[loc+6],new glass(1.5),
        dObjRand[loc],dObjRand[loc+1],dObjRand[loc+2]);
    }
}

__global__ void createWorldRandomsNaive(hittable** dList, int loopBound, float* dObjRand){
    if(threadIdx.x==0 && blockIdx.x==0){
        int i=0;
        int j=0;
        for(int a=-loopBound;a<loopBound;a++){
            for(int b=-loopBound;b<loopBound;b++){
                float materialChoice=dObjRand[j];
                vec3 center((1.25*a)+dObjRand[j+1],0.45*dObjRand[j+6],(1.25*b)+dObjRand[j+2]);
                if(materialChoice<0.25){        //matte
                    dList[i++]=new sphere(center,0.4*dObjRand[j+6],new matte(vec3(dObjRand[j]*dObjRand[(j+1)], dObjRand[(j+2)]*dObjRand[(j+3)], dObjRand[(j+4)]*dObjRand[(j+5)])),
                    dObjRand[j],dObjRand[j+1],dObjRand[j+2]);
                    // j+=6;
                }
                else if(materialChoice<0.5){        //smooth metal
                    dList[i++]=new sphere(center,0.4*dObjRand[j+6],new metal(vec3(0.5*(0.5+dObjRand[j]), 0.5*(0.5+dObjRand[j+1]), 0.5*(0.5+dObjRand[j+2])), 0.0),
                    0,0,0);
                    // j+=4;
                }
                else if(materialChoice<0.75){       //fuzzy metal
                    dList[i++]=new sphere(center,0.4*dObjRand[j+6],new metal(vec3(0.5*(0.5+dObjRand[j]), 0.5*(0.5+dObjRand[j+1]), 0.5*(0.5+dObjRand[j+2])), 0.5*(dObjRand[j+3])),
                    0,0,0);
                    // j+=4;
                }
                else{       // glass
                    dList[i++]=new sphere(center,0.4*dObjRand[j+6],new glass(1.5),
                    dObjRand[j],dObjRand[j+1],dObjRand[j+2]);
                    // j++;
                }
                j++;
            }
        }
    }
}

__global__ void createWorld(hittable** dList, hittable** dWorld, camera** dCamera, int imageWidth, int imageHeight){
    // extern __shared__ hittable* dList[];
    // hittable** dList = (hittable**) dList3;

    // x[0]=blockDim.x*blockIdx.x+threadIdx.x;

    //location in threads
    int loc2=blockDim.x*blockIdx.x+threadIdx.x;
    int loc=blockDim.x*blockIdx.x+threadIdx.x;
    int thread=threadIdx.x-(blockDim.x/2.0);
    int block=blockIdx.x-(blockDim.x/2.0);
    // int bound=((loc/2)-loc)/20;
    if(loc>gridDim.x*blockDim.x-9){
        loc=(gridDim.x*blockDim.x)-9;
        // return;
    }
    // printf("%d\n",gridDim.x);
    if(loc==0){

    }
    if(loc!=0){
        float materialChoice=orc[loc];
        // printf("BlockIdx.x: %d | Block: %d | Thread: %d | threadIdx.x: %d\n",blockIdx.x,block,thread,threadIdx.x);
        vec3 center(block+orc[(loc2+1)],0.2,thread+orc[(loc2+2)]);
        if(materialChoice<0.8){
            dList[loc2]=new sphere(center,0.2,new matte(vec3(orc[loc]*orc[(loc+1)], orc[(loc+2)]*orc[(loc+3)], orc[(loc+4)]*orc[(loc+5)])),
            orc[loc],orc[loc+1],orc[loc+2]);
            loc+=6;
        }
        else if(materialChoice<0.95){
            dList[loc2]=new sphere(center,0.2,new metal(vec3(0.5*(0.5+orc[loc]), 0.5*(0.5+orc[loc+1]), 0.5*(0.5+orc[loc+2])), 0.5*(orc[loc+3])),
            orc[loc],orc[loc+1],orc[loc+2]);
            loc+=4;
        }
        else{
            dList[loc2]=new sphere(center,0.2,new glass(1.5),
            orc[loc],orc[loc+1],orc[loc+2]);
        }
        
        // printf("%d: center[%d]: %f %f %f\n",loc2,loc, center.x(), center.y(), center.z());
        loc+=3;
    // __syncthreads();
    }
    // __syncthreads();
    if(loc==0){

                //ground
        dList[0]=new sphere(vec3(0,-1000,-1),1000,
        new matte(vec3(0.05,0.05,0.05)),
        orc[loc],orc[loc+1],orc[loc+2]);


        //do things that only need to be done once.
        dList[gridDim.x*blockDim.x-3]=new sphere(vec3(0,1,0),1.0,new glass(1.5),
        orc[loc],orc[loc+1],orc[loc+2]);
        dList[gridDim.x*blockDim.x-2]=new sphere(vec3(-4,1,0),1.0,new matte(vec3(0.4,0.2,0.1)),
        orc[loc],orc[loc+1],orc[loc+2]);
        dList[gridDim.x*blockDim.x-1]=new sphere(vec3(4,1,0),1.0,new metal(vec3(0.7,0.6,0.5),0.0),
        orc[loc],orc[loc+1],orc[loc+2]);

        // printf("%d\n",gridDim.x*blockDim.x);
        *dWorld=new hittable_list(dList,gridDim.x*blockDim.x);
        vec3 lookfrom(16,3,-5);
        vec3 lookat(0,0,0);
        float distToFocus=10.0;
        float aperture=0.0;
        *dCamera=new camera(lookfrom,
                lookat,
                vec3(0,1,0),
                30.0,
                float(imageWidth)/float(imageHeight),
                aperture,
                distToFocus);
        // loc+=4;
        
    }
    // if(loc==0){

                                
    // }
    // printf("Done with createWorld\n");
}   
        //single threaded create world
__global__ void createWorldNaive(hittable **dList, hittable **dWorld, camera **dCamera, int imageWidth, int imageHeight, float* dObjRand, int loopBound) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // curandState localRandState=*rand_state;
        int i=1;
        int j=0;
        //ground sphere
        dList[0]=new sphere(vec3(0,-1000.0,-1), 1000,
            new matte(vec3(0.05, 0.05, 0.05)),
            orc[j],orc[j+1],orc[j+2]);
        dList[i++]=new sphere(vec3(0, 1, 0),  1.0, new glass(1.5),orc[j],orc[j+1],orc[j+2]);
        dList[i++]=new sphere(vec3(-4, 1, 0), 1.0, new matte(vec3(0.4, 0.2, 0.1)),orc[j],orc[j+1],orc[j+2]);
        dList[i++]=new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0),orc[j],orc[j+1],orc[j+2]);
        
        // int j=0;
        for(int a=-loopBound; a<loopBound; a+=1) {
            for(int b=-loopBound; b<loopBound; b+=1) {
                float materialChoice=orc[j];
                vec3 center(a+orc[j+1],0.2,b+orc[j+2]);
                // j+=3;
                // if(1||(center-vec3(4,0.2f,0)).length() > 0.9){
                if(materialChoice<0.8f) {
                    dList[i++]=new sphere(center, 0.2,
                    new matte(vec3(orc[j]*orc[j+1], orc[j+2]*orc[j+3], orc[j+4]*orc[j+5])),
                    orc[j],orc[j+1],orc[j+2]);
                    // j+=6;
                }
                else if(materialChoice<0.95f) {
                    dList[i++]=new sphere(center, 0.2,
                    new metal(vec3(0.5f*(0.5+orc[j]), 0.5f*(0.5+orc[j+1]), 0.5f*(0.5+orc[j+2])), 0.5f*orc[j+3]),
                    orc[j],orc[j+1],orc[j+2]);
                    // j+=4;
                }
                else {
                    dList[i++]=new sphere(center, 0.2, 
                    new glass(1.5),orc[j],orc[j+1],orc[j+2]);
                }
                j++;
                // }
            }
        }
        *dWorld=new hittable_list(dList, (loopBound*loopBound*2*2)+1+3);
        // printf("%d\n",(loopBound*loopBound*2*2)+1+3);
            //(+farther -closer, +up -down, +left -right)
                    //13,2,3
        vec3 lookfrom(16,3,-5);
        vec3 lookat(0,0,0);
        float distToFocus=10.0; //(lookfrom-lookat).length();
        float aperture=0.0;
        *dCamera=new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(imageWidth)/float(imageHeight),
                                 aperture,
                                 distToFocus);
    }
}

__global__ void free_world(hittable **dList, hittable **dWorld, camera **dCamera, int numHittables) {
    for(int i=0; i<numHittables; i++) {
        delete ((sphere *)dList[i])->mat_ptr;
        delete dList[i];
    }
    delete *dWorld;
    delete *dCamera;
}

int main() {
    
    int imageWidth=300;
    int imageHeight=200;
    int samplesPerPixel=5;
    int loopBound=11;
    int tx=8;
    int ty=8;
    std::cerr<<"Begin GPU"<<std::endl;
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

    int loopSize=loopBound*loopBound*2*2;

    int num_pixels=imageWidth*imageHeight;
    size_t fb_size=num_pixels*sizeof(vec3);

    // std::vector<float> objectRandoms;
    std::ifstream objRandFile("object_randoms.txt");

    float* objectRandoms=(float*)malloc(loopSize*(1+2+6)*sizeof(float));

    unsigned long randCount1=0;
    float f;
    while(objRandFile >> f && randCount1<loopSize*(1+2+6)){
        objectRandoms[randCount1]=f;
        randCount1++;
    }
    objRandFile.close();

    std::ifstream rtRandFile("raytrace_randoms.txt");

    float* raytraceRandoms=(float*)malloc((100+(samplesPerPixel*4))*sizeof(float));
    std::cerr<<"Reading raytrace randoms file"<<std::endl;
    
    long rtCount=0;
    long total=100+(samplesPerPixel*4);
    while(rtRandFile >> f && rtCount<total){
        raytraceRandoms[rtCount]=f;
        rtCount++;
    }
    rtRandFile.close();

    // float* rands;

    // std::copy(objectRandoms.begin(),objectRandoms.end(),rands);

    // allocate frame buffer
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    // curandState *d_rand_state;
    // checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    // curandState *d_rand_state2;
    // checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    // rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hittables & the camera
    hittable **dList;
    int numHittables=loopSize+1+3;
    checkCudaErrors(cudaMalloc((void **)&dList, numHittables*sizeof(hittable *)));
    hittable **dWorld=NULL;
    checkCudaErrors(cudaMalloc((void **)&dWorld, sizeof(hittable *)));
    camera **dCamera=NULL;
    checkCudaErrors(cudaMalloc((void**)&dCamera, sizeof(camera *)));

    float* dObjRandoms;

    checkCudaErrors(cudaMalloc(&dObjRandoms,sizeof(float)*randCount1));
    checkCudaErrors(cudaMemcpy(dObjRandoms,objectRandoms,sizeof(float)*randCount1,cudaMemcpyHostToDevice));

    cudaMemcpyToSymbol(objRandConst,objectRandoms,sizeof(float)*randCount1);

    cudaMemcpyToSymbol(dRaytraceRandomsConst,raytraceRandoms,sizeof(float)*rtCount);

    // size_t threadMemSize=loopBound*(sizeof(hittable*)+sizeof(hitRecord)+sizeof(ray)+sizeof(vec3));

    float* dRaytraceRandoms;
    checkCudaErrors(cudaMalloc(&dRaytraceRandoms,sizeof(float)*rtCount));
    checkCudaErrors(cudaMemcpy(dRaytraceRandoms,raytraceRandoms,sizeof(float)*rtCount,cudaMemcpyHostToDevice));

    cudaEvent_t startCW,stopCW;     //start create world, stop create world
    float time;
    cudaEventCreate(&startCW);
    cudaEventCreate(&stopCW);
    cudaEventRecord(startCW,0);

    // createWorld<<<loopBound*2,loopBound*2>>>(dList, dWorld, dCamera, imageWidth, imageHeight);
    createWorldRandoms<<<loopBound*2,loopBound*2>>>(dList, dObjRandoms);
    // createWorldRandomsNaive<<<1,1>>>(dList, loopBound, dObjRandoms);
    createWorldConstants<<<1,1>>>(dList, dWorld, dCamera, imageWidth, imageHeight, loopSize);
    // createWorldNaive<<<1,1>>>(dList, dWorld, dCamera, imageWidth, imageHeight, dObjRandoms, loopBound);
    
    cudaEventRecord(stopCW,0);
    cudaEventSynchronize(stopCW);
    cudaEventElapsedTime(&time,startCW,stopCW);
    checkCudaErrors(cudaDeviceSynchronize());

    std::cerr<<"Elapsed time to create world: "<<time<<" ms"<<std::endl;



    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    // Render our buffer
    dim3 blocks(imageWidth/tx+1,imageHeight/ty+1);
    dim3 threads(tx,ty);
    // render_init<<<blocks, threads>>>(imageWidth, imageHeight, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cerr << "Rendering a " << imageWidth << "x" << imageHeight << " image with " << samplesPerPixel << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    dim3 blocks1(2,2);
    dim3 threads1(32,32);
    
    start=clock();
    render<<<blocks, threads>>>(fb, imageWidth, imageHeight, samplesPerPixel, dCamera, dWorld, dRaytraceRandoms);
    // renderNaive<<<1, 1>>>(fb, imageWidth, imageHeight, samplesPerPixel, dCamera, dWorld, dRaytraceRandoms);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop=clock();
    double timer_seconds=((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::cout << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";
    for (int j=imageHeight-1; j >= 0; j--) {
        for (int i=0; i<imageWidth; i++) {
            size_t pixel_index=j*imageWidth + i;
            int ir=int(255.99*fb[pixel_index].r());
            int ig=int(255.99*fb[pixel_index].g());
            int ib=int(255.99*fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // cleanup
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(dList,dWorld,dCamera,numHittables);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(dCamera));
    checkCudaErrors(cudaFree(dWorld));
    checkCudaErrors(cudaFree(dList));
    // checkCudaErrors(cudaFree(d_rand_state));
    // checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(dRaytraceRandoms));

    cudaFree(dObjRandoms);
    free(objectRandoms);
    free(raytraceRandoms);

    cudaDeviceReset();
    std::cerr<<"Done"<<std::endl;
    return 0;
}
