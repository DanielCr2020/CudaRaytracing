#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <fstream>

#include "sphere.h"
#include "hittable_list.h"
#include "float.h"
#include "camera.h"
#include "material.h"

#define FLOAT_MAX 3.402823466e+38F

vec3 color2(const ray& r, hittable *world, float rand1, float rand2, float rand3) {
    ray curRay=r;
    vec3 curAttenuation=vec3(1.0,1.0,1.0);
    for(int i=0; i<50; i++) {
        hitRecord rec;
        // printf("%d ",i);

        if (world->hit(curRay, 0.001, FLOAT_MAX, rec)) {
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
            vec3 c=(1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return curAttenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

vec3 color(const ray& r, hittable* world, int depth, float rand1, float rand2, float rand3) {
    hitRecord rec;
    if (world->hit(r, 0.001, FLOAT_MAX, rec)) { 
        ray scattered;
        vec3 attenuation;           //3 randoms (0-1)
        if (depth<50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered,rand1,rand2,rand3)) {
             return attenuation*color(scattered, world, depth+1,rand1,rand2,rand3);
        }
        else {
            return vec3(0,0,0);
        }
    }
    else {
        vec3 unit_direction=unitVector(r.direction());
        float t=0.5*(unit_direction.y()+1.0);
        return (1.0-t)*vec3(1.0, 1.0, 1.0)+t*vec3(0.5, 0.7, 1.0);
    }
}

hittable* createWorldRandoms(hittable** list, std::vector<float> objRand, int loopBound, int* count){
    // int n=1000;
    // hittable** list=new hittable*[n+1];

    int i=0;
    int j=0;

    for (int a=-loopBound; a<loopBound; a++) {
        for (int b=-loopBound; b<loopBound; b++) {
            float materialChoice=objRand[j];
            vec3 center((1.25*a)+objRand[j+1],0.45*objRand[j+6],(1.25*b)+objRand[j+2]);

            if (materialChoice<0.25) {  // diffuse
                list[i++]=new sphere(center, 0.4*objRand[j+6], 
                new matte(vec3(objRand[j]*objRand[j+1], objRand[j+2]*objRand[j+3], objRand[j+4]*objRand[j+5])),
                objRand[j],objRand[j+1],objRand[j+2]);
            }
            else if (materialChoice<0.5) { // smooth metal
                list[i++]=new sphere(center, 0.4*objRand[j+6],
                new metal(vec3(0.5*(0.5+objRand[j]), 0.5*(0.5+objRand[j+1]), 0.5*(0.5+objRand[j+2])), 0.0),
                0,0,0);
            }
            else if (materialChoice<0.75) { // fuzzy metal
                list[i++]=new sphere(center, 0.4*objRand[j+6],
                new metal(vec3(0.5*(0.5+objRand[j]), 0.5*(0.5+objRand[j+1]), 0.5*(0.5+objRand[j+2])), 0.5*objRand[j+3]),
                0,0,0);
            }
            else {  // glass
                list[i++]=new sphere(center, 0.4*objRand[j+6], 
                new glass(1.5), objRand[j],objRand[j+1],objRand[j+2]);
            }
            j++;
        }
    }
    *count=i;
    return *list;
}

hittable* createWorldConstants(hittable** list, std::vector<float> objRand, int count){
    //ground
    list[count++]=new sphere(vec3(0,-1000,-1),1000,
            new metal(vec3(0.1,0.1,0.1),0.0),1,1,1);
    
    //glass
    list[count++]=new sphere(vec3(0,1.3,0),1.3,new glass(1.5),
            objRand[count],objRand[count+1],objRand[count+2]);

    list[count++]=new sphere(vec3(10,0.75,-1),0.75,new glass(1.5),
    objRand[count],objRand[count+1],objRand[count+2]);

    //matte
    list[count++]=new sphere(vec3(-4,1,-3),1.0,new matte(vec3(1,1,1)),
    objRand[count],objRand[count+1],objRand[count+2]);

    list[count++]=new sphere(vec3(-6,1,6),1.0,new matte(vec3(0.1,0.5,0.8)),
    objRand[count],objRand[count+1],objRand[count+2]);

    //metal
    list[count++]=new sphere(vec3(4,1,0),1.0,new metal(vec3(0.7,0.6,0.5),0.0),
    objRand[count],objRand[count+1],objRand[count+2]);

    list[count++]=new sphere(vec3(-7,1.5,-7),1.5,new metal(vec3(0.7,0.85,1),0.0),
    objRand[count],objRand[count+1],objRand[count+2]);

    list[count++]=new sphere(vec3(-3,2.5,7),2.5,new metal(vec3(0.8,0.85,1),0.0),
    objRand[count],objRand[count+1],objRand[count+2]);

    return new hittable_list(list,count);
}

hittable* randomScene(std::vector<float> objRand, int loopBound) {
    int n=1000;
    hittable **list=new hittable*[n+1];

    int i=0;
    int j=0;

    for (int a=-loopBound; a<loopBound; a++) {
        for (int b=-loopBound; b<loopBound; b++) {
            float materialChoice=objRand[j];
            vec3 center((1.25*a)+objRand[j+1],0.45*objRand[j+6],(1.25*b)+objRand[j+2]);

            if (materialChoice<0.25) {  // diffuse
                list[i++]=new sphere(center, 0.4*objRand[j+6], 
                new matte(vec3(objRand[j]*objRand[j+1], objRand[j+2]*objRand[j+3], objRand[j+4]*objRand[j+5])),
                objRand[j],objRand[j+1],objRand[j+2]);
            }
            else if (materialChoice<0.5) { // smooth metal
                list[i++]=new sphere(center, 0.4*objRand[j+6],
                new metal(vec3(0.5*(0.5+objRand[j]), 0.5*(0.5+objRand[j+1]), 0.5*(0.5+objRand[j+2])), 0.0),
                0,0,0);
            }
            else if (materialChoice<0.75) { // fuzzy metal
                list[i++]=new sphere(center, 0.4*objRand[j+6],
                new metal(vec3(0.5*(0.5+objRand[j]), 0.5*(0.5+objRand[j+1]), 0.5*(0.5+objRand[j+2])), 0.5*objRand[j+3]),
                0,0,0);
            }
            else {  // glass
                list[i++]=new sphere(center, 0.4*objRand[j+6], 
                new glass(1.5), objRand[j],objRand[j+1],objRand[j+2]);
            }
            j++;
        }
    }

        //ground
    list[i++]=new sphere(vec3(0,-1000,-1), 1000, new matte(vec3(0.03, 0.05, 0.03)),1.0,1.0,1.0);
    
    list[i++]=new sphere(vec3(0, 1, 0), 1.0, new glass(1.5),objRand[j],objRand[j+1],objRand[j+2]);
    list[i++]=new sphere(vec3(-4, 1, 0), 1.0, new matte(vec3(0.4, 0.2, 0.1)),objRand[j],objRand[j+1],objRand[j+2]);
    list[i++]=new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0),objRand[j],objRand[j+1],objRand[j+2]);


    return new hittable_list(list,i);
}

int main() {
    int imageWidth=600;
    int imageHeight=400;
    int samplesPerPixel=10;
    int loopBound=11;

    std::cerr<<"Begin"<<std::endl;

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

    // std::cout << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";


    std::vector<float> objectRandoms;
    std::ifstream objRandFile("object_randoms.txt");
    std::cerr<<"Reading object randoms file"<<std::endl;
    float f;
    while(objRandFile >> f){
        objectRandoms.push_back(f);
    }
    objRandFile.close();
    std::cerr<<"Done reading object randoms file"<<std::endl;

    // hittable *list[5];
    // list[0]=new sphere(vec3(0,0,-1), 0.5, new matte(vec3(0.1, 0.2, 0.5)),objectRandoms[0],objectRandoms[1],objectRandoms[2]);
    // list[1]=new sphere(vec3(0,-100.5,-1), 100, new matte(vec3(0.8, 0.8, 0.0)),objectRandoms[0],objectRandoms[1],objectRandoms[2]);
    // list[2]=new sphere(vec3(1,0,-1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 0.0),objectRandoms[0],objectRandoms[1],objectRandoms[2]);
    // list[3]=new sphere(vec3(-1,0,-1), 0.5, new glass(1.5),objectRandoms[0],objectRandoms[1],objectRandoms[2]);

    // hittable *world=new hittable_list(list,5);
    hittable** list=new hittable*[(loopBound*loopBound*2*2)+8];
    int objCount=0;
    createWorldRandoms(list,objectRandoms,loopBound,&objCount);
    hittable* world=createWorldConstants(list,objectRandoms,objCount);
    // hittable *world=new hittable_list(list,5);


    auto cw_start=std::chrono::high_resolution_clock::now();

    // world=randomScene(objectRandoms,loopBound);

    auto cw_end=std::chrono::high_resolution_clock::now();
    auto elapsed=std::chrono::duration<double, std::milli>(cw_end-cw_start);
    std::clog<<"Elapsed time to create world (CPU): "<< std::defaultfloat << elapsed.count()/1000.0<<" seconds"<<std::endl;
                //13,2,3
    vec3 lookfrom(21,4,-5);
    vec3 lookat(0,0,0);
    float distToFocus=10.0;
    float aperture=0.0;       //depth of field effect

    std::vector<std::string> membuf;
    std::string magic_bytes="P3\n"+std::to_string(imageWidth)+" "+std::to_string(imageHeight)+"\n255\n"; 
    membuf.push_back(magic_bytes);

    camera cam(lookfrom, lookat, vec3(0,1,0), 30.0, float(imageWidth)/float(imageHeight), aperture, distToFocus);

    std::vector<float> rtRandoms;
    std::ifstream rtRandFile("raytrace_randoms.txt");
    //std::cerr<<"Reading raytrace randoms file"<<std::endl;
    f=0;
    long count=0;
    int total=samplesPerPixel*4;
    while(rtRandFile >> f && count<total){
        rtRandoms.push_back(f);
        count++;
    }
    std::cerr<<"Done reading raytrace randoms file."<<std::endl;
    rtRandFile.close();
    

    auto t_start=std::chrono::high_resolution_clock::now();

    //int k=0;        //rt random vector index;

    //render loop
    for (int j=imageHeight-1; j>=0; j--) {
        for (int i=0; i<imageWidth; i++) {
            vec3 col(0, 0, 0);
            for (int s=0; s<samplesPerPixel; s++) {
                float u=float(i+rtRandoms[s]) / float(imageWidth);
                float v=float(j+rtRandoms[s+1]) / float(imageHeight);
                            //2 randoms
                ray r=cam.getRay(u, v, rtRandoms[s+2],rtRandoms[s+3]);
                // k+=1;
                vec3 p=r.pointAtParemeter(2.0);
                col += color2(r, world,rtRandoms[s+4],rtRandoms[s+5],rtRandoms[s+6]);

                // col += color(r, world, 0,rtRandoms[s+4],rtRandoms[s+5],rtRandoms[s+6]);
            }
            col /= float(samplesPerPixel);
            col=vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );
            int ir=int(255.99*col[0]); 
            int ig=int(255.99*col[1]); 
            int ib=int(255.99*col[2]); 
            membuf.push_back(std::to_string(ir)+" "+std::to_string(ig)+" "+std::to_string(ib)+"\n");
        }
        std::clog << "\rLines remaining: " << (j) << "/" << imageHeight << ' ' << std::flush;
    }

    auto t_end=std::chrono::high_resolution_clock::now();
    elapsed=std::chrono::duration<double, std::milli>(t_end-t_start);

    for(auto i: membuf){
        std::cout<<i;
    }

    std::clog << "\rDone.                              \n";
    std::clog<<"Elapsed time: "<<elapsed.count()/1000.0<<" seconds"<<std::endl;
    return 0;
}



