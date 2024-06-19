#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "rayGPU.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ vec3 randomInUnitDisk(float rand1, float rand2) {
    vec3 p;
    // do {
    //     p=2.0*vec3(rand1,rand2,0) - vec3(1,1,0);
    // } while (dot(p,p) >= 1.0);
    return 2.0*vec3(rand1,rand2,0) - vec3(1,1,0);//p;
}

class camera {
public:
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
        lensRadius=aperture / 2;
        float theta=vfov*3.14159265358979323846/180;
        float halfHeight=tan(theta/2);
        float halfWidth=aspect * halfHeight;
        origin=lookfrom;
        w=unitVector(lookfrom - lookat);
        u=unitVector(cross(vup, w));
        v=cross(w, u);
        lowerLeftCorner=origin - halfWidth*focus_dist*u -halfHeight*focus_dist*v - focus_dist*w;
        horizontal=2*halfWidth*focus_dist*u;
        vertical=2*halfHeight*focus_dist*v;
    }
    __device__ ray getRay(float s, float t, float rand1, float rand2) {
        vec3 rd=lensRadius*randomInUnitDisk(rand1, rand2);
        vec3 offset=u * rd.x() + v * rd.y();
        return ray(origin + offset, lowerLeftCorner + s*horizontal + t*vertical - origin - offset);
    }

    vec3 origin;
    vec3 lowerLeftCorner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lensRadius;
};

#endif
