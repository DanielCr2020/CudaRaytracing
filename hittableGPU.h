#ifndef hittableH
#define hittableH

#include "rayGPU.h"

class material;

struct hitRecord
{
    float t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
};

class hittable  {
    public:
        __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const=0;
};

#endif
