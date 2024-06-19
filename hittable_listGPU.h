#ifndef hittableLISTH
#define hittableLISTH

#include "hittableGPU.h"

class hittable_list: public hittable  {
    public:
        __device__ hittable_list() {}
        __device__ hittable_list(hittable **l, int n) {list=l; listSize=n; }
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hitRecord& rec) const;
        hittable** list;
        int listSize;
};

__device__ bool hittable_list::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
        hitRecord tempRec;
        bool hitAnything=false;
        float closestSoFar=tMax;
        for (int i=0; i<listSize; i++) {
            if (list[i]->hit(r, tMin, closestSoFar, tempRec)) {
                hitAnything=true;
                closestSoFar=tempRec.t;
                rec=tempRec;
            }
        }
        return hitAnything;
}

#endif
