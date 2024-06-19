#ifndef hittableLISTH
#define hittableLISTH

#include "hittable.h"

class hittable_list: public hittable  {
    public:
        hittable_list() {}
        hittable_list(hittable **l, int n) {list=l; listSize=n; }
        virtual bool hit(const ray& r, float tmin, float tmax, hitRecord& rec) const;
        hittable** list;
        int listSize;
};

bool hittable_list::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
        hitRecord tempRec;
        bool hitAnything=false;
        double closestSoFar=tMax;
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

