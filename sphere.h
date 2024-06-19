// #ifndef SPHEREH
// #define SPHEREH

#include "hittable.h"

class sphere: public hittable  {
    public:
        sphere() {}
        // sphere(vec3 cen, float r, material *m) {};
        sphere(vec3 cen, float r, material *m, float rand1, float rand2, float rand3) : center(cen), radius(r), mat_ptr(m)  {};
        virtual bool hit(const ray& r, float tmin, float tmax, hitRecord& rec) const;
        vec3 center;
        float radius;
        material *mat_ptr;
        float rand1;
        float rand2;
        float rand3;
};

bool sphere::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    vec3 oc=r.origin() - center;
    float a=dot(r.direction(), r.direction());
    float b=dot(oc, r.direction());
    float c=dot(oc, oc) - radius*radius;
    float discriminant=b*b - a*c;
    if (discriminant > 0) {
        float temp=(-b - sqrt(discriminant))/a;
        if (temp<tMax && temp > tMin) {
            rec.t=temp;
            rec.p=r.pointAtParemeter(rec.t);
            rec.normal=(rec.p - center) / radius;
            rec.mat_ptr=mat_ptr;
            return true;
        }
        temp=(-b + sqrt(discriminant)) / a;
        if (temp<tMax && temp > tMin) {
            rec.t=temp;
            rec.p=r.pointAtParemeter(rec.t);
            rec.normal=(rec.p - center) / radius;
            rec.mat_ptr=mat_ptr;
            return true;
        }
    }
    return false;
}


// #endif



