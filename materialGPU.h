#ifndef MATERIALH
#define MATERIALH

struct hitRecord;

#include "rayGPU.h"
#include "hittableGPU.h"


__device__ float schlick(float cosine, float ref_idx) {
    float r0=(1.0-ref_idx) / (1.0+ref_idx);
    r0=r0*r0;
    return r0 + (1.0-r0)*pow((1.0 - cosine),5.0);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv=unitVector(v);
    float dt=dot(uv, n);
    float discriminant=1.0 - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted=ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

// #define RANDVEC3 vec3(curand_uniform(localRandState),curand_uniform(localRandState),curand_uniform(localRandState))

__device__ vec3 randomInUnitSphere(float rand1, float rand2, float rand3) {
    vec3 p;
    // do {
    //     p=2.0f*vec3(rand1,rand2,rand3) - vec3(1,1,1);
    // } while (p.squaredLength() >= 1.0f);
    return 2.0*vec3(rand1, rand2, rand3) - vec3(1,1,1);
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
     return v - 2.0*dot(v,n)*n;
}

class material  {
    public:
        __device__ virtual bool scatter(const ray& r_in, const hitRecord& rec, vec3& attenuation, ray& scattered,float rand1, float rand2, float rand3) const=0;
};

class matte : public material {
    public:
        __device__ matte(const vec3& a) : albedo(a) {}
        __device__ virtual bool scatter(const ray& r_in, const hitRecord& rec, vec3& attenuation, ray& scattered,float rand1, float rand2, float rand3) const  {
             vec3 target=rec.p + rec.normal + randomInUnitSphere(rand1,rand2,rand3);
             scattered=ray(rec.p, target-rec.p);
             attenuation=albedo;
             return true;
        }

        vec3 albedo;
};

class metal : public material {
    public:
        __device__ metal(const vec3& a, float f) : albedo(a) { if (f<1) fuzz=f; else fuzz=1; }
        __device__ virtual bool scatter(const ray& r_in, const hitRecord& rec, vec3& attenuation, ray& scattered,float rand1, float rand2, float rand3) const  {
            vec3 reflected=reflect(unitVector(r_in.direction()), rec.normal);
            scattered=ray(rec.p, reflected + fuzz*randomInUnitSphere(rand1,rand2,rand3));
            attenuation=albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }
        vec3 albedo;
        float fuzz;
};

class glass : public material {
public:
    __device__ glass(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray& r_in,
                         const hitRecord& rec,
                         vec3& attenuation,
                         ray& scattered,
                         float rand1, float rand2, float rand3) const  {
        vec3 outward_normal;
        vec3 reflected=reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation=vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0) {
            outward_normal=-rec.normal;
            ni_over_nt=ref_idx;
            cosine=dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine=sqrt(1.0 - ref_idx*ref_idx*(1-cosine*cosine));
        }
        else {
            outward_normal=rec.normal;
            ni_over_nt=1.0 / ref_idx;
            cosine=-dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob=schlick(cosine, ref_idx);
        else
            reflect_prob=1.0;
        if (rand1<reflect_prob)
            scattered=ray(rec.p, reflected);
        else
            scattered=ray(rec.p, refracted);
        return true;
    }

    float ref_idx;
};
#endif
