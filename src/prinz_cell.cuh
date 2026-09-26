#pragma once

#include <cmath>

#ifdef __CUDACC__
#define PRINZ_HD __host__ __device__
#else
#define PRINZ_HD
#endif

namespace sim {

// Equations and PM_0 conductances adapted from mackelab/pyloric
// (MIT license: LICENSES/pyloric-MIT.txt), based on Prinz et al. 2004.
// Published pyloric currents: Na, CaT, CaS, A, KCa,
// Kd, H and leak conductances. Units: nS, pF, pA, mV, ms, micromolar Ca.
struct PrinzConductance { float g[8]; };
struct PrinzCell { float v, ca, m[7], h[4]; };

PRINZ_HD inline PrinzConductance prinz_pacemaker_conductance() {
    // Published PM_0 profile, converted from mS/cm2 using 0.000628 cm2.
    return {{400*628.0f, 2.5f*628, 6*628.0f, 50*628.0f,
             10*628.0f, 100*628.0f, .01f*628, 0}};
}
PRINZ_HD inline PrinzConductance prinz_follower_conductance() {
    // Published LP_2 profile. The circuit assay uses the separate LP model.
    return {{100*628.0f, 0, 10*628.0f, 50*628.0f,
             5*628.0f, 100*628.0f, 0, .03f*628}};
}
PRINZ_HD inline PrinzCell initial_prinz_cell() {
    PrinzCell s{};
    s.v = -50.0f;
    s.ca = .05f;
    for (int i = 0; i < 4; ++i) s.h[i] = 1.0f;
    return s;
}
PRINZ_HD inline float prinz_gate(float v, float a, float b) {
    return 1.0f / (1.0f + expf((v+a)/b));
}
PRINZ_HD inline float prinz_gate2(float v, float a, float b, float c, float d) {
    return 1.0f / (expf((v+a)/b) + expf((v+c)/d));
}

PRINZ_HD inline void prinz_substep(PrinzCell &s, PrinzConductance gene,
                                  float external_pa, float clamp_ns,
                                  float clamp_reversal_mv) {
    constexpr float dt = .025f;
    const float v = s.v, ca = fmaxf(s.ca, 0.000001f);
    const float eca = (8.31451e3f*283.0f/(2.0f*96485.3415f)) *
                      logf(3000.0f/ca);
    const float c[8] = {
        gene.g[0]*s.m[0]*s.m[0]*s.m[0]*s.h[0],
        gene.g[1]*s.m[1]*s.m[1]*s.m[1]*s.h[1],
        gene.g[2]*s.m[2]*s.m[2]*s.m[2]*s.h[2],
        gene.g[3]*s.m[3]*s.m[3]*s.m[3]*s.h[3],
        gene.g[4]*s.m[4]*s.m[4]*s.m[4]*s.m[4],
        gene.g[5]*s.m[5]*s.m[5]*s.m[5]*s.m[5],
        gene.g[6]*s.m[6], gene.g[7]
    };
    const float reversal[8] = {50, eca, eca, -80, -80, -80, -20, -50};
    float total_g = clamp_ns;
    float weighted = external_pa + clamp_ns*clamp_reversal_mv;
    for (int i = 0; i < 8; ++i) {
        total_g += c[i];
        weighted += c[i]*reversal[i];
    }
    if (fabsf(total_g) < .000001f)
        s.v = v + dt*weighted/628.3f;
    else {
        const float target = weighted/total_g;
        s.v = target + (v-target)*expf(-dt*total_g/628.3f);
    }

    const float ca_current_ua = (c[1]+c[2])*(v-eca)/1000000.0f;
    const float ca_target = .05f-14961.0f*ca_current_ua;
    s.ca = fmaxf(.000001f, ca_target+(ca-ca_target)*expf(-dt/200.0f));

    const float mi[7] = {
        prinz_gate(v,25.5f,-5.29f), prinz_gate(v,27.1f,-7.2f),
        prinz_gate(v,33,-8.1f), prinz_gate(v,27.2f,-8.7f),
        (ca/(ca+3))*prinz_gate(v,28.3f,-12.6f),
        prinz_gate(v,12.3f,-11.8f), prinz_gate(v,75,5.5f)
    };
    const float mt[7] = {
        2.64f-2.52f*prinz_gate(v,120,-25),
        43.4f-42.6f*prinz_gate(v,68.1f,-20.5f),
        2.8f+14*prinz_gate2(v,27,10,70,-13),
        23.2f-20.8f*prinz_gate(v,32.9f,-15.2f),
        180.6f-150.2f*prinz_gate(v,46,-22.7f),
        14.4f-12.8f*prinz_gate(v,28.3f,-19.2f),
        2.0f/(expf(-14.59f-.086f*v)+expf(-1.87f+.0701f*v))
    };
    for (int i = 0; i < 7; ++i)
        s.m[i] = mi[i]+(s.m[i]-mi[i])*expf(-dt/mt[i]);

    const float hi[4] = {prinz_gate(v,48.9f,5.18f),
        prinz_gate(v,32.1f,5.5f), prinz_gate(v,60,6.2f),
        prinz_gate(v,56.9f,4.9f)};
    const float ht[4] = {
        (1.34f/(1+expf((v+62.9f)/-10))) *
          (1.5f+1/(1+expf((v+34.9f)/3.6f))),
        210-179.6f*prinz_gate(v,55,-16.9f),
        120+300*prinz_gate2(v,55,9,65,-16),
        77.2f-58.4f*prinz_gate(v,38.9f,-26.5f)
    };
    for (int i = 0; i < 4; ++i)
        s.h[i] = hi[i]+(s.h[i]-hi[i])*expf(-dt/ht[i]);
}

PRINZ_HD inline bool prinz_advance(PrinzCell &s, PrinzConductance gene,
                                   float external_pa = 0,
                                   float clamp_ns = 0,
                                   float clamp_reversal_mv = -68) {
    bool spiked = false;
    for (int i = 0; i < 40; ++i) {
        const float previous = s.v;
        prinz_substep(s, gene, external_pa, clamp_ns, clamp_reversal_mv);
        spiked |= previous < 0 && s.v >= 0;
    }
    return spiked;
}

} // namespace sim
