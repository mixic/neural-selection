#pragma once

#include <cmath>

#ifdef __CUDACC__
#define LP_HD __host__ __device__
#else
#define LP_HD
#endif

namespace sim {

// Nowotny, Levi & Selverston (2008), fitted Panulirus LP-cell parameters.
// Units follow the published model: mV, ms, microS, microF, and nA.
// The somatic and axonal compartments carry different channel sets.
struct LPCell {
    float x[16];
};
struct LPConductance {
    float calcium, calcium_k, h, leak;
};

LP_HD inline LPCell initial_lp_cell() {
    return {{-56, 0, .2f, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0,
             .04f, .01f, -56}};
}

LP_HD inline float lp_sigmoid(float v, float midpoint, float slope) {
    return 1.0f / (1.0f + expf((v - midpoint) / slope));
}

LP_HD inline float lp_rate_ratio(float numerator, float denominator,
                                  float singular_limit) {
    return fabsf(denominator) < 1e-5f ? singular_limit
                                      : numerator / denominator;
}

// Direct implementation of the paper's two-compartment current and gating
// equations. The input is the controlled DC injection in nA; the paper's
// fitted 2 nA offset is compensated here, as in the authors' public driver.
LP_HD inline void lp_derivative(const float *x, float current_na,
                                LPConductance scale, float *d) {
    const float va = x[0], vs = x[15];
    const float sodium = x[1]*x[1]*x[1]*x[2]*7.15f*(va-50.0f);
    const float delayed_k = x[8]*x[8]*x[8]*x[8]*1.43f*(va+72.0f);
    const float m_k = .2613041227f*x[14]*(va+80.0f);
    const float axon_leak = scale.leak*.001004288227f*(va+50.0f);
    const float coupling = .004017089073f*(vs-va);
    d[0] = (-sodium-delayed_k-m_k-axon_leak+coupling)/.1148312178f;

    const float ev = expf(vs/11.4866f);
    const float ghk = fabsf(vs) < .001f
        ? 11.4866f*(x[13]-15123.5f)
        : vs*(x[13]*ev-15123.5f)/(ev-1.0f);
    const float calcium = scale.calcium*(x[3]*x[4]*.1247796343f+x[5]*.07142141077f)
        * .00155982f * ghk * .9f;
    const float ca_k = scale.calcium_k*x[6]*x[7]*1.663612026f*(vs+72.0f)*.9f;
    const float a_mix = lp_sigmoid(vs,55.0f,-15.0f);
    const float a_k = x[9]*x[9]*x[9]*.8008791415f*(vs+72.0f)
        *(a_mix*x[10]+(1.0f-a_mix)*x[11])*.9f;
    const float h_current = scale.h*x[12]*.0126874385f*(vs+20.0f)*.9f;
    const float soma_leak = scale.leak*.0006901584324f*(vs+50.0f);
    d[15] = (-calcium-ca_k-a_k-h_current-soma_leak-coupling
             +(current_na-2.0f)*.01222505828f)/.0543909161f;

    float a = lp_rate_ratio(-16.64f-.32f*va,
                            expf(-13.0f-va/4.0f)-1.0f, 1.28f);
    float b = lp_rate_ratio(.28f*va+7.0f,
                            expf(va/5.0f+5.0f)-1.0f, 1.4f);
    d[1] = a*(1.0f-x[1])-b*x[1];
    a = .128f*expf(-2.66666666667f-va/18.0f);
    b = 4.0f/(expf(-5.0f-va/5.0f)+1.0f);
    d[2] = a*(1.0f-x[2])-b*x[2];
    a = lp_rate_ratio(-1.6f-.032f*va,
                      expf(-10.0f-va/5.0f)-1.0f, .16f);
    b = .5f*expf(-1.375f-va/40.0f);
    d[8] = a*(1.0f-x[8])-b*x[8];
    d[3] = (lp_sigmoid(vs,15.0f,-9.8f)-x[3])*.045f;
    d[4] = (lp_sigmoid(vs,-40.0f,3.2f)-x[4])
        *.02f*lp_sigmoid(vs,-15.0f,-10.0f);
    d[5] = (lp_sigmoid(vs,29.1336f,-4.43083f)-x[5])*.0608644f;
    const float ca = x[13];
    const float ca_k_target = lp_sigmoid(vs,-.6f*ca,-23.0f)
        *lp_sigmoid(vs,-16.0f-.6f*ca,-5.0f)*ca/(2.5f+ca);
    d[6] = (ca_k_target-x[6])*.6f;
    d[7] = (.7f/(.6f+ca)-x[7])*.035f;
    d[9] = (lp_sigmoid(vs,-12.0f,-26.0f)-x[9])*.14f;
    const float a_inact = lp_sigmoid(vs,-62.0f,6.0f);
    d[10] = (a_inact-x[10])*.05f;
    d[11] = (a_inact-x[11])*.0036f*lp_sigmoid(vs,-40.0f,-12.0f);
    d[12] = (lp_sigmoid(vs,-70.0f,8.0f)-x[12])
        *.00033f/lp_sigmoid(vs,-110.0f,-21.6f);
    d[13] = -.05049639866f*calcium-.01729930764f*(ca-.02f);
    d[14] = (lp_sigmoid(va,-26.9916223f,-5.957441281f)-x[14])
        *.0001386796618f*lp_sigmoid(va,-60.57875745f,-13.25674673f);
}

LP_HD inline void lp_advance(LPCell &cell, float current_na,
                             LPConductance scale = {1, 1, 1, 1}) {
    constexpr float dt = .05f;
    for (int step=0; step<20; ++step) {
        float k1[16], k2[16], k3[16], k4[16], tmp[16];
        lp_derivative(cell.x, current_na, scale, k1);
        for (int j=0; j<16; ++j) tmp[j]=cell.x[j]+.5f*dt*k1[j];
        lp_derivative(tmp, current_na, scale, k2);
        for (int j=0; j<16; ++j) tmp[j]=cell.x[j]+.5f*dt*k2[j];
        lp_derivative(tmp, current_na, scale, k3);
        for (int j=0; j<16; ++j) tmp[j]=cell.x[j]+dt*k3[j];
        lp_derivative(tmp, current_na, scale, k4);
        for (int j=0; j<16; ++j)
            cell.x[j]+=dt*(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6.0f;
    }
}

} // namespace sim
