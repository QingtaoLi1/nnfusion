COMPUTE_V1='- einstein_v2(\
"\
m0[B, S, H] = input0[B, S, H].cast(`float32`);\
m1[B, S, H] = (m0[B, S, H]).call(`pow`, [const(2.0)]);\
m2[B, S] +=! m1[B, S, H];\
m3[B, S] = m2[B, S] / const(4096);\
m4[B, S] = const(1.0) / (m3[B, S] + const(1e-6).call(`sqrt`));\
m5[B, S, H] = m1[B, S, H] * m4[B, S];\
output0[B, S, H] = m5[B, S, H].cast(`float16`) * input1[H];\
", {\
"input0": {"dtype":"float16","shape":[16, 2048, 4096]},\
"input1": {"dtype":"float16","shape":[4096]}\
}\
)' antares