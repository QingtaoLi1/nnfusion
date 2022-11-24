# Multiple Outputs
BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N] = input0[N] + input1[N]; output1[N] = input0[N] * 2; output2[N] = input1[N] + output1[N];", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}, "input1": {"dtype": "float32", "shape": [1024 * 512]}}, extra_outputs=["output0", "output1", "output2"])' antares

# Softmax
COMPUTE_V1='- einstein_v2("temp0[N] >=! input0[N, C]; temp1[N] +=! (input0[N, C] - temp0[N]).call(`exp`); output0[N, C] = (input0[N, C] - temp0[N]).call(`exp`) / temp1[N]", { "input0": {"dtype": "float32", "shape": [32, 1024]} })' antares

# ConvBiasRelu Tail Fusion
# support in other format
COMPUTE_V1='- einstein_v2("conv_out[N, F, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[KH, KW, C, F] where HO in 256, WO in 256; conv_bias[N, F, HO, WO] = conv_out[N, F, HO, WO] + input2[0, 0, 0, F]; output0[N, F, HO, WO] = conv_bias[N, F, HO, WO].when(conv_bias[N, F, HO, WO] > 0.0, 0.0)", input_dict={"input0": {"dtype": "float32", "shape": [1, 16, 256, 256]}, "input1": {"dtype": "float32", "shape": [1, 1, 16, 16]}, "input2": {"dtype": "float32", "shape": [1, 1, 1, 16]}})' antares

# ConvolutionWithPad (Fused reduce axis)
# support in other format
COMPUTE_V1='- _N, _CI, _H, _W, _CO, _KH, _KW, _SH, _SW, _PH, _PW = 16, 64, 32, 32, 256, 3, 3, 1, 1, 0, 0; \
    _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
    einstein_v2(f" \
      output0[N, F, HO, WO] +=! input0[N, CKHKW // {_KW * _KH}, HO * {_SH} + ((CKHKW % {_KW * _KH}) // {_KW}) - {_PH}, WO * {_SW} + ((CKHKW % {_KW * _KH}) % {_KW}) - {_PW}].when([HO * {_SH} + ((CKHKW % {_KW * _KH}) // {_KW}) - {_PH} >= 0, HO * {_SH} + ((CKHKW % {_KW * _KH}) // {_KW}) - {_PH} < {_H}, WO * {_SW} + ((CKHKW % {_KW * _KH}) % {_KW}) - {_PW} >= 0, WO * {_SW} + ((CKHKW % {_KW * _KH}) % {_KW}) - {_PW} < {_W}], 0.0) * input1[F, CKHKW] where HO in {_HO}, WO in {_WO} \
    ", { "input0": {"dtype": "float32", "shape": [_N, _CI, _H, _W]}, "input1": {"dtype": "float32", "shape": [_CO, _CI * _KH * _KW]}})' antares

# ConvWinograd_3x3
COMPUTE_V1='- _N, _CI, _H, _W, _CO = 16, 64, 32, 32, 256; _HO, _WO = _H - 2, _W - 2; _nH, _nW = (_HO + 1) // 2, (_WO + 1) // 2; _P = _N * _nH * _nW; einstein_v2(f"helper4x3[N, M] = const(1.0).when([N * 3 + M == 0, N * 3 + M == 11], const(0.0).when([N * 3 + M == 1, N * 3 + M == 2, N * 3 + M == 9, N * 3 + M == 10], const(-0.5).when([N * 3 + M == 4], 0.5, merge_op=`any`), merge_op=`any`), merge_op=`any`) where N in 4, M in 3; transform_filter[EPS, NU, CI, CO] +=! ((input1[CO, CI, Rkh, Rkw] * helper4x3[EPS, Rkh] * helper4x3[NU, Rkw])); input_tile[C, B, EPS, NU] = input0[B // ({_nH} * {_nW}), C, B // {_nW} % {_nH} * 2 + EPS, B % {_nW} * 2 + NU] where C in {_CI}, B in {_P}, EPS in 4, NU in 4; helper4x4[N, M] = const(1.0).when([N * 4 + M == 0, N * 4 + M == 6, N * 4 + M == 9, N * 4 + M == 10, N * 4 + M == 15], const(-1.0).when([N * 4 + M == 5, N * 4 + M == 7, N * 4 + M == 8], 0.0, merge_op=`any`), merge_op=`any`) where N in 4, M in 4; transform_input[EPS, NU, C, B] +=! input_tile[C, B, K1, K2] * helper4x4[K1, EPS] * helper4x4[K2, NU] where EPS in 4, NU in 4, C in {_CI}, B in {_P}; batch_gemm[EPS, NU, K, B] +=! transform_filter[EPS, NU, CI, K] * transform_input[EPS, NU, CI, B] where EPS in 4, NU in 4, K in {_CO}, B in {_P}; helper4x2[N, M] = const(0.0) .when([N * 2 + M == 1, N * 2 + M == 6], const(-1.0).when([N * 2 + M == 3], 1.0, merge_op=`any`), merge_op=`any`) where N in 4, M in 2; inverse[K, B, VH, VW] +=! batch_gemm[K1, K2, K, B] * helper4x2[K1, VH] * helper4x2[K2, VW] where K in {_CO}, B in {_P}, VH in 2, VW in 2; output0[N, K, H, W] = inverse[K, N * {_nH} * {_nW} + H // 2 * {_nW} + W // 2, H % 2, W % 2] where N in {_N}, K in {_CO}, H in {_HO}, W in {_WO}", {"input0": {"dtype": "float32", "shape": [_N, _CI, _H, _W]}, "input1": {"dtype": "float32", "shape": [_CO, _CI, 3, 3]}})' antares
