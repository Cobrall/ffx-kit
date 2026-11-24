#define A_GPU 1
#define A_HLSL 1
#include "ffx_a.h"

[[vk::binding(0, 0)]]
StructuredBuffer<float4>   InBuf  : register(t0);
[[vk::binding(1, 0)]]
RWStructuredBuffer<float4> OutBuf : register(u0);
[[vk::binding(2, 0)]]
StructuredBuffer<float4>   Params : register(t1); // (W, H, sharpness, 0)

AF3 CasLoad(ASU2 p) {
    int2 q = clamp(p, int2(0,0), int2(Params[0].xy) - int2(1,1));
    return InBuf[q.y * int(Params[0].x) + q.x].rgb;
}
void CasInput(inout AF1 r, inout AF1 g, inout AF1 b) {}
#include "ffx_cas.h"

[numthreads(8,8,1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint W = uint(Params[0].x), H = uint(Params[0].y);
    if (dtid.x >= W || dtid.y >= H) return;
    AU2 ip = AU2(dtid.xy);
    AF1 sharp = -1.0 / lerp(8.0, 5.0, saturate(Params[0].z));
    AU4 const0 = AU4(0,0,0,0);
    AU4 const1 = AU4(asuint(sharp), 0, 0, 0);
    AF1 r, g, b;
    CasFilter(r, g, b, ip, const0, const1, true);
    OutBuf[dtid.y * W + dtid.x] = float4(r, g, b, 1.0);
}
