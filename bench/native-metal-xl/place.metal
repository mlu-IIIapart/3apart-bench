#include <metal_stdlib>
using namespace metal;
// Same operation sequence as stock place(): acos then sin/cos, no sqrt substitution.
kernel void place_markers(device const int* layer [[buffer(0)]],
                          device const float* t [[buffer(1)]],
                          device const float* value [[buffer(2)]],
                          device float* out [[buffer(3)]],
                          constant float* edges [[buffer(4)]],
                          constant uint& n [[buffer(5)]], uint i [[thread_position_in_grid]]) {
  if(i>=n)return;
  const float pi=3.14159265358979323846f;
  float phi=pi*(2.0f*t[i]-1.0f);int l=layer[i];float zt=edges[l],zb=edges[l+1];
  float z=zb+(zt-zb)*(0.12f+(1.0f-2.0f*0.12f)*value[i]);
  float th=acos(clamp(z,-1.0f,1.0f)),st=sin(th);
  out[3*i]=st*sin(phi);out[3*i+1]=-st*cos(phi);out[3*i+2]=cos(th);
}
