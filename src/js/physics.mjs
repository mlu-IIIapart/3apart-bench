// Physics baseline: d3-force-3d, tuned for band membership (the comparison is against physics at its best
// for this task). Same forces as the benchmark page and the server runs. See PROTOCOL.md.
import * as d3f from 'd3-force-3d';
import { EDGES, NL, rng, timeOrderIdx } from './engine.mjs';

export const TICKS = 300;              // d3's standard schedule: alpha 1 -> 0.001 over 300 ticks
export const REHEAT_ALPHA = 0.3;       // d3's standard restart for one new marker

export function buildPhys(d, seed) {
  const n = d.n, R = 30 * Math.sqrt(n / (4 * Math.PI)), r = rng(seed), nodes = new Array(n);
  for (let i = 0; i < n; i++) {
    const u = 2 * r() - 1, a = 2 * Math.PI * r(), s = Math.sqrt(1 - u * u);
    nodes[i] = { x: R * s * Math.cos(a), y: R * s * Math.sin(a), z: R * u, layer: d.layer[i] };
  }
  const ord = timeOrderIdx(d), links = [];
  for (let k = 1; k < n; k++) if (d.layer[ord[k]] === d.layer[ord[k - 1]]) links.push({ source: ord[k - 1], target: ord[k] });
  const zc = l => R * (EDGES[l] + EDGES[l + 1]) / 2;
  const sim = d3f.forceSimulation(nodes, 3).stop()
    .force('charge', d3f.forceManyBody().strength(-10))
    .force('link', d3f.forceLink(links).distance(30))
    .force('radial', d3f.forceRadial(R).strength(1))
    .force('band', d3f.forceZ(nd => zc(nd.layer)).strength(1));
  return { sim, nodes, links, R, ord };
}

// Add one marker to a settled layout and reheat; returns the ticks it took to settle again.
export function addMarkerAndResettle(b, n, seed) {
  const L = Math.floor(rng(seed)() * NL);
  b.nodes.push({ x: b.R, y: 0, z: 0, layer: L });
  let prev = -1; for (let i = 0; i < n; i++) if (b.nodes[i].layer === L) prev = i;
  if (prev >= 0) b.links.push({ source: prev, target: n });
  b.sim.nodes(b.nodes); b.sim.force('link').links(b.links); b.sim.alpha(REHEAT_ALPHA);
  let ticks = 0;
  while (b.sim.alpha() >= b.sim.alphaMin()) { b.sim.tick(); ticks++; }
  return ticks;
}
