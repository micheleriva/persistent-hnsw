/**
 * Runs bench.mjs on Node, Bun, and Deno, then writes bench/RUNTIME_RESULTS.md.
 *
 *   deno run -A scripts/bench_runtimes.ts
 *
 * Requires: deno task build (for .npm/), node, bun on PATH.
 */

interface BenchResult {
  runtime: string
  insertion: Record<string, { opsPerSec: number; timeMs: number }>
  search: Record<string, { usPerQuery: number; qps: number }>
  distance: Record<string, { opsPerSec: number }>
}

const scriptDir = new URL('.', import.meta.url).pathname
const projectRoot = scriptDir.replace(/scripts\/$/, '')
const benchFile = projectRoot + 'tests/node/bench.mjs'

async function run(cmd: string[]): Promise<BenchResult | null> {
  const label = cmd[0]
  try {
    const proc = Deno.run({ cmd, stdout: 'piped', stderr: 'piped', cwd: projectRoot })
    const [status, stdout, stderr] = await Promise.all([
      proc.status(),
      proc.output(),
      proc.stderrOutput(),
    ])
    proc.close()

    if (!status.success) {
      const err = new TextDecoder().decode(stderr)
      console.error(`  ${label}: failed (${err.slice(0, 200)})`)
      return null
    }

    const out = new TextDecoder().decode(stdout).trim()
    return JSON.parse(out)
  } catch (e) {
    console.error(`  ${label}: not available (${(e as Error).message})`)
    return null
  }
}

function fmt(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return String(n)
}

console.log('Running benchmarks on each runtime...\n')

const all: BenchResult[] = []

for (
  const cmd of [
    ['node', benchFile],
    ['bun', benchFile],
    ['deno', 'run', '--allow-read', benchFile],
  ]
) {
  console.log(`  ${cmd[0]}...`)
  const r = await run(cmd)
  if (r) {
    all.push(r)
    console.log(`    ${r.runtime}`)
  }
}

if (all.length === 0) {
  console.error('No runtimes succeeded.')
  Deno.exit(1)
}

// ── Generate markdown ──────────────────────────────────────

const lines: string[] = []
const date = new Date().toISOString().split('T')[0]
const runtimeNames = all.map((r) => r.runtime)

lines.push('# Runtime Comparison\n')
lines.push(`> Benchmarked on ${date} — ${Deno.build.arch}-${Deno.build.os}\n`)
lines.push(`Runtimes tested: ${runtimeNames.join(', ')}\n`)

// Insertion
lines.push('## Insertion (128d, M=16, efConstruction=200)\n')
{
  const sizes = Object.keys(all[0].insertion)
  const headers = ['Vectors', ...runtimeNames]
  const rows = sizes.map((n) => {
    return [fmt(Number(n)), ...all.map((r) => `${fmt(r.insertion[n].opsPerSec)} ops/s`)]
  })
  lines.push(mdTable(headers, rows))
}

// Search
lines.push('\n## Search (10K index, 128d, M=16, k=10)\n')
{
  const efs = Object.keys(all[0].search)
  const headers = ['efSearch', ...runtimeNames.map((r) => `${r} (µs/q)`), ...runtimeNames.map((r) => `${r} (qps)`)]
  const rows = efs.map((ef) => {
    return [
      `ef=${ef}`,
      ...all.map((r) => `${r.search[ef].usPerQuery}`),
      ...all.map((r) => fmt(r.search[ef].qps)),
    ]
  })
  lines.push(mdTable(headers, rows))
}

// Bar chart comparing search qps at ef=100
lines.push('\n### Search Throughput at ef=100\n')
lines.push('```')
const maxQps = Math.max(...all.map((r) => r.search['100'].qps))
for (const r of all) {
  const qps = r.search['100'].qps
  const barLen = Math.max(1, Math.round((qps / maxQps) * 40))
  lines.push(`${r.runtime.padEnd(16)} ${'█'.repeat(barLen)} ${fmt(qps)} qps`)
}
lines.push('```')

// Distance
lines.push('\n## Euclidean Distance Throughput\n')
{
  const dims = Object.keys(all[0].distance)
  const headers = ['Dimensions', ...runtimeNames]
  const rows = dims.map((d) => {
    return [`${d}d`, ...all.map((r) => `${fmt(r.distance[d].opsPerSec)} ops/s`)]
  })
  lines.push(mdTable(headers, rows))
}

// Bar chart comparing distance at 128d
lines.push('\n### Euclidean 128d Throughput\n')
lines.push('```')
const maxDist = Math.max(...all.map((r) => r.distance['128'].opsPerSec))
for (const r of all) {
  const ops = r.distance['128'].opsPerSec
  const barLen = Math.max(1, Math.round((ops / maxDist) * 40))
  lines.push(`${r.runtime.padEnd(16)} ${'█'.repeat(barLen)} ${fmt(ops)} ops/s`)
}
lines.push('```')

const md = lines.join('\n') + '\n'
await Deno.writeTextFile('bench/RUNTIME_RESULTS.md', md)
console.log('\nWritten to bench/RUNTIME_RESULTS.md\n')
console.log(md)

// ── helpers ────────────────────────────────────────────────

function mdTable(headers: string[], rows: string[][]): string {
  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => (r[i] ?? '').length)))
  const out: string[] = []
  out.push('| ' + headers.map((h, i) => h.padEnd(widths[i])).join(' | ') + ' |')
  out.push('| ' + widths.map((w) => '-'.repeat(w)).join(' | ') + ' |')
  for (const row of rows) {
    out.push('| ' + row.map((c, i) => c.padStart(widths[i])).join(' | ') + ' |')
  }
  return out.join('\n')
}
