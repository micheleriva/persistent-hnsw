import { build, emptyDir } from 'https://deno.land/x/dnt@0.40.0/mod.ts'

const denoJson = JSON.parse(Deno.readTextFileSync('./deno.json'))
const outDir = './.npm'

await emptyDir(outDir)

await build({
  entryPoints: ['./mod.ts'],
  outDir,
  shims: {
    deno: 'dev',
  },
  compilerOptions: {
    lib: ['DOM'],
  },
  scriptModule: 'cjs',
  typeCheck: 'both',
  test: false,
  package: {
    name: denoJson.name,
    version: denoJson.version,
    description: 'Pure-TypeScript HNSW vector search. No native dependencies, no WASM. ' +
      'Sub-millisecond search with pluggable persistence, automatic sharding, ' +
      'and three distance metrics.',
    license: 'MIT',
    author: {
      name: 'Michele Riva',
      email: 'riva.michele.95@gmail.com',
    },
    repository: {
      type: 'git',
      url: 'git+https://github.com/micheleriva/persistent-hnsw.git',
    },
    bugs: {
      url: 'https://github.com/micheleriva/persistent-hnsw/issues',
    },
    keywords: [
      'hnsw',
      'vector',
      'search',
      'nearest-neighbor',
      'ann',
      'similarity',
      'embedding',
      'knn',
      'cosine',
      'euclidean',
    ],
    engines: {
      node: '>=18.0.0',
    },
    sideEffects: false,
  },
  postBuild() {
    Deno.copyFileSync('./README.md', `${outDir}/README.md`)
    Deno.copyFileSync('./LICENSE.md', `${outDir}/LICENSE.md`)
  },
})
