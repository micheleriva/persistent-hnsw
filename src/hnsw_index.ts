import { BinaryHeap } from './binary_heap.ts'
import { Bitset } from './bitset.ts'
import { computeNorm, getDistanceFunction } from './distances.ts'
import { createRng } from './random.ts'
import {
  type DistanceFunction,
  type ExternalId,
  type HNSWConfig,
  type InternalId,
  type SearchResult,
  SENTINEL,
  type Vector,
} from './types.ts'

interface Candidate {
  id: InternalId
  distance: number
}

const MIN_CMP = (a: Candidate, b: Candidate) => a.distance - b.distance
const MAX_CMP = (a: Candidate, b: Candidate) => b.distance - a.distance

const INITIAL_CAPACITY = 1024
const GROWTH_FACTOR = 1.5

export class HNSWIndex {
  readonly config: HNSWConfig
  readonly distanceFn: DistanceFunction

  // Flat typed arrays for cache-friendly access
  private vectors: Float32Array
  private norms: Float32Array
  private levels: Uint8Array

  // adjacency[layer] is flat: node i's neighbors at [i * maxNeighbors, (i+1) * maxNeighbors)
  private adjacency: Uint32Array[]
  private neighborCounts: Uint8Array[]

  private deletedSet: Bitset
  private internalToExternal: string[]
  private externalToInternal: Map<string, number>

  private entryPointId: InternalId
  private maxLevel: number
  private count: number
  private capacity: number
  private _deletedCount: number

  private rng: () => number

  // Pooled objects — reused across searchLayer calls to avoid allocation
  private _visited: Bitset
  private _candidates: BinaryHeap<Candidate>
  private _results: BinaryHeap<Candidate>

  constructor(config: HNSWConfig) {
    this.config = config
    this.distanceFn = getDistanceFunction(config.metric)
    this.rng = createRng(config.seed)

    this.capacity = INITIAL_CAPACITY
    this.count = 0
    this._deletedCount = 0
    this.entryPointId = -1
    this.maxLevel = -1

    const dim = config.dimensions
    this.vectors = new Float32Array(this.capacity * dim)
    this.norms = new Float32Array(this.capacity)
    this.levels = new Uint8Array(this.capacity)

    this.adjacency = []
    this.neighborCounts = []

    this.deletedSet = new Bitset(this.capacity)
    this.internalToExternal = []
    this.externalToInternal = new Map()

    // Pooled search structures
    this._visited = new Bitset(this.capacity)
    this._candidates = new BinaryHeap<Candidate>(MIN_CMP)
    this._results = new BinaryHeap<Candidate>(MAX_CMP)

    // Pre-allocate layer 0
    this.ensureLayer(0)
  }

  get size(): number {
    return this.count - this._deletedCount
  }

  get totalAllocated(): number {
    return this.count
  }

  get entryPoint(): InternalId {
    return this.entryPointId
  }

  get topLevel(): number {
    return this.maxLevel
  }

  get dimensions(): number {
    return this.config.dimensions
  }

  /** Insert a vector with an external ID. */
  insert(id: ExternalId, vector: Vector | number[]): void {
    if (this.externalToInternal.has(id)) {
      throw new Error(`Duplicate ID: ${id}`)
    }

    const vec = vector instanceof Float32Array ? vector : new Float32Array(vector)
    if (vec.length !== this.config.dimensions) {
      throw new Error(
        `Vector dimension mismatch: expected ${this.config.dimensions}, got ${vec.length}`,
      )
    }

    // Grow if needed
    if (this.count >= this.capacity) {
      this.grow()
    }

    const internalId = this.count
    this.count++

    // Store vector
    const dim = this.config.dimensions
    this.vectors.set(vec, internalId * dim)

    // Cache norm for cosine
    if (this.config.metric === 'cosine') {
      this.norms[internalId] = computeNorm(vec)
    }

    // Assign random level
    const level = this.randomLevel()
    this.levels[internalId] = level

    // ID mapping
    this.internalToExternal[internalId] = id
    this.externalToInternal.set(id, internalId)

    // Ensure all layers up to this level exist
    for (let l = 0; l <= level; l++) {
      this.ensureLayer(l)
    }

    // First node
    if (this.entryPointId === -1) {
      this.entryPointId = internalId
      this.maxLevel = level
      return
    }

    let currObj = this.entryPointId
    let currDist = this.distance(internalId, currObj)

    // Phase 1: Greedy descent from top layer to level+1
    for (let l = this.maxLevel; l > level; l--) {
      let changed = true
      while (changed) {
        changed = false
        const neighbors = this.getNeighbors(currObj, l)
        for (let i = 0; i < neighbors.length; i++) {
          const neighbor = neighbors[i]
          if (neighbor === SENTINEL) break
          const d = this.distance(internalId, neighbor)
          if (d < currDist) {
            currObj = neighbor
            currDist = d
            changed = true
          }
        }
      }
    }

    // Phase 2: Insert at each layer from min(level, maxLevel) down to 0
    const topInsertLevel = Math.min(level, this.maxLevel)
    for (let l = topInsertLevel; l >= 0; l--) {
      const maxNeighbors = l === 0 ? this.config.Mmax0 : this.config.M

      // Beam search to find candidates
      const candidates = this.searchLayer(
        internalId,
        currObj,
        this.config.efConstruction,
        l,
      )

      // Select neighbors
      const selected = this.config.useHeuristic
        ? this.selectNeighborsHeuristic(candidates, maxNeighbors)
        : this.selectNeighborsSimple(candidates, maxNeighbors)

      // Connect internalId to selected neighbors
      this.setNeighbors(internalId, l, selected)

      // Add bidirectional connections
      for (const neighbor of selected) {
        this.addConnection(neighbor.id, internalId, l)
      }

      // Use the nearest candidate as entry for next layer
      if (candidates.length > 0) {
        currObj = candidates[0].id
        currDist = candidates[0].distance
      }
    }

    // Update entry point if new node has highest level
    if (level > this.maxLevel) {
      this.entryPointId = internalId
      this.maxLevel = level
    }
  }

  /** Search for k nearest neighbors of a query vector. */
  search(
    query: Vector | number[],
    k: number,
    efSearch?: number,
    filter?: (id: ExternalId) => boolean,
  ): SearchResult[] {
    if (this.entryPointId === -1) return []

    const q = query instanceof Float32Array ? query : new Float32Array(query)
    const ef = Math.max(efSearch ?? this.config.efSearch, k)

    let currObj = this.entryPointId
    let currDist = this.distanceToQuery(q, currObj)

    // Phase 1: Greedy descent from top to layer 1
    for (let l = this.maxLevel; l >= 1; l--) {
      let changed = true
      while (changed) {
        changed = false
        const neighbors = this.getNeighbors(currObj, l)
        for (let i = 0; i < neighbors.length; i++) {
          const neighbor = neighbors[i]
          if (neighbor === SENTINEL) break
          const d = this.distanceToQuery(q, neighbor)
          if (d < currDist) {
            currObj = neighbor
            currDist = d
            changed = true
          }
        }
      }
    }

    // Phase 2: Beam search at layer 0
    const candidates = this.searchLayerByQuery(q, currObj, ef, 0)

    // Filter and collect top-k
    const results: SearchResult[] = []
    for (const c of candidates) {
      if (this.deletedSet.has(c.id)) continue
      const extId = this.internalToExternal[c.id]
      if (filter && !filter(extId)) continue
      results.push({ id: extId, distance: c.distance })
      if (results.length === k) break
    }

    return results
  }

  /** Mark a vector as deleted (lazy tombstone). */
  delete(id: ExternalId): boolean {
    const internalId = this.externalToInternal.get(id)
    if (internalId === undefined) return false
    if (this.deletedSet.has(internalId)) return false

    this.deletedSet.set(internalId)
    this._deletedCount++
    return true
  }

  /** Check if an external ID exists and is not deleted. */
  has(id: ExternalId): boolean {
    const internalId = this.externalToInternal.get(id)
    if (internalId === undefined) return false
    return !this.deletedSet.has(internalId)
  }

  /** Get vector by external ID. */
  getVector(id: ExternalId): Vector | null {
    const internalId = this.externalToInternal.get(id)
    if (internalId === undefined) return null
    if (this.deletedSet.has(internalId)) return null
    const dim = this.config.dimensions
    return this.vectors.slice(internalId * dim, (internalId + 1) * dim)
  }

  /** Rebuild the index without deleted nodes. Returns a new compact index. */
  compact(): HNSWIndex {
    const newIndex = new HNSWIndex(this.config)
    for (let i = 0; i < this.count; i++) {
      if (this.deletedSet.has(i)) continue
      const extId = this.internalToExternal[i]
      const dim = this.config.dimensions
      const vec = this.vectors.slice(i * dim, (i + 1) * dim)
      newIndex.insert(extId, vec)
    }
    return newIndex
  }

  /** Shrink all internal arrays to fit the current count exactly. */
  shrinkToFit(): void {
    if (this.capacity === this.count) return
    const newCapacity = Math.max(1, this.count)
    const dim = this.config.dimensions

    const newVectors = new Float32Array(newCapacity * dim)
    newVectors.set(this.vectors.subarray(0, this.count * dim))
    this.vectors = newVectors

    const newNorms = new Float32Array(newCapacity)
    newNorms.set(this.norms.subarray(0, this.count))
    this.norms = newNorms

    const newLevels = new Uint8Array(newCapacity)
    newLevels.set(this.levels.subarray(0, this.count))
    this.levels = newLevels

    for (let l = 0; l < this.adjacency.length; l++) {
      const maxN = l === 0 ? this.config.Mmax0 : this.config.M
      const newAdj = new Uint32Array(newCapacity * maxN)
      newAdj.fill(SENTINEL)
      newAdj.set(this.adjacency[l].subarray(0, this.count * maxN))
      this.adjacency[l] = newAdj

      const newCounts = new Uint8Array(newCapacity)
      newCounts.set(this.neighborCounts[l].subarray(0, this.count))
      this.neighborCounts[l] = newCounts
    }

    this.capacity = newCapacity
  }

  /** Estimate memory usage in bytes. */
  memoryUsage(): number {
    let bytes = this.vectors.byteLength + this.norms.byteLength +
      this.levels.byteLength
    for (const adj of this.adjacency) bytes += adj.byteLength
    for (const nc of this.neighborCounts) bytes += nc.byteLength
    return bytes
  }

  // --- Serialization helpers (used by serialization.ts) ---

  /** Get raw internal state for serialization. */
  getInternalState(): {
    vectors: Float32Array
    norms: Float32Array
    levels: Uint8Array
    adjacency: Uint32Array[]
    neighborCounts: Uint8Array[]
    internalToExternal: string[]
    count: number
    capacity: number
    entryPointId: number
    maxLevel: number
  } {
    return {
      vectors: this.vectors,
      norms: this.norms,
      levels: this.levels,
      adjacency: this.adjacency,
      neighborCounts: this.neighborCounts,
      internalToExternal: this.internalToExternal,
      count: this.count,
      capacity: this.capacity,
      entryPointId: this.entryPointId,
      maxLevel: this.maxLevel,
    }
  }

  /** Restore from serialized state. */
  static fromInternalState(
    config: HNSWConfig,
    state: {
      vectors: Float32Array
      norms: Float32Array
      levels: Uint8Array
      adjacency: Uint32Array[]
      neighborCounts: Uint8Array[]
      internalToExternal: string[]
      count: number
      capacity: number
      entryPointId: number
      maxLevel: number
    },
  ): HNSWIndex {
    const index = new HNSWIndex(config)
    index.vectors = state.vectors
    index.norms = state.norms
    index.levels = state.levels
    index.adjacency = state.adjacency
    index.neighborCounts = state.neighborCounts
    index.internalToExternal = state.internalToExternal
    index.count = state.count
    index.capacity = state.capacity
    index.entryPointId = state.entryPointId
    index.maxLevel = state.maxLevel
    index._deletedCount = 0

    // Rebuild external-to-internal map
    index.externalToInternal = new Map()
    for (let i = 0; i < state.count; i++) {
      index.externalToInternal.set(state.internalToExternal[i], i)
    }

    // Rebuild deleted set and visited pool
    index.deletedSet = new Bitset(state.capacity)
    index._visited = new Bitset(state.capacity)

    return index
  }

  // --- Private methods ---

  private randomLevel(): number {
    return Math.floor(-Math.log(this.rng()) * this.config.mL)
  }

  private distance(a: InternalId, b: InternalId): number {
    const dim = this.config.dimensions
    const vecA = new Float32Array(
      this.vectors.buffer,
      this.vectors.byteOffset + a * dim * 4,
      dim,
    )
    const vecB = new Float32Array(
      this.vectors.buffer,
      this.vectors.byteOffset + b * dim * 4,
      dim,
    )
    return this.distanceFn(vecA, vecB)
  }

  private distanceToQuery(query: Vector, b: InternalId): number {
    const dim = this.config.dimensions
    const vecB = new Float32Array(
      this.vectors.buffer,
      this.vectors.byteOffset + b * dim * 4,
      dim,
    )
    return this.distanceFn(query, vecB)
  }

  private ensureLayer(layer: number): void {
    while (this.adjacency.length <= layer) {
      const l = this.adjacency.length
      const maxN = l === 0 ? this.config.Mmax0 : this.config.M
      const adj = new Uint32Array(this.capacity * maxN)
      adj.fill(SENTINEL)
      this.adjacency.push(adj)
      this.neighborCounts.push(new Uint8Array(this.capacity))
    }
  }

  private getNeighbors(nodeId: InternalId, layer: number): Uint32Array {
    const maxN = layer === 0 ? this.config.Mmax0 : this.config.M
    const offset = nodeId * maxN
    const count = this.neighborCounts[layer][nodeId]
    return this.adjacency[layer].subarray(offset, offset + count)
  }

  private setNeighbors(
    nodeId: InternalId,
    layer: number,
    neighbors: Candidate[],
  ): void {
    const maxN = layer === 0 ? this.config.Mmax0 : this.config.M
    const offset = nodeId * maxN
    const count = Math.min(neighbors.length, maxN)

    for (let i = 0; i < count; i++) {
      this.adjacency[layer][offset + i] = neighbors[i].id
    }
    for (let i = count; i < maxN; i++) {
      this.adjacency[layer][offset + i] = SENTINEL
    }
    this.neighborCounts[layer][nodeId] = count
  }

  private addConnection(
    nodeId: InternalId,
    neighborId: InternalId,
    layer: number,
  ): void {
    const maxN = layer === 0 ? this.config.Mmax0 : this.config.M
    const count = this.neighborCounts[layer][nodeId]
    const offset = nodeId * maxN

    // Check if already connected
    for (let i = 0; i < count; i++) {
      if (this.adjacency[layer][offset + i] === neighborId) return
    }

    if (count < maxN) {
      this.adjacency[layer][offset + count] = neighborId
      this.neighborCounts[layer][nodeId] = count + 1
    } else {
      // Overflow: collect all current neighbors + new one, then select best
      const candidates: Candidate[] = new Array(count + 1)
      for (let i = 0; i < count; i++) {
        const nId = this.adjacency[layer][offset + i]
        candidates[i] = { id: nId, distance: this.distance(nodeId, nId) }
      }
      candidates[count] = {
        id: neighborId,
        distance: this.distance(nodeId, neighborId),
      }

      const selected = this.config.useHeuristic
        ? this.selectNeighborsHeuristic(candidates, maxN)
        : this.selectNeighborsSimple(candidates, maxN)

      this.setNeighbors(nodeId, layer, selected)
    }
  }

  /**
   * Beam search within a single layer, using stored vectors (for insertion).
   * Uses pooled bitset and heaps to avoid per-call allocation.
   */
  private searchLayer(
    queryId: InternalId,
    entryId: InternalId,
    ef: number,
    layer: number,
  ): Candidate[] {
    const entryDist = this.distance(queryId, entryId)

    // Reuse pooled structures
    const visited = this._visited
    visited.grow(this.count)
    visited.clear()
    visited.set(entryId)

    const candidates = this._candidates
    const results = this._results
    candidates.clear()
    results.clear()

    candidates.push({ id: entryId, distance: entryDist })
    results.push({ id: entryId, distance: entryDist })

    while (candidates.size > 0) {
      const nearest = candidates.pop()!
      const farthest = results.peek()!

      if (nearest.distance > farthest.distance) break

      const neighbors = this.getNeighbors(nearest.id, layer)
      for (let i = 0; i < neighbors.length; i++) {
        const neighborId = neighbors[i]
        if (neighborId === SENTINEL) break
        if (visited.has(neighborId)) continue
        visited.set(neighborId)

        const d = this.distance(queryId, neighborId)
        const worstResult = results.peek()!

        if (d < worstResult.distance || results.size < ef) {
          candidates.push({ id: neighborId, distance: d })
          results.push({ id: neighborId, distance: d })
          if (results.size > ef) {
            results.pop()
          }
        }
      }
    }

    // Drain results into sorted array (nearest first)
    const len = results.size
    const sorted: Candidate[] = new Array(len)
    for (let i = len - 1; i >= 0; i--) {
      sorted[i] = results.pop()!
    }
    return sorted
  }

  /** Beam search within a single layer, using a query vector (for search). */
  private searchLayerByQuery(
    query: Vector,
    entryId: InternalId,
    ef: number,
    layer: number,
  ): Candidate[] {
    const entryDist = this.distanceToQuery(query, entryId)

    // Reuse pooled structures
    const visited = this._visited
    visited.grow(this.count)
    visited.clear()
    visited.set(entryId)

    const candidates = this._candidates
    const results = this._results
    candidates.clear()
    results.clear()

    candidates.push({ id: entryId, distance: entryDist })
    results.push({ id: entryId, distance: entryDist })

    while (candidates.size > 0) {
      const nearest = candidates.pop()!
      const farthest = results.peek()!

      if (nearest.distance > farthest.distance) break

      const neighbors = this.getNeighbors(nearest.id, layer)
      for (let i = 0; i < neighbors.length; i++) {
        const neighborId = neighbors[i]
        if (neighborId === SENTINEL) break
        if (visited.has(neighborId)) continue
        visited.set(neighborId)

        const d = this.distanceToQuery(query, neighborId)
        const worstResult = results.peek()!

        if (d < worstResult.distance || results.size < ef) {
          candidates.push({ id: neighborId, distance: d })
          results.push({ id: neighborId, distance: d })
          if (results.size > ef) {
            results.pop()
          }
        }
      }
    }

    const len = results.size
    const sorted: Candidate[] = new Array(len)
    for (let i = len - 1; i >= 0; i--) {
      sorted[i] = results.pop()!
    }
    return sorted
  }

  /** Algorithm 4: Heuristic neighbor selection for spatial diversity. */
  private selectNeighborsHeuristic(
    candidates: Candidate[],
    maxNeighbors: number,
  ): Candidate[] {
    if (candidates.length <= maxNeighbors) {
      return candidates
    }

    // Sort candidates by distance to query
    const sorted = [...candidates].sort(MIN_CMP)
    const selected: Candidate[] = []
    const discarded: Candidate[] = []

    for (const candidate of sorted) {
      if (selected.length >= maxNeighbors) break

      // Keep if candidate is closer to query than to any already-selected neighbor
      let keep = true
      for (const sel of selected) {
        const distCandToSel = this.distance(candidate.id, sel.id)
        if (distCandToSel < candidate.distance) {
          keep = false
          break
        }
      }

      if (keep) {
        selected.push(candidate)
      } else {
        discarded.push(candidate)
      }
    }

    // Fill remaining slots with pruned connections if configured
    if (this.config.keepPrunedConnections && selected.length < maxNeighbors) {
      for (const d of discarded) {
        if (selected.length >= maxNeighbors) break
        if (!selected.some((s) => s.id === d.id)) {
          selected.push(d)
        }
      }
    }

    return selected
  }

  /** Simple neighbor selection: just take the closest. */
  private selectNeighborsSimple(
    candidates: Candidate[],
    maxNeighbors: number,
  ): Candidate[] {
    return [...candidates].sort(MIN_CMP).slice(0, maxNeighbors)
  }

  private grow(): void {
    const newCapacity = Math.max(
      this.capacity + 1,
      Math.ceil(this.capacity * GROWTH_FACTOR),
    )
    const dim = this.config.dimensions

    const newVectors = new Float32Array(newCapacity * dim)
    newVectors.set(this.vectors)
    this.vectors = newVectors

    const newNorms = new Float32Array(newCapacity)
    newNorms.set(this.norms)
    this.norms = newNorms

    const newLevels = new Uint8Array(newCapacity)
    newLevels.set(this.levels)
    this.levels = newLevels

    for (let l = 0; l < this.adjacency.length; l++) {
      const maxN = l === 0 ? this.config.Mmax0 : this.config.M
      const newAdj = new Uint32Array(newCapacity * maxN)
      newAdj.fill(SENTINEL)
      newAdj.set(this.adjacency[l])
      this.adjacency[l] = newAdj

      const newCounts = new Uint8Array(newCapacity)
      newCounts.set(this.neighborCounts[l])
      this.neighborCounts[l] = newCounts
    }

    this.deletedSet.grow(newCapacity)
    this._visited.grow(newCapacity)

    this.capacity = newCapacity
  }
}
