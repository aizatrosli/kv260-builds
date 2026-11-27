"""DMA buffer management.

  FirstFit    offset allocator (alignment, first fit, coalescing free list, reuse statistics)
  DmaPool     sub-allocates aligned Blocks (64 B default, 4 KiB on request) out of large backend
              buffers ("arenas", one driver ALLOC + mmap each), so thousands of tensors cost a few
              ioctls; freed blocks are reused.  Blocks give zero-copy numpy views of DMA memory and
              flush()/invalidate() cache hooks (no-ops on the driver's coherent buffers).
  WeightStore weight residency: named constant blocks written once and shared by every program /
              context ("load weights once, run many inputs").
"""
import bisect

import numpy as np


class FirstFit:
    def __init__(self, size, align=64):
        self.size, self.align = size, align
        self.free_list = [(0, size)]          # sorted (offset, size)
        self.used = {}                        # offset -> size
        self.hwm = 0
        self.reused = self.allocs = 0
        self._freed = []                      # ranges that held a block once (reuse statistics)

    def alloc(self, size, align=None):
        a = max(align or self.align, 1)
        size = (size + self.align - 1) // self.align * self.align
        for i, (off, sz) in enumerate(self.free_list):
            start = (off + a - 1) // a * a
            if start + size <= off + sz:
                del self.free_list[i]
                if start > off:
                    bisect.insort(self.free_list, (off, start - off))
                if start + size < off + sz:
                    bisect.insort(self.free_list, (start + size, off + sz - start - size))
                self.used[start] = size
                self.allocs += 1
                if any(o < start + size and start < o + s_ for o, s_ in self._freed):
                    self.reused += 1
                self.hwm = max(self.hwm, start + size)
                return start
        return None

    def free(self, off):
        size = self.used.pop(off)
        if len(self._freed) < 4096:
            self._freed.append((off, size))
        bisect.insort(self.free_list, (off, size))
        merged = []
        for o, s in self.free_list:
            if merged and merged[-1][0] + merged[-1][1] == o:
                merged[-1] = (merged[-1][0], merged[-1][1] + s)
            else:
                merged.append((o, s))
        self.free_list = merged

    @property
    def in_use(self):
        return sum(self.used.values())


class Block:
    """A slice [offset, offset+size) of an arena buffer."""

    def __init__(self, pool, arena, offset, size, name=""):
        self.pool, self.arena, self.offset, self.size, self.name = pool, arena, offset, size, name
        self.freed = False

    def np(self, dtype=np.int8, shape=None, offset=0, count=None):
        """Zero-copy numpy view of the DMA memory."""
        dt = np.dtype(dtype)
        n = (self.size - offset) // dt.itemsize if count is None else count
        a = np.frombuffer(self.arena.dev.mem, dtype=dt, count=n, offset=self.offset + offset)
        return a if shape is None else a.reshape(shape)

    def rows(self, n=None):
        """View as int8 rows of 16 lanes ([n, 16])."""
        n = self.size // 16 if n is None else n
        return self.np(np.int8, (n, 16), count=n * 16)

    def write(self, data, offset=0):
        b = np.frombuffer(memoryview(np.ascontiguousarray(data)).cast("B"), np.uint8)
        if offset + b.size > self.size:
            raise ValueError(f"write of {b.size} B at {offset} exceeds block {self.name} ({self.size} B)")
        self.np(np.uint8, count=b.size, offset=offset)[:] = b
        self.flush(offset, b.size)

    def read(self, nbytes=None, offset=0):
        self.invalidate(offset, nbytes)
        return self.np(np.uint8, count=self.size - offset if nbytes is None else nbytes, offset=offset).copy()

    def reloc(self, offset=0):
        """(DevBuffer, byte offset) for a relocation entry."""
        return self.arena.dev, self.offset + offset

    def flush(self, offset=0, length=None):
        self.pool.backend.sync_for_device(self.arena.dev, self.offset + offset,
                                          self.size - offset if length is None else length)

    def invalidate(self, offset=0, length=None):
        self.pool.backend.sync_for_cpu(self.arena.dev, self.offset + offset,
                                       self.size - offset if length is None else length)

    def sub(self, offset, size, name=""):
        """Child view [offset, offset+size) of this block (freed with the parent)."""
        if offset < 0 or offset + size > self.size:
            raise ValueError("sub-block outside the parent block")
        b = Block(self.pool, self.arena, self.offset + offset, size, name)
        b.parent = self
        return b

    def free(self):
        if not self.freed:
            self.freed = True
            if getattr(self, "parent", None) is None:
                self.pool._free(self)

    def __repr__(self):
        return f"Block({self.name!r}, h={self.arena.dev.handle}, off={self.offset}, size={self.size})"


class _Arena:
    def __init__(self, dev, size, align):
        self.dev, self.size = dev, size
        self.fit = FirstFit(size, align)


class DmaPool:
    def __init__(self, backend, arena_size=4 << 20, align=64):
        self.backend, self.align = backend, align
        self.arena_size = min(arena_size, backend.max_buf_bytes)
        self.arenas = []
        self.peak = 0
        self.ioctl_allocs = 0

    def alloc(self, nbytes, align=None, name=""):
        if nbytes <= 0:
            raise ValueError("block size must be positive")
        align = align or self.align
        if align & (align - 1) or align < 16:
            raise ValueError("alignment must be a power of two >= 16 (DMA offsets are 16-byte units)")
        for ar in self.arenas:
            off = ar.fit.alloc(nbytes, align)
            if off is not None:
                return self._made(ar, off, nbytes, name)
        size = max(self.arena_size, (nbytes + 4095) & ~4095)
        if size > self.backend.max_buf_bytes:
            raise MemoryError(f"block {name} of {nbytes} B exceeds the backend's per-buffer limit")
        ar = _Arena(self.backend.alloc(size), size, self.align)
        self.ioctl_allocs += 1
        self.arenas.append(ar)
        return self._made(ar, ar.fit.alloc(nbytes, align), nbytes, name)

    def _made(self, ar, off, nbytes, name):
        self.peak = max(self.peak, self.in_use)
        return Block(self, ar, off, nbytes, name)

    def _free(self, blk):
        blk.arena.fit.free(blk.offset)

    @property
    def in_use(self):
        return sum(a.fit.in_use for a in self.arenas)

    def stats(self):
        return dict(arenas=len(self.arenas), arena_bytes=sum(a.size for a in self.arenas), in_use=self.in_use,
                    peak=self.peak, allocs=sum(a.fit.allocs for a in self.arenas),
                    reused=sum(a.fit.reused for a in self.arenas), driver_allocs=self.ioctl_allocs)

    def trim(self):
        """Return completely free arenas to the backend."""
        keep = []
        for a in self.arenas:
            if a.fit.used:
                keep.append(a)
            else:
                self.backend.free(a.dev)
        self.arenas = keep

    def close(self):
        for a in self.arenas:
            self.backend.free(a.dev)
        self.arenas = []


class WeightStore:
    """Resident constant blocks (weights, LUTs, positional embedding), loaded once."""

    def __init__(self, pool):
        self.pool, self.blocks, self.loads = pool, {}, 0
        self.bytes = 0

    def put(self, key, data, align=256):
        if key in self.blocks:
            return self.blocks[key]
        raw = np.ascontiguousarray(data)
        blk = self.pool.alloc(raw.nbytes, align=align, name=f"const:{key}")
        blk.write(raw)
        self.blocks[key] = blk
        self.loads += 1
        self.bytes += raw.nbytes
        return blk

    def get(self, key):
        return self.blocks[key]

    def __contains__(self, key):
        return key in self.blocks
