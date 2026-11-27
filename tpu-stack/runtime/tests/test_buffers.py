import numpy as np
import pytest

from tpurt import get_backend
from tpurt.backends.raw import FakeFabric, FakeRegion, RawBackend
from tpurt.buffers import DmaPool, FirstFit, WeightStore


def test_firstfit_alignment_coalescing_reuse():
    f = FirstFit(1 << 20, align=64)
    a = f.alloc(100)
    b = f.alloc(5000, align=4096)
    c = f.alloc(64)
    assert a == 0 and b % 4096 == 0 and c % 64 == 0
    assert len({a, b, c}) == 3 and f.in_use == 128 + 5056 + 64
    f.free(b)
    d = f.alloc(4096, align=4096)
    assert d == b and f.reused == 1                    # freed space is reused
    f.free(a), f.free(c), f.free(d)
    assert f.free_list == [(0, 1 << 20)] and f.in_use == 0   # fully coalesced
    assert f.alloc(2 << 20) is None


@pytest.fixture(params=["sim", "mock"])
def be(request):
    b = get_backend(request.param)
    yield b
    b.close()


def test_pool_blocks_views_and_reuse(be):
    pool = DmaPool(be, arena_size=1 << 16, align=64)
    blks = [pool.alloc(1000, name=f"b{i}") for i in range(10)]
    assert all(b.offset % 64 == 0 for b in blks)
    assert pool.stats()["driver_allocs"] == 1          # one ALLOC+mmap for ten blocks
    page = pool.alloc(100, align=4096)
    assert page.offset % 4096 == 0
    # zero-copy: numpy view and the raw DMA buffer are the same memory
    v = blks[3].np(np.int8)
    v[:5] = [1, 2, 3, 4, 5]
    raw = bytes(memoryview(blks[3].arena.dev.mem)[blks[3].offset:blks[3].offset + 5])
    assert raw == bytes([1, 2, 3, 4, 5]) and np.shares_memory(v, blks[3].np(np.int8))
    assert blks[3].rows(8).shape == (8, 16)
    blks[3].write(np.arange(16, dtype=np.uint8))
    assert (blks[3].read(16) == np.arange(16)).all()
    # reuse after free
    off = blks[4].offset
    blks[4].free()
    again = pool.alloc(900)
    assert again.offset == off and pool.stats()["reused"] >= 1
    # relocation tuple and dedicated arena for a big block
    dev, o = blks[2].reloc(32)
    assert dev is blks[2].arena.dev and o == blks[2].offset + 32
    big = pool.alloc(200_000)
    assert big.arena.size >= 200_000 and pool.stats()["arenas"] == 2
    sub = big.sub(4096, 256)
    sub.free()                                          # children never free the parent's memory
    assert not big.freed
    with pytest.raises(ValueError):
        pool.alloc(16, align=24)
    pool.close()


def test_weight_store_loads_once(be):
    pool = DmaPool(be)
    ws = WeightStore(pool)
    w = np.arange(512, dtype=np.int8)
    b1 = ws.put("w", w)
    b2 = ws.put("w", w)
    assert b1 is b2 and ws.loads == 1 and "w" in ws
    assert (b1.np(np.int8, count=512) == w).all() and b1.offset % 256 == 0
    pool.close()


def test_cache_hooks_reach_noncoherent_backend():
    reg = FakeRegion()
    be = RawBackend(regs=FakeFabric(reg), region=reg)
    pool = DmaPool(be)
    b = pool.alloc(4096)
    f0, i0 = reg.flushes, reg.invalidates
    b.write(np.ones(64, np.uint8))                      # write -> flush (sync_for_device)
    b.read(64)                                          # read  -> invalidate (sync_for_cpu)
    b.flush()
    assert reg.flushes == f0 + 2 and reg.invalidates == i0 + 1
