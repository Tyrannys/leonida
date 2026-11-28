"""A scalar loop over the lanes of a ballot, inside a wave64 loop that ends on a vote (SHADER-WAVE-LOOP-EXEC-VOTE).

The compiler walks the lanes a ballot found with a scalar loop: `s_and_b64 s[26:27], vcc, exec`, then
`s_ff1_i32_b32` / `s_xor_b32` of s26 and `s_mov_b32 s26, sN` on the back edge, so s26 is a lane mask on the way in and
a number around the loop. The translator keeps a mask as each lane's own bit (0 or ~0) and data as one number, and at
the loop header took s26 as data without gathering the mask's bits: every lane found its own first bit and took its
own number of passes, where the wave takes one (lane_masks.insert_data_merges now gathers them on the way in).

The program is hand-written in the shape of a library compute program that ran its outer loop to the loop guard
(#1608, SHADER-LIB-LOOP-AT-GUARD): an outer loop over the lanes' work items, 64 apart, that ends when
`v_cmpx_le` + `s_andn2_b64 saved, saved, exec` leaves no lane; inside it, one bit loop over each half of a ballot with a
lane-local `s_cbranch_execz` region. No game data.

    python -I tests/semantic/test_ballot_bit_loop.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_loop_ballot_barrier import VSHARP, _translate_cs, run_compute_workgroup  # noqa: E402


def program(count):
    p = [
        f"s_movk_i32 s12, {count}",
        "s_mov_b64 s[14:15], exec",
        "s_add_u32 s16, s12, 63",            # s16 = min(ceil(count / 64), 4) * 64: the passes, at most 4
        "s_lshr_b32 s16, s16, 6",
        "s_min_u32 s16, s16, 4",
        "s_lshl_b32 s16, s16, 6",
        "v_mov_b32_e32 v1, v0",              # the lane, kept
        "v_mov_b32_e32 v2, 0",               # passes of the outer loop
        "v_mov_b32_e32 v3, 0",               # passes of the bit loops' lane-local regions
        "v_cmpx_gt_i32_e32 s16, v0",
        "s_cbranch_execz done",
        "s_mov_b64 s[18:19], exec",
        "s_mov_b64 s[24:25], exec",          # the lanes still looping
        "s_branch body",
        "head:",
        "s_mov_b64 exec, s[24:25]",
        "body:",
        "v_add_nc_u32_e32 v2, 1, v2",
        "v_cmpx_gt_u32_e32 s12, v0",          # the lanes with an item this pass
        "v_and_b32_e32 v4, 1, v0",
        "v_cmp_eq_u32_e32 vcc, 0, v4",        # ... whose item is even: the ballot
        "s_and_b64 s[26:27], vcc, exec",
        "s_mov_b64 exec, s[24:25]",
        # Lanes 0-31 of the ballot, one pass per lane set; lanes below 16 count them.
        "s_cmp_lg_i32 s26, 0",
        "s_cbranch_scc0 after_lo",
        "lo:",
        "s_ff1_i32_b32 s32, s26",
        "s_lshl_b32 s32, 1, s32",
        "s_xor_b32 s35, s32, s26",
        "v_cmp_gt_u32_e32 vcc, 16, v1",
        "s_and_b64 exec, vcc, exec",
        "s_cbranch_execz skip_lo",
        "v_add_nc_u32_e32 v3, 1, v3",
        "skip_lo:",
        "s_mov_b64 exec, s[24:25]",
        "s_cmp_eq_i32 s26, s32",
        "s_cbranch_scc1 after_lo",
        "s_mov_b32 s26, s35",
        "s_branch lo",
        "after_lo:",
        # Lanes 32-63, the same through s27; lanes above 47 count them in the second byte.
        "s_cmp_lg_i32 s27, 0",
        "s_cbranch_scc0 after_hi",
        "hi:",
        "s_ff1_i32_b32 s32, s27",
        "s_lshl_b32 s32, 1, s32",
        "s_xor_b32 s35, s32, s27",
        "v_cmp_lt_u32_e32 vcc, 47, v1",
        "s_and_b64 exec, vcc, exec",
        "s_cbranch_execz skip_hi",
        "v_add_nc_u32_e32 v3, 0x100, v3",
        "skip_hi:",
        "s_mov_b64 exec, s[24:25]",
        "s_cmp_eq_i32 s27, s32",
        "s_cbranch_scc1 after_hi",
        "s_mov_b32 s27, s35",
        "s_branch hi",
        "after_hi:",
        # The next item 64 on; the loop ends when no lane has one left.
        "v_add_nc_u32_e32 v0, 64, v0",
        "v_cmpx_le_i32_e32 s16, v0",
        "s_andn2_b64 s[24:25], s[24:25], exec",
        "s_cbranch_scc1 head",
        "s_mov_b64 exec, s[18:19]",
        "done:",
        "s_mov_b64 exec, s[14:15]",
        "v_lshlrev_b32_e32 v6, 3, v1",
        "buffer_store_dword v2, v6, s[0:3], 0 offen offset:0",
        "buffer_store_dword v3, v6, s[0:3], 0 offen offset:4",
        "s_endpgm",
    ]
    return p


def expected(count):
    """(outer passes, bit-loop visits) per lane, as one wave64 runs it."""
    limit = min((count + 63) >> 6, 4) << 6
    item = {t: t for t in range(64)}
    passes = {t: 0 for t in range(64)}
    visits = {t: 0 for t in range(64)}
    alive = [t for t in range(64) if item[t] < limit]
    while alive:
        for t in alive:
            passes[t] += 1
        ballot = [t for t in alive if item[t] < count and item[t] % 2 == 0]
        for _ in [t for t in ballot if t < 32]:
            for t in alive:
                visits[t] += 1 if t < 16 else 0
        for _ in [t for t in ballot if t >= 32]:
            for t in alive:
                visits[t] += 0x100 if t > 47 else 0
        for t in alive:
            item[t] += 64
        alive = [t for t in alive if item[t] < limit]
    return {t: (passes[t], visits[t]) for t in range(64)}


def mask_use_program(take_mask):
    """A lane mask (even lanes) on one path and a number (lanes 0-15: 0xffff) on the other meet in s[6:7], which then
    only goes into EXEC. Taken as data at the merge, each lane's 0 / ~0 on the mask path still gives its own bit
    (bit L of a data register for lane L), so no gather is needed and none is inserted."""
    return [
        f"s_mov_b32 s40, {1 if take_mask else 0}",
        "s_mov_b64 s[14:15], exec",
        "v_mov_b32_e32 v2, 0",
        "v_and_b32_e32 v4, 1, v0",
        "v_cmp_eq_u32_e32 vcc, 0, v4",
        "s_mov_b64 s[6:7], vcc",
        "s_cmp_eq_u32 s40, 1",
        "s_cbranch_scc1 join",
        "s_mov_b32 s6, 0xffff",
        "s_mov_b32 s7, 0",
        "join:",
        "s_and_b64 exec, s[6:7], exec",
        "v_mov_b32_e32 v2, 1",
        "s_mov_b64 exec, s[14:15]",
        "v_lshlrev_b32_e32 v6, 2, v0",
        "buffer_store_dword v2, v6, s[0:3], 0 offen offset:0",
        "s_endpgm",
    ]


class BallotBitLoop(unittest.TestCase):
    def test_a_merged_mask_used_as_a_mask_needs_no_gather(self):
        for take_mask in (True, False):
            want = {t: int(t % 2 == 0) if take_mask else int(t < 16) for t in range(64)}
            for label, emulate, sg in (("native64", False, 64), ("emulate_sg32", True, 32)):
                with self.subTest(take_mask=take_mask, mode=label):
                    spv, _ = _translate_cs(f"cs_merged_mask_{label}_{int(take_mask)}", mask_use_program(take_mask),
                                           threads=64, emulate=emulate)
                    mem = run_compute_workgroup(spv, 64, sg, push_constants=VSHARP)
                    for t in range(64):
                        self.assertEqual(mem.get(4 * t, -1), want[t], f"{label} lane {t}")

    def test_every_lane_walks_the_wave_ballot(self):
        configs = [
            ("native64", False, 64, True),
            ("emulate_sg32", True, 32, True),
            ("emulate_sg32_rounds", True, 32, False),
        ]
        for count in (100, 200, 7):
            want = expected(count)
            for label, emulate, sg, structured in configs:
                with self.subTest(count=count, mode=label):
                    spv, _ = _translate_cs(f"cs_ballot_bit_loop_{label}_{count}", program(count), threads=64,
                                           emulate=emulate, structured=structured)
                    mem = run_compute_workgroup(spv, 64, sg, push_constants=VSHARP)
                    for t in range(64):
                        got = (mem.get(8 * t, -1), mem.get(8 * t + 4, -1))
                        self.assertEqual(got, want[t], f"{label} count {count} lane {t}")


if __name__ == "__main__":
    unittest.main()
