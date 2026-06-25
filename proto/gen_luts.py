#!/usr/bin/env python3
# Generate proto/t1_luts.metal (decode LUTs as MSL constant arrays) from
# src/lib/openjp2/t1_luts.h. Run from the repo root.
import os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "src", "lib", "openjp2", "t1_luts.h")).read()
out = ["// Auto-generated from t1_luts.h by gen_luts.py — decode LUTs for the Metal T1 kernel", ""]
for name, size in [("lut_ctxno_zc", 2048), ("lut_ctxno_sc", 256), ("lut_spb", 256)]:
    i = src.index(name + "[")
    b = src.index("{", i); e = src.index("};", b)
    nums = [x.strip() for x in src[b+1:e].replace("\n", " ").split(",") if x.strip() != ""]
    assert len(nums) == size, (name, len(nums))
    out.append("constant uchar %s[%d] = {" % (name, size))
    for j in range(0, size, 32):
        out.append("  " + ",".join(nums[j:j+32]) + ",")
    out.append("};\n")
open(os.path.join(HERE, "t1_luts.metal"), "w").write("\n".join(out))
print("wrote proto/t1_luts.metal")
