// Headless Metal benchmark for the T1 codeblock decode kernel.
// Loads proto/corpus.bin, dispatches t1_decode (1 thread = 1 codeblock),
// validates bit-exact vs the oracle, and reports GPU vs CPU throughput.
//
// Build: swiftc -O proto/t1_bench.swift -o /tmp/t1_bench -framework Metal -framework Foundation
// Run:   /tmp/t1_bench proto/corpus.bin proto/t1_luts.metal proto/t1_kernel.metal

import Foundation
import Metal

func die(_ s: String) -> Never { FileHandle.standardError.write((s+"\n").data(using:.utf8)!); exit(1) }

let args = CommandLine.arguments
let corpusPath = args.count > 1 ? args[1] : "proto/corpus.bin"
let lutsPath   = args.count > 2 ? args[2] : "proto/t1_luts.metal"
let kernelPath = args.count > 3 ? args[3] : "proto/t1_kernel.metal"

guard let raw = FileManager.default.contents(atPath: corpusPath) else { die("cannot read \(corpusPath)") }
let bytes = [UInt8](raw)

func u32(_ off: Int) -> UInt32 {
    return UInt32(bytes[off]) | (UInt32(bytes[off+1])<<8) | (UInt32(bytes[off+2])<<16) | (UInt32(bytes[off+3])<<24)
}

// Parse corpus -> packed cdata (+0xFFFF pad), descriptors, oracle
var cdata = [UInt8]()
var desc  = [UInt32]()            // 8 per cblk
var oracle = [Int32]()
var outTotal = 0
var off = 0
var ncblk = 0
var maxFlag = 0
while off < bytes.count {
    let orient = u32(off); let _ = u32(off+4); let _ = u32(off+8)
    let numbps = u32(off+12); let w = u32(off+16); let h = u32(off+20); let nseg = u32(off+24)
    off += 28
    var passes: UInt32 = 0
    for _ in 0..<Int(nseg) { let _ = u32(off); passes += u32(off+4); off += 8 }
    let tot = u32(off); off += 4
    let dataOff = UInt32(cdata.count)
    cdata.append(contentsOf: bytes[off..<off+Int(tot)]); off += Int(tot)
    cdata.append(0xFF); cdata.append(0xFF)               // synthetic marker
    let n = Int(w)*Int(h)
    oracle.append(contentsOf: (0..<n).map { Int32(bitPattern: u32(off + 4*$0)) }); off += 4*n
    desc.append(contentsOf: [dataOff, tot, w, h, numbps, orient, passes, UInt32(outTotal)])
    outTotal += n
    let flagsz = ((Int(h)+3)/4 + 2) * (Int(w)+2)
    if flagsz > maxFlag { maxFlag = flagsz }
    ncblk += 1
}
let flgstride = maxFlag
print("parsed \(ncblk) codeblocks, cdata \(cdata.count) B, out \(outTotal) coeffs, flgstride \(flgstride)")

// Metal setup
guard let dev = MTLCreateSystemDefaultDevice() else { die("no Metal device") }
print("GPU: \(dev.name)")
guard let lutsSrc = try? String(contentsOfFile: lutsPath, encoding: .utf8) else { die("read luts") }
guard let kernSrc = try? String(contentsOfFile: kernelPath, encoding: .utf8) else { die("read kernel") }
let src = lutsSrc + "\n" + kernSrc
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: nil) } catch { die("compile: \(error)") }
guard let fn = lib.makeFunction(name: "t1_decode") else { die("no t1_decode") }
let pso = try! dev.makeComputePipelineState(function: fn)
guard let q = dev.makeCommandQueue() else { die("no queue") }

// Buffers (shared = unified memory, zero-copy)
func buf<T>(_ a: [T]) -> MTLBuffer { a.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! } }
let cdataBuf = buf(cdata)
let descBuf  = buf(desc)
let outBuf   = dev.makeBuffer(length: max(1,outTotal)*MemoryLayout<Int32>.size, options: .storageModeShared)!
let flgBuf   = dev.makeBuffer(length: max(1,ncblk*flgstride)*MemoryLayout<UInt32>.size, options: .storageModePrivate)!
var nc = UInt32(ncblk); var fs = UInt32(flgstride)

func dispatch() -> Double {
    let cb = q.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(cdataBuf, offset: 0, index: 0)
    enc.setBuffer(descBuf,  offset: 0, index: 1)
    enc.setBuffer(outBuf,   offset: 0, index: 2)
    enc.setBuffer(flgBuf,   offset: 0, index: 3)
    enc.setBytes(&nc, length: 4, index: 4)
    enc.setBytes(&fs, length: 4, index: 5)
    let tpt = min(64, pso.maxTotalThreadsPerThreadgroup)
    enc.dispatchThreads(MTLSize(width: ncblk, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tpt, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    return (cb.gpuEndTime - cb.gpuStartTime) * 1000.0   // ms
}

// Correctness
_ = dispatch()
let outPtr = outBuf.contents().bindMemory(to: Int32.self, capacity: outTotal)
var fails = 0, firstBad = -1
for i in 0..<outTotal { if outPtr[i] != oracle[i] { if firstBad < 0 { firstBad = i }; fails += 1 } }
if fails == 0 { print("BIT-EXACT: all \(outTotal) coeffs match oracle ✓") }
else { print("MISMATCH: \(fails)/\(outTotal) coeffs differ, first @\(firstBad) got \(outPtr[firstBad]) want \(oracle[firstBad])") }

// Timing: GPU time per dispatch, best of N
let iters = 50
var best = Double.greatestFiniteMagnitude, sum = 0.0
for _ in 0..<iters { let t = dispatch(); best = min(best, t); sum += t }
let avg = sum/Double(iters)
print(String(format: "GPU T1 decode: best %.3f ms, avg %.3f ms for %d cblks (%.3f us/cblk)",
             best, avg, ncblk, best*1000.0/Double(ncblk)))
print(String(format: "GPU throughput: %.2f Mcblk/s  (frame of %d cblks => %.3f ms/frame)",
             Double(ncblk)/best/1000.0, ncblk, best))
