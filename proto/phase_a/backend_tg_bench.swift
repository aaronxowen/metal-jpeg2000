// Benchmark the THREADGROUP-MEMORY GPU back-end vs the oracle (FEASIBILITY §18 follow-up).
// Mirrors backend_opt_bench.swift but dispatches one THREADGROUP per line with the lifting
// scratch in threadgroup memory (backend_tg.metal) — no device-memory Wpool at all.
//
// Build: swiftc -O proto/phase_a/backend_tg_bench.swift -o /tmp/backend_tg_bench -framework Metal -framework Foundation
// Run:   /tmp/backend_tg_bench proto/backend_corpus.bin proto/phase_a/backend_tg.metal [threadsPerGroup]
// (threadsPerGroup default 256 — sweep 64/128/256/512 when tuning on the M1.)

import Foundation
import Metal

func die(_ s: String) -> Never { FileHandle.standardError.write((s+"\n").data(using:.utf8)!); exit(1) }
let args = CommandLine.arguments
let corpusPath = args.count > 1 ? args[1] : "proto/backend_corpus.bin"
let kernelPath = args.count > 2 ? args[2] : "proto/phase_a/backend_tg.metal"
let tgThreads  = args.count > 3 ? (Int(args[3]) ?? 256) : 256

guard let raw = FileManager.default.contents(atPath: corpusPath) else { die("read corpus") }
let dchars = [UInt8](raw)
func u32(_ o: Int) -> UInt32 { UInt32(dchars[o]) | (UInt32(dchars[o+1])<<8) | (UInt32(dchars[o+2])<<16) | (UInt32(dchars[o+3])<<24) }
func i32(_ o: Int) -> Int32 { Int32(bitPattern: u32(o)) }

var off = 0
if u32(off) != 0x444E4B42 { die("bad magic") }; off += 4
let nc = Int(u32(off)); off += 4
let w = Int(u32(off)); let h = Int(u32(off+4)); off += 8
let numres = Int(u32(off)); off += 4
var boxes = [Int32](); for k in 0..<(numres*4) { boxes.append(i32(off+4*k)) }; off += 16*numres
var prec=[Int](), sgnd=[Int](), dcs=[Int]()
for _ in 0..<nc { prec.append(Int(i32(off))); sgnd.append(Int(i32(off+4))); dcs.append(Int(i32(off+8))); off += 12 }
let mct = Int(i32(off)); off += 4
let n = w*h
// concatenated input (nc*n floats) and oracle (nc*n ints)
var inAll = [Float](repeating: 0, count: nc*n)
for c in 0..<nc { for i in 0..<n { inAll[c*n + i] = Float(bitPattern: u32(off + 4*i)) }; off += 4*n }
var oracle = [Int32](repeating: 0, count: nc*n)
for c in 0..<nc { for i in 0..<n { oracle[c*n + i] = i32(off + 4*i) }; off += 4*n }

guard let dev = MTLCreateSystemDefaultDevice() else { die("no device") }
print("GPU: \(dev.name)  \(w)x\(h), \(nc) comps, \(numres) res, mct=\(mct)  [THREADGROUP back-end, T=\(tgThreads)]")
guard let ksrc = try? String(contentsOfFile: kernelPath, encoding: .utf8) else { die("read kernel") }
let opts = MTLCompileOptions(); opts.fastMathEnabled = false
guard let lib = try? dev.makeLibrary(source: ksrc, options: opts) else { die("compile") }
let psoH = try! dev.makeComputePipelineState(function: lib.makeFunction(name:"idwt97_h_tg")!)
let psoV = try! dev.makeComputePipelineState(function: lib.makeFunction(name:"idwt97_v_tg")!)
let psoF = try! dev.makeComputePipelineState(function: lib.makeFunction(name:"finalize_b")!)
let q = dev.makeCommandQueue()!

func tgBytes(_ len: Int) -> Int { (len * 4 + 15) / 16 * 16 }
if tgBytes(max(w,h)) > dev.maxThreadgroupMemoryLength { die("line too long for threadgroup memory") }

let dataBuf = dev.makeBuffer(length: nc*n*4, options: .storageModeShared)!
let outBuf  = dev.makeBuffer(length: nc*n*4, options: .storageModeShared)!
let boxBuf  = boxes.withUnsafeBytes { dev.makeBuffer(bytes:$0.baseAddress!, length:$0.count, options:.storageModeShared)! }

struct FinalizeParams { var n: UInt32 = 0; var mct: Int32 = 0
    var dc=(Int32(0),Int32(0),Int32(0),Int32(0)); var lo=(Int32(0),Int32(0),Int32(0),Int32(0)); var hi=(Int32(0),Int32(0),Int32(0),Int32(0)) }
var fp = FinalizeParams(); fp.n = UInt32(n); fp.mct = Int32(mct)
func lo(_ c:Int)->Int32 { sgnd[c] != 0 ? Int32(-(1<<(prec[c]-1))) : 0 }
func hi(_ c:Int)->Int32 { sgnd[c] != 0 ? Int32((1<<(prec[c]-1))-1) : Int32((1<<prec[c])-1) }
fp.dc=(Int32(dcs[0]),Int32(dcs[1]),nc>2 ? Int32(dcs[2]):0,0)
fp.lo=(lo(0),lo(1),nc>2 ? lo(2):0,0); fp.hi=(hi(0),hi(1),nc>2 ? hi(2):0,0)
var wv=UInt32(w); var cs=UInt32(n)

func run() -> Double {
    inAll.withUnsafeBytes { memcpy(dataBuf.contents(), $0.baseAddress!, nc*n*4) }
    let cb = q.makeCommandBuffer()!
    for lvl in 1..<numres {
        let x0=Int(boxes[lvl*4+0]), y0=Int(boxes[lvl*4+1])
        let rw=Int(boxes[lvl*4+2])-x0, rh=Int(boxes[lvl*4+3])-y0
        var lv=UInt32(lvl); var rhv=UInt32(rh); var rwv=UInt32(rw)
        let eh=cb.makeComputeCommandEncoder()!; eh.setComputePipelineState(psoH)
        eh.setBuffer(dataBuf,offset:0,index:0); eh.setBuffer(boxBuf,offset:0,index:1)
        eh.setBytes(&wv,length:4,index:2); eh.setBytes(&lv,length:4,index:3)
        eh.setBytes(&cs,length:4,index:4); eh.setBytes(&rhv,length:4,index:5)
        eh.setThreadgroupMemoryLength(tgBytes(rw), index: 0)
        eh.dispatchThreadgroups(MTLSize(width:nc*rh,height:1,depth:1), threadsPerThreadgroup:MTLSize(width:min(tgThreads,psoH.maxTotalThreadsPerThreadgroup),height:1,depth:1)); eh.endEncoding()
        let ev=cb.makeComputeCommandEncoder()!; ev.setComputePipelineState(psoV)
        ev.setBuffer(dataBuf,offset:0,index:0); ev.setBuffer(boxBuf,offset:0,index:1)
        ev.setBytes(&wv,length:4,index:2); ev.setBytes(&lv,length:4,index:3)
        ev.setBytes(&cs,length:4,index:4); ev.setBytes(&rwv,length:4,index:5)
        ev.setThreadgroupMemoryLength(tgBytes(rh), index: 0)
        ev.dispatchThreadgroups(MTLSize(width:nc*rw,height:1,depth:1), threadsPerThreadgroup:MTLSize(width:min(tgThreads,psoV.maxTotalThreadsPerThreadgroup),height:1,depth:1)); ev.endEncoding()
    }
    let ef=cb.makeComputeCommandEncoder()!; ef.setComputePipelineState(psoF)
    ef.setBuffer(dataBuf,offset:0,index:0); ef.setBuffer(outBuf,offset:0,index:1)
    ef.setBytes(&fp,length:MemoryLayout<FinalizeParams>.stride,index:2); ef.setBytes(&cs,length:4,index:3)
    ef.dispatchThreads(MTLSize(width:n,height:1,depth:1), threadsPerThreadgroup:MTLSize(width:min(256,psoF.maxTotalThreadsPerThreadgroup),height:1,depth:1)); ef.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    return (cb.gpuEndTime - cb.gpuStartTime)*1000.0
}

_ = run()
let p = outBuf.contents().bindMemory(to: Int32.self, capacity: nc*n)
var diff = 0; for i in 0..<(nc*n) { if p[i] != oracle[i] { diff += 1 } }
print(diff==0 ? "INTEGER-EXACT vs oracle ✓" : "MISMATCH: \(diff) samples ✗")

var best = Double.greatestFiniteMagnitude
for _ in 0..<30 { best = min(best, run()) }
print(String(format: "THREADGROUP GPU back-end: %.3f ms/frame", best))
