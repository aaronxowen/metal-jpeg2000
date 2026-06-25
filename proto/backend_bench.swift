// Headless Metal benchmark for the full GPU back-end: inverse 9/7 DWT (3 comps)
// + inverse ICT + DC level-shift -> final integer image. Validates integer-exact
// vs OpenJPEG (OPJ_BACKEND_DUMP corpus).
//
// Build: swiftc -O proto/backend_bench.swift -o /tmp/backend_bench -framework Metal -framework Foundation
// Run:   /tmp/backend_bench proto/backend_corpus.bin proto/backend_kernel.metal

import Foundation
import Metal

func die(_ s: String) -> Never { FileHandle.standardError.write((s+"\n").data(using:.utf8)!); exit(1) }
let args = CommandLine.arguments
let corpusPath = args.count > 1 ? args[1] : "proto/backend_corpus.bin"
let kernelPath = args.count > 2 ? args[2] : "proto/backend_kernel.metal"

guard let raw = FileManager.default.contents(atPath: corpusPath) else { die("read corpus") }
let d = [UInt8](raw)
func u32(_ o: Int) -> UInt32 { UInt32(d[o]) | (UInt32(d[o+1])<<8) | (UInt32(d[o+2])<<16) | (UInt32(d[o+3])<<24) }
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
var inp = [[Float]]()
for _ in 0..<nc {
    var a = [Float](repeating: 0, count: n)
    for i in 0..<n { let bits: UInt32 = u32(off + 4*i); a[i] = Float(bitPattern: bits) }
    off += 4*n
    inp.append(a)
}
var oracle = [[Int32]]()
for _ in 0..<nc {
    var a = [Int32](repeating: 0, count: n)
    for i in 0..<n { a[i] = i32(off + 4*i) }
    off += 4*n
    oracle.append(a)
}

guard let dev = MTLCreateSystemDefaultDevice() else { die("no device") }
print("GPU: \(dev.name)  frame \(w)x\(h), \(nc) comps, \(numres) res, mct=\(mct)")
guard let ksrc = try? String(contentsOfFile: kernelPath, encoding: .utf8) else { die("read kernel") }
let opts = MTLCompileOptions(); opts.fastMathEnabled = false
guard let lib = try? dev.makeLibrary(source: ksrc, options: opts) else { die("compile") }
let psoH = try! dev.makeComputePipelineState(function: lib.makeFunction(name:"idwt97_h")!)
let psoV = try! dev.makeComputePipelineState(function: lib.makeFunction(name:"idwt97_v")!)
let psoF = try! dev.makeComputePipelineState(function: lib.makeFunction(name:"backend_finalize")!)
let q = dev.makeCommandQueue()!

let stride = max(w,h)
let boxBuf = boxes.withUnsafeBytes { dev.makeBuffer(bytes:$0.baseAddress!, length:$0.count, options:.storageModeShared)! }
var dataBufs = [MTLBuffer](); for c in 0..<nc { dataBufs.append(dev.makeBuffer(length:n*4, options:.storageModeShared)!) }
var outBufs  = [MTLBuffer](); for _ in 0..<nc { outBufs.append(dev.makeBuffer(length:n*4, options:.storageModeShared)!) }
let wpool = dev.makeBuffer(length: stride*stride*4, options:.storageModePrivate)!

// FinalizeParams: n, mct, dc[4], lo[4], hi[4]  (matches MSL struct layout)
struct FinalizeParams { var n: UInt32; var mct: Int32
    var dc=(Int32(0),Int32(0),Int32(0),Int32(0)); var lo=(Int32(0),Int32(0),Int32(0),Int32(0)); var hi=(Int32(0),Int32(0),Int32(0),Int32(0)) }
var fp = FinalizeParams(n: UInt32(n), mct: Int32(mct))
let dcA=(Int32(dcs[0]),Int32(dcs[1]),Int32(nc>2 ? dcs[2]:0),Int32(0)); fp.dc=dcA
func loFor(_ c:Int)->Int32 { sgnd[c] != 0 ? Int32(-(1<<(prec[c]-1))) : 0 }
func hiFor(_ c:Int)->Int32 { sgnd[c] != 0 ? Int32((1<<(prec[c]-1))-1) : Int32((1<<prec[c])-1) }
fp.lo=(loFor(0),loFor(1),nc>2 ? loFor(2):0,0)
fp.hi=(hiFor(0),hiFor(1),nc>2 ? hiFor(2):0,0)
var wv = UInt32(w)

func run() -> Double {
    for c in 0..<nc { inp[c].withUnsafeBytes { memcpy(dataBufs[c].contents(), $0.baseAddress!, n*4) } }
    let cb = q.makeCommandBuffer()!
    for c in 0..<nc {
        for lvl in 1..<numres {
            let x0=Int(boxes[lvl*4+0]), y0=Int(boxes[lvl*4+1])
            let rw=Int(boxes[lvl*4+2])-x0, rh=Int(boxes[lvl*4+3])-y0
            var lv=UInt32(lvl); var ws=UInt32(stride)
            let eh=cb.makeComputeCommandEncoder()!; eh.setComputePipelineState(psoH)
            eh.setBuffer(dataBufs[c],offset:0,index:0); eh.setBuffer(boxBuf,offset:0,index:1); eh.setBuffer(wpool,offset:0,index:2)
            eh.setBytes(&wv,length:4,index:3); eh.setBytes(&lv,length:4,index:4); eh.setBytes(&ws,length:4,index:5)
            eh.dispatchThreads(MTLSize(width:rh,height:1,depth:1), threadsPerThreadgroup:MTLSize(width:min(64,psoH.maxTotalThreadsPerThreadgroup),height:1,depth:1)); eh.endEncoding()
            let ev=cb.makeComputeCommandEncoder()!; ev.setComputePipelineState(psoV)
            ev.setBuffer(dataBufs[c],offset:0,index:0); ev.setBuffer(boxBuf,offset:0,index:1); ev.setBuffer(wpool,offset:0,index:2)
            ev.setBytes(&wv,length:4,index:3); ev.setBytes(&lv,length:4,index:4); ev.setBytes(&ws,length:4,index:5)
            ev.dispatchThreads(MTLSize(width:rw,height:1,depth:1), threadsPerThreadgroup:MTLSize(width:min(64,psoV.maxTotalThreadsPerThreadgroup),height:1,depth:1)); ev.endEncoding()
        }
    }
    let ef=cb.makeComputeCommandEncoder()!; ef.setComputePipelineState(psoF)
    for c in 0..<nc { ef.setBuffer(dataBufs[c],offset:0,index:c) }
    for c in 0..<nc { ef.setBuffer(outBufs[c],offset:0,index:3+c) }
    ef.setBytes(&fp, length: MemoryLayout<FinalizeParams>.stride, index:6)
    ef.dispatchThreads(MTLSize(width:n,height:1,depth:1), threadsPerThreadgroup:MTLSize(width:min(256,psoF.maxTotalThreadsPerThreadgroup),height:1,depth:1)); ef.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    return (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
}

_ = run()
var ndiff=0, maxd=0
for c in 0..<nc {
    let p = outBufs[c].contents().bindMemory(to: Int32.self, capacity: n)
    for i in 0..<n { if p[i] != oracle[c][i] { ndiff += 1; let dd=Int(abs(p[i]-oracle[c][i])); if dd>maxd {maxd=dd} } }
}
print("samples: \(nc*n)  differ: \(ndiff)  max |diff|: \(maxd)")
print(ndiff==0 ? "INTEGER-EXACT vs OpenJPEG final image ✓" : (maxd<=1 ? "off-by-<=1 (\(ndiff) px)" : "MISMATCH"))

var best = Double.greatestFiniteMagnitude
for _ in 0..<20 { best = min(best, run()) }
print(String(format: "GPU full back-end (iDWT+ICT+level-shift): %.3f ms/frame", best))
