// Headless Metal benchmark for the inverse 9/7 DWT kernel.
// Reads proto/dwt_corpus.bin, runs the multi-level H/V transform per component,
// validates vs oracle (float tolerance), reports GPU vs CPU timing.
//
// Build: swiftc -O proto/dwt_bench.swift -o /tmp/dwt_bench -framework Metal -framework Foundation
// Run:   /tmp/dwt_bench proto/dwt_corpus.bin proto/dwt_kernel.metal

import Foundation
import Metal

func die(_ s: String) -> Never { FileHandle.standardError.write((s+"\n").data(using:.utf8)!); exit(1) }

let args = CommandLine.arguments
let corpusPath = args.count > 1 ? args[1] : "proto/dwt_corpus.bin"
let kernelPath = args.count > 2 ? args[2] : "proto/dwt_kernel.metal"

guard let raw = FileManager.default.contents(atPath: corpusPath) else { die("cannot read \(corpusPath)") }
let data = [UInt8](raw)
func u32(_ o: Int) -> UInt32 { UInt32(data[o]) | (UInt32(data[o+1])<<8) | (UInt32(data[o+2])<<16) | (UInt32(data[o+3])<<24) }
func i32(_ o: Int) -> Int32 { Int32(bitPattern: u32(o)) }
func f32(_ o: Int) -> Float { Float(bitPattern: u32(o)) }

guard let dev = MTLCreateSystemDefaultDevice() else { die("no Metal device") }
print("GPU: \(dev.name)")
guard let ksrc = try? String(contentsOfFile: kernelPath, encoding: .utf8) else { die("read kernel") }
let opts = MTLCompileOptions(); opts.fastMathEnabled = false
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: ksrc, options: opts) } catch { die("compile: \(error)") }
let psoH = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "idwt97_h")!)
let psoV = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "idwt97_v")!)
let q = dev.makeCommandQueue()!

var off = 0, rec = 0
var maxabs = 0.0, ndiff = 0, ntot = 0
var gpuMsTotal = 0.0
while off < data.count {
    let magic = u32(off); off += 4
    if magic != 0x31545744 { die("bad magic @\(off)") }
    let numres = Int(u32(off)); off += 4
    var boxes = [Int32](); for k in 0..<(numres*4) { boxes.append(i32(off + 4*k)) }; off += 16*numres
    let w = Int(u32(off)); let h = Int(u32(off+4)); off += 8
    let n = w*h
    var input = [Float](repeating: 0, count: n); for i in 0..<n { input[i] = f32(off + 4*i) }; off += 4*n
    var oracle = [Float](repeating: 0, count: n); for i in 0..<n { oracle[i] = f32(off + 4*i) }; off += 4*n

    let dataBuf = dev.makeBuffer(bytes: &input, length: n*4, options: .storageModeShared)!
    let boxBuf  = boxes.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
    let stride = max(w, h)
    let wpool  = dev.makeBuffer(length: stride * stride * 4, options: .storageModePrivate)!
    var wv = UInt32(w)

    func run() -> Double {
        // reset data to input each run
        memcpy(dataBuf.contents(), &input, n*4)
        let cb = q.makeCommandBuffer()!
        for lvl in 1..<numres {
            let x0 = Int(boxes[lvl*4+0]), y0 = Int(boxes[lvl*4+1])
            let rw = Int(boxes[lvl*4+2]) - x0, rh = Int(boxes[lvl*4+3]) - y0
            var lv = UInt32(lvl); var ws = UInt32(stride)
            // horizontal: rh threads
            let eh = cb.makeComputeCommandEncoder()!
            eh.setComputePipelineState(psoH)
            eh.setBuffer(dataBuf, offset: 0, index: 0); eh.setBuffer(boxBuf, offset: 0, index: 1)
            eh.setBuffer(wpool, offset: 0, index: 2)
            eh.setBytes(&wv, length: 4, index: 3); eh.setBytes(&lv, length: 4, index: 4); eh.setBytes(&ws, length: 4, index: 5)
            eh.dispatchThreads(MTLSize(width: rh, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(64, psoH.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            eh.endEncoding()
            // vertical: rw threads
            let ev = cb.makeComputeCommandEncoder()!
            ev.setComputePipelineState(psoV)
            ev.setBuffer(dataBuf, offset: 0, index: 0); ev.setBuffer(boxBuf, offset: 0, index: 1)
            ev.setBuffer(wpool, offset: 0, index: 2)
            ev.setBytes(&wv, length: 4, index: 3); ev.setBytes(&lv, length: 4, index: 4); ev.setBytes(&ws, length: 4, index: 5)
            ev.dispatchThreads(MTLSize(width: rw, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(64, psoV.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            ev.endEncoding()
        }
        cb.commit(); cb.waitUntilCompleted()
        return (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
    }

    _ = run()
    let out = dataBuf.contents().bindMemory(to: Float.self, capacity: n)
    for i in 0..<n { let d = abs(Double(out[i]) - Double(oracle[i])); if d != 0 { ndiff += 1; if d > maxabs { maxabs = d } } }
    ntot += n

    var best = Double.greatestFiniteMagnitude
    for _ in 0..<20 { best = min(best, run()) }
    gpuMsTotal += best
    print(String(format: "  component %d: %dx%d, %d res levels -> GPU iDWT %.3f ms", rec, w, h, numres, best))
    rec += 1
}
print("---")
print("records: \(rec)  coeffs: \(ntot)  differ: \(ndiff)  max abs diff: \(maxabs)")
print(String(format: "GPU iDWT: %.3f ms total (%.3f ms/component)", gpuMsTotal, gpuMsTotal/Double(rec)))
print(maxabs < 1e-2 ? "WITHIN TOLERANCE (rounds to same integer image) ✓" : "OUT OF TOLERANCE ✗")
