// Phase A step 2 — standalone overlapped hybrid decode harness.
//
// Pipeline per frame: CPU decodes a J2K codestream through Tier-1 only (via the
// fork's opj_set_t1_output_callback), the GPU runs the back-end (iDWT + inverse
// ICT + DC level-shift) on the previous frame's post-T1 buffers — overlapped, so
// steady-state frame time = max(CPU T1, GPU back-end). Reports sustained fps and
// validates the GPU output integer-exact vs an oracle.
//
// Build (from repo root):
//   swiftc -O proto/phase_a/hybrid_harness.swift \
//     -import-objc-header proto/phase_a/bridge.h \
//     -I src/lib/openjp2 -I build/src/lib/openjp2 \
//     -L build/bin -lopenjp2 -Xlinker -rpath -Xlinker "$(pwd)/build/bin" \
//     -framework Metal -framework Foundation -o /tmp/hybrid_harness
// Run:
//   /tmp/hybrid_harness <frame>.j2c proto/backend_kernel.metal [frames] [proto/backend_corpus.bin]

import Foundation
import Metal

func die(_ s: String) -> Never { FileHandle.standardError.write((s+"\n").data(using:.utf8)!); exit(1) }

struct FinalizeParams { var n: UInt32 = 0; var mct: Int32 = 0
    var dc = (Int32(0),Int32(0),Int32(0),Int32(0))
    var lo = (Int32(0),Int32(0),Int32(0),Int32(0))
    var hi = (Int32(0),Int32(0),Int32(0),Int32(0)) }

final class Slot {
    var comp: MTLBuffer!   // nc components concatenated (post-T1 floats; iDWT in place)
    var out:  MTLBuffer!   // nc components concatenated (final image ints)
}

final class HybridDecoder {
    // Metal
    let dev: MTLDevice
    let queue: MTLCommandQueue
    let psoH: MTLComputePipelineState, psoV: MTLComputePipelineState, psoF: MTLComputePipelineState
    // geometry (captured on first T1 callback)
    var nc = 0, w = 0, h = 0, numres = 0, mct: Int32 = 0
    var boxes: [Int32] = [], prec: [Int32] = [], sgnd: [Int32] = [], dcs: [Int32] = []
    var ready = false
    // resources
    var slots: [Slot] = []
    var boxBuf: MTLBuffer!
    var wpool: MTLBuffer!
    var fp = FinalizeParams()
    var fillSlot = 0
    var n = 0

    init(kernelPath: String) {
        guard let d = MTLCreateSystemDefaultDevice() else { die("no Metal device") }
        dev = d
        guard let ksrc = try? String(contentsOfFile: kernelPath, encoding: .utf8) else { die("read kernel") }
        let opts = MTLCompileOptions(); opts.fastMathEnabled = false
        guard let lib = try? d.makeLibrary(source: ksrc, options: opts) else { die("compile kernel") }
        psoH = try! d.makeComputePipelineState(function: lib.makeFunction(name: "idwt97_h_b")!)
        psoV = try! d.makeComputePipelineState(function: lib.makeFunction(name: "idwt97_v_b")!)
        psoF = try! d.makeComputePipelineState(function: lib.makeFunction(name: "finalize_b")!)
        queue = d.makeCommandQueue()!
    }

    // Called from the C decode at the T1/back-end split point.
    func onT1(_ info: opj_t1_output_t) {
        if !ready {
            nc = Int(info.numcomps); w = Int(info.w); h = Int(info.h); numres = Int(info.numres)
            mct = info.mct
            boxes = Array(UnsafeBufferPointer(start: info.boxes, count: numres*4))
            prec  = Array(UnsafeBufferPointer(start: info.prec,  count: nc))
            sgnd  = Array(UnsafeBufferPointer(start: info.sgnd,  count: nc))
            dcs   = Array(UnsafeBufferPointer(start: info.dc_shift, count: nc))
            n = w*h
            allocate()
            ready = true
        }
        let slot = slots[fillSlot]
        let base = slot.comp.contents()
        for c in 0..<nc {
            if let src = info.comp_data[c] {
                memcpy(base + c * n * MemoryLayout<Int32>.size, src, n * MemoryLayout<Int32>.size)
            }
        }
    }

    func allocate() {
        let stride = max(w, h)
        boxBuf = boxes.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        wpool = dev.makeBuffer(length: nc*stride*stride*4, options: .storageModePrivate)!
        for _ in 0..<2 {
            let s = Slot()
            s.comp = dev.makeBuffer(length: nc*n*4, options: .storageModeShared)!
            s.out  = dev.makeBuffer(length: nc*n*4, options: .storageModeShared)!
            slots.append(s)
        }
        fp.n = UInt32(n); fp.mct = mct
        func lo(_ c: Int) -> Int32 { sgnd[c] != 0 ? Int32(-(1 << (prec[c]-1))) : 0 }
        func hi(_ c: Int) -> Int32 { sgnd[c] != 0 ? Int32((1 << (prec[c]-1))-1) : Int32((1 << prec[c])-1) }
        fp.dc = (Int32(dcs[0]), Int32(dcs[1]), nc>2 ? Int32(dcs[2]):0, 0)
        fp.lo = (lo(0), lo(1), nc>2 ? lo(2):0, 0)
        fp.hi = (hi(0), hi(1), nc>2 ? hi(2):0, 0)
    }

    func encodeBackend(_ slotIdx: Int) -> MTLCommandBuffer {
        let s = slots[slotIdx]
        let cb = queue.makeCommandBuffer()!
        var wv = UInt32(w); let stride = max(w, h); var cs = UInt32(n)
        for lvl in 1..<numres {
            let x0 = Int(boxes[lvl*4+0]), y0 = Int(boxes[lvl*4+1])
            let rw = Int(boxes[lvl*4+2]) - x0, rh = Int(boxes[lvl*4+3]) - y0
            var lv = UInt32(lvl); var ws = UInt32(stride); var rhv = UInt32(rh); var rwv = UInt32(rw)
            let eh = cb.makeComputeCommandEncoder()!; eh.setComputePipelineState(psoH)
            eh.setBuffer(s.comp, offset:0, index:0); eh.setBuffer(boxBuf, offset:0, index:1); eh.setBuffer(wpool, offset:0, index:2)
            eh.setBytes(&wv,length:4,index:3); eh.setBytes(&lv,length:4,index:4); eh.setBytes(&ws,length:4,index:5)
            eh.setBytes(&cs,length:4,index:6); eh.setBytes(&rhv,length:4,index:7)
            eh.dispatchThreads(MTLSize(width:nc*rh,height:1,depth:1), threadsPerThreadgroup:MTLSize(width:min(64,psoH.maxTotalThreadsPerThreadgroup),height:1,depth:1)); eh.endEncoding()
            let ev = cb.makeComputeCommandEncoder()!; ev.setComputePipelineState(psoV)
            ev.setBuffer(s.comp, offset:0, index:0); ev.setBuffer(boxBuf, offset:0, index:1); ev.setBuffer(wpool, offset:0, index:2)
            ev.setBytes(&wv,length:4,index:3); ev.setBytes(&lv,length:4,index:4); ev.setBytes(&ws,length:4,index:5)
            ev.setBytes(&cs,length:4,index:6); ev.setBytes(&rwv,length:4,index:7)
            ev.dispatchThreads(MTLSize(width:nc*rw,height:1,depth:1), threadsPerThreadgroup:MTLSize(width:min(64,psoV.maxTotalThreadsPerThreadgroup),height:1,depth:1)); ev.endEncoding()
        }
        let ef = cb.makeComputeCommandEncoder()!; ef.setComputePipelineState(psoF)
        ef.setBuffer(s.comp, offset:0, index:0); ef.setBuffer(s.out, offset:0, index:1)
        ef.setBytes(&fp, length: MemoryLayout<FinalizeParams>.stride, index:2); ef.setBytes(&cs,length:4,index:3)
        ef.dispatchThreads(MTLSize(width:n,height:1,depth:1), threadsPerThreadgroup:MTLSize(width:min(256,psoF.maxTotalThreadsPerThreadgroup),height:1,depth:1)); ef.endEncoding()
        return cb
    }
}

// ----- C interop: decode one J2K frame through Tier-1 into the decoder's current slot -----
let t1cb: @convention(c) (UnsafePointer<opj_t1_output_t>?, UnsafeMutableRawPointer?) -> Void = { infoPtr, user in
    guard let info = infoPtr?.pointee, let user = user else { return }
    Unmanaged<HybridDecoder>.fromOpaque(user).takeUnretainedValue().onT1(info)
}

var decodeThreads: Int32 = Int32(ProcessInfo.processInfo.activeProcessorCount)

func decodeToT1(_ path: String) {
    guard let stream = opj_stream_create_default_file_stream(path, 1) else { die("stream") }
    let codec = opj_create_decompress(OPJ_CODEC_J2K)
    var params = opj_dparameters_t(); opj_set_default_decoder_parameters(&params)
    _ = opj_setup_decoder(codec, &params)
    _ = opj_codec_set_threads(codec, decodeThreads)   // multithreaded T1 (the hybrid's CPU stage)
    var image: UnsafeMutablePointer<opj_image_t>? = nil
    if opj_read_header(stream, codec, &image) == 0 { die("read_header") }
    if opj_decode(codec, stream, image) == 0 { die("decode") }
    _ = opj_end_decompress(codec, stream)
    opj_image_destroy(image)
    opj_destroy_codec(codec)
    opj_stream_destroy(stream)
}

// ----- main -----
let args = CommandLine.arguments
guard args.count >= 3 else { die("usage: hybrid_harness <frame>.j2c <backend_kernel.metal> [frames] [oracle.bin]") }
let j2cPath = args[1]
let kernelPath = args[2]
let nFrames = args.count > 3 ? Int(args[3])! : 120
let oraclePath = args.count > 4 ? args[4] : nil

let ctx = HybridDecoder(kernelPath: kernelPath)
opj_set_t1_output_callback(t1cb, Unmanaged.passUnretained(ctx).toOpaque(), 1)

// Prime slot 0 (captures geometry + allocates).
ctx.fillSlot = 0
decodeToT1(j2cPath)
print("GPU: \(ctx.dev.name)  frame \(ctx.w)x\(ctx.h), \(ctx.nc) comps, \(ctx.numres) res, mct=\(ctx.mct)")
print("frames: \(nFrames)")

// Overlapped pipeline: commit GPU back-end for frame i (async), decode frame i+1 (CPU) concurrently, then wait.
let t0 = DispatchTime.now().uptimeNanoseconds
for i in 0..<nFrames {
    let cur = i % 2, nxt = (i + 1) % 2
    let cb = ctx.encodeBackend(cur)
    cb.commit()
    if i + 1 < nFrames { ctx.fillSlot = nxt; decodeToT1(j2cPath) }
    cb.waitUntilCompleted()
}
let t1 = DispatchTime.now().uptimeNanoseconds
let ms = Double(t1 - t0) / 1e6
let fps = Double(nFrames) / (ms / 1000.0)
print(String(format: "\nOVERLAPPED hybrid: %.1f ms for %d frames => %.3f ms/frame, %.1f fps", ms, nFrames, ms/Double(nFrames), fps))

// Calibration: CPU stage (decode-to-T1) and GPU stage (back-end) => max() bounds the pipeline.
ctx.fillSlot = 0
var cpuMs = 0.0
do {
    let c0 = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<nFrames { decodeToT1(j2cPath) }
    let c1 = DispatchTime.now().uptimeNanoseconds
    cpuMs = Double(c1 - c0)/1e6/Double(nFrames)
    print(String(format: "  (calibration) CPU decode-to-T1: %.3f ms/frame", cpuMs))
}
var gbest = Double.greatestFiniteMagnitude
for _ in 0..<30 { let cb = ctx.encodeBackend(0); cb.commit(); cb.waitUntilCompleted(); gbest = min(gbest, (cb.gpuEndTime - cb.gpuStartTime)*1000.0) }
print(String(format: "  (calibration) GPU back-end:     %.3f ms/frame", gbest))
print(String(format: "  max(CPU, GPU) = %.3f ms/frame (overlap target)", max(cpuMs, gbest)))

// Clean correctness pass: one fresh decode-to-T1 + one back-end (the calibration
// loops leave slot state transformed-in-place, so re-run cleanly before validating).
ctx.fillSlot = 0
decodeToT1(j2cPath)
do { let cb = ctx.encodeBackend(0); cb.commit(); cb.waitUntilCompleted() }
opj_set_t1_output_callback(nil, nil, 0)

// Validation vs oracle final integer image (last nc*n int buffers of backend_corpus.bin).
if let op = oraclePath, let data = FileManager.default.contents(atPath: op) {
    let bytes = [UInt8](data)
    func u32(_ o: Int) -> UInt32 { UInt32(bytes[o]) | (UInt32(bytes[o+1])<<8) | (UInt32(bytes[o+2])<<16) | (UInt32(bytes[o+3])<<24) }
    let outBytes = ctx.nc * ctx.n * 4
    var off = bytes.count - outBytes
    var diff = 0
    let p = ctx.slots[0].out.contents().bindMemory(to: Int32.self, capacity: ctx.nc * ctx.n)
    for c in 0..<ctx.nc {
        for i in 0..<ctx.n { if p[c*ctx.n + i] != Int32(bitPattern: u32(off + 4*i)) { diff += 1 } }
        off += ctx.n*4
    }
    print(diff == 0 ? "VALIDATION: integer-exact vs oracle ✓" : "VALIDATION: \(diff) samples differ ✗")
}

print(String(format: "\nbudget 41.6 ms/frame (2K@24): overlapped %.3f ms => %@",
             ms/Double(nFrames), fps >= 24.0 ? "MEETS 24fps ✓" : "below 24fps"))
