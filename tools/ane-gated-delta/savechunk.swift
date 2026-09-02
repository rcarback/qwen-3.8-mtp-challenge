// `savechunk` mode: compile a SINGLE-TILE gated-delta layer (NT=1, i.e.
// S == St) to a .mlmodelc on disk and print its path plus its FLOP/memory
// profile. Building with S == St reuses buildTiledLayer's per-tile MIL graph
// but drops the internal multi-tile loop, so the model takes external
// recurrence state (h0 in, hst1 out) for exactly one St-token chunk -- the
// unit an external cross-engine pipeline scheduler needs to drive chunk by
// chunk itself. Used by ANEMetalPipelineTests.swift, which cannot import
// this directory's loose sources directly (it is not part of the SwiftPM
// dependency graph), so it shells out to the compiled `gdprobe` binary once
// per (St, K) to materialize the artifact, then loads it via
// MLModel(contentsOf:).
import CoreML
import Foundation

@available(macOS 15.0, *)
func runSaveChunk() async {
    let args = CommandLine.arguments
    guard args.count >= 5, let St = Int(args[3]), let K = Int(args[4]) else {
        print("usage: gdprobe savechunk outDir St K")
        exit(2)
    }
    let outDir = args[2]
    let b = buildTiledLayer(S: St, St: St, K: K)
    do {
        try FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        let key = "gd_St\(St)_K\(K)"
        let specURL = URL(fileURLWithPath: outDir).appendingPathComponent("\(key).mlmodel")
        try b.spec.write(to: specURL)
        let compiled = try await MLModel.compileModel(at: specURL)
        let dest = URL(fileURLWithPath: outDir).appendingPathComponent("\(key).mlmodelc")
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.moveItem(at: compiled, to: dest)
        print(String(format: "SAVED\t%@\tusefulGF=%.4f\tmaxLiveMB=%.3f", dest.path, b.usefulGF, b.maxLiveMB))
    } catch {
        print("SAVE FAILED \(error)")
        exit(1)
    }
}
