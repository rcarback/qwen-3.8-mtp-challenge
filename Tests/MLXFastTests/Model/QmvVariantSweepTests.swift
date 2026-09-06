import Foundation
import MLX
import MLXFast
import MLXRandom
import Testing

@testable import MLXFastCore

/// B3: is the stock `qmv_fast` shape the reason q4 streams at 55 percent of
/// peak while bf16 gemv reaches 80?
///
/// The stock kernel (quantized.h `qmv_fast_impl`) fixes two simdgroups per
/// threadgroup, four output rows per simdgroup and two packs per lane. This
/// sweep re-implements the same arithmetic as a custom kernel with those three
/// knobs as template parameters, checks every variant against `quantizedMM`,
/// and times chains over distinct weights at the two shapes decode pays for:
/// the gated-delta `in_proj_qkv` [2560 -> 10240] and `lm_head` [2560 -> 248320].
///
///     MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test -c release \
///       --force-resolved-versions --filter qmvVariantSweep
@Suite(.serialized)
struct QmvVariantSweepTests {
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    static let kernel = MLXFast.metalKernel(
        name: "qmv_variant",
        inputNames: ["w", "scales", "biases", "x"],
        outputNames: ["y"],
        source: """
            constexpr int pack_factor = (BITS == 4) ? 8 : 4;
            constexpr int bytes_per_pack = 4;
            constexpr int values_per_thread = pack_factor * PACKS;
            constexpr int block_size = values_per_thread * 32;
            constexpr int scale_step_per_thread = GROUP / values_per_thread;

            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint tg_y = threadgroup_position_in_grid.y;

            const int in_vec_size_w = K * bytes_per_pack / pack_factor;
            const int in_vec_size_g = K / GROUP;
            const int out_row = (int)tg_y * (NUM_SG * ROWS) + (int)simd_gid * ROWS;
            if (out_row >= N) { return; }

            const device uint8_t* ws = (const device uint8_t*)w;
            const device uint8_t* wp = ws + out_row * in_vec_size_w + simd_lid * PACKS * bytes_per_pack;
            const device half* sp = scales + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
            const device half* bp = biases + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
            const device half* xp = x + simd_lid * values_per_thread;

            float x_thread[values_per_thread];
            float result[ROWS];
            for (int r = 0; r < ROWS; r++) { result[r] = 0.0f; }

            for (int k = 0; k < K; k += block_size) {
                const bool live = (k + (int)simd_lid * values_per_thread) < K;
                float sum = 0.0f;
                if (live) {
                    if (BITS == 4) {
                        for (int i = 0; i < values_per_thread; i += 4) {
                            float a = xp[i], b = xp[i + 1], c = xp[i + 2], d = xp[i + 3];
                            sum += a + b + c + d;
                            x_thread[i] = a;
                            x_thread[i + 1] = b / 16.0f;
                            x_thread[i + 2] = c / 256.0f;
                            x_thread[i + 3] = d / 4096.0f;
                        }
                    } else {
                        for (int i = 0; i < values_per_thread; i++) {
                            float a = xp[i];
                            sum += a;
                            x_thread[i] = a;
                        }
                    }
                    for (int row = 0; row < ROWS; row++) {
                        const device uint8_t* wl = wp + row * in_vec_size_w;
                        const float s = sp[row * in_vec_size_g];
                        const float b = bp[row * in_vec_size_g];
                        float accum = 0.0f;
                        if (BITS == 4) {
                            const device uint16_t* w16 = (const device uint16_t*)wl;
                            for (int i = 0; i < values_per_thread / 4; i++) {
                                const uint16_t v = w16[i];
                                accum += x_thread[4 * i] * (float)(v & 0x000f)
                                    + x_thread[4 * i + 1] * (float)(v & 0x00f0)
                                    + x_thread[4 * i + 2] * (float)(v & 0x0f00)
                                    + x_thread[4 * i + 3] * (float)(v & 0xf000);
                            }
                        } else {
                            for (int i = 0; i < values_per_thread; i++) {
                                accum += x_thread[i] * (float)wl[i];
                            }
                        }
                        result[row] += s * accum + b * sum;
                    }
                }
                wp += block_size * bytes_per_pack / pack_factor;
                sp += block_size / GROUP;
                bp += block_size / GROUP;
                xp += block_size;
            }

            for (int row = 0; row < ROWS; row++) {
                const float r = simd_sum(result[row]);
                if (simd_lid == 0 && out_row + row < N) {
                    y[out_row + row] = (half)r;
                }
            }
            """,
        ensureRowContiguous: true)

    static func variant(
        _ x: MLXArray, _ wq: MLXArray, _ s: MLXArray, _ b: MLXArray,
        n: Int, k: Int, bits: Int, group: Int, rows: Int, sgs: Int, packs: Int
    ) -> MLXArray {
        let perTG = rows * sgs
        let tgs = (n + perTG - 1) / perTG
        return kernel(
            [wq, s, b, x],
            template: [
                ("BITS", bits), ("GROUP", group), ("ROWS", rows), ("NUM_SG", sgs),
                ("PACKS", packs), ("K", k), ("N", n),
            ],
            grid: (32, sgs * tgs, 1),
            threadGroup: (32, sgs, 1),
            outputShapes: [[n]],
            outputDTypes: [.float16])[0]
    }

    @Test("qmv variant sweep", .enabled(if: enabled), .timeLimit(.minutes(30)))
    func qmvVariantSweep() throws {
        let k = 2560
        let group = 32
        let x = MLXRandom.normal([k]).asType(.float16)
        eval(x)

        struct Shape { let n: Int; let distinct: Int; let label: String }
        let shapes = [
            Shape(n: 10240, distinct: 36, label: "in_proj_qkv [2560->10240]"),
            Shape(n: 248320, distinct: 4, label: "lm_head [2560->248320]"),
        ]
        // (rows per simdgroup, simdgroups per threadgroup, packs per lane).
        // The stock kernel is (4, 2, 2). values_per_thread must not exceed the
        // group so one scale covers a lane's slice.
        let variants4: [(Int, Int, Int)] = [
            (4, 2, 2), (4, 2, 1), (4, 4, 2), (8, 2, 2), (8, 4, 2), (4, 8, 2), (4, 2, 4), (8, 4, 4),
        ]
        let variants8: [(Int, Int, Int)] = [
            (4, 2, 2), (4, 2, 4), (4, 4, 4), (8, 2, 4), (8, 4, 4), (4, 8, 4), (4, 4, 8),
        ]

        // Where does the stock kernel fall off? Stock vs the plain (4,2,2)
        // mapping across N, q4 only, 4 distinct weights each.
        print("[qmv-sweep] N sweep, q4 g32, stock vs variant(4,2,2)")
        for n in [10240, 32768, 65536, 98304, 131072, 196608, 248320] {
            var sets = [(MLXArray, MLXArray, MLXArray)]()
            for _ in 0 ..< 4 {
                let w = MLXRandom.normal([n, k]).asType(.float16)
                let (wq, sc, b) = quantized(w, groupSize: group, bits: 4)
                eval(wq, sc, b!)
                sets.append((wq, sc, b!))
            }
            let bytes = Double(n * k) / 2 + Double(n * k / group) * 4
            func time(_ f: (Int) -> MLXArray) -> Double {
                func chain() -> MLXArray {
                    var acc = MLXArray(Float(0)).asType(.float16)
                    for i in 0 ..< sets.count { acc = acc + f(i).sum() }
                    return acc
                }
                eval(chain())
                var samples = [Double]()
                for _ in 0 ..< 7 {
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    eval(chain())
                    samples.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
                }
                return samples.sorted()[samples.count / 2] / Double(sets.count)
            }
            let stock = time { i in
                quantizedMM(
                    x.reshaped([1, k]), sets[i].0, scales: sets[i].1, biases: sets[i].2,
                    transpose: true, groupSize: group, bits: 4)
            }
            let variant = time { i in
                Self.variant(
                    x, sets[i].0, sets[i].1, sets[i].2, n: n, k: k, bits: 4, group: group,
                    rows: 4, sgs: 2, packs: 2)
            }
            print(String(
                format: "  N=%7d  stock %7.3f ms %6.1f GB/s   variant %7.3f ms %6.1f GB/s",
                n, stock * 1000, bytes / stock / 1e9, variant * 1000, bytes / variant / 1e9))
        }

        print("[qmv-sweep] one row, K=\(k), g\(group); ms per call is a chain median over distinct weights")
        for bits in [4, 8] {
            let variants = bits == 4 ? variants4 : variants8
            for shape in shapes {
                // Distinct weights so nothing is served from cache.
                var sets = [(MLXArray, MLXArray, MLXArray)]()
                for _ in 0 ..< shape.distinct {
                    let w = MLXRandom.normal([shape.n, k]).asType(.float16)
                    let (wq, s, b) = quantized(w, groupSize: group, bits: bits)
                    eval(wq, s, b!)
                    sets.append((wq, s, b!))
                }
                let wBytes = Double(shape.n * k) * Double(bits) / 8
                let sBytes = Double(shape.n * k / group) * 4
                let bytes = wBytes + sBytes

                // Correctness of every variant against the stock kernel on set 0.
                let ref = quantizedMM(
                    x.reshaped([1, k]), sets[0].0, scales: sets[0].1, biases: sets[0].2,
                    transpose: true, groupSize: group, bits: bits
                ).reshaped([shape.n]).asType(.float32)
                eval(ref)

                func time(_ f: (Int) -> MLXArray) -> Double {
                    func chain() -> MLXArray {
                        var acc = MLXArray(Float(0)).asType(.float16)
                        for i in 0 ..< sets.count { acc = acc + f(i).sum() }
                        return acc
                    }
                    eval(chain())
                    var samples = [Double]()
                    for _ in 0 ..< 7 {
                        let t0 = DispatchTime.now().uptimeNanoseconds
                        eval(chain())
                        samples.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
                    }
                    return samples.sorted()[samples.count / 2] / Double(sets.count)
                }

                let stock = time { i in
                    quantizedMM(
                        x.reshaped([1, k]), sets[i].0, scales: sets[i].1, biases: sets[i].2,
                        transpose: true, groupSize: group, bits: bits)
                }
                print(String(
                    format: "  q%d %@  stock qmv_fast        %7.3f ms  %6.1f GB/s",
                    bits, shape.label as NSString, stock * 1000, bytes / stock / 1e9))

                for (rows, sgs, packs) in variants {
                    let out = Self.variant(
                        x, sets[0].0, sets[0].1, sets[0].2, n: shape.n, k: k, bits: bits,
                        group: group, rows: rows, sgs: sgs, packs: packs
                    ).asType(.float32)
                    let err = (abs(out - ref).max() / (abs(ref).max() + 1e-6)).item(Float.self)
                    let t = time { i in
                        Self.variant(
                            x, sets[i].0, sets[i].1, sets[i].2, n: shape.n, k: k, bits: bits,
                            group: group, rows: rows, sgs: sgs, packs: packs)
                    }
                    print(String(
                        format: "  q%d %@  rows=%d sgs=%d packs=%d  %7.3f ms  %6.1f GB/s  %+.1f%%  maxrelerr %.2e",
                        bits, shape.label as NSString, rows, sgs, packs, t * 1000, bytes / t / 1e9,
                        (stock / t - 1) * 100, err))
                    #expect(err < 2e-2, "variant \(rows)/\(sgs)/\(packs) q\(bits) diverges: \(err)")
                }
            }
        }
    }
}
