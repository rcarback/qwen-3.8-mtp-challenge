import Darwin
import Foundation
import MLXFastCore
import MLXFastModel
import MLXFastRuntimeWorkerSupport

let exitCode = ParticipantWorkerCLI.run(
    arguments: Array(CommandLine.arguments.dropFirst())
)
exit(Int32(exitCode))

private enum ParticipantWorkerCLI {
    static func run(arguments: [String]) -> Int {
        do {
            guard let command = arguments.first else {
                printUsage()
                return 0
            }
            if ["help", "--help", "-h"].contains(command) {
                printUsage()
                return 0
            }
            if Array(arguments.dropFirst()) == ["--help"]
                || Array(arguments.dropFirst()) == ["-h"]
            {
                printUsage()
                return 0
            }
            let options = try WorkerOptions(
                Array(arguments.dropFirst())
            )
            switch command {
            case "runtime-worker":
                try options.requireOnly(
                    values: ["--weights"]
                )
                let weightsPath = options.value(
                    for: "--weights",
                    default: ProcessInfo.processInfo.environment[
                        "MLXFAST_WEIGHTS_PATH"
                    ] ?? MLXFastConstants.defaultWeightsPath
                )
                try QwenRuntime.runWorker(weightsPath: weightsPath)

            case "dflash-runtime-worker":
                // DFlash block-decode track worker. Takes the organizer-pinned
                // target weights and the organizer-provisioned DFlash drafter;
                // serves dflash_decode_begin/_block/_diagnostics only.
                try options.requireOnly(
                    values: ["--weights", "--drafter"]
                )
                let weightsPath = options.value(
                    for: "--weights",
                    default: ProcessInfo.processInfo.environment[
                        "MLXFAST_WEIGHTS_PATH"
                    ] ?? MLXFastConstants.defaultWeightsPath
                )
                let drafterPath = options.value(
                    for: "--drafter",
                    default: ProcessInfo.processInfo.environment[
                        "MLXFAST_DFLASH_DRAFTER_DIR"
                    ] ?? ""
                )
                guard !drafterPath.isEmpty else {
                    throw MLXFastError.invalidInput(
                        "dflash-runtime-worker requires --drafter (or "
                            + "MLXFAST_DFLASH_DRAFTER_DIR)"
                    )
                }
                try QwenRuntime.runExperimentalDFlashWorker(
                    targetWeightsPath: weightsPath,
                    drafterPath: drafterPath
                )

            case "mtp-runtime-worker":
                // Qwen 3.6 native-MTP track worker. Takes the organizer-pinned
                // backbone and the SEPARATELY pinned MTP head (operator Q8:
                // separate trees, merge at load); serves the mtp_* kinds only.
                try options.requireOnly(
                    values: ["--weights", "--mtp-head", "--decode-ceiling"]
                )
                let weightsPath = options.value(
                    for: "--weights",
                    default: ProcessInfo.processInfo.environment[
                        "MLXFAST_WEIGHTS_PATH"
                    ] ?? MLXFastConstants.defaultWeightsPath
                )
                let mtpHeadPath = options.value(
                    for: "--mtp-head",
                    default: ProcessInfo.processInfo.environment[
                        "MLXFAST_QWEN_MTP_HEAD_DIR"
                    ] ?? ""
                )
                // LOCAL RESEARCH ESCAPE: `none` selects the headless backbone
                // for sibling `qwen3_5_text` towers, which publish no MTP
                // head. See `Qwen35CheckpointValidation.resolved`.
                let headless = mtpHeadPath == "none"
                    && ProcessInfo.processInfo
                        .environment["DARKBLOOM_QWEN_GEOMETRY_UNPINNED"] == "1"
                guard headless || !mtpHeadPath.isEmpty else {
                    throw MLXFastError.invalidInput(
                        "mtp-runtime-worker requires --mtp-head (or "
                            + "MLXFAST_QWEN_MTP_HEAD_DIR)"
                    )
                }
                // Per-session OUTPUT ceiling. Absent means the pinned
                // default; the benchmark and ranked paths never pass it.
                let decodeCeilingText = options.value(
                    for: "--decode-ceiling", default: ""
                )
                let decodeCeiling = decodeCeilingText.isEmpty
                    ? nil : Int(decodeCeilingText)
                if !decodeCeilingText.isEmpty, decodeCeiling == nil {
                    throw MLXFastError.invalidInput(
                        "--decode-ceiling requires an integer"
                    )
                }
                try QwenRuntime.runQwenMTPWorker(
                    targetWeightsPath: weightsPath,
                    mtpHeadPath: headless ? "" : mtpHeadPath,
                    decodeCeiling: decodeCeiling
                )

            case "qwen4exp-transform":
                // Local fork only: Qwen3.8-Flash-Next bf16 source -> runtime tree.
                try options.requireOnly(
                    values: ["--source", "--destination", "--expert-group-size", "--expert-bits"]
                )
                let source = options.value(for: "--source", default: "")
                let destination = options.value(for: "--destination", default: "")
                guard !source.isEmpty, !destination.isEmpty else {
                    throw MLXFastError.invalidInput(
                        "usage: qwen4exp-transform --source DIR --destination DIR [--expert-group-size 32] [--expert-bits 4]")
                }
                let groupSize = Int(options.value(for: "--expert-group-size", default: "32")) ?? 32
                let bits = Int(options.value(for: "--expert-bits", default: "4")) ?? 4
                try Qwen4ExpTransform.run(
                    .init(
                        source: URL(fileURLWithPath: source),
                        destination: URL(fileURLWithPath: destination),
                        expertGroupSize: groupSize, expertBits: bits))
                print("qwen4exp-transform: wrote \(destination)")

            case "preflight":
                try options.requireOnly(
                    values: ["--weights"]
                )
                let weightsPath = options.value(
                    for: "--weights",
                    default: ProcessInfo.processInfo.environment[
                        "MLXFAST_WEIGHTS_PATH"
                    ] ?? MLXFastConstants.defaultWeightsPath
                )
                try QwenRuntime.runPreflightWorker(
                    weightsPath: weightsPath
                )

            default:
                throw MLXFastError.invalidInput(
                    "unknown participant worker command '\(command)'"
                )
            }
            return 0
        } catch {
            fputs("mlxfast-runtime-worker: \(error)\n", stderr)
            return 1
        }
    }

    private static func printUsage() {
        print(
            """
            Usage:
              mlxfast-runtime-worker runtime-worker [--weights PATH]
              mlxfast-runtime-worker dflash-runtime-worker [--weights PATH] --drafter PATH
              mlxfast-runtime-worker mtp-runtime-worker [--weights PATH] --mtp-head PATH
              mlxfast-runtime-worker preflight [--weights PATH]

            Participant-side MLX runtime worker for mlxfast-swift.
            """
        )
    }
}

private struct WorkerOptions {
    private let values: [String: String]

    init(_ arguments: [String]) throws {
        var values: [String: String] = [:]
        var index = 0
        while index < arguments.count {
            let option = arguments[index]
            guard option.hasPrefix("--") else {
                throw MLXFastError.invalidInput(
                    "unexpected participant worker argument '\(option)'"
                )
            }
            guard values[option] == nil else {
                throw MLXFastError.invalidInput(
                    "duplicate participant worker option \(option)"
                )
            }
            guard index + 1 < arguments.count else {
                throw MLXFastError.invalidInput(
                    "participant worker option \(option) requires a value"
                )
            }
            values[option] = arguments[index + 1]
            index += 2
        }
        self.values = values
    }

    func value(for option: String, default defaultValue: String = "") -> String {
        values[option] ?? defaultValue
    }

    func requireOnly(values allowedValues: Set<String>) throws {
        if let unexpected = Set(values.keys).subtracting(allowedValues).sorted().first {
            throw MLXFastError.invalidInput(
                "unexpected participant worker option \(unexpected)"
            )
        }
    }
}
