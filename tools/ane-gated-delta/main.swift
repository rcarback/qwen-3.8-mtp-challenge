import CoreML
import Foundation

@available(macOS 15.0, *)
func dispatch() async {
    setvbuf(stdout, nil, _IONBF, 0)
    let mode = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "ops"
    switch mode {
    case "ops": await runOpProbes()
    case "ops2": await runOpProbes2()
    case "ops3": await runOpProbes3()
    case "ops4": await runOpProbes4()
    case "ops5": await runOpProbes5()
    case "ops6": await runOpProbes6()
    case "ops7": await runOpProbes7()
    case "ops8": await runOpProbes8()
    case "chunk": await runChunkProbe()
    case "chunk2": await runChunkProbe2()
    case "layer": await runLayerBench()
    case "tiled": await runTiledBench()
    case "ycmp": runYCompare()
    case "savechunk": await runSaveChunk()
    default: print("modes: ops | chunk | layer | tiled | savechunk"); exit(2)
    }
}
if #available(macOS 15.0, *) { await dispatch() } else { exit(3) }
