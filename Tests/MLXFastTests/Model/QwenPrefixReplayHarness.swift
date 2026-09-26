import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
import Tokenizers

@testable import MLXFastCore
@testable import MLXFastModel

/// Local fitting instrument. Archive provenance is checked by the experiment driver.
@Suite(.serialized)
struct QwenPrefixReplayHarness {
  private struct Plan: Codable {
    let seedTokens: [Int]
    var forcedTokens: [Int]
  }

  private struct Row: Codable {
    let predictionIndex: Int
    let draftToken: Int
    let fullProjectionToken: Int
    let secondDraftToken: Int?
    let targetToken: Int
    let forcedToken: Int
    let targetTop2Tokens: [Int]
    let targetTop2Logits: [Double]
  }

  private struct Report: Codable {
    let matched: Int
    let count: Int
    let secondEligibleCount: Int
    let secondMatched: Int
    let rows: [Row]
  }

  private struct Head: Decodable {
    let id: String
    let normPath: String
  }

  private struct Frame {
    let inputHidden: MLXArray
    let inputTokens: [Int]
    let targetTop2Tokens: [Int]
    let targetTop2Logits: [Double]
  }

  private struct SavedFrame: Codable {
    let inputTokens: [Int]
    let targetTop2Tokens: [Int]
    let targetTop2Logits: [Double]
  }

  private struct Archive: Codable {
    let version: Int
    let plan: Plan
    let bulkSeed: Int
    let singleSeed: Int
    let frames: [SavedFrame]
  }

  @Test("head-only replay over frozen native target frames")
  func measure() throws {
    let env = ProcessInfo.processInfo.environment
    guard env["MLXFAST_PREFIX_REPLAY_CONTEXT"] == "1" else { return }
    func required(_ key: String) throws -> String {
      guard let value = env[key], !value.isEmpty else {
        throw MLXFastError.invalidInput("prefix replay requires \(key)")
      }
      return value
    }
    let weights = try required("MLXFAST_QWEN_PREFILL_WEIGHTS")
    let head = try required("MLXFAST_QWEN_PREFILL_HEAD")
    let output = try required("MLXFAST_PREFIX_REPLAY_OUT")
    try #require(env["DARKBLOOM_QWEN_MTP_FORCE_DEPTH"] == "2")
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let plan = try decoder.decode(Plan.self, from: Data(contentsOf:
      URL(fileURLWithPath: required("MLXFAST_PREFIX_REPLAY_PLAN"))))
    let heads = try decoder.decode([Head].self, from: Data(contentsOf:
      URL(fileURLWithPath: required("MLXFAST_PREFIX_REPLAY_HEADS"))))
    try #require(!heads.isEmpty && Set(heads.map { $0.id }).count == heads.count)
    try #require(!plan.seedTokens.isEmpty && plan.forcedTokens.count >= 2)
    let targetURL = URL(fileURLWithPath: weights)
    let context = try Qwen36MTPHeadAttachment.withHeadAttached(
      backboneDirectory: targetURL, headDirectory: URL(fileURLWithPath: head)
    ) { _ in
      let box = UnsafeSendableBox<Result<ModelContext, Error>>()
      let semaphore = DispatchSemaphore(value: 0)
      Task {
        do {
          box.value = .success(
            try await LLMModelFactory.shared.load(
              from: targetURL, using: #huggingFaceTokenizerLoader()))
        } catch {
          box.value = .failure(error)
        }
        semaphore.signal()
      }
      semaphore.wait()
      guard let result = box.value else {
        throw MLXFastError.invalidInput("model loader completed without a result")
      }
      return try result.get()
    }
    let model = try #require(context.model as? any Qwen36MTPTarget)
    try #require(model.hasMTPHead)
    // MLX eval materializes tensor graphs; it does not evaluate source text.
    eval(model)

    let normKeys = context.model.parameters().flattened().map { $0.0 }.filter {
      $0 == "mtp.norm.weight" || $0.hasSuffix(".mtp.norm.weight")
    }
    try #require(normKeys.count == 1)
    let normKey = try #require(normKeys.first)
    let normParameter = try #require(
      context.model.parameters().flattened().first { $0.0 == normKey }?.1)
    let originalNorm = normParameter[0...]
    eval(originalNorm)
    defer {
      context.model.update(parameters: ModuleParameters.unflattened([normKey: originalNorm]))
    }
    let archiveURL = URL(fileURLWithPath: try required("MLXFAST_PREFIX_REPLAY_ARCHIVE"))
    let (frames, _, singleSeed) = try loadFrames(from: archiveURL, plan: plan)
    try #require(singleSeed == plan.forcedTokens[0])
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    var reports: [String: Report] = [:]
    for selected in heads {
      let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: selected.normPath))
      let norm = try #require(arrays["norm.weight"])
      try #require(norm.shape == originalNorm.shape && norm.dtype == originalNorm.dtype)
      eval(norm)
      try context.model.update(
        parameters: ModuleParameters.unflattened([normKey: norm]), verify: [.noUnusedKeys])
      let rows = try replay(model: model, plan: plan, frames: frames)
      let eligible = zip(rows, rows.dropFirst()).filter { first, _ in
        first.draftToken == first.targetToken && first.draftToken == first.forcedToken
      }
      reports[selected.id] = Report(
        matched: rows.filter { $0.draftToken == $0.targetToken }.count, count: rows.count,
        secondEligibleCount: eligible.count,
        secondMatched: eligible.filter { $0.0.secondDraftToken == $0.1.targetToken }.count,
        rows: rows)
      try encoder.encode(reports).write(to: URL(fileURLWithPath: output), options: .atomic)
      FileHandle.standardError.write(Data("prefix replay head complete: \(selected.id)\n".utf8))
    }
  }

  private func loadFrames(from directory: URL, plan: Plan) throws -> ([Frame], Int, Int) {
    let archive = try JSONDecoder().decode(Archive.self,
      from: Data(contentsOf: directory.appendingPathComponent("frames.json")))
    try #require(archive.version == 1)
    try #require(archive.plan.seedTokens == plan.seedTokens)
    try #require(archive.plan.forcedTokens == plan.forcedTokens)
    try #require(archive.frames.count == plan.forcedTokens.count - 1)
    let tensors = try MLX.loadArrays(url: directory.appendingPathComponent("frames.safetensors"))
    try #require(Set(tensors.keys) == Set(archive.frames.indices.map(String.init)))
    let prime = Qwen36MTPBlockSession.headPrimingRange(
      primeCount: plan.seedTokens.count - 1, cap: Qwen36MTPBlockSession.headPrimingRowCap)
    var frames: [Frame] = []
    for (index, saved) in archive.frames.enumerated() {
      let expectedTokens = index == 0
        ? Array(plan.seedTokens[prime.tokenIndices]) + [plan.forcedTokens[0]]
        : [plan.forcedTokens[index]]
      try #require(saved.inputTokens == expectedTokens)
      try #require(saved.targetTop2Tokens.count == 2 && saved.targetTop2Logits.count == 2)
      let hidden = try #require(tensors[String(index)])
      try #require(hidden.shape == [1, expectedTokens.count, 5120])
      frames.append(Frame(inputHidden: hidden, inputTokens: saved.inputTokens,
        targetTop2Tokens: saved.targetTop2Tokens, targetTop2Logits: saved.targetTop2Logits))
    }
    return (frames, archive.bulkSeed, archive.singleSeed)
  }

  private func replay(model: any Qwen36MTPTarget, plan: Plan, frames: [Frame]) throws -> [Row] {
    let headCache = model.makeMTPCache()
    var rows: [Row] = []
    for (offset, frame) in frames.enumerated() {
      let index = offset + 1
      let tokens = MLXArray(frame.inputTokens.map(Int32.init)).reshaped([1, frame.inputTokens.count])
      let hidden = model.mtpHeadLastHiddenWithKVOnlyHistory(
        hidden: frame.inputHidden, nextTokenIds: tokens, cache: headCache)
        ?? model.mtpHeadHiddenForward(hidden: frame.inputHidden, nextTokenIds: tokens, cache: headCache)
      let last = hidden[0..., (hidden.dim(1) - 1)..<hidden.dim(1), 0...]
      let draft = model.draftTokenID(last).item(Int.self)
      let fullProjection = argMax(model.applyLMHead(last), axis: -1).item(Int.self)
      eval(headCache.flatMap { $0.state })
      var second: Int?
      if index + 1 < plan.forcedTokens.count {
        let offsets = headCache.map { $0.offset }
        try #require(headCache.allSatisfy { $0.isTrimmable })
        let next = model.mtpHeadHiddenForward(hidden: last,
          nextTokenIds: MLXArray([Int32(draft)]).reshaped([1, 1]), cache: headCache)
        second = model.draftTokenID(next).item(Int.self)
        eval(headCache.flatMap { $0.state })
        for (entry, oldOffset) in zip(headCache, offsets) {
          try #require(entry.offset == oldOffset + 1)
          try #require(entry.trim(1) == 1)
          try #require(entry.offset == oldOffset)
        }
      }
      rows.append(Row(predictionIndex: index, draftToken: draft,
        fullProjectionToken: fullProjection, secondDraftToken: second,
        targetToken: frame.targetTop2Tokens[0], forcedToken: plan.forcedTokens[index],
        targetTop2Tokens: frame.targetTop2Tokens, targetTop2Logits: frame.targetTop2Logits))
    }
    return rows
  }
}
