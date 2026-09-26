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

/// Local training instrument with serial target-frame export and verified replay.
@Suite(.serialized)
struct QwenPrefixContextHarness {
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
    let seedPrediction: Int
    let bulkSeedPrediction: Int?
    let productionFirstDraft: Int
    let productionSecondDraft: Int
    let serialReferenceVerified: Bool
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

  @Test("exported target frames reproduce native first and second drafts")
  func measure() throws {
    let env = ProcessInfo.processInfo.environment
    guard env["MLXFAST_PREFIX_CONTEXT"] == "1" else { return }
    func required(_ key: String) throws -> String {
      guard let value = env[key], !value.isEmpty else {
        throw MLXFastError.invalidInput("prefix context requires \(key)")
      }
      return value
    }
    let weights = try required("MLXFAST_QWEN_PREFILL_WEIGHTS")
    let head = try required("MLXFAST_QWEN_PREFILL_HEAD")
    let output = try required("MLXFAST_PREFIX_OUT")
    try #require(env["DARKBLOOM_QWEN_MTP_FORCE_DEPTH"] == "2")
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let plan = try decoder.decode(Plan.self, from: Data(contentsOf:
      URL(fileURLWithPath: required("MLXFAST_PREFIX_PLAN"))))
    let heads = try decoder.decode([Head].self, from: Data(contentsOf:
      URL(fileURLWithPath: required("MLXFAST_PREFIX_HEADS"))))
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
    let archivePath = try required("MLXFAST_PREFIX_ARCHIVE")
    let mode = try required("MLXFAST_PREFIX_MODE")
    try #require(mode == "export" || mode == "replay")
    let archiveURL = URL(fileURLWithPath: archivePath)
    let frames: [Frame]
    let bulkSeed: Int
    let singleSeed: Int
    if mode == "replay" {
      (frames, bulkSeed, singleSeed) = try loadFrames(from: archiveURL, plan: plan)
    } else {
      try #require(!FileManager.default.fileExists(atPath: archivePath))
      (frames, bulkSeed, singleSeed) = try targetFrames(model: model, plan: plan)
    }
    try #require(singleSeed == plan.forcedTokens[0])
    let reference = Qwen36MTPReferenceSession(model: model)
    let referenceSeed = try reference.prefillSeed(plan.seedTokens)
    try #require(referenceSeed == bulkSeed)
    let answer = try reference.rows(
      tokens: plan.seedTokens + plan.forcedTokens,
      seedTokenCount: plan.seedTokens.count, startOffset: plan.seedTokens.count,
      count: frames.count)
    try #require(answer.rows.count == frames.count)
    for (frame, expected) in zip(frames, answer.rows) {
      try #require(frame.targetTop2Tokens == expected.top2Tokens)
      try #require(frame.targetTop2Logits == expected.top2Logits)
    }
    if mode == "export" {
      try saveFrames(frames, plan: plan, bulkSeed: bulkSeed,
        singleSeed: singleSeed, to: archiveURL)
    }
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
      let session = try Qwen36MTPBlockSession(model: model, stopTokens: [])
      let seed = try session.begin(seedTokens: plan.seedTokens,
        expectedTotalTokens: plan.seedTokens.count + plan.forcedTokens.count)
      try #require(seed == singleSeed)
      let round = try session.generateRound(depth: 2)
      try #require(round.draftTokens.count == 2)
      let rows = try replay(model: model, plan: plan, frames: frames,
        production: (round.draftTokens[0], round.draftTokens[1]))
      let eligible = zip(rows, rows.dropFirst()).filter { first, _ in
        first.draftToken == first.targetToken && first.draftToken == first.forcedToken
      }
      reports[selected.id] = Report(
        seedPrediction: seed, bulkSeedPrediction: bulkSeed,
        productionFirstDraft: round.draftTokens[0], productionSecondDraft: round.draftTokens[1],
        serialReferenceVerified: true,
        matched: rows.filter { $0.draftToken == $0.targetToken }.count, count: rows.count,
        secondEligibleCount: eligible.count,
        secondMatched: eligible.filter { $0.0.secondDraftToken == $0.1.targetToken }.count,
        rows: rows)
      try encoder.encode(reports).write(to: URL(fileURLWithPath: output), options: .atomic)
      FileHandle.standardError.write(Data("prefix context head complete: \(selected.id)\n".utf8))
    }
  }

  private func saveFrames(_ frames: [Frame], plan: Plan, bulkSeed: Int,
    singleSeed: Int, to directory: URL) throws {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    var tensors: [String: MLXArray] = [:]
    for (index, frame) in frames.enumerated() {
      tensors[String(index)] = frame.inputHidden
    }
    eval(Array(tensors.values))
    try MLX.save(arrays: tensors, url: directory.appendingPathComponent("frames.safetensors"))
    let archive = Archive(version: 1, plan: plan, bulkSeed: bulkSeed, singleSeed: singleSeed,
      frames: frames.map { SavedFrame(inputTokens: $0.inputTokens,
        targetTop2Tokens: $0.targetTop2Tokens, targetTop2Logits: $0.targetTop2Logits) })
    try JSONEncoder().encode(archive).write(
      to: directory.appendingPathComponent("frames.json"), options: .atomic)
    // The round trip checks tensor dtype and values before admitting the archive.
    let (loaded, loadedBulk, loadedSingle) = try loadFrames(from: directory, plan: plan)
    try #require(loadedBulk == bulkSeed && loadedSingle == singleSeed)
    for (before, after) in zip(frames, loaded) {
      try #require(before.inputHidden.dtype == after.inputHidden.dtype)
      try #require(arrayEqual(before.inputHidden, after.inputHidden).item(Bool.self))
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

  private func targetFrames(model: any Qwen36MTPTarget, plan: Plan) throws
    -> ([Frame], Int, Int) {
    let cache = model.newCache(parameters: nil)
    let (logits, hidden) = model.callWithHidden(
      input: LMInput.Text(tokens: MLXArray(plan.seedTokens).reshaped([1, plan.seedTokens.count])),
      cache: cache, nConfirmed: 0)
    eval(cache.flatMap { $0.state })
    let bulkSeed = argMax(logits[0, logits.dim(1) - 1], axis: -1).item(Int.self)
    let last = hidden[0..., (plan.seedTokens.count - 1)..<plan.seedTokens.count, 0...]
    let projected = model.applyLMHead(model.applyFinalNorm(last)).reshaped([-1])
    let singleSeed = argMax(projected, axis: -1).item(Int.self)
    let prime = Qwen36MTPBlockSession.headPrimingRange(
      primeCount: plan.seedTokens.count - 1, cap: Qwen36MTPBlockSession.headPrimingRowCap)
    var inputHidden = model.applyFinalNorm(
      hidden[0..., prime.hiddenRows.lowerBound..<plan.seedTokens.count, 0...])
    var inputTokens = Array(plan.seedTokens[prime.tokenIndices]) + [plan.forcedTokens[0]]
    var frames: [Frame] = []
    for index in 1..<plan.forcedTokens.count {
      eval(inputHidden)
      let (stepLogits, stepHidden) = model.callWithHidden(
        input: LMInput.Text(tokens: MLXArray([plan.forcedTokens[index - 1]]).reshaped([1, 1])),
        cache: cache, nConfirmed: 0)
      eval(cache.flatMap { $0.state })
      let (ids, values) = Qwen36MTPBlockSession.topTwo(of: stepLogits[0, 0])
      frames.append(Frame(inputHidden: inputHidden, inputTokens: inputTokens,
        targetTop2Tokens: ids, targetTop2Logits: values))
      inputHidden = model.applyFinalNorm(stepHidden)
      inputTokens = [plan.forcedTokens[index]]
    }
    return (frames, bulkSeed, singleSeed)
  }

  private func replay(model: any Qwen36MTPTarget, plan: Plan, frames: [Frame],
    production: (Int, Int)) throws -> [Row] {
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
      if index == 1 { try #require(draft == production.0) }
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
        if index == 1 { try #require(second == production.1) }
      }
      rows.append(Row(predictionIndex: index, draftToken: draft,
        fullProjectionToken: fullProjection, secondDraftToken: second,
        targetToken: frame.targetTop2Tokens[0], forcedToken: plan.forcedTokens[index],
        targetTop2Tokens: frame.targetTop2Tokens, targetTop2Logits: frame.targetTop2Logits))
    }
    return rows
  }
}
