import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers

@testable import MLXFastCore
@testable import MLXFastModel

private final class UnsafeSendableBox<T>: @unchecked Sendable {
  var value: T?
}

/// Untimed, opt-in instrument. Forced tokens are diagnostic inputs, never labels
/// for fitting a head. Each proposal is judged against this model's serial target.
@Suite(.serialized)
struct QwenHeadAgreementHarness {
  private struct Plan: Codable {
    let seedTokens: [Int]
    var forcedTokens: [Int]
  }

  private struct Article: Decodable {
    let text: String
    let sha256: String
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

  @Test("teacher-forced head agreement with serial reference and production checks")
  func measure() throws {
    let env = ProcessInfo.processInfo.environment
    guard env["MLXFAST_HEAD_AGREEMENT"] == "1" else { return }
    func required(_ key: String) throws -> String {
      guard let value = env[key], !value.isEmpty else {
        throw MLXFastError.invalidInput("head agreement requires \(key)")
      }
      return value
    }
    let weights = try required("MLXFAST_QWEN_PREFILL_WEIGHTS")
    let head = try required("MLXFAST_QWEN_PREFILL_HEAD")
    let planPath = env["MLXFAST_HEAD_AGREEMENT_PLAN"]
    let corpusPath = env["MLXFAST_HEAD_AGREEMENT_CORPUS"]
    try #require((planPath == nil) != (corpusPath == nil))
    let output = try required("MLXFAST_HEAD_AGREEMENT_OUT")
    guard env["DARKBLOOM_QWEN_MTP_FORCE_DEPTH"] == "2" else {
      throw MLXFastError.invalidInput(
        "set DARKBLOOM_QWEN_MTP_FORCE_DEPTH=2 for the production check")
    }
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
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

    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    if let planPath {
      let plan = try decoder.decode(
        Plan.self, from: Data(contentsOf: URL(fileURLWithPath: planPath)))
      let (report, _) = try measurePlan(model: model, input: plan)
      try encoder.encode(report).write(to: URL(fileURLWithPath: output), options: .atomic)
    } else if let corpusPath {
      let articles = try String(contentsOfFile: corpusPath, encoding: .utf8)
        .split(separator: "\n").map { try decoder.decode(Article.self, from: Data($0.utf8)) }
      let limit = Int(env["MLXFAST_HEAD_AGREEMENT_LIMIT"] ?? "") ?? articles.count
      let start = Int(env["MLXFAST_HEAD_AGREEMENT_START"] ?? "") ?? 0
      try #require(limit > 0 && limit <= articles.count)
      try #require(start >= 0 && start < limit)
      let directory = URL(fileURLWithPath: output)
      if start == 0 {
        try #require(!FileManager.default.fileExists(atPath: directory.path))
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      } else {
        try #require(
          !FileManager.default.fileExists(atPath: directory.appendingPathComponent("COMPLETE").path)
        )
        for index in 0..<start {
          let stem = String(format: "article-%04d", index)
          let identity = try String(
            contentsOf: directory.appendingPathComponent(stem + "-source.sha256"), encoding: .utf8)
          try #require(identity == articles[index].sha256)
          let previous = try decoder.decode(
            Report.self,
            from: Data(contentsOf: directory.appendingPathComponent(stem + "-report.json")))
          let previousPlan = try decoder.decode(
            Plan.self, from: Data(contentsOf: directory.appendingPathComponent(stem + "-plan.json"))
          )
          try #require(previous.serialReferenceVerified && previous.rows.count == 127)
          try #require(
            previousPlan.seedTokens.count == 512 && previousPlan.forcedTokens.count == 128)
        }
      }
      for (index, article) in articles.prefix(limit).enumerated().dropFirst(start) {
        let stem = String(format: "article-%04d", index)
        for suffix in ["-report.json", "-plan.json", "-source.sha256"] {
          try #require(
            !FileManager.default.fileExists(
              atPath: directory.appendingPathComponent(stem + suffix).path))
        }
        let tokens = context.tokenizer.encode(text: article.text)
        try #require(tokens.count >= 640)
        let input = Plan(
          seedTokens: Array(tokens.prefix(512)), forcedTokens: Array(tokens[512..<640]))
        let (report, plan) = try measurePlan(model: model, input: input, replaceFirstToken: true)
        try encoder.encode(report).write(
          to: directory.appendingPathComponent(stem + "-report.json"), options: .atomic)
        try encoder.encode(plan).write(
          to: directory.appendingPathComponent(stem + "-plan.json"), options: .atomic)
        try Data(article.sha256.utf8).write(
          to: directory.appendingPathComponent(stem + "-source.sha256"), options: .atomic)
        FileHandle.standardError.write(
          Data("head evaluation: \(index + 1)/\(limit) articles\n".utf8))
      }
      try Data("complete: \(limit) articles\n".utf8).write(
        to: directory.appendingPathComponent("COMPLETE"), options: .atomic)
    }
  }

  private func measurePlan(
    model: any Qwen36MTPTarget, input: Plan, replaceFirstToken: Bool = false
  ) throws -> (Report, Plan) {
    var plan = input
    guard !plan.seedTokens.isEmpty, plan.forcedTokens.count >= 2 else {
      throw MLXFastError.invalidInput("head agreement needs a seed and at least two forced tokens")
    }

    // Scope the production session so its caches are released before the walk.
    let production: (Int, Int, Int) = try {
      let session = try Qwen36MTPBlockSession(model: model, stopTokens: [])
      session.draftPolicy = { offeredDepth, _ in min(offeredDepth, 2) }
      let seedPrediction = try session.begin(seedTokens: plan.seedTokens)
      if replaceFirstToken {
        // The first pending primary always commits in production. This same
        // target prediction must be used in both heads' saved evaluation plans.
        plan.forcedTokens[0] = seedPrediction
      }
      try #require(seedPrediction == plan.forcedTokens[0])
      let round = try session.generateRound(depth: 2)
      try #require(round.draftTokens.count == 2)
      return (seedPrediction, round.draftTokens[0], round.draftTokens[1])
    }()

    var bulkSeedPrediction: Int?
    let rows: [Row] = try {
      let cache = model.newCache(parameters: nil)
      let headCache = model.makeMTPCache()
      let (seedLogits, seedHidden) = model.callWithHidden(
        input: LMInput.Text(
          tokens: MLXArray(plan.seedTokens).reshaped([1, plan.seedTokens.count])),
        cache: cache, nConfirmed: 0)
      eval(cache.flatMap { $0.state })
      let seedID = argMax(seedLogits[0, seedLogits.dim(1) - 1], axis: -1).item(Int.self)
      bulkSeedPrediction = seedID
      let last = seedHidden[0..., (plan.seedTokens.count - 1)..<plan.seedTokens.count, 0...]
      let single = model.applyLMHead(model.applyFinalNorm(last)).reshaped([-1])
      let singleSeedID = argMax(single, axis: -1).item(Int.self)
      if seedID != production.0 {
        let (bulkIDs, bulkValues) = Qwen36MTPBlockSession.topTwo(
          of: seedLogits[0, seedLogits.dim(1) - 1])
        let (singleIDs, singleValues) = Qwen36MTPBlockSession.topTwo(of: single)
        FileHandle.standardError.write(
          Data(
            "seed frame: production=\(production.0), bulk=\(bulkIDs) \(bulkValues), single=\(singleIDs) \(singleValues)\n"
              .utf8))
      }
      try #require(singleSeedID == production.0)
      // This baseline primes the head with the full seed history.
      var inputHidden = model.applyFinalNorm(seedHidden)
      var inputTokens = Array(plan.seedTokens.dropFirst()) + [plan.forcedTokens[0]]
      var measured: [Row] = []
      for index in 1..<plan.forcedTokens.count {
        let tokens = MLXArray(inputTokens.map(Int32.init)).reshaped([1, inputTokens.count])
        let hidden =
          model.mtpHeadLastHiddenWithKVOnlyHistory(
            hidden: inputHidden, nextTokenIds: tokens, cache: headCache)
          ?? model.mtpHeadHiddenForward(
            hidden: inputHidden, nextTokenIds: tokens, cache: headCache)
        let last = hidden[0..., (hidden.dim(1) - 1)..<hidden.dim(1), 0...]
        let draft = model.draftTokenID(last).item(Int.self)
        let fullProjection = argMax(model.applyLMHead(last), axis: -1).item(Int.self)
        eval(headCache.flatMap { $0.state })
        if index == 1 { try #require(draft == production.1) }

        // Only the extra head row is speculative. Restore its cache offset
        // before advancing the continuous teacher-forced head history.
        var secondDraft: Int?
        if index + 1 < plan.forcedTokens.count {
          let offsets = headCache.map { $0.offset }
          try #require(headCache.allSatisfy { $0.isTrimmable })
          let secondHidden = model.mtpHeadHiddenForward(
            hidden: last, nextTokenIds: MLXArray([Int32(draft)]).reshaped([1, 1]),
            cache: headCache)
          secondDraft = model.draftTokenID(secondHidden).item(Int.self)
          eval(headCache.flatMap { $0.state })
          for (entry, offset) in zip(headCache, offsets) {
            try #require(entry.offset == offset + 1)
            try #require(entry.trim(1) == 1)
            try #require(entry.offset == offset)
          }
          if index == 1 { try #require(secondDraft == production.2) }
        }

        let (logits, targetHidden) = model.callWithHidden(
          input: LMInput.Text(tokens: MLXArray([plan.forcedTokens[index - 1]]).reshaped([1, 1])),
          cache: cache, nConfirmed: 0)
        eval(cache.flatMap { $0.state })
        let (ids, values) = Qwen36MTPBlockSession.topTwo(of: logits[0, 0])
        measured.append(
          Row(
            predictionIndex: index, draftToken: draft,
            fullProjectionToken: fullProjection, secondDraftToken: secondDraft,
            targetToken: ids[0],
            forcedToken: plan.forcedTokens[index],
            targetTop2Tokens: ids, targetTop2Logits: values))
        inputHidden = model.applyFinalNorm(targetHidden)
        inputTokens = [plan.forcedTokens[index]]
      }
      return measured
    }()

    // An independent existing instrument checks every target row, including
    // logits. The head walk must not alter target state or shift the frame.
    let reference = Qwen36MTPReferenceSession(model: model)
    let referenceSeed = try reference.prefillSeed(plan.seedTokens)
    // The reference prefill projects a bulk row set, while begin() projects
    // only the last hidden row. Bind each argmax to the frame that produced it.
    try #require(referenceSeed == bulkSeedPrediction)
    let answer = try reference.rows(
      tokens: plan.seedTokens + plan.forcedTokens,
      seedTokenCount: plan.seedTokens.count, startOffset: plan.seedTokens.count,
      count: rows.count)
    for (row, expected) in zip(rows, answer.rows) {
      try #require(row.targetTop2Tokens == expected.top2Tokens)
      try #require(row.targetTop2Logits == expected.top2Logits)
    }
    let eligibleSeconds = zip(rows, rows.dropFirst()).filter { first, _ in
      first.draftToken == first.targetToken && first.draftToken == first.forcedToken
    }
    let report = Report(
      seedPrediction: production.0, bulkSeedPrediction: bulkSeedPrediction,
      productionFirstDraft: production.1,
      productionSecondDraft: production.2,
      serialReferenceVerified: true,
      matched: rows.filter { $0.draftToken == $0.targetToken }.count,
      count: rows.count,
      secondEligibleCount: eligibleSeconds.count,
      secondMatched: eligibleSeconds.filter { $0.0.secondDraftToken == $0.1.targetToken }.count,
      rows: rows)
    print(
      "head agreement: \(report.matched)/\(report.count); production first draft and all serial rows verified"
    )
    return (report, plan)
  }
}
