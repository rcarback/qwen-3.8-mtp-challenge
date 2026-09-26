import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers

@testable import MLXFastCore
@testable import MLXFastModel

@Suite(.serialized)
struct QwenCorpusTrajectoryHarness {
  private struct Input: Decodable {
    let title: String
    let domain: String
    let text: String
    let sha256: String
    let seedLimit: Int?
    let stopAtEnd: Bool?
  }

  @Test("export serial target trajectories for a domain-balanced corpus")
  func exportTrajectories() throws {
    let env = ProcessInfo.processInfo.environment
    guard env["MLXFAST_CORPUS_TRAJECTORIES"] == "1" else { return }
    func required(_ key: String) throws -> String {
      guard let value = env[key], !value.isEmpty else {
        throw MLXFastError.invalidInput("trajectory export requires \(key)")
      }
      return value
    }
    let weights = URL(fileURLWithPath: try required("MLXFAST_QWEN_PREFILL_WEIGHTS"))
    let head = URL(fileURLWithPath: try required("MLXFAST_QWEN_PREFILL_HEAD"))
    let destination = URL(fileURLWithPath: try required("MLXFAST_CORPUS_OUT"))
    let input = try String(contentsOfFile: required("MLXFAST_CORPUS_INPUT"), encoding: .utf8)
      .split(separator: "\n").map {
        try JSONDecoder().decode(Input.self, from: Data($0.utf8))
      }
    try #require(!input.isEmpty)
    let context = try Qwen36MTPHeadAttachment.withHeadAttached(
      backboneDirectory: weights, headDirectory: head
    ) { _ in
      let box = UnsafeSendableBox<Result<ModelContext, Error>>()
      let semaphore = DispatchSemaphore(value: 0)
      Task {
        do {
          box.value = .success(try await LLMModelFactory.shared.load(
            from: weights, using: #huggingFaceTokenizerLoader()))
        } catch {
          box.value = .failure(error)
        }
        semaphore.signal()
      }
      semaphore.wait()
      return try #require(box.value).get()
    }
    let model = try #require(context.model as? any Qwen36MTPTarget)
    // MLX eval materializes tensor graphs; it does not execute source text.
    eval(model)
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    for (index, article) in input.enumerated() {
      let resultPath = destination.appendingPathComponent(String(format: "rows-%04d.json", index))
      if FileManager.default.fileExists(atPath: resultPath.path) {
        let existing = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: resultPath)) as? [String: Any])
        try #require(existing["input_sha256"] as? String == article.sha256)
        try #require(existing["title"] as? String == article.title)
        try #require(existing["collector_version"] as? Int == 2)
        continue
      }
      let encoded = context.tokenizer.encode(text: article.text)
      if let limit = article.seedLimit {
        try #require(limit > 0 && encoded.count >= limit)
      }
      let seed = Array(encoded.prefix(article.seedLimit ?? encoded.count))
      try #require(!seed.isEmpty && seed.count <= 4096)
      // Fixed-length diagnostic trajectories match the benchmark's decode window.
      // Stop tokens do not truncate this diagnostic. No tools or code are executed.
      let session = try Qwen36MTPBlockSession(model: model, stopTokens: [])
      let firstPrediction = try session.begin(seedTokens: seed, expectedTotalTokens: seed.count + 512)
      var emitted: [Int] = []
      while emitted.count < 512 {
        let round = try session.generateRound(depth: 0)
        try #require(!round.tokens.isEmpty && round.draftTokens.isEmpty)
        if emitted.isEmpty {
          // begin previews this token; generateRound commits and emits it once.
          try #require(round.tokens[0] == firstPrediction)
        }
        emitted.append(contentsOf: round.tokens)
        try #require(round.targetCacheOffset == seed.count + emitted.count)
        if article.stopAtEnd == true, let token = emitted.last, token == 248044 || token == 248046 {
          break
        }
      }
      emitted = Array(emitted.prefix(512))
      let record: [String: Any] = [
        "title": article.title, "domain": article.domain,
        "input_sha256": article.sha256, "seed_tokens": seed,
        "emitted": emitted, "draft_depth": 0,
        "collector_version": 2,
        "decoded_continuation": context.tokenizer.decode(tokenIds: emitted),
      ]
      let data = try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
      try data.write(to: resultPath, options: .atomic)
      FileHandle.standardError.write(Data("trajectory: \(index + 1)/\(input.count) \(article.domain), seed=\(seed.count), generated=\(emitted.count)\n".utf8))
    }
    try Data("complete: \(input.count) serial trajectories\n".utf8)
      .write(to: destination.appendingPathComponent("COMPLETE"), options: .atomic)
  }
}
