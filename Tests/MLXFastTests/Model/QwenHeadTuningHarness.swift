import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers

@testable import MLXFastCore
@testable import MLXFastModel

/// Offline training instrument. Bulk-frame features are not correctness goldens.
@Suite(.serialized)
struct QwenHeadTuningHarness {
  private struct Article: Decodable {
    let title: String
    let text: String
    let sha256: String
    let tokens: [Int]?
    let seedCount: Int?
    let examples: Int?
    let targetTokens: [Int]?
  }

  @Test("export frozen-head features and target labels for tuning")
  func exportFeatures() throws {
    let env = ProcessInfo.processInfo.environment
    guard env["MLXFAST_HEAD_EXPORT"] == "1" else { return }
    func required(_ name: String) throws -> String {
      guard let value = env[name], !value.isEmpty else {
        throw MLXFastError.invalidInput("head export requires \(name)")
      }
      return value
    }
    let weights = URL(fileURLWithPath: try required("MLXFAST_QWEN_PREFILL_WEIGHTS"))
    let head = URL(fileURLWithPath: try required("MLXFAST_QWEN_PREFILL_HEAD"))
    let source = try required("MLXFAST_HEAD_EXPORT_INPUT")
    let destination = URL(fileURLWithPath: try required("MLXFAST_HEAD_EXPORT_OUT"))
    let limit = Int(env["MLXFAST_HEAD_EXPORT_LIMIT"] ?? "") ?? 2
    let defaultExampleCount = Int(env["MLXFAST_HEAD_EXPORT_EXAMPLES"] ?? "") ?? 128
    let articles = try String(contentsOfFile: source, encoding: .utf8)
      .split(separator: "\n").map { line in
        try JSONDecoder().decode(Article.self, from: Data(line.utf8))
      }
    try #require(limit > 0 && limit <= articles.count)
    try #require(!FileManager.default.fileExists(atPath: destination.path))
    let context = try Qwen36MTPHeadAttachment.withHeadAttached(
      backboneDirectory: weights, headDirectory: head
    ) { _ in
      let box = UnsafeSendableBox<Result<ModelContext, Error>>()
      let semaphore = DispatchSemaphore(value: 0)
      Task {
        do {
          box.value = .success(
            try await LLMModelFactory.shared.load(
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
    try #require(model.hasMTPHead)
    // MLX eval materializes tensor graphs; it does not execute source text.
    eval(model)
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    var projection: [String: MLXArray] = [:]
    for (key, value) in context.model.parameters().flattened() {
      for leaf in ["weight", "scales", "biases"] {
        if key == "lm_head.\(leaf)" || key.hasSuffix(".lm_head.\(leaf)") {
          try #require(projection[leaf] == nil)
          projection[leaf] = value
        }
      }
    }
    try #require(Set(projection.keys) == Set(["weight", "scales", "biases"]))
    try MLX.save(
      arrays: projection, metadata: ["bits": "4", "group_size": "64"],
      url: destination.appendingPathComponent("projection.safetensors"))

    for (index, article) in articles.prefix(limit).enumerated() {
      let seedCount = article.seedCount ?? 512
      let exampleCount = article.examples ?? defaultExampleCount
      try #require(seedCount > 0 && exampleCount > 0 && exampleCount <= 512)
      let tokenCount = seedCount + exampleCount
      let encoded = article.tokens ?? context.tokenizer.encode(text: article.text)
      try #require(encoded.count >= tokenCount)
      let tokens = Array(encoded.prefix(tokenCount))
      let cache = model.newCache(parameters: nil)
      let headCache = model.makeMTPCache()
      let (logits, hidden) = model.callWithHidden(
        input: LMInput.Text(tokens: MLXArray(tokens).reshaped([1, tokenCount])),
        cache: cache, nConfirmed: 0)
      let headHidden = model.mtpHeadHiddenForward(
        hidden: model.applyFinalNorm(hidden[0..., 0..<(tokenCount - 1), 0...]),
        nextTokenIds: MLXArray(Array(tokens.dropFirst())).reshaped([1, tokenCount - 1]),
        cache: headCache)
      // Hidden row i paired with token i+1 predicts i+2; target row i+1
      // supplies the bulk label. Supplied serial labels preserve the native history.
      let features = headHidden[0, (seedCount - 1)..<(tokenCount - 1), 0...]
      let bulkLabels = argMax(logits[0, seedCount..<tokenCount, 0...], axis: -1)
      if let supplied = article.targetTokens { try #require(supplied.count == exampleCount) }
      let labels = article.targetTokens.map { MLXArray($0) } ?? bulkLabels
      let projected = model.applyLMHead(features)
      let proposalIDs = argMax(projected, axis: -1)
      let first = features[0..<1, 0...].reshaped([1, 1, features.dim(1)])
      let controlLogits = model.applyLMHead(first).reshaped([-1])
      let controlIDs = MLXArray([0, 1, 100, 1024, 32000, 98000, 248044])
      let arrays = [
        "features": features, "target_tokens": labels, "draft_tokens": proposalIDs,
        "bulk_target_tokens": bulkLabels,
        "text_tokens": MLXArray(tokens),
        "control_ids": controlIDs, "control_logits": controlLogits[controlIDs],
      ]
      eval(Array(arrays.values))
      try #require(features.shape == [exampleCount, 5120])
      try #require(labels.shape == [exampleCount])
      try MLX.save(
        arrays: arrays,
        metadata: [
          "text_sha256": article.sha256, "title": article.title,
          "frame": "bulk", "seed_tokens": String(seedCount),
        ],
        url: destination.appendingPathComponent(String(format: "rows-%04d.safetensors", index)))
      FileHandle.standardError.write(Data("head export: \(index + 1)/\(limit) articles\n".utf8))
    }
    let totalRows = articles.prefix(limit).reduce(0) { $0 + ($1.examples ?? defaultExampleCount) }
    try Data("complete: \(limit) articles, \(totalRows) rows, bulk frame\n".utf8)
      .write(to: destination.appendingPathComponent("COMPLETE"), options: .atomic)
  }
}
