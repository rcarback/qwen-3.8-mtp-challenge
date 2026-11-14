import Foundation
import MLXFastCore

struct EmbeddingHTTPError: Error {
    let status: Int
    let message: String
}

extension QwenRuntime.QwenServeOptions {
    func resolvedEmbeddingModelName() throws -> String? {
        guard let path = embeddingWeightsPath else {
            guard embeddingModelName == nil else {
                throw MLXFastError.invalidInput(
                    "--embedding-model-name requires --embedding-weights")
            }
            return nil
        }
        let name = embeddingModelName ?? "qwen3-embedding-4b"
        guard !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            name != modelName
        else {
            throw MLXFastError.invalidInput(
                "embedding path and model name must be nonempty; the name must differ from the chat model"
            )
        }
        return name
    }
}

extension QwenRuntime {
    static func servedModelNames(options: QwenServeOptions) throws -> [String] {
        let embeddingName = try options.resolvedEmbeddingModelName()
        return [options.modelName] + (embeddingName.map { [$0] } ?? [])
    }

    static func embeddingResponse(
        body: Data, modelName: String?,
        embed: (EmbeddingWorkerInput) throws -> EmbeddingBatchResult
    ) throws -> Data {
        guard let modelName else {
            throw EmbeddingHTTPError(
                status: 503, message: "start serve with --embedding-weights to enable embeddings")
        }
        let request: EmbeddingRequest
        do {
            request = try JSONDecoder().decode(EmbeddingRequest.self, from: body)
            try request.validate(modelName: modelName)
            // Check before sending: a rejected oversized write must not consume a worker request ID.
            let wire = RuntimeWorkerRequest(
                id: Int.max, kind: "embedding_encode", embeddingInput: request.input)
            guard
                try JSONEncoder().encode(wire).count
                    <= BufferedFileLineReader.defaultMaximumLineByteCount
            else {
                throw EmbeddingInputError(
                    "embedding request exceeds the worker protocol size limit")
            }
        } catch {
            throw EmbeddingHTTPError(status: 400, message: "\(error)")
        }
        do {
            let result = try embed(request.input)
            try result.validate(inputCount: request.input.count)
            return try JSONEncoder().encode(
                EmbeddingResponse(
                    result: result, model: modelName, format: request.encodingFormat ?? .float))
        } catch let error as EmbeddingInputError {
            throw EmbeddingHTTPError(status: 400, message: error.description)
        } catch {
            throw EmbeddingHTTPError(status: 500, message: "embedding request failed: \(error)")
        }
    }

    static func handleEmbeddings(
        request: HTTPRequest, responder: HTTPResponder,
        context: ServeContext, options: QwenServeOptions
    ) {
        do {
            let body = try embeddingResponse(
                body: request.body, modelName: options.resolvedEmbeddingModelName(),
                embed: { try context.client.embed(input: $0) })
            responder.sendJSON(status: 200, body: body)
        } catch let error as EmbeddingHTTPError {
            responder.sendError(status: error.status, message: error.message)
        } catch {
            responder.sendError(status: 500, message: "\(error)")
        }
    }
}
