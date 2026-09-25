import Foundation

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
}

@MainActor
class LlamaState: ObservableObject {
    @Published var messageLog = ""
    @Published var cacheCleared = false
    @Published var downloadedModels: [Model] = []
    @Published var undownloadedModels: [Model] = []
    let NS_PER_S = 1_000_000_000.0

    private var isLoadingModel = false
    private var currentModelPath: String?

    // Rolling memory: everything older gets folded into `conversationSummary`;
    // the last few exchanges stay verbatim in `recentMessages` for tone/detail.
    private var conversationSummary: String = ""
    private var recentMessages: [(role: String, content: String)] = []
    private let summarizeThresholdChars = 5000
    private let keepRecentCount = 4

    private var defaultModelUrl: URL? {
        Bundle.main.url(forResource: "ggml-model", withExtension: "gguf", subdirectory: "models")
        // Bundle.main.url(forResource: "llama-2-7b-chat", withExtension: "Q2_K.gguf", subdirectory: "models")
    }

    init() {
        loadModelsFromDisk()
        Task {
            await loadDefaultModels()
        }
    }

    private func loadModelsFromDisk() {
        do {
            let documentsURL = getDocumentsDirectory()
            let modelURLs = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
            for modelURL in modelURLs {
                let modelName = modelURL.deletingPathExtension().lastPathComponent
                downloadedModels.append(Model(name: modelName, url: "", filename: modelURL.lastPathComponent, status: "downloaded"))
            }
        } catch {
            print("Error loading models from disk: \(error)")
        }
    }

    private func loadDefaultModels() async {
        do {
            try await loadModel(modelUrl: defaultModelUrl)
        } catch {
            messageLog += "Error!\n"
        }

        for model in defaultModels {
            let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
            if FileManager.default.fileExists(atPath: fileURL.path) {

            } else {
                var undownloadedModel = model
                undownloadedModel.status = "download"
                undownloadedModels.append(undownloadedModel)
            }
        }
    }

    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
    private let defaultModels: [Model] = [
        Model(name: "TinyLlama-1.1B (Q4_0, 0.6 GiB)",url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_0.gguf?download=true",filename: "tinyllama-1.1b-1t-openorca.Q4_0.gguf", status: "download"),
        Model(
            name: "TinyLlama-1.1B Chat (Q8_0, 1.1 GiB)",
            url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true",
            filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf", status: "download"
        ),

        Model(
            name: "TinyLlama-1.1B (F16, 2.2 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-f16.gguf?download=true",
            filename: "tinyllama-1.1b-f16.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q4_0, 1.6 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q4_0.gguf?download=true",
            filename: "phi-2-q4_0.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q8_0, 2.8 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q8_0.gguf?download=true",
            filename: "phi-2-q8_0.gguf", status: "download"
        ),

        Model(
            name: "Mistral-7B-v0.1 (Q4_0, 3.8 GiB)",
            url: "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_0.gguf?download=true",
            filename: "mistral-7b-v0.1.Q4_0.gguf", status: "download"
        ),
        Model(
            name: "OpenHermes-2.5-Mistral-7B (Q3_K_M, 3.52 GiB)",
            url: "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q3_K_M.gguf?download=true",
            filename: "openhermes-2.5-mistral-7b.Q3_K_M.gguf", status: "download"
        )
    ]
    func loadModel(modelUrl: URL?) async throws {
        guard !isLoadingModel else {
            messageLog += "A model is already loading, please wait...\n"
            return
        }
        isLoadingModel = true
        defer { isLoadingModel = false }

        if let modelUrl {
            // DIAGNOSTIC BUILD: no validation context here — just record the
            // path. Each message creates its own fresh context when it's
            // actually needed, so we're not creating and immediately tearing
            // one down here for no reason.
            currentModelPath = modelUrl.path()
            messageLog += "Model selected: \(modelUrl.lastPathComponent)\n"

            updateDownloadedModels(modelName: modelUrl.lastPathComponent, status: "downloaded")
        } else {
            messageLog += "Load a model from the list below\n"
        }
    }


    private func updateDownloadedModels(modelName: String, status: String) {
        undownloadedModels.removeAll { $0.name == modelName }
    }

    /// Combines the running summary (if any) with the verbatim recent
    /// messages, into the actual list sent to the model.
    private func buildPromptMessages() -> [(role: String, content: String)] {
        var messages: [(role: String, content: String)] = []
        if !conversationSummary.isEmpty {
            messages.append((role: "user", content: "Here is a summary of our conversation so far, for context: \(conversationSummary)"))
            messages.append((role: "assistant", content: "Got it, I'll keep that in mind as we continue."))
        }
        messages.append(contentsOf: recentMessages)
        return messages
    }

    /// Folds the oldest recent messages into the running summary, keeping
    /// only the last `keepRecentCount` verbatim. Uses a fresh context, same
    /// as everything else in this diagnostic build.
    private func compactHistoryIfNeeded() async {
        guard let currentModelPath else { return }

        let approxSize = conversationSummary.count + recentMessages.reduce(0) { $0 + $1.content.count }
        guard approxSize > summarizeThresholdChars, recentMessages.count > keepRecentCount else {
            return
        }

        let toFold = recentMessages.prefix(recentMessages.count - keepRecentCount)
        let toKeep = Array(recentMessages.suffix(keepRecentCount))

        var summarizePrompt = "Summarize the important facts, names, and context from the following conversation in a few concise sentences, so it can be used as background context later. Do not add commentary, just the summary.\n\n"
        if !conversationSummary.isEmpty {
            summarizePrompt += "Existing summary: \(conversationSummary)\n\n"
        }
        for msg in toFold {
            summarizePrompt += "\(msg.role): \(msg.content)\n"
        }

        guard let summaryContext = try? await LlamaContext.create_context(path: currentModelPath) else {
            return
        }
        await summaryContext.completion_init(messages: [(role: "user", content: summarizePrompt)])
        var newSummary = ""
        while !summaryContext.is_done {
            newSummary += await summaryContext.completion_loop()
        }

        conversationSummary = newSummary.trimmingCharacters(in: .whitespacesAndNewlines)
        recentMessages = toKeep
        messageLog += "\n[older context compacted to stay within memory]\n\n"
    }

    func complete(text: String) async {
        guard let currentModelPath else {
            return
        }

        recentMessages.append((role: "user", content: text))
        messageLog += "\(text)\n\n"

        await compactHistoryIfNeeded()

        let t_start = DispatchTime.now().uptimeNanoseconds
        // DIAGNOSTIC: brand-new context for this one message, used once,
        // then discarded — no reuse across messages at all.
        guard let llamaContext = try? await LlamaContext.create_context(path: currentModelPath) else {
            messageLog += "Failed to load model for this message.\n"
            return
        }
        await llamaContext.completion_init(messages: buildPromptMessages())
        let t_heat_end = DispatchTime.now().uptimeNanoseconds
        let t_heat = Double(t_heat_end - t_start) / NS_PER_S

        Task.detached {
            var fullResponse = ""
            while !llamaContext.is_done {
                let result = await llamaContext.completion_loop()
                fullResponse += result
                await MainActor.run {
                    self.messageLog += "\(result)"
                }
            }

            let t_end = DispatchTime.now().uptimeNanoseconds
            let t_generation = Double(t_end - t_heat_end) / self.NS_PER_S
            let tokens_per_second = Double(llamaContext.n_len) / t_generation

            await MainActor.run {
                self.recentMessages.append((role: "assistant", content: fullResponse))
                self.messageLog += """
                    \n
                    Done
                    Heat up took \(t_heat)s
                    Generated \(tokens_per_second) t/s\n
                    """
            }
        }
    }

    func bench() async {
        guard let currentModelPath, let llamaContext = try? await LlamaContext.create_context(path: currentModelPath) else {
            return
        }

        messageLog += "\n"
        messageLog += "Running benchmark...\n"
        messageLog += "Model info: "
        messageLog += await llamaContext.model_info() + "\n"

        let t_start = DispatchTime.now().uptimeNanoseconds
        let _ = await llamaContext.bench(pp: 8, tg: 4, pl: 1) // heat up
        let t_end = DispatchTime.now().uptimeNanoseconds

        let t_heat = Double(t_end - t_start) / NS_PER_S
        messageLog += "Heat up time: \(t_heat) seconds, please wait...\n"

        // if more than 5 seconds, then we're probably running on a slow device
        if t_heat > 5.0 {
            messageLog += "Heat up time is too long, aborting benchmark\n"
            return
        }

        let result = await llamaContext.bench(pp: 512, tg: 128, pl: 1, nr: 3)

        messageLog += "\(result)"
        messageLog += "\n"
    }

    func clear() async {
        recentMessages.removeAll()
        conversationSummary = ""
        messageLog = ""
    }
}
