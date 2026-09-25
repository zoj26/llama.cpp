import Foundation
import llama

enum LlamaError: Error {
    case couldNotInitializeContext
}

func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], _ logits: Bool) {
    batch.token   [Int(batch.n_tokens)] = id
    batch.pos     [Int(batch.n_tokens)] = pos
    batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
    for i in 0..<seq_ids.count {
        batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
    }
    batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0

    batch.n_tokens += 1
}

/// A single, permanent background thread that every llama.cpp call is
/// funneled through, so the C++ engine never sees work arrive from a
/// different OS thread than the one it was set up on. Also responsible
/// for calling llama_backend_init() exactly once, ever, for the app's
/// whole lifetime.
final class LlamaWorker {
    static let shared = LlamaWorker()

    private var thread = Thread()
    private let lock = NSLock()
    private var pendingWork: [() -> Void] = []
    private let semaphore = DispatchSemaphore(value: 0)

    private init() {
        thread = Thread { [weak self] in
            self?.runLoop()
        }
        thread.name = "com.pocketpal.llama.worker"
        thread.stackSize = 8 * 1024 * 1024
        thread.start()
    }

    private func runLoop() {
        llama_backend_init()
        print("llama_backend_init() called once for the process lifetime")

        while true {
            semaphore.wait()
            lock.lock()
            let work = pendingWork.isEmpty ? nil : pendingWork.removeFirst()
            lock.unlock()
            work?()
        }
    }

    private func enqueue(_ work: @escaping () -> Void) {
        lock.lock()
        pendingWork.append(work)
        lock.unlock()
        semaphore.signal()
    }

    func run<T>(_ block: @escaping () -> T) async -> T {
        await withCheckedContinuation { continuation in
            enqueue {
                continuation.resume(returning: block())
            }
        }
    }

    func runThrowing<T>(_ block: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            enqueue {
                do {
                    continuation.resume(returning: try block())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Blocking variant, only for use from deinit (which can't be async).
    func runSync(_ block: @escaping () -> Void) {
        let sem = DispatchSemaphore(value: 0)
        enqueue {
            block()
            sem.signal()
        }
        sem.wait()
    }
}

final class LlamaContext {
    private var model: OpaquePointer
    private var context: OpaquePointer
    private var vocab: OpaquePointer
    private var sampling: UnsafeMutablePointer<llama_sampler>
    private var batch: llama_batch
    private var tokens_list: [llama_token]
    var is_done: Bool = false

    /// This variable is used to store temporarily invalid cchars
    private var temporary_invalid_cchars: [CChar]

    var n_len: Int32 = 1024
    var n_cur: Int32 = 0

    var n_decode: Int32 = 0

    init(model: OpaquePointer, context: OpaquePointer) {
        self.model = model
        self.context = context
        self.tokens_list = []
        self.batch = llama_batch_init(2048, 0, 1)
        self.temporary_invalid_cchars = []
        let sparams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(sparams)
        llama_sampler_chain_add(self.sampling, llama_sampler_init_top_k(40))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_top_p(0.9, 1))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.4))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(1234))
        vocab = llama_model_get_vocab(model)
    }

    deinit {
        let m = model
        let c = context
        let s = sampling
        let b = batch
        LlamaWorker.shared.runSync {
            llama_sampler_free(s)
            llama_batch_free(b)
            llama_model_free(m)
            llama_free(c)
        }
    }

    static func create_context(path: String) async throws -> LlamaContext {
        try await LlamaWorker.shared.runThrowing {
            var model_params = llama_model_default_params()

            model_params.n_gpu_layers = 99
            print("Offloading up to 99 layers to GPU (Metal)")

            let model = llama_model_load_from_file(path, model_params)
            guard let model else {
                print("Could not load model at \(path)")
                throw LlamaError.couldNotInitializeContext
            }

            let n_threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
            print("Using \(n_threads) threads")

            var ctx_params = llama_context_default_params()
            ctx_params.n_ctx = 2048
            ctx_params.n_threads       = Int32(n_threads)
            ctx_params.n_threads_batch = Int32(n_threads)

            let context = llama_init_from_model(model, ctx_params)
            guard let context else {
                print("Could not load context!")
                throw LlamaError.couldNotInitializeContext
            }

            return LlamaContext(model: model, context: context)
        }
    }

    func model_info() async -> String {
        await LlamaWorker.shared.run { [model] in
            let result = UnsafeMutablePointer<Int8>.allocate(capacity: 256)
            result.initialize(repeating: Int8(0), count: 256)
            defer {
                result.deallocate()
            }

            let nChars = llama_model_desc(model, result, 256)
            let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nChars))

            var SwiftString = ""
            for char in bufferPointer {
                SwiftString.append(Character(UnicodeScalar(UInt8(char))))
            }

            return SwiftString
        }
    }

    func get_n_tokens() async -> Int32 {
        await LlamaWorker.shared.run { [batch] in
            batch.n_tokens
        }
    }

    func completion_init(messages: [(role: String, content: String)]) async {
        await LlamaWorker.shared.run { [self] in
            self.is_done = false

            var formattedPrompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, concise assistant.<|eot_id|>"
            for msg in messages {
                formattedPrompt += "<|start_header_id|>\(msg.role)<|end_header_id|>\n\n\(msg.content)<|eot_id|>"
            }
            formattedPrompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

            print("attempting to complete \"\(formattedPrompt)\"")

            self.tokens_list = self.tokenize(text: formattedPrompt, add_bos: false)
            self.temporary_invalid_cchars = []

            let n_ctx = Int(llama_n_ctx(self.context))

            let reservedForResponse = 64
            let maxPromptTokens = max(1, n_ctx - reservedForResponse)
            if self.tokens_list.count > maxPromptTokens {
                let dropped = self.tokens_list.count - maxPromptTokens
                self.tokens_list = Array(self.tokens_list.suffix(maxPromptTokens))
                print("Conversation too long for context window — dropped \(dropped) oldest tokens, kept \(self.tokens_list.count)")
            }
            self.n_len = Int32(min(1024, n_ctx - self.tokens_list.count))

            print("\n n_len = \(self.n_len), n_ctx = \(n_ctx), prompt_tokens = \(self.tokens_list.count)")

            for id in self.tokens_list {
                print(String(cString: self.token_to_piece(token: id) + [0]))
            }

            llama_batch_clear(&self.batch)

            for i1 in 0..<self.tokens_list.count {
                let i = Int(i1)
                llama_batch_add(&self.batch, self.tokens_list[i], Int32(i), [0], false)
            }
            self.batch.logits[Int(self.batch.n_tokens) - 1] = 1 // true

            if llama_decode(self.context, self.batch) != 0 {
                print("llama_decode() failed")
            }
            llama_synchronize(self.context)

            self.n_cur = self.batch.n_tokens
        }
    }

    func completion_loop() async -> String {
        await LlamaWorker.shared.run { [self] in
            var new_token_id: llama_token = 0

            new_token_id = llama_sampler_sample(self.sampling, self.context, self.batch.n_tokens - 1)

            if llama_vocab_is_eog(self.vocab, new_token_id) || self.n_cur == self.n_len {
                if self.n_decode == 0 {
                    let reason = llama_vocab_is_eog(self.vocab, new_token_id) ? "immediate EOG token (id \(new_token_id))" : "n_cur==n_len before any token"
                    print("[DIAGNOSTIC] Stopped after zero generated tokens. tokens_list.count=\(self.tokens_list.count), n_cur=\(self.n_cur), n_len=\(self.n_len), reason=\(reason)")
                    self.temporary_invalid_cchars.append(contentsOf: Array("[debug: 0 tokens, \(reason), prompt_tokens=\(self.tokens_list.count)]".utf8.map { CChar(bitPattern: $0) }))
                }
                self.is_done = true
                let new_token_str = String(cString: self.temporary_invalid_cchars + [0])
                self.temporary_invalid_cchars.removeAll()
                return new_token_str
            }

            let new_token_cchars = self.token_to_piece(token: new_token_id)
            self.temporary_invalid_cchars.append(contentsOf: new_token_cchars)
            let new_token_str: String
            if let string = String(validatingUTF8: self.temporary_invalid_cchars + [0]) {
                self.temporary_invalid_cchars.removeAll()
                new_token_str = string
            } else if (0 ..< self.temporary_invalid_cchars.count).contains(where: {$0 != 0 && String(validatingUTF8: Array(self.temporary_invalid_cchars.suffix($0)) + [0]) != nil}) {
                let string = String(cString: self.temporary_invalid_cchars + [0])
                self.temporary_invalid_cchars.removeAll()
                new_token_str = string
            } else {
                new_token_str = ""
            }
            print(new_token_str)

            llama_batch_clear(&self.batch)
            llama_batch_add(&self.batch, new_token_id, self.n_cur, [0], true)

            self.n_decode += 1
            self.n_cur    += 1

            if llama_decode(self.context, self.batch) != 0 {
                print("failed to evaluate llama!")
            }
            llama_synchronize(self.context)

            return new_token_str
        }
    }

    func bench(pp: Int, tg: Int, pl: Int, nr: Int = 1) async -> String {
        await LlamaWorker.shared.run { [self] in
            var pp_avg: Double = 0
            var tg_avg: Double = 0

            var pp_std: Double = 0
            var tg_std: Double = 0

            for _ in 0..<nr {
                llama_batch_clear(&self.batch)

                let n_tokens = pp

                for i in 0..<n_tokens {
                    llama_batch_add(&self.batch, 0, Int32(i), [0], false)
                }
                self.batch.logits[Int(self.batch.n_tokens) - 1] = 1 // true

                llama_memory_clear(llama_get_memory(self.context), false)

                let t_pp_start = DispatchTime.now().uptimeNanoseconds / 1000;

                if llama_decode(self.context, self.batch) != 0 {
                    print("llama_decode() failed during prompt")
                }
                llama_synchronize(self.context)

                let t_pp_end = DispatchTime.now().uptimeNanoseconds / 1000;

                llama_memory_clear(llama_get_memory(self.context), false)

                let t_tg_start = DispatchTime.now().uptimeNanoseconds / 1000;

                for i in 0..<tg {
                    llama_batch_clear(&self.batch)

                    for j in 0..<pl {
                        llama_batch_add(&self.batch, 0, Int32(i), [Int32(j)], true)
                    }

                    if llama_decode(self.context, self.batch) != 0 {
                        print("llama_decode() failed during text generation")
                    }
                    llama_synchronize(self.context)
                }

                let t_tg_end = DispatchTime.now().uptimeNanoseconds / 1000;

                llama_memory_clear(llama_get_memory(self.context), false)

                let t_pp = Double(t_pp_end - t_pp_start) / 1000000.0
                let t_tg = Double(t_tg_end - t_tg_start) / 1000000.0

                let speed_pp = Double(pp)    / t_pp
                let speed_tg = Double(pl*tg) / t_tg

                pp_avg += speed_pp
                tg_avg += speed_tg

                pp_std += speed_pp * speed_pp
                tg_std += speed_tg * speed_tg

                print("pp \(speed_pp) t/s, tg \(speed_tg) t/s")
            }

            pp_avg /= Double(nr)
            tg_avg /= Double(nr)

            if nr > 1 {
                pp_std = sqrt(pp_std / Double(nr - 1) - pp_avg * pp_avg * Double(nr) / Double(nr - 1))
                tg_std = sqrt(tg_std / Double(nr - 1) - tg_avg * tg_avg * Double(nr) / Double(nr - 1))
            } else {
                pp_std = 0
                tg_std = 0
            }

            let result_model = UnsafeMutablePointer<Int8>.allocate(capacity: 256)
            result_model.initialize(repeating: Int8(0), count: 256)
            let nChars = llama_model_desc(self.model, result_model, 256)
            let bufferPointer = UnsafeBufferPointer(start: result_model, count: Int(nChars))
            var model_desc = ""
            for char in bufferPointer {
                model_desc.append(Character(UnicodeScalar(UInt8(char))))
            }
            result_model.deallocate()

            let model_size     = String(format: "%.2f GiB", Double(llama_model_size(self.model)) / 1024.0 / 1024.0 / 1024.0);
            let model_n_params = String(format: "%.2f B", Double(llama_model_n_params(self.model)) / 1e9);
            let backend        = "Metal";
            let pp_avg_str     = String(format: "%.2f", pp_avg);
            let tg_avg_str     = String(format: "%.2f", tg_avg);
            let pp_std_str     = String(format: "%.2f", pp_std);
            let tg_std_str     = String(format: "%.2f", tg_std);

            var result = ""

            result += String("| model | size | params | backend | test | t/s |\n")
            result += String("| --- | --- | --- | --- | --- | --- |\n")
            result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | pp \(pp) | \(pp_avg_str) ± \(pp_std_str) |\n")
            result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | tg \(tg) | \(tg_avg_str) ± \(tg_std_str) |\n")

            return result
        }
    }

    func clear() async {
        await LlamaWorker.shared.run { [self] in
            self.tokens_list.removeAll()
            self.temporary_invalid_cchars.removeAll()
            llama_memory_clear(llama_get_memory(self.context), false)
            llama_sampler_reset(self.sampling)
        }
    }

    private func tokenize(text: String, add_bos: Bool) -> [llama_token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (add_bos ? 1 : 0) + 1
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        let tokenCount = llama_tokenize(vocab, text, Int32(utf8Count), tokens, Int32(n_tokens), add_bos, true)

        var swiftTokens: [llama_token] = []
        for i in 0..<tokenCount {
            swiftTokens.append(tokens[Int(i)])
        }

        tokens.deallocate()

        return swiftTokens
    }

    /// - note: The result does not contain null-terminator
    private func token_to_piece(token: llama_token) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }
        let nTokens = llama_token_to_piece(vocab, token, result, 8, 0, false)

        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nNewTokens = llama_token_to_piece(vocab, token, newResult, -nTokens, 0, false)
            let bufferPointer = UnsafeBufferPointer(start: newResult, count: Int(nNewTokens))
            return Array(bufferPointer)
        } else {
            let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer)
        }
    }
}