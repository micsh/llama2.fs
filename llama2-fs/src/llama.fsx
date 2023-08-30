#nowarn "3391"

open System
open System.IO
open System.Numerics
open System.Runtime.InteropServices
open System.Threading.Tasks

type span = Span<float32>
type rospan = ReadOnlySpan<float32>
type memory = Memory<float32>
type romemory = ReadOnlyMemory<float32>

// helper functions
module Vectorized =
    type vec = Vector<float32>
    let vcount = vec.Count

    let inline (<<<) (out: span) (x: vec) = x.CopyTo(out)

    let inline max (x: rospan) =
        let len = x.Length
        let mutable max = x[0]
        let mutable MAX = vec(max)

        let mutable i = 0
        if len >= vcount then
            while i <= len-vcount do
               MAX <- Vector.Max(vec(x.Slice(i)), MAX)
               i <- i + vcount

            for j in 0..vcount-1 do
                if MAX[j] > max then max <- MAX[j]
    
        i <- len-len%vcount
        while i < len do
            if x[i] > max then max <- x[i]
            i <- i + 1

        max

    let inline dot (x : rospan) (w : rospan) =
        assert(x.Length = w.Length)
        let len = x.Length

        let mutable result = 0f
        let mutable i = 0
        while i <= len-vcount do
            result <- result + Vector.Dot(vec(x.Slice(i)), vec(w.Slice(i)))
            i <- i + vcount
    
        i <- len-len%vcount
        while i < len do
            result <- result + (x[i] * w[i])
            i <- i + 1

        result

    /// out = x + y
    let inline add (out: span) (x: rospan) (y: rospan) =
        assert(x.Length = y.Length && x.Length = out.Length)
        let len = out.Length

        let mutable i = 0
        while i <= len-vcount do
            out.Slice(i) <<< vec(x.Slice(i)) + vec(y.Slice(i))
            i <- i + vcount

        i <- len-len%vcount   
        while i < len do
            out[i] <- x[i] + y[i]
            i <- i + 1

    /// out = out + x * k
    let inline addscaled (out: span) (x: rospan) (k: float32) =
        assert(out.Length = x.Length)
        let len = x.Length
        let mutable i = 0
        while i <= len-vcount do
            out.Slice(i) <<< vec(out.Slice(i)) + vec(x.Slice(i)) * k
            i <- i + vcount

        i <- len-len%vcount   
        while i < len do
            out[i] <- out[i] + x[i] * k
            i <- i + 1

    /// out = out * x (scalar)
    let inline scale (out: span) (x: float32) =
        let len = out.Length
    
        let mutable i = 0
        while i <= len-vcount do
            out.Slice(i) <<< vec(out.Slice(i)) * x
            i <- i + vcount
            
        i <- len-len%vcount   
        while i < len do
            out[i] <- out[i] * x
            i <- i + 1

    /// out = x * y
    let inline mult (out: span) (x: rospan) (y: rospan) (k: float32) =
        assert(out.Length = x.Length && x.Length = y.Length)
        let len = out.Length

        let mutable i = 0
        while i <= len-vcount do
            out.Slice(i) <<< vec(x.Slice(i)) * vec(y.Slice(i)) * k
            i <- i + vcount

        i <- len-len%vcount
        while i < len do
            out[i] <- x[i] * y[i] * k
            i <- i + 1

module Span =
    let inline maxi (x : rospan) =
        let mutable max = x[0]
        let mutable index = 0

        let mutable i = 1
        while i < x.Length do
            if x[i] > max then
                max <- x[i]
                index <- i
            i <- i + 1

        index

// --------------------------------------------------------------------------------------------------------------

type Vocab = {
    vocab: string[]
    map: Map<string, float32 * int>
}

module Vocab =
    let read_vocab vocab_size path =
        use stream = File.OpenRead(path)
        use reader = new BinaryReader(stream)

        let max_token_length = reader.ReadInt32()

        let vocab = [| 
            for i in 0..vocab_size-1 ->
                let score = reader.ReadSingle()
                let len = reader.ReadInt32()
                Text.Encoding.UTF8.GetString(reader.ReadBytes(len)), score
        |]

        { vocab = vocab |> Array.map fst; map = vocab |> Array.mapi (fun i (token: string, score: float32) -> token, (score, i)) |> Map.ofArray }

    let score tok vocab = if vocab.map |> Map.containsKey tok then fst vocab.map[tok] else Single.MinValue
    let index tok vocab = if vocab.map |> Map.containsKey tok then snd vocab.map[tok] else -1
    let symbol i vocab = if i >= 0 && i < vocab.vocab.Length then vocab.vocab[i] else "<unk>"

    let bpe_encode text vocab =
        let mutable encoded = [| for c in text -> c.ToString() |]
    
        let rec loop () =
            let bestPair = encoded
                            |> Array.pairwise
                            |> Array.mapi (fun i (a, b) -> i, vocab |> score (a + b), a + b)
                            |> Array.maxBy (fun (_,scr,_) -> scr)

            let i, scr, tok = bestPair
            if scr > Single.MinValue then
                encoded[i] <- tok
                Array.Copy(encoded, i + 2, encoded, i + 1, encoded.Length - i - 2)
                Array.Resize(&encoded, encoded.Length - 1)
                loop ()

        loop ()
        encoded |> Array.map (fun tok -> vocab |> index tok) |> List.ofArray

// --------------------------------------------------------------------------------------------------------------

type Config = {
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    shared_weights: bool
}

type TransformerWeights = {
    /// (vocab_size, dim)
    token_embedding_table: romemory
    /// (layer, dim) rmsnorm weights
    rms_att_weight: romemory
    /// (layer, dim)
    rms_ffn_weight: romemory
    // weights for matmuls
    /// (layer, dim, dim)
    wq: romemory
    /// (layer, dim, dim)
    wk: romemory
    /// (layer, dim, dim)
    wv: romemory
    /// (layer, dim, dim)
    wo: romemory
    // weights for ffn
    /// (layer, hidden_dim, dim)
    w1: romemory
    /// (layer, dim, hidden_dim)
    w2: romemory
    /// (layer, hidden_dim, dim)
    w3: romemory
    // final rmsnorm
    /// (dim,)
    rms_final_weight: romemory
    /// Last layer classifier: (vocab_size, dim)
    wcls: romemory
}

let read_config_and_weights path =
    use stream = File.OpenRead(path)
    use reader = new BinaryReader(stream)
    
    let cfg = match ({ 
                    dim = reader.ReadInt32()
                    hidden_dim = reader.ReadInt32()
                    n_layers = reader.ReadInt32()
                    n_heads = reader.ReadInt32()
                    n_kv_heads = reader.ReadInt32()
                    vocab_size = reader.ReadInt32()
                    seq_len = reader.ReadInt32()
                    shared_weights = false
                }) with 
                | c when c.vocab_size >= 0 -> c 
                | c -> { c with vocab_size = -c.vocab_size; shared_weights = true }

    let read size =
        let floats = Array.zeroCreate<float32>(size)
        let maxChunkSize = Int32.MaxValue / 4
        let times, remainder = floats.Length / maxChunkSize, floats.Length % maxChunkSize

        for i in 0..times-1 do
            let bytes = MemoryMarshal.AsBytes(floats.AsSpan(i * maxChunkSize, maxChunkSize))
            reader.Read(bytes) |> ignore

        let bytes = MemoryMarshal.AsBytes(floats.AsSpan(times * maxChunkSize, remainder))
        reader.Read(bytes) |> ignore

        memory floats

    let layers_dim_size = cfg.n_layers * cfg.dim
    let token_embedding_table = read (cfg.vocab_size * cfg.dim)

    cfg, {
        token_embedding_table = token_embedding_table
        rms_att_weight = read layers_dim_size
        wq = read (layers_dim_size * cfg.dim)
        wk = read (layers_dim_size * cfg.dim)
        wv = read (layers_dim_size * cfg.dim)
        wo = read (layers_dim_size * cfg.dim)
        rms_ffn_weight = read layers_dim_size
        w1 = read (layers_dim_size * cfg.hidden_dim)
        w2 = read (layers_dim_size * cfg.hidden_dim)
        w3 = read (layers_dim_size * cfg.hidden_dim)
        rms_final_weight = read cfg.dim
        wcls = if cfg.shared_weights then read (cfg.vocab_size * cfg.dim) else token_embedding_table
    }

// --------------------------------------------------------------------------------------------------------------

let rmsnorm (out: memory) (x: romemory) (w: romemory) =
    let s = 1f / sqrt(1e-5f + (Vectorized.dot x.Span x.Span) / (float32 x.Length))
    Vectorized.mult out.Span x.Span w.Span s

let matmul (out: memory) (x: romemory) (w: romemory) =
    assert(out.Length = w.Length / x.Length)
    let len = x.Length
    Parallel.For(0, out.Length, fun i -> out.Span[i] <- Vectorized.dot x.Span (w.Slice(i * len, len).Span)) |> ignore

let inplace_softmax (x: span) =
    let maxx = Vectorized.max x

    let mutable denom = 0f
    let mutable i = 0
    while i < x.Length do
        x[i] <- exp(x[i] - maxx)
        denom <- denom + x[i]
        i <- i + 1

    Vectorized.scale x (1f / denom)

type RunState(cfg: Config, weights: TransformerWeights) = 
    let forLayer layer (weights: romemory) = 
        let len = weights.Length / cfg.n_layers
        weights.Slice(len * layer, len)

    let allocate size = memory (Array.zeroCreate(size))

    let x = allocate cfg.dim // activation at current time stamp (dim,)
    let xb = allocate cfg.dim // same, but inside a residual branch (dim,)
    let xb2 = allocate cfg.dim // an additional buffer just for convenience (dim,)
    let hb = allocate cfg.hidden_dim // buffer for hidden dimension in the ffn (hidden_dim,)
    let hb2 = allocate cfg.hidden_dim // buffer for hidden dimension in the ffn (hidden_dim,)
    let q =  allocate cfg.dim // query (dim,)
    let k =  allocate cfg.dim // key (dim,)
    let v =  allocate cfg.dim // value (dim,)
    let att =  allocate (cfg.seq_len * cfg.n_heads) // buffer for scores/attention values (seq_len,)
    let logits = allocate cfg.vocab_size // output logits
    // kv cache
    let key_cache = allocate (cfg.n_layers * cfg.seq_len * cfg.dim) // (layer, seq_len, dim)
    let value_cache = allocate (cfg.n_layers * cfg.seq_len * cfg.dim) // (layer, seq_len, dim)

    let qkv_for_layer layer =
        matmul q xb (weights.wq |> forLayer layer)
        matmul k xb (weights.wk |> forLayer layer)
        matmul v xb (weights.wv |> forLayer layer)

    let cache_kv pos layer =
        let offset = layer * cfg.dim * cfg.seq_len + pos * cfg.dim
        let length = cfg.dim

        k.CopyTo(key_cache.Slice(offset, length))
        v.CopyTo(value_cache.Slice(offset, length))

    let rope pos =
        let head_size = cfg.dim / cfg.n_heads
        let kv_dim = head_size * cfg.n_kv_heads

        let mutable i = 0
        while i < cfg.dim do
            let head_dim = i % head_size
            let freq = 1f / MathF.Pow(1000000f, float32 head_dim / float32 head_size)
            let vall = float32 pos * freq
            let fcr = cos vall
            let fci = sin vall
            let rotn = if i < kv_dim then 2 else 1

            let mutable v = 0
            while v < rotn do
                let vec = if v = 0 then q else k
                let v0 = vec.Span[i]
                let v1 = vec.Span[i+1]
                vec.Span[i] <- v0 * fcr - v1 * fci
                vec.Span[i + 1] <- v0 * fci + v1 * fcr
                v <- v + 1

            i <- i + 2

    let attention_head (xb: memory) (att: memory) (q: romemory) (layer_cached_keys: romemory) (layer_cached_vals: romemory) =
        let head_size = cfg.dim / cfg.n_heads

        let multi = 1f / sqrt(float32 head_size)
        let mutable t = 0
        while t < att.Length do
            let k = layer_cached_keys.Slice(t * cfg.dim, head_size)
            att.Span[t] <- (Vectorized.dot k.Span q.Span) * multi
            t <- t + 1

        inplace_softmax (att.Span)

        xb.Span.Clear()
        let mutable t = 0
        while t < att.Length do
            let v = layer_cached_vals.Slice(t * head_size * cfg.n_heads, head_size)
            Vectorized.addscaled xb.Span v.Span att.Span[t]
            t <- t + 1

    let attention pos layer =
        assert (pos < cfg.seq_len)

        let length = cfg.seq_len * cfg.dim
        let layer_cached_keys = key_cache.Slice(layer * length, length)
        let layer_cached_vals = value_cache.Slice(layer * length, length)

        let head_size = cfg.dim / cfg.n_heads

        Parallel.For(0, cfg.n_heads, fun h -> attention_head
                                                (xb.Slice(h * head_size, head_size))
                                                (att.Slice(h * cfg.seq_len, pos + 1))
                                                (q.Slice(h * head_size, head_size))
                                                (layer_cached_keys.Slice(h * head_size))
                                                (layer_cached_vals.Slice(h * head_size))) |> ignore

    let ffn layer =
        rmsnorm xb x (weights.rms_ffn_weight |> forLayer layer)

        matmul hb xb (weights.w1 |> forLayer layer)
        matmul hb2 xb (weights.w3 |> forLayer layer)

        let mutable i = 0
        while i < hb.Length do
            hb.Span[i] <- hb.Span[i] * (1f / (1f + exp(-hb.Span[i]))) * hb2.Span[i]
            i <- i + 1

        matmul xb hb (weights.w2 |> forLayer layer)

    let mutable last_pos = 0
    member _.pos = last_pos
    member _.max_length = cfg.seq_len
    member _.out_logits with get() = logits

    member _.step token pos =
        last_pos <- pos
        weights.token_embedding_table
            .Slice(token * cfg.dim, cfg.dim)
            .CopyTo(x)

        for layer in 0..cfg.n_layers-1 do
            rmsnorm xb x (weights.rms_att_weight |> forLayer layer)
            qkv_for_layer layer
            rope pos
            cache_kv pos layer
            attention pos layer
            matmul xb2 xb (weights.wo |> forLayer layer)
            Vectorized.add x.Span x.Span xb2.Span
            ffn layer
            Vectorized.add x.Span x.Span xb.Span

        rmsnorm x x weights.rms_final_weight
        matmul logits x weights.wcls


let generate pos prompt (state: RunState) =
    let generator (token, pos, prompt) =
        if token = 2 || pos >= state.max_length then
            None
        else
            state.step token pos
            let next = match prompt with | h::t -> h | [] -> Span.maxi state.out_logits.Span 
            Some(next, (next, pos + 1, (match prompt with | h::t -> t | [] -> prompt)))

    Seq.unfold generator (1, pos, prompt)

// -----------------------------------------------------------------------------------------------

type ChatBot(state: RunState, voc) = 
    let watch = Diagnostics.Stopwatch()
    let mutable next_pos = 0

    member _.restart () =
        next_pos <- 0

    member _.chat (text: string) =
        let prompt = voc |> Vocab.bpe_encode $" [INST] {text} [/INST]"
        let to_symbol tok = voc |> Vocab.symbol tok

        let spos = next_pos
        watch.Restart()
        if next_pos >= state.max_length then
            printfn "Reached the end. restart is required"
        else
            state |> generate next_pos prompt |> Seq.iter (to_symbol >> printf "%s")
            next_pos <- state.pos + 1

        watch.Stop()
        let ps = double(int64 (next_pos - spos) * 1000L) / double(watch.ElapsedMilliseconds)
        printfn $"\n {ps} Tokens/Sec"

//-----------------------------------------------------------------------------------------------------

let cfg, weights = read_config_and_weights @"../../CodeLlama-7b-Instruct.bin"//"../../stories110M.bin"//
let vocab = Vocab.read_vocab cfg.vocab_size @"../../tokenizer.bin"
let state = RunState(cfg, weights)

//-----------------------------------------------------------------------------------------------------

#time "on"

let readCodeLines st en =
    let lines = File.ReadAllLines(@"llama.fsx")[st..en]
    "\n```\n" + (lines |> String.concat "\n") + "\n```\n"
    
let bot = ChatBot(state, vocab)
bot.chat ("Can you help me improve how we interact? Here's the code: " + readCodeLines 269 474)


//"You're job is to help reduce the numbers of tokens, while retaining contextual information !
//* New tokens should replace existing tokens from the start, and retain the final tokens as they are.
//* The tokens will be presented immediately following the instructions.


//<s>[INST] As an expert in the field, can you figure out how to replace tokens at the start of your short term memory with fewer tokens that keep the relevant information to the rest of the conversation? [/INST]  As an AI language model, I can provide some insights on how to optimize the use of tokens in short-term memory for conversational AI systems. Here are some strategies that can help reduce the number of tokens required to retain relevant information during a conversation:

//1. Use a hierarchical tokenization approach: Instead of treating each token as an independent unit, group related tokens into hierarchical categories. For example, in a conversation about a restaurant, you could group tokens related to the restaurant's name, location, and cuisine into separate categories. This can help reduce the overall number of tokens required to retain relevant information.
//2. Apply context-dependent tokenization: Tokenize the conversation based on the context of the conversation. For instance, in a conversation about a specific topic, you could tokenize related tokens based on the topic, such as separating tokens related to the topic's definition, examples, and applications.
//3. Use a hybrid tokenization approach: Combine both hierarchical and context-dependent tokenization approaches to create a hybrid tokenization system. This can help capture both the structural and contextual information in the conversation.
//4. Use a smaller set of core tokens: Instead of using a large set of tokens to represent each concept, use a smaller set of core tokens that can be combined to represent more complex ideas. For example, in a conversation about a movie, you could use a smaller set of core tokens to represent the movie's title, director, genre, and plot.
//5. Use a token pruning approach: As the conversation progresses, prune out tokens that are no longer relevant or useful. This can help reduce the overall number of tokens required to retain relevant information.
//6. Use a memory-based approach: Instead of relying solely on tokenization, use a memory-based approach to store and retrieve relevant information. This can help reduce the overall number of tokens required to retain relevant information by leveraging the memory of the system.
//7. Use a transfer learning approach: If possible, use a transfer learning approach to leverage pre-trained models and fine-tune them for the specific conversation task at hand. This can help reduce the overall number of tokens required to retain relevant information by leveraging the knowledge learned from similar conversations.
//8. Use a multi-modal representation: Instead of relying solely on textual tokens, use a multi-modal representation that incorporates other modalities such as speech, vision, or even physiological signals. This can help reduce the overall number of tokens required to retain relevant information by leveraging the complementary information from other modalities.
//9. Use a generative model: Use a generative model, such as a Generative Adversarial Network (GAN) or Variational Autoencoder (VAE), to generate new tokens based on the existing ones. This can help reduce the overall number of tokens required to retain relevant information by generating new tokens that are similar to the existing ones.
//10. Use a reinforcement learning approach: Use a reinforcement learning approach to learn the optimal tokenization strategy based on the feedback from the conversation. This can help reduce the overall number of tokens required to retain relevant information by learning the most effective tokenization strategy for the specific conversation task at hand.

//By applying these strategies, you can reduce the number of tokens required to retain relevant information during a conversation while still maintaining the accuracy and relevance of the information being retained.
//</s>"
//You can ask me for the list of files and their sizes by typing '[please provide list of files]' or '[get length of *filename]' and you can ask me for their content by '[get content of *filename]' or '[get content of *filename from offset x with length y]'):\n")

//bot.chat "Here's the list of files:\n\
//tokenizer.bin
//tokenizer.model
//"

//bot.chat "Here are lengths:\n\
//423KB
//489KB
//"

//bot.chat "Here's the content for tokenizer.bin:\n\
//let rmsnorm (out: span) (x: rospan) (w: rospan) =
//    let s = 1f / sqrt(1e-5f + (Vectorized.dot x x) / (float32 x.Length))
//    Vectorized.mult out x w
//    Vectorized.scale out s
//"


//bot.chat "Here's the content for tokenizer.model:\n\
//let matmul (out: memory) (x: romemory) (w: romemory) =
//    assert(out.Length = w.Length / x.Length)
//    let len = x.Length
//    Parallel.For(0, out.Length, fun i -> out.Span[i] <- Vectorized.dot x.Span (w.Slice(i * len, len).Span)) |> ignore
//"

