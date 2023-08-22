open System
open System.Numerics
open System.Threading.Tasks

type span = Span<float32>
type memory = Memory<float32>

// helper functions
module Vectorized =
    type vec = Vector<float32>
    let vcount = vec.Count

    let inline (<<<) (out: span) (x: vec) = x.CopyTo(out)

    let inline max (x: span) =
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

    let inline dot (x : span) (w : span) =
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
    let inline add (out: span) (x: span) (y: span) =
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
    let inline addscaled (out: span) (x: span) (k: float32) =
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
    let inline mult (out: span) (x: span) (y: span) =
        let len = out.Length

        let mutable i = 0
        while i <= len-vcount do
            out.Slice(i) <<< vec(x.Slice(i)) * vec(y.Slice(i))
            i <- i + vcount

        i <- len-len%vcount
        while i < len do
            out[i] <- x[i] * y[i]
            i <- i + 1

module Span =
    let inline maxi (x : span) =
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


let rmsnorm (out: span) (x: span) (w: span) =
    let s = 1f / sqrt(1e-5f + (Vectorized.dot x x) / (float32 x.Length))
    Vectorized.mult out x w
    Vectorized.scale out s

let matmul (out: memory) (x: memory) (w: memory) =
    let len = x.Length
    Parallel.For(0, w.Length / len, fun i -> out.Span[i] <- Vectorized.dot x.Span (w.Slice(i * len, len).Span)) |> ignore

let inplace_softmax (x: span) =
    let maxx = Vectorized.max x

    let mutable denom = 0f
    let mutable i = 0
    while i < x.Length do
        x[i] <- exp(x[i] - maxx)
        denom <- denom + x[i]
        i <- i + 1

    Vectorized.scale x (1f / denom)


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
    token_embedding_table: memory
    /// (layer, dim) rmsnorm weights
    rms_att_weight: memory
    /// (layer, dim)
    rms_ffn_weight: memory
    // weights for matmuls
    /// (layer, dim, dim)
    wq: memory
    /// (layer, dim, dim)
    wk: memory
    /// (layer, dim, dim)
    wv: memory
    /// (layer, dim, dim)
    wo: memory
    // weights for ffn
    /// (layer, hidden_dim, dim)
    w1: memory
    /// (layer, dim, hidden_dim)
    w2: memory
    /// (layer, hidden_dim, dim)
    w3: memory
    // final rmsnorm
    /// (dim,)
    rms_final_weight: memory
    // freq_cis for RoPE relatively positional embeddings
    /// (seq_len, dim/2)
    freq_cis_real: memory
    /// (seq_len, dim/2)
    freq_cis_imag: memory
    /// Last layer classifier: (vocab_size, dim)
    wcls: Option<memory>
}

let allocate size =
    memory (Array.zeroCreate(size))

type RunState(cfg: Config, weights: TransformerWeights) = 
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
        let offset = layer * cfg.dim * cfg.dim
        let length = cfg.dim * cfg.dim

        let wq = weights.wq.Slice(offset, length)
        let wk = weights.wk.Slice(offset, length)
        let wv = weights.wv.Slice(offset, length)
        
        matmul q xb wq
        matmul k xb wk
        matmul v xb wv

    let cache_kv pos layer =
        let offset = layer * cfg.dim * cfg.seq_len + pos * cfg.dim
        let length = cfg.dim

        k.CopyTo(key_cache.Slice(offset, length))
        v.CopyTo(value_cache.Slice(offset, length))

    let rope pos =
        let head_size = cfg.dim / cfg.n_heads

        let cis_real = weights.freq_cis_real.Slice(pos * head_size / 2, head_size / 2).Span
        let cis_imag = weights.freq_cis_imag.Slice(pos * head_size / 2, head_size / 2).Span

        let mutable h = 0
        while h < cfg.n_heads do
            let q' = q.Slice(h * head_size, head_size).Span
            let k' = k.Slice(h * head_size, head_size).Span

            let mutable j = 0
            while j < head_size do
                let (q0, q1) = q'[j], q'[j+1]
                let (k0, k1) = k'[j], k'[j+1]

                let fcr = cis_real[j/2]
                let fci = cis_imag[j/2]

                q'[j] <- q0 * fcr - q1 * fci
                q'[j+1] <- q0 * fci + q1 * fcr
                k'[j] <- k0 * fcr - k1 * fci
                k'[j+1] <- k0 * fci + k1 * fcr

                j <- j + 2

            h <- h + 1

    let attention_head (q: memory) (xb: memory) (layer_cached_keys: memory) (layer_cached_vals: memory) (att: memory) pos =
        let head_size = cfg.dim / cfg.n_heads

        let multi = 1f / sqrt(float32 head_size)
        Parallel.For(0, pos + 1, fun t -> 
                                    let k = layer_cached_keys.Slice(t * cfg.dim, head_size).Span
                                    att.Span[t] <- (Vectorized.dot k q.Span) * multi) |> ignore

        inplace_softmax (att.Slice(0, pos+1).Span)
        xb.Span.Clear()

        let mutable t = 0
        while t <= pos do
            let v = layer_cached_vals.Slice(t * head_size * cfg.n_heads, head_size).Span
            Vectorized.addscaled xb.Span v (att.Span[t])
            t <- t + 1

    let attention pos layer =
        assert (pos < cfg.seq_len)

        let length = cfg.seq_len * cfg.dim
        let layer_cached_keys = key_cache.Slice(layer * length, length)
        let layer_cached_vals = value_cache.Slice(layer * length, length)

        let head_size = cfg.dim / cfg.n_heads

        Parallel.For(0, cfg.n_heads, fun h -> attention_head
                                                (q.Slice(h * head_size, head_size))
                                                (xb.Slice(h * head_size, head_size))
                                                (layer_cached_keys.Slice(h * head_size))
                                                (layer_cached_vals.Slice(h * head_size))
                                                (att.Slice(h * cfg.seq_len, cfg.seq_len))
                                                pos) |> ignore

    let ffn layer =
        let rms_ffn_w = weights.rms_ffn_weight.Slice(layer * cfg.dim, cfg.dim)
        rmsnorm xb.Span x.Span rms_ffn_w.Span

        let length = cfg.dim * cfg.hidden_dim
        let w1 = weights.w1.Slice(layer * length, length)
        let w2 = weights.w2.Slice(layer * length, length)
        let w3 = weights.w3.Slice(layer * length, length)

        matmul hb xb w1
        matmul hb2 xb w3

        let mutable i = 0
        while i < hb.Length do
            hb.Span[i] <- hb.Span[i] * (1f / (1f + exp(-hb.Span[i]))) * hb2.Span[i]
            i <- i + 1

        matmul xb hb w2

    member _.out_logits with get() = logits

    member _.step token pos =
        weights.token_embedding_table
            .Slice(token * cfg.dim, cfg.dim)
            .CopyTo(x)

        for layer in 0..cfg.n_layers-1 do
            rmsnorm xb.Span x.Span (weights.rms_att_weight.Slice(layer * cfg.dim, cfg.dim).Span)
            qkv_for_layer layer
            rope pos
            cache_kv pos layer
            attention pos layer
            matmul xb2 xb (weights.wo.Slice(layer * cfg.dim * cfg.dim, cfg.dim * cfg.dim))
            Vectorized.add x.Span x.Span xb2.Span
            ffn layer
            Vectorized.add x.Span x.Span xb.Span

        rmsnorm x.Span x.Span (weights.rms_final_weight.Span)
        matmul logits x (if weights.wcls.IsSome then weights.wcls.Value else weights.token_embedding_table)


let read_vocab vocab_size path =
    use stream = System.IO.File.OpenRead(path)
    use reader = new System.IO.BinaryReader(stream)

    let max_token_length = reader.ReadInt32()

    [| 
        for i in 0..vocab_size-1 ->
            let score = reader.ReadSingle()
            let len = reader.ReadInt32()
            Text.Encoding.UTF8.GetString(reader.ReadBytes(len))
    |]

let read_config_and_weights path =
    use stream = System.IO.File.OpenRead(path)
    use reader = new System.IO.BinaryReader(stream)
    
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
        memory [| for i in 0..size - 1 -> reader.ReadSingle() |]

    let head_size = cfg.dim / cfg.n_heads

    cfg, {
        token_embedding_table = read (cfg.vocab_size * cfg.dim)
        rms_att_weight = read (cfg.n_layers * cfg.dim)
        wq = read (cfg.n_layers * cfg.dim * cfg.dim)
        wk = read (cfg.n_layers * cfg.dim * cfg.dim)
        wv = read (cfg.n_layers * cfg.dim * cfg.dim)
        wo = read (cfg.n_layers * cfg.dim * cfg.dim)
        rms_ffn_weight = read (cfg.n_layers * cfg.dim)
        w1 = read (cfg.n_layers * cfg.dim * cfg.hidden_dim)
        w2 = read (cfg.n_layers * cfg.dim * cfg.hidden_dim)
        w3 = read (cfg.n_layers * cfg.dim * cfg.hidden_dim)
        rms_final_weight = read (cfg.dim)
        freq_cis_real = read (cfg.seq_len * (head_size / 2))
        freq_cis_imag = read (cfg.seq_len * (head_size / 2))
        wcls = if cfg.shared_weights then Some (read (cfg.vocab_size * cfg.dim)) else None
    }

let cfg, weights = read_config_and_weights @"stories110M.bin"
let voc = read_vocab (cfg.vocab_size) @"tokenizer.bin"

let state = RunState(cfg, weights)

let mutable token = 1

#time "on"

let mutable pos = 0
let watch = System.Diagnostics.Stopwatch.StartNew()

while pos < cfg.seq_len do
    state.step token pos
    let next = Span.maxi state.out_logits.Span
                
    printf "%s" (voc[next])
    pos <- pos + 1
    token <- next

watch.Stop()
let ps = double(int64 pos * 1000L) / double(watch.ElapsedMilliseconds)

printfn $"\n {ps} Tokens/Sec"


