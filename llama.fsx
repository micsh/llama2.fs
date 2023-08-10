open System

type span = Span<float32>
type memory = Memory<float32>

// helper functions

open System.Numerics

module Span =
    let inline maxi (x : span) =
        let mutable max = x[0]
        let mutable index = 0

        for i in 1..x.Length - 1 do
            if x[i] > max then
                max <- x[i]
                index <- i

        index


let inline dot (x : span) (w : span) =
    let count = Vector<float32>.Count
    
    let mutable result = 0f
    let mutable i = 0
    while i <= x.Length - count do
        result <- result + Vector.Dot(Vector<float32>(x.Slice(i)), Vector<float32>(w.Slice(i)))
        i <- i + count
    
    i <- x.Length - x.Length % count
    while i < x.Length do
        result <- result + (x.[i] * w.[i])
        i <- i + 1

    result

let inline max (x: span) =
    let len = x.Length
    let mutable max = x[0]
    let count = Vector<float32>.Count
    let minValue = Single.MinValue

    let mutable i = 0
    let mutable maxV = Vector<float32>(max)
    if len >= count then
        while i <= len-count do
           let v = Vector<float32>(x.Slice(i))
           maxV <- Vector.Max(v, maxV)
           i <- i + count

        for j=0 to count-1 do
            if maxV[j] > max then max <- maxV[j]
    
    i <- len-len%count
    while i < x.Length do
        if x[i] > max then max <- x[i]
        i <- i + 1
    max

let inline mult_scalar (out: span) (x: float32) =
    let len = out.Length
    let count = Vector<float32>.Count
    
    let mutable i = 0
    while i <= len-count do
        (Vector<float32>(out.Slice(i)) * x).CopyTo(out.Slice(i))
        i <- i + count
            
    i <- len-len%count   
    while i < out.Length do
        out[i] <- out[i] * x
        i <- i + 1

//let inline mult_scalar (out: span) x =
//    for i in 0..out.Length - 1 do
//        out[i] <- out[i] * x

let inline mult (out: span) (x: span) (y: span) =
    let len = out.Length
    let count = Vector<float32>.Count

    let mutable i = 0
    while i <= len-count do
        (Vector<float32>(x.Slice(i)) * Vector<float32>(y.Slice(i))).CopyTo(out.Slice(i))
        i <- i + count

    i <- len-len%count   
    while i < out.Length do
        out[i] <- x[i] * y[i]
        i <- i + 1


//let inline mult (out: span) (x: span) (y: span) =
//    for i in 0..x.Length - 1 do
//        out[i] <- x[i] * y[i]

//let inline dot (x: span) (w: span) =
//    let mutable s = 0f
//    for i in 0..x.Length - 1 do
//        s <- s + x[i] * w[i]
//    s

// --------------------------------------------------------------------------------------------------------------


let rmsnorm (out: span) (x: span) (w: span) =
    let s = 1f / sqrt(1e-5f + (dot x x) / (float32 x.Length))
    mult out x w
    mult_scalar out s

let matmul (out: memory) (x: memory) (w: memory) =
    let len = x.Length
    let parts = w.Length / len

    System.Threading.Tasks.Parallel.For(0, parts, fun i -> out.Span[i] <- dot (w.Span.Slice(i * len, len)) x.Span) |> ignore
    //for i in 0..parts-1 do
    //    out.Span[i] <- dot (w.Span.Slice(i * len, len)) x.Span

let inplace_softmax (x: span) =
    let len = x.Length

    let maxx = max x
    //for i in 0..len-1 do
    //    if x[i] > maxx then
    //        maxx <- x[i]

    let mutable denom = 0f

    for i in 0..len-1 do
        x[i] <- exp(x[i] - maxx)
        denom <- denom + x[i]

    for i in 0..len-1 do
        x[i] <- x[i] / denom


type Config = {
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
}

let inline _attention_head layer h (q: memory) (xb: memory) (layer_cached_keys: memory) (layer_cached_vals: memory) (att: memory) pos (cfg:Config) =
    let head_size = cfg.dim / cfg.n_heads

    for t in 0..pos do
        let k = layer_cached_keys.Slice(t * head_size * cfg.n_heads, head_size)
        let score = (dot k.Span q.Span) / sqrt(float32 head_size)
        att.Span[t] <- score

    inplace_softmax (att.Span.Slice(0, pos+1))

    for i in 0..xb.Length - 1 do
        xb.Span[i] <- 0f

    for t in 0..pos do
        let v = layer_cached_vals.Slice(t * head_size * cfg.n_heads, head_size)
        let attn_w = att.Span[t]

        let len = v.Length
        for i in 0..len - 1 do
            xb.Span[i] <- xb.Span[i] + v.Span[i] * attn_w

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

type RunState(cfg: Config) = 
    /// activation at current time stamp (dim,)
    let x = allocate cfg.dim
    /// same, but inside a residual branch (dim,)
    let xb = allocate cfg.dim
    /// an additional buffer just for convenience (dim,)
    let xb2 = allocate cfg.dim
    /// buffer for hidden dimension in the ffn (hidden_dim,)
    let hb = allocate cfg.hidden_dim
    /// buffer for hidden dimension in the ffn (hidden_dim,)
    let hb2 = allocate cfg.hidden_dim
    /// query (dim,)
    let q =  allocate cfg.dim
    /// key (dim,)
    let k =  allocate cfg.dim
    /// value (dim,)
    let v =  allocate cfg.dim
    /// buffer for scores/attention values (seq_len,)
    let att =  allocate (cfg.seq_len * cfg.n_heads)
    /// output logits
    let logits = allocate cfg.vocab_size
    // kv cache
    /// (layer, seq_len, dim)
    let key_cache = allocate (cfg.n_layers * cfg.seq_len * cfg.dim)
    /// (layer, seq_len, dim)
    let value_cache = allocate (cfg.n_layers * cfg.seq_len * cfg.dim)

    let qkv_for_layer l (w: TransformerWeights) =
        let wq = w.wq.Slice(l * cfg.dim * cfg.dim, cfg.dim * cfg.dim)
        let wk = w.wk.Slice(l * cfg.dim * cfg.dim, cfg.dim * cfg.dim)
        let wv = w.wv.Slice(l * cfg.dim * cfg.dim, cfg.dim * cfg.dim)
        
        matmul q xb wq
        matmul k xb wk
        matmul v xb wv

    let cache_kv pos layer =
        let offset = layer * cfg.dim * cfg.seq_len + pos * cfg.dim
        let kc = key_cache.Slice(offset, cfg.dim)
        let vc = value_cache.Slice(offset, cfg.dim)
        
        k.CopyTo(kc)
        v.CopyTo(vc)

    let rope pos (w: TransformerWeights) =
        let head_size = cfg.dim / cfg.n_heads
        let parts = q.Length / head_size

        let cis_real = w.freq_cis_real.Span.Slice(pos * head_size / 2, head_size / 2)
        let cis_imag = w.freq_cis_imag.Span.Slice(pos * head_size / 2, head_size / 2)

        for i in 0..parts-1 do
            let q' = q.Span.Slice(i * head_size, head_size)
            let k' = k.Span.Slice(i * head_size, head_size)

            for j in 0..2..head_size-1 do
                let (q0, q1) = q'[j], q'[j+1]
                let (k0, k1) = k'[j], k'[j+1]

                let fcr = cis_real[j/2]
                let fci = cis_imag[j/2]

                q'[j] <- q0 * fcr - q1 * fci
                q'[j+1] <- q0 * fci + q1 * fcr
                k'[j] <- k0 * fcr - k1 * fci
                k'[j+1] <- k0 * fci + k1 * fcr

    let attention pos layer =
        assert (pos < cfg.seq_len)

        let head_size = cfg.dim / cfg.n_heads
        let layer_cached_keys = key_cache.Slice(layer * cfg.seq_len * cfg.dim, cfg.seq_len * cfg.dim)
        let layer_cached_vals = value_cache.Slice(layer * cfg.seq_len * cfg.dim, cfg.seq_len * cfg.dim)

        System.Threading.Tasks.Parallel.For(
            0,
            cfg.n_heads,
            fun h ->  _attention_head
                        layer
                        h
                        (q.Slice(h * head_size, head_size))
                        (xb.Slice(h * head_size, head_size))
                        (layer_cached_keys.Slice(h * head_size))
                        (layer_cached_vals.Slice(h * head_size))
                        (att.Slice(h * cfg.seq_len, cfg.seq_len))
                        pos
                        cfg) |> ignore

        //for h in 0..cfg.n_heads - 1 do
        //    _attention_head
        //                    layer
        //                    h
        //                    (q.Slice(h * head_size, head_size))
        //                    (xb.Slice(h * head_size, head_size))
        //                    layer_cached_keys
        //                    (value_cache.Slice(0, cfg.n_layers * cfg.seq_len * cfg.dim))
        //                    (att.Slice(h * cfg.seq_len, cfg.seq_len))
        //                    pos
        //                    cfg

    let ffn l (w: TransformerWeights) =
        let rms_ffn_w = w.rms_ffn_weight.Span.Slice(l * cfg.dim, cfg.dim)
        rmsnorm xb.Span x.Span rms_ffn_w

        let w1 = w.w1.Slice(cfg.dim * cfg.hidden_dim * l, cfg.hidden_dim * cfg.dim)
        let w2 = w.w2.Slice(cfg.dim * cfg.hidden_dim * l, cfg.hidden_dim * cfg.dim)
        let w3 = w.w3.Slice(cfg.dim * cfg.hidden_dim * l, cfg.hidden_dim * cfg.dim)

        matmul hb xb w1
        matmul hb2 xb w3

        for i in 0..hb.Length - 1 do
            hb.Span[i] <- hb.Span[i] * (1f / (1f + exp(-hb.Span[i]))) * hb2.Span[i]

        matmul xb hb w2

    member _.out_logits with get() = logits

    member _.step token pos (w: TransformerWeights) =
        w.token_embedding_table
            .Slice(token * cfg.dim, cfg.dim)
            .CopyTo(x)

        for l in 0..cfg.n_layers-1 do
            let rms_attn_w = w.rms_att_weight.Span.Slice(l * cfg.dim, cfg.dim)
            rmsnorm xb.Span x.Span rms_attn_w

            qkv_for_layer l w
            rope pos w
            cache_kv pos l
            attention pos l

            let wo = w.wo.Slice(l * cfg.dim * cfg.dim, cfg.dim * cfg.dim)
            matmul xb2 xb wo

            for i in 0..x.Length - 1 do
                x.Span[i] <- x.Span[i] + xb2.Span[i]

            ffn l w

            for i in 0..x.Length - 1 do
                x.Span[i] <- x.Span[i] + xb.Span[i]

        rmsnorm x.Span x.Span (w.rms_final_weight.Span)

        if w.wcls.IsSome then
            matmul logits x (w.wcls.Value)
        else
            matmul logits x (w.token_embedding_table)



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
    
    let cfg = { 
        dim = reader.ReadInt32()
        hidden_dim = reader.ReadInt32()
        n_layers = reader.ReadInt32()
        n_heads = reader.ReadInt32()
        n_kv_heads = reader.ReadInt32()
        vocab_size = Math.Abs(reader.ReadInt32())
        seq_len = reader.ReadInt32()
    }

    let read size =
        let bytes = reader.ReadBytes(size * sizeof<float32>)
        let floats = Array.zeroCreate<float32>(size)
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length)
        memory(floats)
        //[| for i in 0..size - 1 -> reader.ReadSingle() |]

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
        wcls = None
    }

let cfg, weights = read_config_and_weights @"stories110M.bin"
let voc = read_vocab (cfg.vocab_size) @"tokenizer.bin"

let state = RunState(cfg)

let mutable token = 1

#time "on"

let mutable pos = 0
let watch = System.Diagnostics.Stopwatch.StartNew()

while pos < cfg.seq_len do
    state.step token pos weights
    let next = Span.maxi state.out_logits.Span
                
    printf "%s" (voc[next])
    pos <- pos + 1
    token <- next

watch.Stop()
let ps = double(int64 pos * 1000L) / double(watch.ElapsedMilliseconds)

printfn $"\n {ps} Tokens/Sec"