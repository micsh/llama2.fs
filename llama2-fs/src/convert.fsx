open System
open System.IO
open System.Numerics
open System.Runtime.InteropServices

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

type Vec = Vector
type vec<^a> = Vector<^a>

let read_config (reader: BinaryReader) = 
    match   ({ 
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

let write_config cfg (writer: BinaryWriter) =
    writer.Write(cfg.dim)
    writer.Write(cfg.hidden_dim)
    writer.Write(cfg.n_layers)
    writer.Write(cfg.n_heads)
    writer.Write(cfg.n_kv_heads)
    writer.Write(if cfg.shared_weights then -cfg.vocab_size else cfg.vocab_size)
    writer.Write(cfg.seq_len)


let SIZE = 2_000_000_000
let ONE = Memory (Array.zeroCreate<float32>(SIZE))

let read size (reader: BinaryReader) =
    let buffer = ONE.Slice(0, size)
    buffer.Span.Clear()

    let maxChunkSize = Int32.MaxValue / 4
    let times, remainder = buffer.Length / maxChunkSize, buffer.Length % maxChunkSize

    for i in 0..times-1 do
        let bytes = MemoryMarshal.AsBytes(buffer.Slice(i * maxChunkSize, maxChunkSize).Span)
        reader.Read(bytes) |> ignore

    let bytes = MemoryMarshal.AsBytes(buffer.Slice(times * maxChunkSize, remainder).Span)
    reader.Read(bytes) |> ignore
        
    buffer

let write (mem: _ Memory) (writer: BinaryWriter) =
    printfn "writing %d values" mem.Length

    let maxChunkSize = Int32.MaxValue / 4
    let times, remainder = mem.Length / maxChunkSize, mem.Length % maxChunkSize

    for i in 0..times-1 do
        let bytes = MemoryMarshal.AsBytes(mem.Slice(i * maxChunkSize, maxChunkSize).Span)
        writer.Write(bytes) |> ignore

    let bytes = MemoryMarshal.AsBytes(mem.Slice(times * maxChunkSize, remainder).Span)
    writer.Write(bytes) |> ignore

let TWO = Memory (Array.zeroCreate<int16>(SIZE))

let downsize (w: Span<float32>) =
    let out = TWO.Slice(0, w.Length)
    let sp = out.Span
    
    let mutable i = 0
    while i < w.Length do
        let a = vec<float32>(w.Slice(i))
        let b = vec<float32>(w.Slice(i + 8))
        Vec.Narrow(Vec.ShiftRightArithmetic(Vec.AsVectorInt32(a), 16), Vec.ShiftRightArithmetic(Vec.AsVectorInt32(b), 16)).CopyTo(sp.Slice(i))
        i <- i + 16

    out

let copy (reader: BinaryReader) (writer: BinaryWriter) =
    let read size = read size reader
    let write span = write span writer

    let cfg = read_config reader
    write_config cfg writer

    let layers_dim_size = cfg.n_layers * cfg.dim

    // copy token_embedding_table
    let b = read (cfg.vocab_size * cfg.dim) in write (downsize b.Span)
    // copy rms_att_weight
    let b = read (layers_dim_size) in write b
    
    // copy wq
    let b = read (layers_dim_size * cfg.dim) in write (downsize b.Span)
    // copy wk
    let b = read (layers_dim_size * cfg.dim) in write (downsize b.Span)
    // copy wv
    let b = read (layers_dim_size * cfg.dim) in write (downsize b.Span)
    // copy wo
    let b = read (layers_dim_size * cfg.dim) in write (downsize b.Span)
    
    // copy rms_ffn_weight
    let b = read (layers_dim_size) in write b
    // copy w1
    let b = read (layers_dim_size * cfg.hidden_dim) in write (downsize b.Span)
    // copy w2
    let b = read (layers_dim_size * cfg.hidden_dim) in write (downsize b.Span)
    // copy w3
    let b = read (layers_dim_size * cfg.hidden_dim) in write (downsize b.Span)
    
    // copy rms_final_weight
    let b = read (cfg.dim) in write b

    // copy wcls
    if cfg.shared_weights then
        let b = read (cfg.vocab_size * cfg.dim) in write (downsize b.Span)


let start origin_path dest_path =
    use ostream = File.OpenRead(origin_path)
    use reader = new BinaryReader(ostream)


    use dstream = File.OpenWrite(dest_path)
    use writer = new BinaryWriter(dstream)

    copy reader writer

let origin_path = "../../CodeLlama-7b-Instruct.bin"
let dest_path = "../../CodeLlama-7b-Instruct-short.bin"

start origin_path dest_path