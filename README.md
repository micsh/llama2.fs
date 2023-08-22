
## llama2.fs 🦀

> Have you ever wanted to inference a baby [Llama 2](https://ai.meta.com/llama/) model in pure C? No? Well, now you can!

## Full Llama2 Support 🚀🚀
We can now run the full **llama2-7B**!!  No memory mapping for now, so all the weights must fit in memory (~26Gb). On my codespaces VM with 16 cores and 64Gb memory, the inference runs at 1.4 tokens per second.



## Performance

It's fast: ~1 token per sec on 7B chat model, on a 6 core machine.

## Keeping up with the original
I'm pretty sure that `llama2.c` is going to move fast and get lots of contributions. 

So any contribution is welcome here!

### Contribution Ideas
- 8bit vectorization.

## License
MIT
