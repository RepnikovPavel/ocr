// Safetensors reader (pure C++, mmap'd) and a multi-shard ModelWeights store.
//
// We mmap each shard read-only and keep the mapping alive for the lifetime of
// the model — the weights total ~6 GiB, and an mmap lets the OS page them in
// lazily instead of allocating a 6 GiB host buffer up front. Tensors are
// uploaded to the GPU lazily as each module initialises, so cold memory stays
// out of device VRAM until needed.
#pragma once

#include "tensor.h"

#include <nlohmann/json.hpp>

#include <sys/mman.h>   // MAP_FAILED
#include <string>
#include <vector>
#include <unordered_map>

namespace dots {

struct TensorDescriptor {
    std::string         name;
    Dtype               dtype;
    std::vector<int>    dims;     // PyTorch shape, e.g. [out, in] for Linear
    uint8_t*            host_ptr; // points into the mmap'd shard
    std::size_t         nbytes;
};

struct MappedFile {
    void*       base = nullptr;
    std::size_t len  = 0;
};

struct ModelWeights {
    std::string                            ckpt_dir;
    int                                    device = 0;
    std::vector<TensorDescriptor>          tensors;
    std::unordered_map<std::string, const TensorDescriptor*> by_name;
    std::vector<MappedFile>                mappings;   // keeps shards mapped

    static ModelWeights load(const std::string& ckpt_dir, int device = 0);

    ModelWeights() = default;
    ~ModelWeights();
    ModelWeights(const ModelWeights&) = delete;
    ModelWeights& operator=(const ModelWeights&) = delete;
    // Movable so load() can return by value: the mmap mappings are just
    // pointer+length pairs that transfer ownership cleanly.
    ModelWeights(ModelWeights&& o) noexcept
        : ckpt_dir(std::move(o.ckpt_dir)), device(o.device),
          tensors(std::move(o.tensors)), by_name(std::move(o.by_name)),
          mappings(std::move(o.mappings)) { o.mappings.clear(); }
    ModelWeights& operator=(ModelWeights&& o) noexcept {
        if (this != &o) {
            for (auto& m : mappings)
                if (m.base && m.base != MAP_FAILED) ::munmap(m.base, m.len);
            ckpt_dir = std::move(o.ckpt_dir); device = o.device;
            tensors = std::move(o.tensors); by_name = std::move(o.by_name);
            mappings = std::move(o.mappings); o.mappings.clear();
        }
        return *this;
    }

    // Upload a named tensor into `dst` (device memory). Casts f32->bf16 if the
    // checkpoint stored a weight in f32 (e.g. some norm tables). Returns false
    // if the tensor is missing or the cast is unsupported.
    bool copy_to_device(const std::string& name, void* dst, Dtype dst_dtype) const;

    std::vector<int> shape(const std::string& name) const;
    bool has(const std::string& name) const;
};

}  // namespace dots
