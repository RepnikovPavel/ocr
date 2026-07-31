#include "safetensors_loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace dots {

// Parse a single .safetensors file (8-byte LE header length, JSON, raw bf16).
// One object maps exactly one file; the .index.json across shards is handled
// by the higher-level ModelWeights loader below.
static bool load_one_safetensors(const std::string& path,
                                 std::vector<TensorDescriptor>& out,
                                 MappedFile& mapping) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror(("open " + path).c_str()); return false; }

    struct stat st;
    if (::fstat(fd, &st) != 0) { perror("fstat"); ::close(fd); return false; }
    std::size_t file_size = static_cast<std::size_t>(st.st_size);

    void* base = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); ::close(fd); return false; }
    // fd can be closed after mmap; the mapping stays valid until munmap.
    ::close(fd);

    mapping.base = base;
    mapping.len  = file_size;

    // 8-byte LE header length.
    if (file_size < 8) { fprintf(stderr, "%s: too small\n", path.c_str()); return false; }
    auto* u8 = static_cast<uint8_t*>(base);
    uint64_t header_len = 0;
    for (int i = 0; i < 8; ++i) header_len |= uint64_t(u8[i]) << (8 * i);
    if (8 + header_len > file_size) {
        fprintf(stderr, "%s: header length %llu exceeds file size %zu\n",
                path.c_str(), (unsigned long long)header_len, file_size);
        return false;
    }

    std::string header_json(reinterpret_cast<const char*>(u8 + 8),
                            static_cast<std::size_t>(header_len));
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(header_json);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s: bad header JSON: %s\n", path.c_str(), e.what());
        return false;
    }
    if (!j.is_object()) { fprintf(stderr, "%s: header not an object\n", path.c_str()); return false; }

    const uint8_t* data_region = u8 + 8 + header_len;
    std::size_t    data_size   = file_size - (8 + header_len);

    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& key = it.key();
        if (key == "__metadata__") continue;
        const auto& v = it.value();
        if (!v.is_object()) continue;
        if (!v.contains("dtype") || !v.contains("shape") || !v.contains("data_offsets"))
            continue;
        const std::string& dtype_s = v["dtype"];
        const auto& shape = v["shape"];
        auto off0 = v["data_offsets"][0].get<uint64_t>();
        auto off1 = v["data_offsets"][1].get<uint64_t>();

        Dtype dt = Dtype::BF16;
        if      (dtype_s == "BF16") dt = Dtype::BF16;
        else if (dtype_s == "F32")  dt = Dtype::F32;
        else if (dtype_s == "F16")  dt = Dtype::BF16;       // treat as bf16-sized
        else if (dtype_s == "I64")  dt = Dtype::I64;
        else if (dtype_s == "I32")  dt = Dtype::I32;
        else if (dtype_s == "BOOL") dt = Dtype::BOOL;
        else if (dtype_s == "U8")   dt = Dtype::U8;
        else continue;

        std::vector<int> dims;
        for (const auto& d : shape) dims.push_back(d.get<int>());

        TensorDescriptor t;
        t.name   = key;
        t.dtype  = dt;
        t.dims   = dims;
        t.host_ptr = const_cast<uint8_t*>(data_region + off0);
        t.nbytes   = static_cast<std::size_t>(off1 - off0);
        if (off1 > data_size) {
            fprintf(stderr, "%s: tensor %s offsets exceed data region\n",
                    path.c_str(), key.c_str());
            return false;
        }
        out.push_back(std::move(t));
    }
    return true;
}

// ---- ModelWeights: collects all shards, uploads to device on demand ---------

ModelWeights ModelWeights::load(const std::string& ckpt_dir, int device) {
    ModelWeights w;
    w.ckpt_dir = ckpt_dir;
    w.device   = device;

    // If model.safetensors.index.json exists, load shards in the listed order;
    // otherwise look for a single model.safetensors.
    std::string index_path = ckpt_dir + "/model.safetensors.index.json";

    std::ifstream idxf(index_path);
    std::vector<std::string> shards;
    if (idxf) {
        nlohmann::json idx;
        idxf >> idx;
        std::map<std::string, std::string> uniq;
        for (auto it = idx["weight_map"].begin(); it != idx["weight_map"].end(); ++it)
            uniq[it.value()] = it.key();   // dedup shard filenames
        for (const auto& kv : uniq) shards.push_back(kv.first);
        std::sort(shards.begin(), shards.end());
    } else {
        std::string single = ckpt_dir + "/model.safetensors";
        std::ifstream test(single, std::ios::binary);
        if (test) shards.push_back("model.safetensors");
    }
    if (shards.empty()) {
        fprintf(stderr, "no safetensors shards found in %s\n", ckpt_dir.c_str());
        return w;
    }

    w.mappings.reserve(shards.size());
    for (const auto& s : shards) {
        std::string full = ckpt_dir + "/" + s;
        w.mappings.emplace_back();
        MappedFile& m = w.mappings.back();
        if (!load_one_safetensors(full, w.tensors, m)) {
            fprintf(stderr, "failed to load shard %s\n", full.c_str());
            w.tensors.clear();
            return w;
        }
    }
    // Index by name for O(1) lookup.
    for (auto& t : w.tensors) w.by_name[t.name] = &t;
    fprintf(stderr, "[loader] %zu tensors from %zu shard(s), mapped host-side\n",
            w.tensors.size(), shards.size());
    return w;
}

ModelWeights::~ModelWeights() {
    for (auto& m : mappings)
        if (m.base && m.base != MAP_FAILED) ::munmap(m.base, m.len);
}

// Upload a named tensor to the device into a caller-provided device buffer.
// Reshapes a 2-D weight to [rows, cols] matching the conventional Linear
// storage: PyTorch nn.Linear.weight is [out_features, in_features].
bool ModelWeights::copy_to_device(const std::string& name, void* dst,
                                  Dtype dst_dtype) const {
    auto it = by_name.find(name);
    if (it == by_name.end()) {
        fprintf(stderr, "[loader] tensor '%s' not found\n", name.c_str());
        return false;
    }
    const TensorDescriptor& t = *it->second;
    if (dst_dtype == t.dtype) {
        cudaError_t e = cudaMemcpy(dst, t.host_ptr, t.nbytes, cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            fprintf(stderr, "[loader] cudaMemcpy H2D failed for '%s': %s (dst=%p host=%p nbytes=%zu dtype=%d)\n",
                    name.c_str(), cudaGetErrorString(e), dst, t.host_ptr, t.nbytes, (int)t.dtype);
            return false;
        }
        return true;
    }
    // bf16<->f32 host-side conversion is not used by the engine (everything
    // ships as bf16), but keep a f32->bf16 path for a tokenizer/embedding table
    // that the HF checkpoint may have stored as f32.
    if (t.dtype == Dtype::F32 && dst_dtype == Dtype::BF16) {
        std::size_t n = t.nbytes / 4;
        std::vector<float> tmp_f(n);
        std::memcpy(tmp_f.data(), t.host_ptr, t.nbytes);
        std::vector<bf16> tmp_b(n);
        for (std::size_t i = 0; i < n; ++i)
            tmp_b[i] = __float2bfloat16(tmp_f[i]);
        DOTS_CUDA_CHECK(cudaMemcpy(dst, tmp_b.data(), n * sizeof(bf16),
                                   cudaMemcpyHostToDevice));
        return true;
    }
    fprintf(stderr, "[loader] unsupported cast %d->%d for '%s'\n",
            (int)t.dtype, (int)dst_dtype, name.c_str());
    return false;
}

std::vector<int> ModelWeights::shape(const std::string& name) const {
    auto it = by_name.find(name);
    return it == by_name.end() ? std::vector<int>{} : it->second->dims;
}

bool ModelWeights::has(const std::string& name) const {
    return by_name.count(name) != 0;
}

}  // namespace dots
