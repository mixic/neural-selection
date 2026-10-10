// Included from main.cu inside namespace sim, after BehaviorTotals is defined.
constexpr uint32_t CHECKPOINT_VERSION = 1;
constexpr size_t CHECKPOINT_IO_CHUNK = 1 << 20;
constexpr int CHECKPOINT_RECORD_CHUNK = 8192;

struct CheckpointHeader {
    char magic[8];
    uint32_t version, endian;
    uint64_t layout;
    int64_t step;
    int64_t checkpoint_every;
    unsigned long long seed;
    int initial, trace_animal;
    float food_renewal;
    uint32_t flags, neuron_count, synapse_count;
    uint64_t csv_size, behavior_size, lives_size;
    char output[512];
};

uint64_t checkpoint_layout() {
    const uint64_t values[] = {P, NEURON_CAPACITY, SYNAPSE_CAPACITY, S, M,
        W, H, DENDRITES, HISTORY, TICKS, sizeof(Animal), sizeof(NeuronGene),
        sizeof(NeuronState), sizeof(SynapseGene), sizeof(SynapseState),
        sizeof(CheckpointNeuron), sizeof(CheckpointSynapse),
        sizeof(BehaviorTotals), sizeof(LifeRecord), sizeof(BiteEvent)};
    uint64_t hash = 1469598103934665603ull;
    for (uint64_t value : values) {
        hash ^= value;
        hash *= 1099511628211ull;
    }
    return hash;
}

void checkpoint_write_bytes(gzFile file, const void *data, size_t bytes) {
    const auto *source = static_cast<const unsigned char *>(data);
    while (bytes) {
        unsigned int chunk = static_cast<unsigned int>(
            std::min(bytes, CHECKPOINT_IO_CHUNK));
        if (gzwrite(file, source, chunk) != static_cast<int>(chunk))
            throw std::runtime_error(std::string("Checkpoint write: ") +
                                     gzerror(file, nullptr));
        source += chunk;
        bytes -= chunk;
    }
}

void checkpoint_read_bytes(gzFile file, void *data, size_t bytes) {
    auto *target = static_cast<unsigned char *>(data);
    while (bytes) {
        unsigned int chunk = static_cast<unsigned int>(
            std::min(bytes, CHECKPOINT_IO_CHUNK));
        if (gzread(file, target, chunk) != static_cast<int>(chunk))
            throw std::runtime_error(std::string("Checkpoint is incomplete or corrupt: ") +
                                     gzerror(file, nullptr));
        target += chunk;
        bytes -= chunk;
    }
}

template <typename T> void checkpoint_write_gpu(gzFile file, const T *device,
                                                 size_t count) {
    std::vector<unsigned char> buffer(CHECKPOINT_IO_CHUNK);
    const size_t bytes = count * sizeof(T);
    const auto *source = reinterpret_cast<const unsigned char *>(device);
    for (size_t offset = 0; offset < bytes; offset += buffer.size()) {
        size_t chunk = std::min(buffer.size(), bytes - offset);
        check(cudaMemcpy(buffer.data(), source + offset, chunk,
                         cudaMemcpyDeviceToHost), "checkpoint GPU download");
        checkpoint_write_bytes(file, buffer.data(), chunk);
    }
}

template <typename T> void checkpoint_read_gpu(gzFile file, T *device,
                                                size_t count) {
    std::vector<unsigned char> buffer(CHECKPOINT_IO_CHUNK);
    const size_t bytes = count * sizeof(T);
    auto *target = reinterpret_cast<unsigned char *>(device);
    for (size_t offset = 0; offset < bytes; offset += buffer.size()) {
        size_t chunk = std::min(buffer.size(), bytes - offset);
        checkpoint_read_bytes(file, buffer.data(), chunk);
        check(cudaMemcpy(target + offset, buffer.data(), chunk,
                         cudaMemcpyHostToDevice), "checkpoint GPU upload");
    }
}

void checkpoint_write_observer(gzFile file, const BehaviorTotals &dead_totals,
        const std::deque<std::pair<int64_t, BehaviorTotals>> &behavior_history,
        const std::deque<std::pair<uint64_t, BiteEvent>> &bite_history,
        uint64_t next_bite_event_id) {
    checkpoint_write_bytes(file, &dead_totals, sizeof(dead_totals));
    uint32_t count = static_cast<uint32_t>(behavior_history.size());
    checkpoint_write_bytes(file, &count, sizeof(count));
    for (const auto &[step, totals] : behavior_history) {
        checkpoint_write_bytes(file, &step, sizeof(step));
        checkpoint_write_bytes(file, &totals, sizeof(totals));
    }
    count = static_cast<uint32_t>(bite_history.size());
    checkpoint_write_bytes(file, &count, sizeof(count));
    for (const auto &[id, event] : bite_history) {
        checkpoint_write_bytes(file, &id, sizeof(id));
        checkpoint_write_bytes(file, &event, sizeof(event));
    }
    checkpoint_write_bytes(file, &next_bite_event_id,
                           sizeof(next_bite_event_id));
}

void checkpoint_read_observer(gzFile file, BehaviorTotals &dead_totals,
        std::deque<std::pair<int64_t, BehaviorTotals>> &behavior_history,
        std::deque<std::pair<uint64_t, BiteEvent>> &bite_history,
        uint64_t &next_bite_event_id) {
    checkpoint_read_bytes(file, &dead_totals, sizeof(dead_totals));
    uint32_t count = 0;
    checkpoint_read_bytes(file, &count, sizeof(count));
    if (count > 10002) throw std::runtime_error("Invalid checkpoint behavior history");
    for (uint32_t i = 0; i < count; ++i) {
        int64_t step;
        BehaviorTotals totals;
        checkpoint_read_bytes(file, &step, sizeof(step));
        checkpoint_read_bytes(file, &totals, sizeof(totals));
        behavior_history.emplace_back(step, totals);
    }
    checkpoint_read_bytes(file, &count, sizeof(count));
    if (count > 1024) throw std::runtime_error("Invalid checkpoint bite history");
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t id;
        BiteEvent event;
        checkpoint_read_bytes(file, &id, sizeof(id));
        checkpoint_read_bytes(file, &event, sizeof(event));
        bite_history.emplace_back(id, event);
    }
    checkpoint_read_bytes(file, &next_bite_event_id,
                          sizeof(next_bite_event_id));
}

void checkpoint_write_state(gzFile file, Device d, void *selection_temp,
        size_t selection_temp_bytes, CheckpointHeader &header,
        const BehaviorTotals &dead_totals,
        const std::deque<std::pair<int64_t, BehaviorTotals>> &behavior_history,
        const std::deque<std::pair<uint64_t, BiteEvent>> &bite_history,
        uint64_t next_bite_event_id) {
    auto slots = thrust::make_counting_iterator<int>(0);
    check(cub::DeviceSelect::If(selection_temp, selection_temp_bytes, slots,
          d.occupied_neurons, d.occupied_neuron_count, NEURON_CAPACITY,
          OwnedSlot{d.neuron_owner}), "checkpoint neuron selection");
    check(cub::DeviceSelect::If(selection_temp, selection_temp_bytes, slots,
          d.occupied_synapses, d.occupied_synapse_count, SYNAPSE_CAPACITY,
          OwnedSlot{d.synapse_owner}), "checkpoint synapse selection");
    int neurons = 0, synapses = 0, pending_lives = 0;
    check(cudaMemcpy(&neurons, d.occupied_neuron_count, sizeof(int),
                     cudaMemcpyDeviceToHost), "checkpoint neuron count");
    check(cudaMemcpy(&synapses, d.occupied_synapse_count, sizeof(int),
                     cudaMemcpyDeviceToHost), "checkpoint synapse count");
    check(cudaMemcpy(&pending_lives, d.life_count, sizeof(int),
                     cudaMemcpyDeviceToHost), "checkpoint life count");
    if (pending_lives != 0)
        throw std::runtime_error("Checkpoint requires a completed report");
    std::vector<Animal> animals;
    download(animals, d.animals, P);
    uint64_t allocated_neurons = 0, allocated_synapses = 0;
    for (const Animal &animal : animals)
        if (animal.alive == 1) {
            allocated_neurons += animal.neuron_count;
            allocated_synapses += animal.synapse_count;
        }
    if (allocated_neurons != static_cast<uint64_t>(neurons) ||
        allocated_synapses != static_cast<uint64_t>(synapses))
        throw std::runtime_error("Checkpoint pool ownership disagrees with living brains");
    header.neuron_count = neurons;
    header.synapse_count = synapses;
    checkpoint_write_bytes(file, &header, sizeof(header));
    checkpoint_write_observer(file, dead_totals, behavior_history,
                              bite_history, next_bite_event_id);
    checkpoint_write_gpu(file, d.animals, P);
    checkpoint_write_gpu(file, d.food, W * H);
    checkpoint_write_gpu(file, d.kind, W * H);
    checkpoint_write_gpu(file, d.senses, P * S);
    checkpoint_write_gpu(file, d.motors, P * M);
    checkpoint_write_gpu(file, d.births, 1);
    checkpoint_write_gpu(file, d.deaths, 1);
    checkpoint_write_gpu(file, d.predation_deaths, 1);
    checkpoint_write_gpu(file, d.bites, 1);
    checkpoint_write_gpu(file, d.next_id, 1);
    checkpoint_write_gpu(file, d.founder_motor_steps, M);
    checkpoint_write_gpu(file, d.founder_observed_steps, 1);
    checkpoint_write_gpu(file, d.trace_exc_arrivals, M);
    checkpoint_write_gpu(file, d.trace_exc_releases, M);
    checkpoint_write_gpu(file, d.trace_inh_arrivals, M);
    checkpoint_write_gpu(file, d.trace_exc_ns, M);
    checkpoint_write_gpu(file, d.trace_inh_ns, M);
    checkpoint_write_gpu(file, d.trace_peak_mv, M);
    if (neurons) {
        CheckpointNeuron *packed = nullptr;
        allocate(packed, std::min(neurons, CHECKPOINT_RECORD_CHUNK));
        for (int offset = 0; offset < neurons; offset += CHECKPOINT_RECORD_CHUNK) {
            int count = std::min(neurons - offset, CHECKPOINT_RECORD_CHUNK);
            pack_checkpoint_neurons<<<(count + 127) / 128, 128>>>(
                d, d.occupied_neurons + offset, count, packed);
            check(cudaGetLastError(), "checkpoint neuron pack");
            checkpoint_write_gpu(file, packed, count);
        }
        cudaFree(packed);
    }
    if (synapses) {
        CheckpointSynapse *packed = nullptr;
        allocate(packed, std::min(synapses, CHECKPOINT_RECORD_CHUNK));
        for (int offset = 0; offset < synapses; offset += CHECKPOINT_RECORD_CHUNK) {
            int count = std::min(synapses - offset, CHECKPOINT_RECORD_CHUNK);
            pack_checkpoint_synapses<<<(count + 127) / 128, 128>>>(
                d, d.occupied_synapses + offset, count, packed);
            check(cudaGetLastError(), "checkpoint synapse pack");
            checkpoint_write_gpu(file, packed, count);
        }
        cudaFree(packed);
    }
    if (header.flags & 64u) {
        uint64_t extension[] = {0x4e53504c41535431ull, 1, sizeof(PlasticityCohort), P};
        checkpoint_write_bytes(file, extension, sizeof(extension));
        checkpoint_write_gpu(file, d.plasticity_cohort, P);
    }
}

void checkpoint_read_state(gzFile file, Device d, const CheckpointHeader &header,
        BehaviorTotals &dead_totals,
        std::deque<std::pair<int64_t, BehaviorTotals>> &behavior_history,
        std::deque<std::pair<uint64_t, BiteEvent>> &bite_history,
        uint64_t &next_bite_event_id) {
    checkpoint_read_observer(file, dead_totals, behavior_history,
                             bite_history, next_bite_event_id);
    checkpoint_read_gpu(file, d.animals, P);
    checkpoint_read_gpu(file, d.food, W * H);
    checkpoint_read_gpu(file, d.kind, W * H);
    checkpoint_read_gpu(file, d.senses, P * S);
    checkpoint_read_gpu(file, d.motors, P * M);
    checkpoint_read_gpu(file, d.births, 1);
    checkpoint_read_gpu(file, d.deaths, 1);
    checkpoint_read_gpu(file, d.predation_deaths, 1);
    checkpoint_read_gpu(file, d.bites, 1);
    checkpoint_read_gpu(file, d.next_id, 1);
    checkpoint_read_gpu(file, d.founder_motor_steps, M);
    checkpoint_read_gpu(file, d.founder_observed_steps, 1);
    checkpoint_read_gpu(file, d.trace_exc_arrivals, M);
    checkpoint_read_gpu(file, d.trace_exc_releases, M);
    checkpoint_read_gpu(file, d.trace_inh_arrivals, M);
    checkpoint_read_gpu(file, d.trace_exc_ns, M);
    checkpoint_read_gpu(file, d.trace_inh_ns, M);
    checkpoint_read_gpu(file, d.trace_peak_mv, M);
    std::vector<Animal> animals;
    download(animals, d.animals, P);
    uint64_t allocated_neurons = 0, allocated_synapses = 0;
    for (const Animal &animal : animals) {
        if (animal.alive == 1) {
            if (animal.neuron_count < 0 || animal.synapse_count < 0)
                throw std::runtime_error("Invalid checkpoint animal counts");
            allocated_neurons += animal.neuron_count;
            allocated_synapses += animal.synapse_count;
        }
    }
    if (allocated_neurons != header.neuron_count ||
        allocated_synapses != header.synapse_count)
        throw std::runtime_error("Checkpoint brain allocation counts disagree");
    if (header.neuron_count) {
        std::vector<CheckpointNeuron> host(std::min(header.neuron_count,
                                                   static_cast<uint32_t>(CHECKPOINT_RECORD_CHUNK)));
        CheckpointNeuron *packed = nullptr;
        allocate(packed, host.size());
        int previous = -1;
        for (uint32_t offset = 0; offset < header.neuron_count;
             offset += CHECKPOINT_RECORD_CHUNK) {
            int count = std::min(header.neuron_count - offset,
                                 static_cast<uint32_t>(CHECKPOINT_RECORD_CHUNK));
            checkpoint_read_bytes(file, host.data(), count * sizeof(host[0]));
            for (int i = 0; i < count; ++i) {
                const auto &record = host[i];
                if (record.slot <= previous || record.slot >= NEURON_CAPACITY ||
                    record.owner < 0 || record.owner >= P ||
                    animals[record.owner].alive != 1 ||
                    record.next < -1 || record.next >= NEURON_CAPACITY ||
                    record.port < -1 || record.port >= S + M)
                    throw std::runtime_error("Invalid checkpoint neuron record");
                previous = record.slot;
            }
            check(cudaMemcpy(packed, host.data(), count * sizeof(host[0]),
                             cudaMemcpyHostToDevice), "checkpoint neuron upload");
            unpack_checkpoint_neurons<<<(count + 127) / 128, 128>>>(
                d, packed, count);
            check(cudaGetLastError(), "checkpoint neuron unpack");
        }
        cudaFree(packed);
    }
    if (header.synapse_count) {
        std::vector<CheckpointSynapse> host(std::min(header.synapse_count,
                                                    static_cast<uint32_t>(CHECKPOINT_RECORD_CHUNK)));
        CheckpointSynapse *packed = nullptr;
        allocate(packed, host.size());
        int previous = -1;
        for (uint32_t offset = 0; offset < header.synapse_count;
             offset += CHECKPOINT_RECORD_CHUNK) {
            int count = std::min(header.synapse_count - offset,
                                 static_cast<uint32_t>(CHECKPOINT_RECORD_CHUNK));
            checkpoint_read_bytes(file, host.data(), count * sizeof(host[0]));
            for (int i = 0; i < count; ++i) {
                const auto &record = host[i];
                if (record.slot <= previous || record.slot >= SYNAPSE_CAPACITY ||
                    record.owner < 0 || record.owner >= P ||
                    animals[record.owner].alive != 1 ||
                    record.next < -1 || record.next >= SYNAPSE_CAPACITY ||
                    record.gene.pre < 0 || record.gene.pre >= NEURON_CAPACITY ||
                    record.gene.post < 0 || record.gene.post >= NEURON_CAPACITY)
                    throw std::runtime_error("Invalid checkpoint synapse record");
                previous = record.slot;
            }
            check(cudaMemcpy(packed, host.data(), count * sizeof(host[0]),
                             cudaMemcpyHostToDevice), "checkpoint synapse upload");
            unpack_checkpoint_synapses<<<(count + 127) / 128, 128>>>(
                d, packed, count);
            check(cudaGetLastError(), "checkpoint synapse unpack");
        }
        cudaFree(packed);
    }
    if (header.flags & 64u) {
        uint64_t extension[4];
        checkpoint_read_bytes(file, extension, sizeof(extension));
        if (extension[0] != 0x4e53504c41535431ull || extension[1] != 1 ||
            extension[2] != sizeof(PlasticityCohort) || extension[3] != P)
            throw std::runtime_error("Incompatible plasticity intervention extension");
        checkpoint_read_gpu(file, d.plasticity_cohort, P);
    }
    unsigned char extra;
    if (gzread(file, &extra, 1) != 0)
        throw std::runtime_error("Unexpected trailing checkpoint data");
    check(cudaDeviceSynchronize(), "checkpoint restore");
}

void checkpoint_validate_header(const CheckpointHeader &header) {
    if (std::memcmp(header.magic, "NSCHKP01", 8) != 0 ||
        header.version != CHECKPOINT_VERSION ||
        header.endian != 0x01020304u ||
        header.layout != checkpoint_layout() ||
        header.step < 0 || header.step > INT64_MAX / TICKS ||
        header.checkpoint_every <= 0 ||
        header.initial < 1 || header.initial > P ||
        header.trace_animal < -1 || header.trace_animal >= header.initial ||
        (header.flags & ~127u) != 0 ||
        !std::isfinite(header.food_renewal) ||
        header.food_renewal < 0 || header.food_renewal > 1 ||
        header.neuron_count > NEURON_CAPACITY ||
        header.synapse_count > SYNAPSE_CAPACITY ||
        !std::memchr(header.output, 0, sizeof(header.output)))
        throw std::runtime_error("Incompatible or invalid checkpoint header");
}

void checkpoint_atomic_save(const std::string &path, Device d,
        void *selection_temp, size_t selection_temp_bytes,
        CheckpointHeader header, const BehaviorTotals &dead_totals,
        const std::deque<std::pair<int64_t, BehaviorTotals>> &behavior_history,
        const std::deque<std::pair<uint64_t, BiteEvent>> &bite_history,
        uint64_t next_bite_event_id) {
    std::string temporary = path + ".tmp";
    gzFile file = gzopen(temporary.c_str(), "wb1");
    if (!file) throw std::runtime_error("Cannot open checkpoint for writing: " + temporary);
    gzbuffer(file, CHECKPOINT_IO_CHUNK);
    try {
        checkpoint_write_state(file, d, selection_temp, selection_temp_bytes,
                               header, dead_totals, behavior_history,
                               bite_history, next_bite_event_id);
    } catch (...) {
        gzclose(file);
        throw;
    }
    if (gzclose(file) != Z_OK)
        throw std::runtime_error("Cannot finish checkpoint: " + temporary);
    int descriptor = open(temporary.c_str(), O_RDONLY);
    if (descriptor < 0 || fsync(descriptor) != 0) {
        if (descriptor >= 0) close(descriptor);
        throw std::runtime_error("Cannot sync checkpoint: " + temporary);
    }
    close(descriptor);
    std::filesystem::rename(temporary, path);
    std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (parent.empty()) parent = ".";
    descriptor = open(parent.c_str(), O_RDONLY | O_DIRECTORY);
    if (descriptor >= 0) { fsync(descriptor); close(descriptor); }
}
