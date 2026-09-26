#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

namespace sim {

class LiveView {
public:
    explicit LiveView(int port);
    ~LiveView();
    LiveView(const LiveView &) = delete;
    LiveView &operator=(const LiveView &) = delete;

    void publish(std::string snapshot);
    void finish();
    bool paused() const { return paused_.load(); }
    bool stop_requested() const { return stop_requested_.load(); }
    int speed() const { return speed_.load(); }

private:
    void serve();
    void handle_client(int client);
    int socket_ = -1;
    std::thread server_;
    std::mutex snapshot_mutex_;
    std::string snapshot_ = "{}";
    std::atomic<bool> running_{true};
    std::atomic<bool> paused_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> finished_{false};
    std::atomic<int> speed_{1};
};

} // namespace sim
