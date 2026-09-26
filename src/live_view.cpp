#include "live_view.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace sim {
namespace {
void send_response(int client, int status, const char *type,
                   const std::string &body) {
    std::string header = "HTTP/1.1 " + std::to_string(status) +
        (status == 200 ? " OK" : " Not Found") +
        "\r\nContent-Type: " + type +
        "\r\nContent-Length: " + std::to_string(body.size()) +
        "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    auto write_all = [client](const std::string &data) {
        size_t sent = 0;
        while (sent < data.size()) {
            ssize_t n = send(client, data.data() + sent, data.size() - sent,
                             MSG_NOSIGNAL);
            if (n <= 0) break;
            sent += static_cast<size_t>(n);
        }
    };
    write_all(header);
    write_all(body);
}

std::string asset(const char *name) {
    std::ifstream file(std::string(LIVE_ASSET_DIR) + "/" + name,
                       std::ios::binary);
    if (!file) return {};
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
} // namespace

LiveView::LiveView(int port) {
    socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_ < 0) throw std::runtime_error("Cannot create live view socket");
    int reuse = 1;
    setsockopt(socket_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(socket_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) ||
        listen(socket_, 16)) {
        std::string error = std::strerror(errno);
        close(socket_);
        socket_ = -1;
        throw std::runtime_error("Cannot start live view: " + error);
    }
    server_ = std::thread(&LiveView::serve, this);
}

LiveView::~LiveView() {
    running_ = false;
    shutdown(socket_, SHUT_RDWR);
    close(socket_);
    if (server_.joinable()) server_.join();
}

void LiveView::publish(std::string snapshot) {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_ = std::move(snapshot);
}

void LiveView::finish() { finished_ = true; }

void LiveView::serve() {
    while (running_) {
        int client = accept(socket_, nullptr, nullptr);
        if (client < 0) {
            if (running_) std::cerr << "Live view connection error: "
                                    << std::strerror(errno) << '\n';
            break;
        }
        handle_client(client);
        close(client);
    }
}

void LiveView::handle_client(int client) {
    timeval timeout{2, 0};
    setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    char request[4096]{};
    ssize_t length = recv(client, request, sizeof(request) - 1, 0);
    if (length <= 0) return;
    std::string line(request, static_cast<size_t>(length));
    size_t end = line.find("\r\n");
    if (end == std::string::npos) return;
    line.resize(end);
    if (line == "GET /api/state HTTP/1.1") {
        std::string snapshot;
        {
            std::lock_guard<std::mutex> lock(snapshot_mutex_);
            snapshot = snapshot_;
        }
        snapshot.pop_back();
        snapshot += ",\"paused\":" + std::string(paused_ ? "true" : "false") +
                    ",\"finished\":" + std::string(finished_ ? "true" : "false") +
                    ",\"speed\":" + std::to_string(speed_) + "}";
        send_response(client, 200, "application/json; charset=utf-8", snapshot);
    } else if (line == "POST /api/pause HTTP/1.1" &&
               std::strstr(request, "X-Live-View: 1")) {
        paused_ = !paused_;
        send_response(client, 200, "application/json", "{}");
    } else if (line == "POST /api/stop HTTP/1.1" &&
               std::strstr(request, "X-Live-View: 1")) {
        stop_requested_ = true;
        send_response(client, 200, "application/json", "{}");
    } else if (line.rfind("POST /api/speed/", 0) == 0 &&
               std::strstr(request, "X-Live-View: 1")) {
        const std::string value = line.substr(16, line.find(' ', 16) - 16);
        if (value == "1") speed_ = 1;
        else if (value == "4") speed_ = 4;
        else if (value == "0") speed_ = 0;
        else { send_response(client, 404, "text/plain", "Unknown speed"); return; }
        send_response(client, 200, "application/json", "{}");
    } else if (line == "GET / HTTP/1.1") {
        send_response(client, 200, "text/html; charset=utf-8", asset("index.html"));
    } else if (line == "GET /style.css HTTP/1.1") {
        send_response(client, 200, "text/css; charset=utf-8", asset("style.css"));
    } else if (line == "GET /app.js HTTP/1.1") {
        send_response(client, 200, "text/javascript; charset=utf-8", asset("app.js"));
    } else {
        send_response(client, 404, "text/plain", "Not found");
    }
}
} // namespace sim
