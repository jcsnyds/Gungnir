#include <librealsense2/rs.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
    #define L515_API __declspec(dllexport)
#else
    #define L515_API __attribute__((visibility("default")))
#endif

namespace {

class L515Native {
public:
    explicit L515Native(int width, int height, int fps)
        : width_(width), height_(height), fps_(fps) {}

    ~L515Native() {
        stop_noexcept();
    }

    void start() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (running_) {
            return;
        }

        rs2::context ctx;
        auto devices = ctx.query_devices();
        if (devices.size() == 0) {
            throw std::runtime_error(
                "No RealSense device found. Connect the L515 with USB 3 and "
                "verify that librealsense 2.54.2 can access it."
            );
        }

        bool found_l515 = false;
        std::string found_names;
        for (int i = 0; i < static_cast<int>(devices.size()); ++i) {
            auto dev = devices[i];
            const char *name = dev.supports(RS2_CAMERA_INFO_NAME)
                ? dev.get_info(RS2_CAMERA_INFO_NAME)
                : "Unknown RealSense device";
            const char *product_line = dev.supports(RS2_CAMERA_INFO_PRODUCT_LINE)
                ? dev.get_info(RS2_CAMERA_INFO_PRODUCT_LINE)
                : "";

            if (!found_names.empty()) {
                found_names += ", ";
            }
            found_names += name;

            // L515 reports product line L500. Keep the name check too so the
            // error message remains useful with unusual/older firmware.
            if (std::string(product_line).find("L500") != std::string::npos ||
                std::string(name).find("L515") != std::string::npos) {
                found_l515 = true;
                break;
            }
        }

        if (!found_l515) {
            throw std::runtime_error(
                "A RealSense device was found, but it does not appear to be an "
                "L515. Devices: " + found_names
            );
        }

        config_ = rs2::config();
        config_.enable_stream(
            RS2_STREAM_DEPTH,
            0,
            width_,
            height_,
            RS2_FORMAT_Z16,
            fps_
        );
        config_.enable_stream(
            RS2_STREAM_COLOR,
            0,
            width_,
            height_,
            RS2_FORMAT_BGR8,
            fps_
        );

        profile_ = pipeline_.start(config_);

        auto depth_sensor = profile_.get_device().first<rs2::depth_sensor>();
        depth_scale_ = depth_sensor.get_depth_scale();

        auto depth_stream = profile_.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
        auto color_stream = profile_.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        depth_intrinsics_ = depth_stream.get_intrinsics();
        color_intrinsics_ = color_stream.get_intrinsics();

        align_ = std::make_unique<rs2::align>(RS2_STREAM_COLOR);

        // Let auto exposure and the depth pipeline settle.
        for (int i = 0; i < 5; ++i) {
            pipeline_.wait_for_frames();
        }

        depth_buffer_.resize(static_cast<size_t>(width_) * static_cast<size_t>(height_));
        color_buffer_.resize(static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3u);
        running_ = true;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_) {
            return;
        }
        pipeline_.stop();
        align_.reset();
        running_ = false;
    }

    void capture() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_) {
            throw std::runtime_error("L515 pipeline is not running.");
        }

        auto frames = pipeline_.wait_for_frames();
        auto aligned = align_->process(frames);

        auto depth = aligned.get_depth_frame();
        auto color = aligned.get_color_frame();

        if (!depth || !color) {
            throw std::runtime_error("RealSense did not return both depth and color frames.");
        }

        const int depth_width = depth.get_width();
        const int depth_height = depth.get_height();
        const int color_width = color.get_width();
        const int color_height = color.get_height();

        if (depth_width != width_ || depth_height != height_ ||
            color_width != width_ || color_height != height_) {
            throw std::runtime_error("Unexpected frame dimensions returned by the L515.");
        }

        const auto *depth_bytes = static_cast<const unsigned char *>(depth.get_data());
        const auto *color_bytes = static_cast<const unsigned char *>(color.get_data());
        const int depth_stride = depth.get_stride_in_bytes();
        const int color_stride = color.get_stride_in_bytes();
        const int depth_row_bytes = width_ * static_cast<int>(sizeof(uint16_t));
        const int color_row_bytes = width_ * 3;

        // Copy rows instead of assuming the SDK stride equals width*bytes/pixel.
        for (int y = 0; y < height_; ++y) {
            std::memcpy(
                depth_buffer_.data() + static_cast<size_t>(y) * width_,
                depth_bytes + static_cast<size_t>(y) * depth_stride,
                static_cast<size_t>(depth_row_bytes)
            );
            std::memcpy(
                color_buffer_.data() + static_cast<size_t>(y) * color_row_bytes,
                color_bytes + static_cast<size_t>(y) * color_stride,
                static_cast<size_t>(color_row_bytes)
            );
        }
    }

    const uint16_t *depth_data() const {
        return depth_buffer_.empty() ? nullptr : depth_buffer_.data();
    }

    const uint8_t *color_data() const {
        return color_buffer_.empty() ? nullptr : color_buffer_.data();
    }

    float depth_scale() const {
        return depth_scale_;
    }

    int width() const { return width_; }
    int height() const { return height_; }

    void get_depth_k(float out[9]) const {
        fill_k(depth_intrinsics_, out);
    }

    void get_color_k(float out[9]) const {
        fill_k(color_intrinsics_, out);
    }

    bool running() const { return running_; }

private:
    static void fill_k(const rs2_intrinsics &intrinsics, float out[9]) {
        out[0] = intrinsics.fx;
        out[1] = 0.0f;
        out[2] = intrinsics.ppx;
        out[3] = 0.0f;
        out[4] = intrinsics.fy;
        out[5] = intrinsics.ppy;
        out[6] = 0.0f;
        out[7] = 0.0f;
        out[8] = 1.0f;
    }

    void stop_noexcept() noexcept {
        try {
            if (running_) {
                pipeline_.stop();
                running_ = false;
            }
        } catch (...) {
            // Never allow an exception to escape a C++ destructor.
        }
    }

    int width_;
    int height_;
    int fps_;

    rs2::pipeline pipeline_;
    rs2::config config_;
    rs2::pipeline_profile profile_;
    std::unique_ptr<rs2::align> align_;

    rs2_intrinsics depth_intrinsics_{};
    rs2_intrinsics color_intrinsics_{};
    float depth_scale_ = 0.001f;

    std::vector<uint16_t> depth_buffer_;
    std::vector<uint8_t> color_buffer_;
    bool running_ = false;
    mutable std::mutex mutex_;
};

void set_error(char *buffer, size_t buffer_size, const std::string &message) {
    if (!buffer || buffer_size == 0) {
        return;
    }
    const size_t count = std::min(buffer_size - 1, message.size());
    std::memcpy(buffer, message.data(), count);
    buffer[count] = '\0';
}

template <typename Fn>
int call_safely(char *error_buffer, size_t error_buffer_size, Fn &&fn) {
    try {
        fn();
        set_error(error_buffer, error_buffer_size, "");
        return 0;
    } catch (const rs2::error &e) {
        set_error(error_buffer, error_buffer_size, e.what());
        return -1;
    } catch (const std::exception &e) {
        set_error(error_buffer, error_buffer_size, e.what());
        return -1;
    } catch (...) {
        set_error(error_buffer, error_buffer_size, "Unknown native RealSense error.");
        return -1;
    }
}

} // namespace

extern "C" {

L515_API void *l515_create(
    int width,
    int height,
    int fps,
    char *error_buffer,
    size_t error_buffer_size
) {
    try {
        auto *camera = new L515Native(width, height, fps);
        set_error(error_buffer, error_buffer_size, "");
        return camera;
    } catch (const rs2::error &e) {
        set_error(error_buffer, error_buffer_size, e.what());
    } catch (const std::exception &e) {
        set_error(error_buffer, error_buffer_size, e.what());
    } catch (...) {
        set_error(error_buffer, error_buffer_size, "Unknown native RealSense error.");
    }
    return nullptr;
}

L515_API int l515_start(void *handle, char *error_buffer, size_t error_buffer_size) {
    if (!handle) {
        set_error(error_buffer, error_buffer_size, "Invalid L515 handle.");
        return -1;
    }
    return call_safely(error_buffer, error_buffer_size, [&] {
        static_cast<L515Native *>(handle)->start();
    });
}

L515_API int l515_stop(void *handle, char *error_buffer, size_t error_buffer_size) {
    if (!handle) {
        set_error(error_buffer, error_buffer_size, "Invalid L515 handle.");
        return -1;
    }
    return call_safely(error_buffer, error_buffer_size, [&] {
        static_cast<L515Native *>(handle)->stop();
    });
}

L515_API int l515_capture(void *handle, char *error_buffer, size_t error_buffer_size) {
    if (!handle) {
        set_error(error_buffer, error_buffer_size, "Invalid L515 handle.");
        return -1;
    }
    return call_safely(error_buffer, error_buffer_size, [&] {
        static_cast<L515Native *>(handle)->capture();
    });
}

L515_API void l515_destroy(void *handle) {
    delete static_cast<L515Native *>(handle);
}

L515_API const uint16_t *l515_depth_data(void *handle) {
    if (!handle) {
        return nullptr;
    }
    return static_cast<L515Native *>(handle)->depth_data();
}

L515_API const uint8_t *l515_color_data(void *handle) {
    if (!handle) {
        return nullptr;
    }
    return static_cast<L515Native *>(handle)->color_data();
}

L515_API float l515_depth_scale(void *handle) {
    if (!handle) {
        return 0.0f;
    }
    return static_cast<L515Native *>(handle)->depth_scale();
}

L515_API int l515_width(void *handle) {
    return handle ? static_cast<L515Native *>(handle)->width() : 0;
}

L515_API int l515_height(void *handle) {
    return handle ? static_cast<L515Native *>(handle)->height() : 0;
}

L515_API int l515_get_depth_k(void *handle, float out[9], char *error_buffer, size_t error_buffer_size) {
    if (!handle || !out) {
        set_error(error_buffer, error_buffer_size, "Invalid L515 handle or output buffer.");
        return -1;
    }
    return call_safely(error_buffer, error_buffer_size, [&] {
        static_cast<L515Native *>(handle)->get_depth_k(out);
    });
}

L515_API int l515_get_color_k(void *handle, float out[9], char *error_buffer, size_t error_buffer_size) {
    if (!handle || !out) {
        set_error(error_buffer, error_buffer_size, "Invalid L515 handle or output buffer.");
        return -1;
    }
    return call_safely(error_buffer, error_buffer_size, [&] {
        static_cast<L515Native *>(handle)->get_color_k(out);
    });
}

L515_API int l515_is_running(void *handle) {
    return handle && static_cast<L515Native *>(handle)->running() ? 1 : 0;
}

} // extern "C"
