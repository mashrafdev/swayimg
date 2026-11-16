// SPDX-License-Identifier: MIT
// Very large image support via libvips.
// Copyright (C) 2026 Artem Senichev <artemsen@gmail.com>

#include "../application.hpp"
#include "../formatfactory.hpp"
#include "../imageformat.hpp"
#include "../log.hpp"
#include "../render.hpp"

#include <unistd.h>
#include <vips/vips.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

/** Vips image deleter. */
struct VipsImageDeleter {
    void operator()(VipsImage* image) const
    {
        if (image) {
            g_object_unref(image);
        }
    }
};

/** Vips image smart pointer. */
using VipsImagePtr = std::unique_ptr<VipsImage, VipsImageDeleter>;

/**
 * Convert vips image to pixmap.
 * @param image source image
 * @return pixmap or empty pixmap on errors
 */
Pixmap to_pixmap(VipsImage* image)
{
    // convert to 8-bit sRGB
    VipsImage* raw_srgb = nullptr;
    if (vips_colourspace(image, &raw_srgb, VIPS_INTERPRETATION_sRGB, nullptr) !=
        0) {
        return {};
    }
    const VipsImagePtr srgb(raw_srgb);

    VipsImage* target = srgb.get();

    VipsImagePtr cast;
    if (vips_image_get_format(target) != VIPS_FORMAT_UCHAR) {
        VipsImage* raw_cast = nullptr;
        if (vips_cast(target, &raw_cast, VIPS_FORMAT_UCHAR, "shift", TRUE,
                      nullptr) != 0) {
            return {};
        }
        cast.reset(raw_cast);
        target = cast.get();
    }

    // add alpha channel if needed
    VipsImagePtr alpha;
    const int bands = vips_image_get_bands(target);
    if (bands == 3) {
        VipsImage* raw_alpha = nullptr;
        if (vips_addalpha(target, &raw_alpha, nullptr) != 0) {
            return {};
        }
        alpha.reset(raw_alpha);
        target = alpha.get();
    } else if (bands != 4) {
        return {};
    }

    // decode image to memory
    size_t size = 0;
    void* memory = vips_image_write_to_memory(target, &size);
    if (!memory) {
        return {};
    }

    const size_t width = vips_image_get_width(target);
    const size_t height = vips_image_get_height(target);
    if (size != width * height * 4) {
        g_free(memory);
        return {};
    }

    // libvips uses RGBA order, convert it to ARGB
    Pixmap pm;
    pm.create(Pixmap::ARGB, width, height);
    const uint8_t* src = static_cast<const uint8_t*>(memory);
    argb_t* dst = reinterpret_cast<argb_t*>(pm.ptr(0, 0));
    for (size_t i = 0; i < width * height; ++i) {
        dst[i] = argb_t(src[3], src[0], src[1], src[2]);
        src += 4;
    }

    g_free(memory);
    return pm;
}

/**
 * Image rendered from a disk-backed tiled pyramid.
 *
 * The image is shown as a small overview until the pyramid cache is ready.
 * The pyramid is built in the background on the first draw and contains
 * lossless, full-resolution tiles at every 2x scale level (a TIFF pyramid).
 * Only the visible part of the appropriate level is decoded for each frame,
 * so memory usage does not depend on the image size.
 */
class TiledImage final : public Image {
public:
    /** Pyramid parameters. */
    struct Config {
        std::filesystem::path source; ///< Source image path
        std::filesystem::path cache;  ///< Path to the pyramid cache
        Size size;                    ///< Full image size
        bool rotate;                  ///< Apply EXIF orientation
        bool alpha;                   ///< Image has alpha channel
        std::string loader;           ///< Loader name
    };

    /**
     * Constructor.
     * @param config pyramid parameters
     * @param overview downscaled image preview
     */
    TiledImage(Config config, Pixmap overview)
        : cfg(std::move(config))
        , state(std::make_shared<State>())
    {
        frames.resize(1);
        frames[0].pm = std::move(overview);
        format = std::format("VIPS {} {}x{} -> tiled", cfg.loader,
                             cfg.size.width, cfg.size.height);
        if (open_cache()) {
            state->ready = true;
        }
    }

    ~TiledImage() override
    {
        std::lock_guard lock(state->mutex);
        state->stop = true;
        if (state->source) {
            vips_image_set_kill(state->source, TRUE);
        }
    }

    void draw(const size_t /*frame*/, Pixmap& target, const double scale,
              const ssize_t x, const ssize_t y) override
    {
        if (scale <= 0.0 || !target) {
            return;
        }

        bool ready;
        {
            const std::lock_guard lock(state->mutex);
            ready = state->ready;
            if (!ready && !state->failed && !state->started) {
                state->started = true;
                Log::verbose("libvips: building tiled pyramid {}x{}: {}",
                             cfg.size.width, cfg.size.height,
                             cfg.cache.string());
                std::thread([st = state, conf = cfg] {
                    build(st, conf);
                }).detach();
            }
        }

        if (!ready) {
            draw_overview(target, scale, x, y);
            return;
        }

        const double width = static_cast<double>(cfg.size.width);
        const double height = static_cast<double>(cfg.size.height);

        // visible part of the window
        const double wx0 = std::max(0.0, static_cast<double>(x));
        const double wy0 = std::max(0.0, static_cast<double>(y));
        const double wx1 = std::min(static_cast<double>(target.width()),
                                    static_cast<double>(x) + width * scale);
        const double wy1 = std::min(static_cast<double>(target.height()),
                                    static_cast<double>(y) + height * scale);
        if (wx1 <= wx0 || wy1 <= wy0) {
            return;
        }

        // visible part of the image in source coordinates
        double sx0 = std::clamp((wx0 - x) / scale, 0.0, width);
        double sy0 = std::clamp((wy0 - y) / scale, 0.0, height);
        double sx1 = std::clamp((wx1 - x) / scale, 0.0, width);
        double sy1 = std::clamp((wy1 - y) / scale, 0.0, height);
        if (sx1 <= sx0 || sy1 <= sy0) {
            return;
        }

        // add a margin to make small panning steps cheap
        constexpr double MARGIN = 128.0;
        const double margin = MARGIN / scale;
        sx0 = std::max(0.0, sx0 - margin);
        sy0 = std::max(0.0, sy0 - margin);
        sx1 = std::min(width, sx1 + margin);
        sy1 = std::min(height, sy1 + margin);

        // choose the largest level that still has enough detail for the scale
        size_t level = 0;
        if (scale < 1.0) {
            const double l = std::floor(std::log2(1.0 / scale));
            if (l > 0.0) {
                level = static_cast<size_t>(l);
            }
        }

        const std::lock_guard lock(state->mutex);
        if (!state->ready || state->levels.empty()) {
            return;
        }
        if (level >= state->levels.size()) {
            level = state->levels.size() - 1;
        }

        const uint64_t factor = uint64_t(1) << level;
        const double factor_d = static_cast<double>(factor);
        VipsImage* level_img = state->levels[level].get();
        const ssize_t level_width = vips_image_get_width(level_img);
        const ssize_t level_height = vips_image_get_height(level_img);

        ssize_t lx0 = static_cast<ssize_t>(std::floor(sx0 / factor_d));
        ssize_t ly0 = static_cast<ssize_t>(std::floor(sy0 / factor_d));
        ssize_t lx1 = static_cast<ssize_t>(std::ceil(sx1 / factor_d));
        ssize_t ly1 = static_cast<ssize_t>(std::ceil(sy1 / factor_d));
        lx0 = std::clamp<ssize_t>(lx0, 0, level_width - 1);
        ly0 = std::clamp<ssize_t>(ly0, 0, level_height - 1);
        lx1 = std::clamp<ssize_t>(lx1, lx0 + 1, level_width);
        ly1 = std::clamp<ssize_t>(ly1, ly0 + 1, level_height);

        // reuse the previously decoded region if it covers the requested area
        const bool cache_hit = region_level == level && region_pm &&
            region_x0 <= lx0 && region_y0 <= ly0 && region_x1 >= lx1 &&
            region_y1 >= ly1;
        if (!cache_hit) {
            VipsImage* raw = nullptr;
            if (vips_extract_area(level_img, &raw, lx0, ly0, lx1 - lx0,
                                  ly1 - ly0, nullptr) != 0) {
                Log::verbose("libvips: unable to extract image area: {}",
                             vips_error_buffer());
                vips_error_clear();
                return;
            }
            const VipsImagePtr region(raw);

            Pixmap pm = to_pixmap(region.get());
            if (!pm) {
                Log::verbose("libvips: unable to convert image area: {}",
                             vips_error_buffer());
                vips_error_clear();
                return;
            }

            region_pm = std::move(pm);
            region_level = level;
            region_x0 = lx0;
            region_y0 = ly0;
            region_x1 = lx1;
            region_y1 = ly1;
        }

        const ssize_t px =
            std::llround(static_cast<double>(x) +
                         static_cast<double>(region_x0) * factor_d * scale);
        const ssize_t py =
            std::llround(static_cast<double>(y) +
                         static_cast<double>(region_y0) * factor_d * scale);
        Render::self().draw(target, region_pm, { .x = px, .y = py },
                            scale * factor_d);
    }

    [[nodiscard]] Size size(const size_t /*frame*/) const override
    {
        return cfg.size;
    }

    [[nodiscard]] bool has_alpha(const size_t /*frame*/) const override
    {
        return cfg.alpha;
    }

    [[nodiscard]] bool is_tiled() const override { return true; }

    void flip_vertical() override
    {
        Log::verbose("libvips: transforms are not supported for tiled images");
    }

    void flip_horizontal() override
    {
        Log::verbose("libvips: transforms are not supported for tiled images");
    }

    void rotate(const size_t /*angle*/) override
    {
        Log::verbose("libvips: transforms are not supported for tiled images");
    }

    bool save(const std::filesystem::path& path,
              const size_t /*frame*/) const override
    {
        const std::lock_guard lock(state->mutex);
        if (!state->ready || state->levels.empty()) {
            return false;
        }

        if (vips_pngsave(state->levels[0].get(), path.c_str(), "compression", 6,
                         nullptr) != 0) {
            Log::error("libvips: unable to export image: {}",
                       vips_error_buffer());
            vips_error_clear();
            return false;
        }
        return true;
    }

private:
    /** Background build state. */
    struct State {
        std::mutex mutex;
        std::vector<VipsImagePtr> levels; ///< Pyramid levels
        VipsImage* source = nullptr;      ///< Source for cancellation
        bool ready = false;               ///< Pyramid is ready
        bool failed = false;              ///< Build failed
        bool started = false;             ///< Build thread started
        bool stop = false;                ///< Stop the build thread
    };

    Config cfg;                   ///< Pyramid parameters
    std::shared_ptr<State> state; ///< Background build state

    // last decoded region, used to avoid re-decoding during small pans
    Pixmap region_pm;
    size_t region_level = std::numeric_limits<size_t>::max();
    ssize_t region_x0 = 0;
    ssize_t region_y0 = 0;
    ssize_t region_x1 = 0;
    ssize_t region_y1 = 0;

    /**
     * Draw the downscaled overview.
     * @param target surface to draw on
     * @param scale image scale factor
     * @param x,y top-left coordinates on target surface
     */
    void draw_overview(Pixmap& target, const double scale, const ssize_t x,
                       const ssize_t y) const
    {
        const Pixmap& pm = frames[0].pm;
        if (!pm || !cfg.size.width) {
            return;
        }
        const double ov_scale =
            scale * static_cast<double>(pm.width()) / cfg.size.width;
        Render::self().draw(target, pm, { .x = x, .y = y }, ov_scale);
    }

    /**
     * Open an existing pyramid cache.
     * @return true if the cache is valid
     */
    bool open_cache()
    {
        if (cfg.cache.empty()) {
            return false;
        }

        std::error_code ec;
        if (!std::filesystem::is_regular_file(cfg.cache, ec)) {
            return false;
        }

        VipsImage* first = vips_image_new_from_file(
            cfg.cache.c_str(), "access", VIPS_ACCESS_RANDOM, nullptr);
        if (!first) {
            vips_error_clear();
            std::filesystem::remove(cfg.cache, ec);
            return false;
        }
        VipsImagePtr first_ptr(first);

        if (vips_image_get_width(first) != static_cast<int>(cfg.size.width) ||
            vips_image_get_height(first) != static_cast<int>(cfg.size.height)) {
            std::filesystem::remove(cfg.cache, ec);
            return false;
        }

        std::vector<VipsImagePtr> levels;
        levels.emplace_back(std::move(first_ptr));

        const int pages = std::max(1, vips_image_get_n_pages(first));
        for (int page = 1; page < pages; ++page) {
            VipsImage* image =
                vips_image_new_from_file(cfg.cache.c_str(), "page", page,
                                         "access", VIPS_ACCESS_RANDOM, nullptr);
            if (!image) {
                vips_error_clear();
                std::filesystem::remove(cfg.cache, ec);
                return false;
            }
            levels.emplace_back(image);
        }

        state->levels = std::move(levels);
        Log::verbose("libvips: using tiled pyramid cache {}: {} levels",
                     cfg.cache.string(), state->levels.size());
        return true;
    }

    /**
     * Build pyramid cache in a background thread.
     * @param state build state
     * @param cfg pyramid parameters
     */
    static void build(const std::shared_ptr<State>& state, const Config& cfg)
    {
        std::error_code ec;
        const std::filesystem::path tmp = std::format(
            "{}.tmp.{}", cfg.cache.string(), static_cast<long>(getpid()));
        std::filesystem::remove(tmp, ec);

        VipsImagePtr source(vips_image_new_from_file(
            cfg.source.c_str(), "access", VIPS_ACCESS_SEQUENTIAL, nullptr));
        if (!source) {
            fail(state, "unable to open source");
            return;
        }

        {
            const std::lock_guard lock(state->mutex);
            if (state->stop) {
                return;
            }
            state->source = source.get();
        }

        VipsImagePtr prepared;
        if (!prepare(source.get(), cfg.rotate, prepared)) {
            clear_source(state);
            fail(state, "unable to prepare image");
            return;
        }

        const int rc = vips_tiffsave(
            prepared.get(), tmp.c_str(), "tile", TRUE, "tile-width", 512,
            "tile-height", 512, "pyramid", TRUE, "compression",
            VIPS_FOREIGN_TIFF_COMPRESSION_DEFLATE, "level", 3, "predictor",
            VIPS_FOREIGN_TIFF_PREDICTOR_HORIZONTAL, "bigtiff", TRUE, nullptr);

        bool stopped;
        {
            const std::lock_guard lock(state->mutex);
            state->source = nullptr;
            stopped = state->stop;
        }
        if (stopped) {
            std::filesystem::remove(tmp, ec);
            return;
        }
        if (rc != 0) {
            std::filesystem::remove(tmp, ec);
            fail(state, "unable to create pyramid");
            return;
        }

        std::filesystem::rename(tmp, cfg.cache, ec);
        if (ec) {
            std::filesystem::remove(tmp, ec);
            fail(state, "unable to store pyramid");
            return;
        }

        // open all pyramid levels
        std::vector<VipsImagePtr> levels;
        VipsImage* first = vips_image_new_from_file(
            cfg.cache.c_str(), "access", VIPS_ACCESS_RANDOM, nullptr);
        if (!first) {
            std::filesystem::remove(cfg.cache, ec);
            fail(state, "unable to open pyramid");
            return;
        }
        levels.emplace_back(first);

        const int pages = std::max(1, vips_image_get_n_pages(first));
        for (int page = 1; page < pages; ++page) {
            VipsImage* image =
                vips_image_new_from_file(cfg.cache.c_str(), "page", page,
                                         "access", VIPS_ACCESS_RANDOM, nullptr);
            if (!image) {
                vips_error_clear();
                std::filesystem::remove(cfg.cache, ec);
                fail(state, "unable to open pyramid level");
                return;
            }
            levels.emplace_back(image);
        }

        if (levels[0]->Xsize != static_cast<int>(cfg.size.width) ||
            levels[0]->Ysize != static_cast<int>(cfg.size.height)) {
            std::filesystem::remove(cfg.cache, ec);
            fail(state, "pyramid has unexpected size");
            return;
        }

        bool notify = false;
        size_t level_count = 0;
        {
            const std::lock_guard lock(state->mutex);
            if (!state->stop) {
                state->levels = std::move(levels);
                state->ready = true;
                level_count = state->levels.size();
                notify = true;
            }
        }
        if (notify) {
            Log::verbose("libvips: tiled pyramid ready {}: {} levels",
                         cfg.cache.string(), level_count);
            if (Application::self().initialized()) {
                Application::redraw();
            }
        }
    }

    /**
     * Prepare source image for writing to the pyramid.
     * @param source source image
     * @param rotate apply EXIF orientation
     * @param target prepared image
     * @return true on success
     */
    static bool prepare(VipsImage* source, const bool rotate,
                        VipsImagePtr& target)
    {
        VipsImagePtr work;
        if (rotate) {
            VipsImage* rotated = nullptr;
            if (vips_autorot(source, &rotated, nullptr) != 0) {
                return false;
            }
            work.reset(rotated);
        } else {
            work.reset(g_object_ref(source));
        }

        const bool alpha = vips_image_hasalpha(work.get());

        // convert to 8-bit sRGB
        VipsImage* raw_srgb = nullptr;
        if (vips_colourspace(work.get(), &raw_srgb, VIPS_INTERPRETATION_sRGB,
                             nullptr) != 0) {
            return false;
        }
        VipsImagePtr srgb(raw_srgb);

        VipsImage* image = srgb.get();
        VipsImagePtr cast;
        VipsImagePtr alpha_channel;
        if (vips_image_get_format(image) != VIPS_FORMAT_UCHAR) {
            VipsImage* raw_cast = nullptr;
            if (vips_cast(image, &raw_cast, VIPS_FORMAT_UCHAR, "shift", TRUE,
                          nullptr) != 0) {
                return false;
            }
            cast.reset(raw_cast);
            image = cast.get();
        }

        const int bands = vips_image_get_bands(image);
        if (bands == 3 && alpha) {
            VipsImage* raw_alpha = nullptr;
            if (vips_addalpha(image, &raw_alpha, nullptr) != 0) {
                return false;
            }
            alpha_channel.reset(raw_alpha);
            image = alpha_channel.get();
        } else if (bands == 4 && !alpha) {
            VipsImage* raw_rgb = nullptr;
            if (vips_extract_band(image, &raw_rgb, 0, "n", 3, nullptr) != 0) {
                return false;
            }
            alpha_channel.reset(raw_rgb);
            image = alpha_channel.get();
        } else if (bands != 3 && bands != 4) {
            return false;
        }

        // the cache is plain sRGB, never apply a source profile again
        vips_image_remove(image, VIPS_META_ICC_NAME);
        vips_image_remove(image, VIPS_META_ORIENTATION);

        target.reset(g_object_ref(image));
        return true;
    }

    /**
     * Clear source image pointer in build state.
     * @param state build state
     */
    static void clear_source(const std::shared_ptr<State>& state)
    {
        const std::lock_guard lock(state->mutex);
        state->source = nullptr;
    }

    /**
     * Mark build as failed.
     * @param state build state
     * @param message error message
     */
    static void fail(const std::shared_ptr<State>& state, const char* message)
    {
        if (*vips_error_buffer()) {
            Log::warning("libvips: {}: {}", message, vips_error_buffer());
            vips_error_clear();
        }

        const std::lock_guard lock(state->mutex);
        state->failed = true;
        state->source = nullptr;
    }
};

class ImageFormatVips : public ImageFormat {
public:
    ImageFormatVips() noexcept
        : ImageFormat(Priority::Top, "vips")
    {
    }

    [[nodiscard]] ImagePtr decode(const Data& data) const override
    {
        const VipsImagePtr source = open_large(data);
        if (!source) {
            return nullptr;
        }

        int width = 0;
        int height = 0;
        if (!get_size(source.get(), width, height)) {
            return nullptr;
        }

        const bool do_rotate = FormatFactory::self().fix_orientation;
        const bool has_alpha = vips_image_hasalpha(source.get());

        VipsImage* raw = nullptr;
        if (vips_thumbnail_buffer(const_cast<uint8_t*>(data.data), data.size,
                                  &raw, static_cast<int>(max_size), "height",
                                  static_cast<int>(max_size), "size",
                                  VIPS_SIZE_DOWN, "no-rotate",
                                  do_rotate ? FALSE : TRUE, nullptr) != 0) {
            Log::verbose("libvips: unable to downscale image: {}",
                         vips_error_buffer());
            vips_error_clear();
            return nullptr;
        }
        const VipsImagePtr thumb(raw);

        Pixmap pm = to_pixmap(thumb.get());
        if (!pm) {
            Log::verbose("libvips: unable to convert image: {}",
                         vips_error_buffer());
            vips_error_clear();
            return nullptr;
        }

        // orientation may swap width and height
        Size full_size { .width = static_cast<size_t>(width),
                         .height = static_cast<size_t>(height) };
        if (do_rotate && vips_image_get_orientation_swap(source.get())) {
            std::swap(full_size.width, full_size.height);
        }

        // use tiled rendering for very large images
        const uint64_t pixels =
            static_cast<uint64_t>(full_size.width) * full_size.height;
        const uint64_t tile_limit =
            static_cast<uint64_t>(tile_threshold) * 1024 * 1024;
        if (tiled && pixels > tile_limit) {
            const std::filesystem::path cache =
                pyramid_path(data, full_size, do_rotate, has_alpha);
            if (!cache.empty()) {
                TiledImage::Config config {
                    .source = data.path,
                    .cache = cache,
                    .size = full_size,
                    .rotate = do_rotate,
                    .alpha = has_alpha,
                    .loader = loader_name(source.get()),
                };
                return std::make_shared<TiledImage>(std::move(config),
                                                    std::move(pm));
            }
        }

        ImagePtr image = std::make_shared<Image>();
        image->frames.resize(1);
        image->frames[0].pm = std::move(pm);
        image->format = std::format(
            "VIPS {} {}x{} -> {}x{}", loader_name(source.get()), width, height,
            image->frames[0].pm.width(), image->frames[0].pm.height());
        return image;
    }

    [[nodiscard]] Pixmap preview(const Data& data, const size_t sz,
                                 const bool fill) const override
    {
        if (sz == 0) {
            return {};
        }

        const VipsImagePtr source = open_large(data);
        if (!source) {
            return {};
        }

        const bool do_rotate = FormatFactory::self().fix_orientation;
        VipsImage* raw = nullptr;
        int rc;
        if (fill) {
            rc = vips_thumbnail_buffer(
                const_cast<uint8_t*>(data.data), data.size, &raw,
                static_cast<int>(sz), "height", static_cast<int>(sz), "size",
                VIPS_SIZE_DOWN, "no-rotate", do_rotate ? FALSE : TRUE, "crop",
                VIPS_INTERESTING_CENTRE, nullptr);
        } else {
            rc = vips_thumbnail_buffer(
                const_cast<uint8_t*>(data.data), data.size, &raw,
                static_cast<int>(sz), "height", static_cast<int>(sz), "size",
                VIPS_SIZE_DOWN, "no-rotate", do_rotate ? FALSE : TRUE, nullptr);
        }
        if (rc != 0) {
            Log::verbose("libvips: unable to create preview: {}",
                         vips_error_buffer());
            vips_error_clear();
            return {};
        }
        const VipsImagePtr thumb(raw);

        Pixmap pm = to_pixmap(thumb.get());
        if (!pm) {
            Log::verbose("libvips: unable to convert preview: {}",
                         vips_error_buffer());
            vips_error_clear();
        }
        return pm;
    }

    void set_config(Config& params) override
    {
        ImageFormat::set_config(params);
        params.get("max_size", max_size, 16, 32768);
        params.get("threshold", threshold, 0, 1000000);
        params.get("tiled", tiled);
        params.get("tile_threshold", tile_threshold, 0, 1000000);
        params.get("cache_dir", cache_dir, 1);
    }

    // ignore, done by decoder
    void fix_orientation(ImagePtr&, const int) const override {}

private:
    size_t max_size { 4096 };     ///< Max width/height of overview
    size_t threshold { 64 };      ///< Vips activation threshold in megapixels
    bool tiled { true };          ///< Use tiled rendering for large images
    size_t tile_threshold { 64 }; ///< Tiled rendering threshold in megapixels
    std::string cache_dir;        ///< Pyramid cache directory

    /**
     * Initialize libvips.
     * @return true if libvips is ready
     */
    static bool initialize()
    {
        static const bool ready = [] {
            if (VIPS_INIT("swayimg") != 0) {
                Log::error("Unable to initialize libvips: {}",
                           vips_error_buffer());
                return false;
            }
            // avoid keeping references to the source image data in cache
            vips_cache_set_max(0);
            return true;
        }();
        return ready;
    }

    /**
     * Open image and check that it is large enough for libvips.
     * @param data source image data
     * @return opened image or nullptr if image is not supported or too small
     */
    [[nodiscard]] VipsImagePtr open_large(const Data& data) const
    {
        if (!initialize()) {
            return {};
        }
        vips_error_clear();

        // check source size before decoding
        VipsImagePtr image(
            vips_image_new_from_buffer(data.data, data.size, nullptr, nullptr));
        if (!image) {
            vips_error_clear();
            return {};
        }

        int width = 0;
        int height = 0;
        if (!get_size(image.get(), width, height)) {
            vips_error_clear();
            return {};
        }

        const uint64_t pixels =
            static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
        const uint64_t limit = static_cast<uint64_t>(threshold) * 1024 * 1024;
        if (pixels <= limit) {
            return {}; // let format-specific decoder handle this image
        }

        return image;
    }

    /**
     * Get image size.
     * @param image vips image
     * @param width,height image size to write
     * @return true if image has valid size
     */
    static bool get_size(VipsImage* image, int& width, int& height)
    {
        width = vips_image_get_width(image);
        height = vips_image_get_height(image);
        return width > 0 && height > 0;
    }

    /**
     * Get image loader name.
     * @param image vips image
     * @return loader name
     */
    static std::string loader_name(VipsImage* image)
    {
        const char* name = nullptr;
        if (vips_image_get_string(image, VIPS_META_LOADER, &name) != 0 ||
            !name || !*name) {
            return "unknown";
        }

        std::string loader = name;
        if (loader.ends_with("load_buffer") ||
            loader.ends_with("load_source")) {
            loader.resize(loader.size() - 11);
        } else if (loader.ends_with("load")) {
            loader.resize(loader.size() - 4);
        }
        return loader;
    }

    /**
     * Get path to the tiled pyramid cache.
     * @param data source image data
     * @param size full image size
     * @param rotate apply EXIF orientation
     * @param alpha image has alpha channel
     * @return cache file path or empty path if cache is not available
     */
    [[nodiscard]] std::filesystem::path pyramid_path(const Data& data,
                                                     const Size& size,
                                                     const bool rotate,
                                                     const bool alpha) const
    {
        const std::string source = data.path.string();
        if (source.empty() || ImageEntry::is_special(source)) {
            return {};
        }

        std::error_code ec;
        std::filesystem::path dir;
        if (!cache_dir.empty()) {
            dir = cache_dir;
        } else {
            const char* xdg = std::getenv("XDG_CACHE_HOME");
            if (xdg && *xdg) {
                dir = std::filesystem::path(xdg) / "swayimg" / "pyramids";
            } else {
                const char* home = std::getenv("HOME");
                if (home && *home) {
                    dir = std::filesystem::path(home) / ".cache" / "swayimg" /
                        "pyramids";
                } else {
                    dir = std::filesystem::temp_directory_path(ec) / "swayimg" /
                        "pyramids";
                    if (ec) {
                        return {};
                    }
                }
            }
        }

        std::filesystem::create_directories(dir, ec);
        if (!std::filesystem::is_directory(dir, ec)) {
            Log::warning("Unable to create cache directory {}", dir.string());
            return {};
        }

        // check free space, the cache is usually smaller than the raw image
        const uint64_t pixels = static_cast<uint64_t>(size.width) * size.height;
        const uint64_t raw = pixels * (alpha ? 4 : 3);
        const uint64_t required = raw / 2 + 256ULL * 1024 * 1024;
        const std::filesystem::space_info space =
            std::filesystem::space(dir, ec);
        if (!ec && space.available < required) {
            Log::warning("Not enough disk space for tiled cache in {}",
                         dir.string());
            return {};
        }

        const uint64_t file_size = std::filesystem::file_size(data.path, ec);
        if (ec) {
            return {};
        }
        const std::filesystem::file_time_type mtime =
            std::filesystem::last_write_time(data.path, ec);
        if (ec) {
            return {};
        }

        const std::string key = std::format(
            "v1|{}|{}|{}|{}|{}", source, file_size,
            mtime.time_since_epoch().count(), rotate ? 1 : 0, alpha ? 1 : 0);
        uint64_t hash = 1469598103934665603ULL;
        for (const char chr : key) {
            hash ^= static_cast<unsigned char>(chr);
            hash *= 1099511628211ULL;
        }

        return dir / std::format("pyramid-{:016x}.tif", hash);
    }
};

// register format in factory
ImageFormatVips format_vips;

} // anonymous namespace
