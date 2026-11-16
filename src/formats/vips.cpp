// SPDX-License-Identifier: MIT
// Very large image support via libvips.
// Copyright (C) 2026 Artem Senichev <artemsen@gmail.com>

#include "../formatfactory.hpp"
#include "../imageformat.hpp"
#include "../log.hpp"

#include <vips/vips.h>

#include <format>
#include <memory>
#include <string>

namespace {

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
    }

    // ignore, done by decoder
    void fix_orientation(ImagePtr&, const int) const override {}

private:
    /** Vips image wrapper. */
    struct ImageDeleter {
        void operator()(VipsImage* image) const
        {
            if (image) {
                g_object_unref(image);
            }
        }
    };
    using VipsImagePtr = std::unique_ptr<VipsImage, ImageDeleter>;

    size_t max_size { 4096 }; ///< Max width/height of decoded image
    size_t threshold { 64 };  ///< Source size threshold in megapixels

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
     * Convert vips image to pixmap.
     * @param image source image
     * @return pixmap or empty pixmap on errors
     */
    static Pixmap to_pixmap(VipsImage* image)
    {
        // convert to 8-bit sRGB
        VipsImage* raw_srgb = nullptr;
        if (vips_colourspace(image, &raw_srgb, VIPS_INTERPRETATION_sRGB,
                             nullptr) != 0) {
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
};

// register format in factory
ImageFormatVips format_vips;

} // anonymous namespace
