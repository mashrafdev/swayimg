// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Artem Senichev <artemsen@gmail.com>

#include "buildconf.hpp"

#ifdef HAVE_LIBVIPS

#include "formatfactory.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <thread>

namespace {

void set_vips_config(const size_t threshold, const size_t max_size,
                     const bool tiled = true, const size_t tile_threshold = 64,
                     const std::string& cache_dir = {})
{
    ImageFormat* fmt = FormatFactory::self().get("vips");
    ASSERT_NE(fmt, nullptr);

    ImageFormat::Config config;
    config.set("threshold", threshold);
    config.set("max_size", max_size);
    config.set("tiled", tiled);
    config.set("tile_threshold", tile_threshold);
    if (!cache_dir.empty()) {
        config.set("cache_dir", cache_dir);
    }
    fmt->set_config(config);
}

ImagePtr load_image(const char* path)
{
    const ImageEntryPtr entry = std::make_shared<ImageEntry>();
    entry->path = path;
    return FormatFactory::self().load(entry);
}

class VipsTest : public testing::Test {
protected:
    void SetUp() override { set_vips_config(0, 32); }
    void TearDown() override { set_vips_config(64, 4096); }
};

} // anonymous namespace

TEST_F(VipsTest, DecodeLargeImage)
{
    const ImagePtr image = load_image(TEST_DATA_DIR "/image.tiff");
    ASSERT_TRUE(image);
    ASSERT_FALSE(image->frames.empty());
    EXPECT_EQ(image->format.find("VIPS"), 0UL);
    EXPECT_LE(image->frames[0].pm.width(), 32UL);
    EXPECT_LE(image->frames[0].pm.height(), 32UL);
}

TEST_F(VipsTest, PreviewLargeImage)
{
    const ImageEntryPtr entry = std::make_shared<ImageEntry>();
    entry->path = TEST_DATA_DIR "/image.tiff";

    const Pixmap preview = FormatFactory::self().preview(entry, 16, false);
    ASSERT_TRUE(preview);
    EXPECT_LE(preview.width(), 16UL);
    EXPECT_LE(preview.height(), 16UL);
}

TEST_F(VipsTest, SmallImageIgnored)
{
    set_vips_config(1000000, 4096);

    const ImagePtr image = load_image(TEST_DATA_DIR "/image.tiff");
    ASSERT_TRUE(image);
    EXPECT_EQ(image->format.find("VIPS"), std::string::npos);
}

TEST_F(VipsTest, TiledImage)
{
    namespace fs = std::filesystem;

    const fs::path cache_dir = fs::temp_directory_path() / "swayimg-vips-test";
    fs::create_directories(cache_dir);
    const fs::path source = cache_dir / "source.png";
    const fs::path exported = cache_dir / "exported.png";

    constexpr size_t width = 600;
    constexpr size_t height = 600;

    // create a source image with unique pixel values
    Pixmap pm;
    pm.create(Pixmap::ARGB, width, height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            argb_t* pixel = static_cast<argb_t*>(pm.ptr(x, y));
            *pixel =
                argb_t(0xff, static_cast<uint8_t>(x), static_cast<uint8_t>(y),
                       static_cast<uint8_t>(x * 3 + y * 5));
        }
    }
    ASSERT_TRUE(FormatFactory::save(pm, {}, source));

    // force tiled mode for the small test image
    set_vips_config(0, 16, true, 0, cache_dir.string());

    const ImagePtr image = load_image(source.c_str());
    ASSERT_TRUE(image);
    ASSERT_TRUE(image->is_tiled());
    EXPECT_NE(image->format.find("tiled"), std::string::npos);
    EXPECT_EQ(image->size(0).width, width);
    EXPECT_EQ(image->size(0).height, height);
    EXPECT_LT(image->frames[0].pm.width(), width); // overview is downscaled

    // the pyramid is built in the background, wait until it is ready
    Pixmap target;
    target.create(Pixmap::ARGB, width, height);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    bool rendered = false;
    while (!rendered && std::chrono::steady_clock::now() < deadline) {
        image->draw(0, target, 1.0, 0, 0);
        rendered = std::memcmp(pm.ptr(0, 0), target.ptr(0, 0),
                               width * height * sizeof(argb_t)) == 0;
        if (!rendered) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    EXPECT_TRUE(rendered);

    // full resolution export uses the pyramid level 0
    ASSERT_TRUE(image->save(exported, 0));
    EXPECT_TRUE(fs::is_regular_file(exported));
    EXPECT_GT(fs::file_size(exported), 0U);

    fs::remove(source);
    fs::remove(exported);
    for (const auto& entry : fs::directory_iterator(cache_dir)) {
        if (entry.path().extension() == ".tif") {
            fs::remove(entry.path());
        }
    }
}

TEST_F(VipsTest, TiledOverviewCoversImage)
{
    namespace fs = std::filesystem;
    const fs::path cache_dir =
        fs::temp_directory_path() / "swayimg-vips-overview-test";
    fs::create_directories(cache_dir);
    const fs::path source = cache_dir / "source.png";

    constexpr size_t width = 256;
    constexpr size_t height = 192;
    Pixmap src;
    src.create(Pixmap::ARGB, width, height);
    src.fill({ 0, 0, width, height }, argb_t(0xff, 0x40, 0x80, 0xc0));
    ASSERT_TRUE(FormatFactory::save(src, {}, source));

    // force tiled mode with a small overview
    set_vips_config(0, 32, true, 0, cache_dir.string());
    const ImagePtr image = load_image(source.c_str());
    ASSERT_TRUE(image);
    ASSERT_TRUE(image->is_tiled());
    ASSERT_LT(image->frames[0].pm.width(), width);

    // the overview is shown until the pyramid and first region are ready
    Pixmap target;
    target.create(Pixmap::ARGB, width, height);
    const argb_t sentinel(0xff, 0xff, 0x00, 0xff);
    target.fill({ 0, 0, width, height }, sentinel);
    image->draw(0, target, 1.0, 0, 0);

    size_t holes = 0;
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            if (target.at(x, y) == sentinel) {
                ++holes;
            }
        }
    }
    EXPECT_EQ(holes, 0U);

    fs::remove(source);
    for (const auto& entry : fs::directory_iterator(cache_dir)) {
        fs::remove(entry.path());
    }
    fs::remove(cache_dir);
}

TEST_F(VipsTest, TiledPanFillsUncoveredArea)
{
    namespace fs = std::filesystem;
    const fs::path cache_dir =
        fs::temp_directory_path() / "swayimg-vips-pan-test";
    fs::create_directories(cache_dir);
    const fs::path source = cache_dir / "source.png";

    constexpr size_t width = 2048;
    constexpr size_t height = 1536;
    Pixmap src;
    src.create(Pixmap::ARGB, width, height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            auto* pixel = reinterpret_cast<argb_t*>(src.ptr(x, y));
            *pixel = argb_t(0xff, x & 0xff, y & 0xff, (x ^ y) & 0xff);
        }
    }
    ASSERT_TRUE(FormatFactory::save(src, {}, source));

    set_vips_config(0, 128, true, 0, cache_dir.string());
    const ImagePtr image = load_image(source.c_str());
    ASSERT_TRUE(image);
    ASSERT_TRUE(image->is_tiled());

    Pixmap target;
    target.create(Pixmap::ARGB, 400, 300);

    // move the view and wait until the pyramid and region are decoded
    image->draw(0, target, 0.5, -200, -150);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    bool cached = false;
    while (!cached && std::chrono::steady_clock::now() < deadline) {
        for (const auto& entry : fs::directory_iterator(cache_dir)) {
            if (entry.path().extension() == ".tif") {
                cached = true;
                break;
            }
        }
        if (!cached) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    EXPECT_TRUE(cached);
    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    // jump back: the decoded region covers only part of the visible area
    const argb_t sentinel(0x00, 0xff, 0x00, 0xff);
    target.fill({ 0, 0, target.width(), target.height() }, sentinel);
    image->draw(0, target, 0.5, 0, 0);

    size_t holes = 0;
    for (size_t y = 0; y < target.height(); ++y) {
        for (size_t x = 0; x < target.width(); ++x) {
            if (target.at(x, y).a == 0) {
                ++holes;
            }
        }
    }
    EXPECT_EQ(holes, 0U);

    fs::remove(source);
    for (const auto& entry : fs::directory_iterator(cache_dir)) {
        fs::remove(entry.path());
    }
    fs::remove(cache_dir);
}

#endif // HAVE_LIBVIPS
