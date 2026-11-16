// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Artem Senichev <artemsen@gmail.com>

#include "buildconf.hpp"

#ifdef HAVE_LIBVIPS

#include "formatfactory.hpp"

#include <gtest/gtest.h>

namespace {

void set_vips_config(const size_t threshold, const size_t max_size)
{
    ImageFormat* fmt = FormatFactory::self().get("vips");
    ASSERT_NE(fmt, nullptr);

    ImageFormat::Config config;
    config.set("threshold", threshold);
    config.set("max_size", max_size);
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

#endif // HAVE_LIBVIPS
