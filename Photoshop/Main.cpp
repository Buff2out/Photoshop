#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>
#include "lodepng/lodepng.h"

using uint = unsigned int;
using byte = unsigned char;
using PixData = std::vector<byte>;

struct Image
{
	uint width = 0;
	uint height = 0;
	PixData data;
};

Image LoadImage(const std::string& filename)
{
	Image image;
	lodepng::decode(image.data, image.width, image.height, filename);
	
	return image;
}

void SaveImage(const Image& image, const std::string& filename)
{
	lodepng::encode("out.png", image.data, image.width, image.height);
}

void InvertFilterSerial(Image& image)
{
	for (size_t i = 0u; i < image.data.size(); i+=4)
	{
		image.data[i] = 255 - image.data[i];
		image.data[i + 1] = 255 - image.data[i + 1];
		image.data[i + 2] = 255 - image.data[i + 2];
	}
}

void InvertFilterOMP(Image& image)
{
#pragma omp parallel for
	for (int64_t i = 0u; i < (int64_t)image.data.size(); i += 4)
	{
		image.data[i] = 255 - image.data[i];
		image.data[i + 1] = 255 - image.data[i + 1];
		image.data[i + 2] = 255 - image.data[i + 2];
	}
}

void InvertFilterSIMD(Image& image)
{
	int rest = image.data.size() % 32;
	size_t i = 0u;
	for (; i < image.data.size() - rest; i += 32)
	{
		__m256i reg = _mm256_loadu_si256((const __m256i*)&image.data[i]);
		__m256i reg_255 = _mm256_set1_epi8(255);

		reg = _mm256_subs_epu8(reg_255, reg);
		__m256i alpha = _mm256_set1_epi32(0xff000000);
		reg = _mm256_or_si256(reg, alpha);
		_mm256_storeu_si256((__m256i*) &image.data[i], reg);
	}

	for (; i < image.data.size(); i += 4)
	{
		image.data[i] = 255 - image.data[i];
		image.data[i + 1] = 255 - image.data[i + 1];
		image.data[i + 2] = 255 - image.data[i + 2];
	}
}

int clamp(int x, int min, int max)
{
	if (x < min)
		x = min;

	if (x > max)
		x = max;

	return x;
}

void MedianFilterSerial(Image& image)
{
	int w = image.width;
	int h = image.height;
	Image copy = image;

	constexpr uint diameter = 15;
	for (int x = 0; x < h; x++)
		for (int y = 0; y < w; y++)
		{
			constexpr uint size = diameter * diameter;
			constexpr int64_t num_channels = 4;

			byte r[size];
			byte g[size];
			byte b[size];

			uint k = 0;
			for (int i = 0; i < diameter; i++)
			{
				for (int j = 0; j < diameter; j++)
				{
					int ri = i - (diameter / 2);
					int rj = j - (diameter / 2);

					int x1 = x + rj;
					int y1 = y + ri;

					x1 = clamp(x1, 0, h - 1);
					y1 = clamp(y1, 0, w - 1);

					int64_t coords = y1 * w * num_channels + x1 * num_channels;
					r[k] = copy.data[coords + 0];
					g[k] = copy.data[coords + 1];
					b[k] = copy.data[coords + 2];
					k++;
				}
			}

			std::sort(r, r + size);
			std::sort(g, g + size);
			std::sort(b, b + size);

			byte med_r = r[size / 2];
			byte med_g = g[size / 2];
			byte med_b = b[size / 2];

			int64_t coords = y * w * num_channels + x * num_channels;
			image.data[coords + 0] = med_r;
			image.data[coords + 1] = med_g;
			image.data[coords + 2] = med_b;
		}
}

void MedianFilterOMP(Image& image)
{
	int w = image.width;
	int h = image.height;
	Image copy = image;

	constexpr uint diameter = 15;

#pragma omp parallel for collapse(2)
	for (int x = 0; x < h; x++)
	{
		for (int y = 0; y < w; y++)
		{
			constexpr uint size = diameter * diameter;
			constexpr int64_t num_channels = 4;

			byte r[size];
			byte g[size];
			byte b[size];

			for (int i = 0; i < diameter; i++)
			{
				for (int j = 0; j < diameter; j++)
				{
					int ri = i - (diameter / 2);
					int rj = j - (diameter / 2);

					int x1 = x + rj;
					int y1 = y + ri;

					x1 = clamp(x1, 0, h - 1);
					y1 = clamp(y1, 0, w - 1);

					int64_t coords = y1 * w * num_channels + x1 * num_channels;

					uint k = i * diameter + j;

					r[k] = copy.data[coords + 0];
					g[k] = copy.data[coords + 1];
					b[k] = copy.data[coords + 2];
				}
			}

			std::sort(r, r + size);
			std::sort(g, g + size);
			std::sort(b, b + size);

			byte med_r = r[size / 2];
			byte med_g = g[size / 2];
			byte med_b = b[size / 2];

			int64_t coords = y * w * num_channels + x * num_channels;
			image.data[coords + 0] = med_r;
			image.data[coords + 1] = med_g;
			image.data[coords + 2] = med_b;
		}
	}
}

int main(int argc, char** argv)
{
	if (argc == 1)
	{
		std::cout << "Usage: Photoshop image <M|I><S|M|G|V>\nimage - path to the image\nfilter to apply: 'M' - median, 'I' - invert\n" << std::endl;
		return 0;
	}

	Image img = LoadImage(argv[1]);
	
	auto t0 = std::chrono::high_resolution_clock::now();

	if (argc >= 3)
	{
		if (argv[2][0] == 'M')
		{
			if (argv[2][1] == 'S')
				MedianFilterSerial(img);
			else if (argv[2][1] == 'M')
				MedianFilterOMP(img);
		}
		else if (argv[2][0] == 'I')
		{
			if (argv[2][1] == 'S')
				InvertFilterSerial(img);
			else if (argv[2][1] == 'M')
				InvertFilterOMP(img);
			else if (argv[2][1] == 'V')
				InvertFilterSIMD(img);
		}
		else
		{
			std::cerr << "Unknown filter" << std::endl;
		}
	}
	else
	{
		InvertFilterSerial(img);
	}

	auto t1 = std::chrono::high_resolution_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << " us" << std::endl;

	SaveImage(img, "out.png");

	return 0;
}