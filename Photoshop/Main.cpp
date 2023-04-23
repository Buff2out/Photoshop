#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "lodepng/lodepng.h"

using uint = unsigned int;
using byte = unsigned char;
using PixData = std::vector<byte>;

struct Image
{
	uint width;
	uint height;
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

void NegateSerial(Image& image)
{
	for (size_t i = 0u; i < image.data.size(); i+4)
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

	constexpr uint diameter = 7;
	for (int x = 0; x < w; x++)
		for (int y = 0; y < h; y++)
		{
			constexpr uint size = diameter * diameter;
			byte r[size];
			byte g[size];
			byte b[size];

			uint k = 0;
			for (int i = 0; i < diameter; i++)
				for (int j = 0; j < diameter; j++)
				{
					int ri = i - (diameter / 2);
					int rj = j - (diameter / 2);

					int x1 = x + ri;
					int y1 = y + rj;

					x1 = clamp(x1, 0, w - 1);
					y1 = clamp(y1, 0, h - 1);

					int coords = y1 * w * 4 + x1 * 4;
					r[k] = copy.data[coords + 0];
					g[k] = copy.data[coords + 1];
					b[k] = copy.data[coords + 2];
					k++;
				}

			std::sort(r, r + size);
			std::sort(g, g + size);
			std::sort(b, b + size);

			byte med_r = r[size / 2];
			byte med_g = g[size / 2];
			byte med_b = b[size / 2];

			int coords = y * w * 4 + x * 4;
			image.data[coords + 0] = med_r;
			image.data[coords + 1] = med_g;
			image.data[coords + 2] = med_b;
		}
}

int main()
{
	std::cout << "Photoshop by Epsicore" << std::endl;
	Image img = LoadImage("images/300x300.png");
	
	MedianFilterSerial(img);

	SaveImage(img, "out.png");

	return 0;
}