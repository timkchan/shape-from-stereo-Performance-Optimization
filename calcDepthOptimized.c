// CS 61C Fall 2015 Project 4

// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

// include OpenMP
#if !defined(_MSC_VER)
#include <pthread.h>
#endif
#include <omp.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "calcDepthOptimized.h"
#include "calcDepthNaive.h"

/* DO NOT CHANGE ANYTHING ABOVE THIS LINE. */

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{
	/* The two outer for loops iterate through each pixel */
	#pragma omp parallel for
	for (int y = 0; y < imageHeight; y++)
	{
		for (int x = 0; x < imageWidth; x++)
		{	
			/* Set the depth to 0 if looking at edge of the image where a feature box cannot fit. */
			if ((y < featureHeight) || (y >= imageHeight - featureHeight) || (x < featureWidth) || (x >= imageWidth - featureWidth))
			{
				depth[y * imageWidth + x] = 0;
				continue;
			}

			float minimumSquaredDifference = -1;
			int minimumDy = 0;
			int minimumDx = 0;

			/* Iterate through all feature boxes that fit inside the maximum displacement box. 
			   centered around the current pixel. */
			for (int dy = -maximumDisplacement; dy <= maximumDisplacement; dy++)
			{
				for (int dx = -maximumDisplacement; dx <= maximumDisplacement; dx++)
				{
					/* Skip feature boxes that dont fit in the displacement box. */
					if (y + dy - featureHeight < 0 || y + dy + featureHeight >= imageHeight || x + dx - featureWidth < 0 || x + dx + featureWidth >= imageWidth)
					{
						continue;
					}

					float squaredDifference = 0;

					/* Sum the squared difference within a box of +/- featureHeight and +/- featureWidth. */
					//int *sum_pointer = malloc(4*sizeof(int));
    				// __m128i boxX = _mm_setzero_si128();
    				// __m128i boxY = _mm_setzero_si128();
    				// __m128i leftX = _mm_setzero_si128();
    				// __m128i leftY = _mm_setzero_si128();
    				// __m128i rightX = _mm_setzero_si128();
    				// __m128i rightY = _mm_setzero_si128();

    				float difference = 0.0;
					for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
					{
						/* No. of roll (y) we are at. */
						int left_roll = (y + boxY) * imageWidth;
						int right_roll = (y + dy + boxY) * imageWidth;

						int leftX = 0;
						int rightX = 0;
						for (int boxX = 0; boxX < featureWidth / 2 * 4; boxX += 4)	//f/2*4 = 2f/4*4
						{
							leftX = x + boxX - featureWidth;
							rightX = x + dx + boxX - featureWidth;

							difference = left[left_roll + leftX] - right[right_roll + rightX];
							squaredDifference += difference * difference;

							leftX ++;
							rightX ++;

							difference = left[left_roll + leftX] - right[right_roll + rightX];
							squaredDifference += difference * difference;

							leftX ++;
							rightX ++;

							difference = left[left_roll + leftX] - right[right_roll + rightX];
							squaredDifference += difference * difference;

							leftX ++;
							rightX ++;

							difference = left[left_roll + leftX] - right[right_roll + rightX];
							squaredDifference += difference * difference;
						}

					    // tail case
						for (int boxX = featureWidth / 2 * 4; boxX <= featureWidth * 2; boxX++)
						{
							leftX ++;
							rightX ++;

							float difference = left[left_roll + leftX] - right[right_roll + rightX];
							squaredDifference += difference * difference;
						}



					}

					/* 
					Check if you need to update minimum square difference. 
					This is when either it has not been set yet, the current
					squared displacement is equal to the min and but the new
					displacement is less, or the current squared difference
					is less than the min square difference.
					*/
					if ((minimumSquaredDifference == -1) || ((minimumSquaredDifference == squaredDifference) && (displacementNaive(dx, dy) < displacementNaive(minimumDx, minimumDy))) || (minimumSquaredDifference > squaredDifference))
					{
						minimumSquaredDifference = squaredDifference;
						minimumDx = dx;
						minimumDy = dy;
					}
				}
			}

			/* 
			Set the value in the depth map. 
			If max displacement is equal to 0, the depth value is just 0.
			*/
			if (minimumSquaredDifference != -1)
			{
				if (maximumDisplacement == 0)
				{
					depth[y * imageWidth + x] = 0;
				}
				else
				{
					depth[y * imageWidth + x] = displacementNaive(minimumDx, minimumDy);
				}
			}
			else
			{
				depth[y * imageWidth + x] = 0;
			}
		}
	}

}
//test git