//SSE
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

	int ih_minus_fh = imageHeight - featureHeight;
	int iw_minus_fw = imageWidth - featureWidth;

	/* The two outer for loops iterate through each pixel */
	#pragma omp parallel for collapse(2)
	for (int y = 0; y < imageHeight; y++) {
		for (int x = 0; x < imageWidth; x++) {	
			/* Set the depth to 0 if looking at edge of the image where a feature box cannot fit. */
			if ((y < featureHeight) || (y >= ih_minus_fh) || (x < featureWidth) || (x >= iw_minus_fw)) {
				depth[y * imageWidth + x] = 0;
			}
		}
	}

	#pragma omp parallel for collapse(2)
	for (int y = featureHeight; y < ih_minus_fh; y++)
	{
		for (int x = featureWidth; x < iw_minus_fw; x++)
		{	

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
					float difference = 0.0;
    				int leftRX = 0;
					int rightRX = 0;
					__m128 squaredDifferenceX = _mm_setzero_ps();

					/* Sum the squared difference within a box of +/- featureHeight and +/- featureWidth. */
					for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
					{
						/* No. of roll (y) we are at plus offsets. */
						int left_roll = (y + boxY) * imageWidth - featureWidth + x;
						int right_roll = (y + dy + boxY) * imageWidth - featureWidth + x + dx;

						
						__m128 sumX = _mm_setzero_ps();
						for (int boxX = 0; boxX < featureWidth / 2 * 4; boxX += 4)	//f/2*4 = 2f/4*4
						{
							leftRX = left_roll + boxX;
							rightRX = right_roll + boxX;

							__m128 left0 = _mm_loadu_ps ((left + leftRX));
							__m128 right0 =  _mm_loadu_ps ((right + rightRX));
							__m128 difference = _mm_sub_ps (left0, right0);

							difference = _mm_mul_ps (difference, difference);
							sumX = _mm_add_ps(sumX, difference);
						}
						squaredDifferenceX = _mm_add_ps(sumX, squaredDifferenceX);


					    // tail case
						for (int boxX = featureWidth / 2 * 4; boxX <= featureWidth * 2; boxX++)
						{
							difference = left[left_roll + boxX] - right[right_roll + boxX];
							squaredDifference += difference * difference;
						}

					}
					float temp[4];
					_mm_storeu_ps (temp, squaredDifferenceX);
				    for (int i = 0; i < 4; i++){
				    	squaredDifference += temp[i];
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