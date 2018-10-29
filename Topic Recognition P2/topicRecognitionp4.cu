/** @file topicRecognition.c 
 *
 * @brief Hyper-Dimension computing based topic detector 
 * 
 * @Author Mohammed Aashyk Mohaiteen Hebsur Rahman
 *
 * NOTE: boost library must be installed and included in 
 * the Additional Include Directories of the project 
 * Properties for this code.
 *
 */

#include <iostream>
#include <math.h>
#include <conio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <map>
#include <iterator>
#include <fstream>
#include <streambuf>
#include <string>
#include <dirent.h>
#include <boost/algorithm/string.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <random>


//Algorithmic parameters for size of HD vectors and N-Grams
//
const int HD_VECTOR_SIZE = 10000;
const int N_GRAMS = 3;

/*!
 * @brief Generate Hyper Vectors with random values
 * @return The Randomized Hyper Vector
 */
std::vector<int> genRandomHV()
{
	std::vector<int> 	randomIndex(HD_VECTOR_SIZE);
	std::vector<int> 	randomHV(HD_VECTOR_SIZE);
	std::mt19937 		r{ std::random_device{}() };
	
	if ((HD_VECTOR_SIZE % 2) == 1)
	{
		std::cout << "Dimension is odd";
	}
	else
	{
		for (int i = 0; i < HD_VECTOR_SIZE; i++)
		{
			randomIndex[i] = i;
		}
		std::shuffle(randomIndex.begin(), randomIndex.end(), r);
		for (int i = 0; i < HD_VECTOR_SIZE / 2; i++)
		{
			randomHV[randomIndex[i]] = 1;
		}
		for (int i = HD_VECTOR_SIZE / 2;i < HD_VECTOR_SIZE;i++)
		{
			randomHV[randomIndex[i]] = -1;
		}
	}
	return randomHV;
} /* genRandomHV() */

/*!
 * @brief Create the Item Memory for each alphabet from which the corresponding
 * HV values will be derived
 * @param[in] iM Item Memory
 * @return The Item Memory	
 */
std::map<char, std::vector<int>> createItemMemory(std::map<char, std::vector<int>> iM)
{	
	char alphabet = 'a';
	int counter = 0;
	while ( counter < 26 )
	{
		iM[alphabet] = genRandomHV();
		alphabet = alphabet + 1;
		counter = counter + 1;
	}
	iM[char(32)] = genRandomHV();
	return iM;
} /* createItemMemory() */

/*!
 * @brief Retrieve the Character's corresponding Hyper vector
 * @param[in] iM Item Memory
 * @param[in] key Alphabet whose item memory needs to be retrieved
 * @return The Randomized HyperVector
 */
std::vector<int> lookUpitemMemory(std::map<char, std::vector<int>> iM, char key)
{
	std::vector<int> randomHV(HD_VECTOR_SIZE);
	
	randomHV = iM[key];
	return randomHV;
} /* lookUpitemMemory() */

/*!
 * @brief Finding the Jaccard similarity between Two Hyper vectors
 * @param[in] A Hyper vector of current character in the string
 * @param[in] B Hyper vector of next character in the string
 * @return Similarity
 */
double jaccard_similarity(std::vector<int> A, std::vector<int> B) {
	double 	m[4] = { 0,0,0,0 };
	int 	check;
	
	for (unsigned int i = 0; i < HD_VECTOR_SIZE; i++) {
		check = A[i] + B[i];
		if (check == 2) {
			m[3]++;
		}
		else if (check == 0) {
			if (B[i] == 1) {
				m[1]++;
			}
			else {
				m[2]++;
			}
		}
		else if (check == -2) {
			m[0]++;
		}
	}
	return (m[3] / (m[1] + m[2] + m[3]));
} /* jaccard_similarity() */

/*!
 * @brief Converting the vectors into values +1 and -1 before comparison(jaccard)
 * @param[in] langHV Hyper vector of character in the string
 * @return Binarized Hyper Vector
 */
std::vector<int> binarizeHV(std::vector<int> langHV)
{
	int threshold = 0;

	for (int i = 0; i < HD_VECTOR_SIZE; i++)
	{
		if (langHV[i] > threshold)
		{
			langHV[i] = 1;
		}
		else
		{
			langHV[i] = -1;
		}
	}
	return langHV;
} /* binarizeHV() */

/*!
 * @brief Generates the hyper vector for each input text file
 * @param[in] langHV Hyper vector of character in the string
 * @return The Hyper Vector generated 
 * NOTE: thrust algorithm is used which allocates the memory in the device (i.e, GPU)
 * and performs all operations in the GPU
 */
std::vector<int> computeSumHV(std::map<char, std::vector<int>> iM, size_t bufferSize, std::string  buffer)
{
	thrust::device_vector<int> st_block2(HD_VECTOR_SIZE, 1);
	thrust::device_vector<int> st_block3(HD_VECTOR_SIZE, 1);
	thrust::device_vector<int> st_block4(HD_VECTOR_SIZE, 1);
	thrust::device_vector<int> st_block5(HD_VECTOR_SIZE, 1);
	thrust::device_vector<int> st_block6(HD_VECTOR_SIZE, 1);
	thrust::device_vector<int> st_block7(HD_VECTOR_SIZE, 1);
	thrust::device_vector<int> block0(HD_VECTOR_SIZE, 1);
	thrust::device_vector<int> block1(HD_VECTOR_SIZE, 1);
	thrust::device_vector<int> block2(HD_VECTOR_SIZE, 1);
	thrust::device_vector<int> block3(HD_VECTOR_SIZE, 1);
	thrust::device_vector<int> block4(HD_VECTOR_SIZE, 1);
	thrust::device_vector<int> block5(HD_VECTOR_SIZE, 1);
	thrust::device_vector<int> block6(HD_VECTOR_SIZE, 1);
	thrust::device_vector<int> block7(HD_VECTOR_SIZE, 1);
	thrust::device_vector<int> nGrams(HD_VECTOR_SIZE, 1);
	thrust::device_vector <int> d_sumHV(HD_VECTOR_SIZE, 0);
	std::vector<int> sumHV(HD_VECTOR_SIZE, 0);
	
	if (N_GRAMS == 3)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int 	i = j;
			char 	key = buffer[i];
			#if 0 
			std::cout << key;
			#endif
			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			st_block2 = lookUpitemMemory(iM, key);
			thrust::copy(st_block2.begin(), st_block2.end(), block2.begin());
			if (j >= 2)
			{
				thrust::copy(block2.begin(), block2.end(), nGrams.begin());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());
				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());
			}
		}
	}

	else if (N_GRAMS == 4)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int 	i = j;
			char 	key = buffer[i];

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			thrust::copy(block3.begin() + 1, block3.end(), block2.begin());
			st_block3 = lookUpitemMemory(iM, key);
			thrust::copy(st_block3.begin(), st_block3.end(), block3.begin());
			if (j >= 3)
			{
				thrust::copy(block3.begin(), block3.end(), nGrams.begin());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block2.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());
				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());
			}
		}
	}

	else if (N_GRAMS == 5)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int 	i = j;
			char 	key = buffer[i];
			#if 0
			std::cout << key;
			#endif
			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			thrust::copy(block3.begin() + 1, block3.end(), block2.begin());
			thrust::copy(block4.begin() + 1, block4.end(), block3.begin());
			st_block4 = lookUpitemMemory(iM, key);
			thrust::copy(st_block4.begin(), st_block4.end(), block4.begin());
			if (j >= 4)
			{
				thrust::copy(block4.begin(), block4.end(), nGrams.begin());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block3.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block2.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());
				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());
			}
		}
	}

	else if (N_GRAMS == 6)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int 	i = j;
			char 	key = buffer[i];

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			thrust::copy(block3.begin() + 1, block3.end(), block2.begin());
			thrust::copy(block4.begin() + 1, block4.end(), block3.begin());
			thrust::copy(block5.begin() + 1, block5.end(), block4.begin());
			st_block5 = lookUpitemMemory(iM, key);
			thrust::copy(st_block5.begin(), st_block5.end(), block5.begin());
			if (j >= 5)
			{
				thrust::copy(block5.begin(), block5.end(), nGrams.begin());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block4.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block3.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block2.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());
				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());
			}
		}
	}

	else if (N_GRAMS == 7)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int 	i = j;
			char 	key = buffer[i];

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			thrust::copy(block3.begin() + 1, block3.end(), block2.begin());
			thrust::copy(block4.begin() + 1, block4.end(), block3.begin());
			thrust::copy(block5.begin() + 1, block5.end(), block4.begin());
			thrust::copy(block6.begin() + 1, block6.end(), block5.begin());
			st_block6 = lookUpitemMemory(iM, key);
			thrust::copy(st_block6.begin(), st_block6.end(), block6.begin());
			if (j >= 6)
			{
				thrust::copy(block6.begin(), block6.end(), nGrams.begin());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block5.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block4.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block3.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block2.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());
				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());
			}
		}
	}
	else if (N_GRAMS == 8)
	{
		for (size_t j = 0; j < bufferSize; j++)
		{
			int 	i = j;
			char 	key = buffer[i];

			thrust::copy(block1.begin() + 1, block1.end(), block0.begin());
			thrust::copy(block2.begin() + 1, block2.end(), block1.begin());
			thrust::copy(block3.begin() + 1, block3.end(), block2.begin());
			thrust::copy(block4.begin() + 1, block4.end(), block3.begin());
			thrust::copy(block5.begin() + 1, block5.end(), block4.begin());
			thrust::copy(block6.begin() + 1, block6.end(), block5.begin());
			thrust::copy(block7.begin() + 1, block7.end(), block6.begin());
			st_block7 = lookUpitemMemory(iM, key);
			thrust::copy(st_block7.begin(), st_block7.end(), block7.begin());
			if (j >= 7)
			{
				thrust::copy(block7.begin(), block7.end(), nGrams.begin());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block6.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block5.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block4.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block3.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block2.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block1.begin(), thrust::multiplies<int>());
				thrust::transform(nGrams.begin(), nGrams.end(), nGrams.begin(), block0.begin(), thrust::multiplies<int>());
				thrust::transform(d_sumHV.begin(), d_sumHV.end(), nGrams.begin(), d_sumHV.begin(), thrust::plus<int>());
			}

		}

	}
	thrust::copy(d_sumHV.begin(), d_sumHV.end(), sumHV.begin());
	return sumHV;
} /* computeSumHV() */

/*!
 * @brief Training the Associative Memory From the training Files
 * @param[in] iM item Memory
 * @return The generated Associative memory
 */
std::map<std::string, std::vector<int>> buildLanguage(std::map<char, std::vector<int>> iM)
{
	std::map<std::string, std::vector<int>> 	langAM;
	std::vector<int> 							langHV(HD_VECTOR_SIZE);
	int 										count = 0;
	std::string 								langLabels[64];

	DIR *pdir = NULL;
	pdir = opendir("C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\Topic Recognition P2\\Topic Recognition P2\\Training Files P2"); // "." will refer to the current directory
	struct dirent *pent = NULL;
	if (pdir == NULL) 
	{
		std::cout << "\nERROR! pdir could not be initialised correctly";
		exit(3);
	} 
	while (pent = readdir(pdir))
	{
		if (pent == NULL)
		{
			std::cout << "\nERROR! pent could not be initialised correctly";
			exit(3);
		}
		if (strcmp(pent->d_name, ".") != 0 && strcmp(pent->d_name, "..") != 0)
		{
			std::cout << pent->d_name << std::endl;
			std::string name = pent->d_name;
			langLabels[count] = name.substr(0, 4);
			std::vector<std::string> list{ "C:", "Users", "Mohammed Aashyk", "Documents", "Visual Studio 2015", "Projects", "Topic Recognition P2", "Topic Recognition P2", "Training Files P2", name };
			std::string joined = boost::algorithm::join(list, "\\");
			std::ifstream t(joined);
			std::string str;

			t.seekg(0, std::ios::end);
			size_t size = t.tellg();
			std::string buffer(size, ' ');
			t.seekg(0);
			t.read(&buffer[0], size);
			std::cout << "Training File:" << langLabels[count] << std::endl;
			langHV = computeSumHV(iM, size, buffer);
			langAM[(langLabels[count])] = binarizeHV(langHV);
			count += 1;
		}
	}
	return langAM;
} /* buildLanguage() */

/*!
 * @brief Testing the associative memory by comparing it with testing files
 * @param[in] iM item Memory
 * @return The accuracy of the associative memory
 */
double test(std::map<char, std::vector<int>> iM, std::map<std::string, std::vector<int>> langAM)
{
	double 				total = 0.0;
	double 				correct = 0.0;
	double 				accuracy = 0;
	double 				maxAngle, angle = 0;
	std::string 		predictLang;
	std::vector<int> 	textHV;
	std::string 		langLabels[64];
	std::string			tmp;
	
	for ( int i = 0; i < 63; i++ ){
		if ( i < 8 )
		{
			tmp = "acq";
			tmp += std::to_string(i);
			langLabels[i] =  tmp;
		}
		else if ( i < 16 )
		{
			tmp = "cru";
			tmp += std::to_string(i);
			langLabels[i] = tmp; 
		} 
		else if ( i < 24 )
		{
			tmp = "gra";
			tmp += std::to_string(i);
			langLabels[i] = tmp;
		} 
		else if ( i < 32 )
		{
			tmp = "int";
			tmp += std::to_string(i);
			langLabels[i] = tmp;
		} 
		else if ( i < 40 )
		{
			tmp = "mon";
			tmp += std::to_string(i);
			langLabels[i] = tmp;
		} 
		else if ( i < 48 )
		{
			tmp = "ear";
			tmp += std::to_string(i);
			langLabels[i] = tmp;
		} 
		else if ( i < 56 )
		{
			tmp = "shi";
			tmp += std::to_string(i);
			langLabels[i] = tmp;
		} 
		else if ( i < 64 )
		{
			tmp = "tra";
			tmp += std::to_string(i);
			langLabels[i] = tmp;
		} 
		else
		{
			break;
		}
	}

	DIR *pdir = NULL; 
	pdir = opendir ("C:\\Users\\Mohammed Aashyk\\Documents\\Visual Studio 2015\\Projects\\Topic Recognition P2\\Topic Recognition P2\\testing_texts"); // "." will refer to the current directory
	struct dirent *pent = NULL;

	if (pdir == NULL) 
	{
	    std::cout << "\nERROR! pdir could not be initialised correctly";
	    exit (3);
	} 
	while (pent = readdir (pdir))
	{
	    if (pent == NULL)
	    { 
	        std::cout << "\nERROR! pent could not be initialised correctly";
	        exit (3);
	    }
		if (strcmp(pent->d_name, ".") != 0 && strcmp(pent->d_name, "..") != 0)
		{
			std::cout << pent->d_name << std::endl;
			std::string name = pent->d_name;
			std::string actualLabel = name.substr(0, 3);
			std::vector<std::string> list{ "C:", "Users", "Mohammed Aashyk", "Documents", "Visual Studio 2015", "Projects", "Topic Recognition P2", "Topic Recognition P2", "testing_texts", name };
			std::string joined = boost::algorithm::join(list, "\\");
			std::ifstream t(joined);
			std::string str;
				
			t.seekg(0, std::ios::end);
			size_t size = t.tellg();
			std::string buffer(size, ' ');
			t.seekg(0);
			t.read(&buffer[0], size);

			std::cout << "Loading test file:" << pent->d_name << std::endl;
			textHV = computeSumHV(iM, size, buffer);
			textHV = binarizeHV(textHV);
			maxAngle = -1;
			for (int i = 0; i < 64; i++)
			{
				angle = jaccard_similarity(langAM[langLabels[i]], textHV);
				if (angle > maxAngle)
				{
					maxAngle = angle;
					predictLang = langLabels[i].substr(0, 3);
				}
			}
			if (predictLang == actualLabel)
			{
				correct = correct + 1.0;
			}
			else
			{
				std::cout << predictLang << "  -->  " << actualLabel <<  std::endl;
			}
		}
		total = total + 1.0;
	}
	closedir (pdir);
	accuracy = correct / total * 100;
	return accuracy;
} /* test() */



int main()
{
	std::vector<int> rand;
	std::map<char, std::vector<int>> iM;
	std::map<std::string, std::vector<int>> langAM;
	double correct;
	iM = createItemMemory(iM);										//creates Item Memory to initaiate the program
	langAM = buildLanguage(iM);										//Builds the associative memory from the train files
	correct = test(iM, langAM);										//Compares the test documents with the associative memory
	std::cout << correct << "%" << std::endl << "Run Success!";	
		//Displays Accuracy
}

/*** end of file ***/