# Topic-Recognition
C++/Cuda implementation of topic recognition algorithm using hyperdimensional computing. 

It has two major functions:

1. buildLanguage (iM): that is a training function. 
This function returns [iM]. iM is an item memory where hypervectors are stored. 
langAM is a memory where language hypervectors are stored and can be used as an associative memory.


2. test (iM, langAM): that is a test function.
This test function tests unseen sentences and tries to recognizes their languages by querying into langAM.
 
 
Notes:
Boost Library must be installed for this code to function
This program will be albe to run only on PCs which has Nvidia GPUs
Locations for test and train files must be changed inside the code.

Let us know if you face any problem or discover any bugs!


This implementation is based on the work of Prof. Abbas Rahimi. More info can be found on the paper, "A Robust and Energy-Efficient Classifier Using Brain-Inspired Hyperdimensional Computing" at ISLPED 2016.
