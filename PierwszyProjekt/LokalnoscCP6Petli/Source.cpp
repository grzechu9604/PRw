#include <stdio.h>
#include <time.h>
#include <windows.h>
#include <iostream>
#include <exception>
#include <string>
#include <fstream>
#include <sstream>
#include "omp.h"

#include "../Constants.h"

using namespace std;

#define threeForsTag "3 petle"
#define sixForsTag "6 petli"
#define seqTag "Seq"
#define parTag "Par"
#define messagePattern "%s: Czas wykonania programu: %8.4f sec (%6.4f sec rozdzielczosc pomiaru)"
#define startMessagePattern "6 petlowy algorytm mnozenia maciezy dla r = %d, liczba watkow %d"

#define USE_MULTIPLE_THREADS true
#define MAXTHREADS 128
int NumThreads;
double start;
double stop;

static const int ROWS = N6cp;     // liczba wierszy macierzy
static const int COLUMNS = ROWS;  // lizba kolumn macierzy
static const double EPSILON = 0.00001;

float matrix_a[ROWS][COLUMNS];    // lewy operand 
float matrix_b[ROWS][COLUMNS];    // prawy operand
float matrix_r[ROWS][COLUMNS];    // wynik
float true_r[ROWS][COLUMNS];    // wynik

/// rownolegle mnozenie maciezy metoda 6 petli
void parallel_multiply_matrices_IKJ_6_fors(int r)
{
#pragma omp parallel
	{
		HANDLE thread_handle = GetCurrentThread();
		int th_id = omp_get_thread_num();
		DWORD_PTR mask = (1 << (th_id % liczbaProcesorow));
		DWORD_PTR result = SetThreadAffinityMask(thread_handle, mask);
#pragma omp for
		for (int i = 0; i < ROWS; i += r)
			for (int j = 0; j < ROWS; j += r)
				for (int k = 0; k < ROWS; k += r) // kolejne fragmenty
					for (int ii = i; ii < i + r && ii < ROWS; ii++)
						for (int kk = k; kk < k + r && kk < ROWS; kk++)
							for (int jj = j; jj < j + r && jj < ROWS; jj++)
								matrix_r[ii][jj] += matrix_a[ii][kk] * matrix_b[kk][jj];

	}
}

/// zdefiniowanie zawarosci poczatkowej macierzy
void initialize_matrices()
{
#pragma omp parallel for 
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			matrix_a[i][j] = (float)rand() / RAND_MAX;
			matrix_b[i][j] = (float)rand() / RAND_MAX;
			matrix_r[i][j] = 0.0;
		}
	}
}

/// wypisanie na konsole i do pliku wynikow czasowych obliczen
void print_elapsed_time(string title, double time, int i, ofstream &file)
{
	double resolution = 1.0 / CLK_TCK;

	if (i > 0)
	{
		title = title.append(" ").append(to_string(i));
	}
	char buff[500];
	sprintf(buff, messagePattern, title.c_str(), time, resolution);
	string message(buff);
	file << message << endl;
	cout << message << endl;
}

int main(int argc, char* argv[])
{
	ofstream file;
	file.open("classic.txt", std::ofstream::out | std::ofstream::app);

	//Determine the number of threads to use
	if (USE_MULTIPLE_THREADS) {
		SYSTEM_INFO SysInfo;
		GetSystemInfo(&SysInfo);
		NumThreads = SysInfo.dwNumberOfProcessors;
		if (NumThreads > MAXTHREADS)
			NumThreads = MAXTHREADS;
	}
	else
		NumThreads = 1;

	char buff[500];
	sprintf(buff, startMessagePattern, Rcp, NumThreads);
	string message(buff);

	file << message << endl;
	cout << message << endl;

	initialize_matrices();
	start = (double)clock() / CLK_TCK;
	parallel_multiply_matrices_IKJ_6_fors(Rcp);
	stop = (double)clock() / CLK_TCK;
	print_elapsed_time(parTag, stop - start, 0, file);
	file.close();

	return(0);
}