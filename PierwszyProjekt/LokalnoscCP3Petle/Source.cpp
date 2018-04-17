
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
#define startMessagePattern "Klasyczny algorytm mnozenia macierzy (IKJ) M = %d, liczba watkow %d"

#define USE_MULTIPLE_THREADS true
#define MAXTHREADS 128
int NumThreads;
double start;
double stop;


static const int ROWS = Ncp;     // liczba wierszy macierzy
static const int COLUMNS = ROWS;  // lizba kolumn macierzy
static const double EPSILON = 0.00001;

float matrix_a[ROWS][COLUMNS];    // lewy operand 
float matrix_b[ROWS][COLUMNS];    // prawy operand
float matrix_r[ROWS][COLUMNS];    // wynik
float true_r[ROWS][COLUMNS];    // wynik

								/// zweryfikowanie poprawnosci obliczen wzgledem kodu sekwencyjnego
void verify()
{
	for (int i = 0; i < ROWS; i++)
		for (int j = 0; j < COLUMNS; j++)
			if (true_r[i][j] - matrix_r[i][j] >= EPSILON)
			{
				throw new exception("Bledny wynik");
			}
}

/// zdefiniowanie zawarosci poczatkowej macierzy
void initialize_matrices()
{
#pragma omp parallel 
#pragma omp for nowait 
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			matrix_a[i][j] = (float)rand() / RAND_MAX;
			matrix_b[i][j] = (float)rand() / RAND_MAX;
			matrix_r[i][j] = 0.0;
		}
	}
}

/// rownolegle mnozenie macierzy metoda trzech petli
void parallel_multiply_matrices_IKJ()
{
#pragma omp parallel
	{
		HANDLE thread_handle = GetCurrentThread();
		int th_id = omp_get_thread_num();
		DWORD_PTR mask = (1 << (th_id % liczbaProcesorow));
		DWORD_PTR result = SetThreadAffinityMask(thread_handle, mask);
#pragma omp for
		for (int i = 0; i < ROWS; i++)
			for (int k = 0; k < COLUMNS; k++)
				for (int j = 0; j < COLUMNS; j++)
					matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];
	}
}

/// sekwencyjne mnozenie macierzy metoda trzech petli
void sequentially_multiply_matrices_IKJ()
{
	for (int i = 0; i < ROWS; i++)
		for (int k = 0; k < COLUMNS; k++)
			for (int j = 0; j < COLUMNS; j++)
				true_r[i][j] += matrix_a[i][k] * matrix_b[k][j];

}

void initialize_matricesZ()
{
	//#pragma omp parallel for 
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			matrix_r[i][j] = 0.0;
		}
	}
}

/// wypisanie na konsole i do pliku wynikow czasowych obliczen
void print_elapsed_time(string title, double time, int i, ofstream &file)
{
	double elapsed;
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


int main()
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
	sprintf(buff, startMessagePattern, ROWS, NumThreads);
	string message(buff);

	file << message << endl;
	cout << message << endl;

	initialize_matrices();
	start = (double)clock() / CLK_TCK;
	sequentially_multiply_matrices_IKJ();
	stop = (double)clock() / CLK_TCK;
	print_elapsed_time(seqTag, stop - start, 0, file);

	initialize_matricesZ();
	start = (double)clock() / CLK_TCK;
	parallel_multiply_matrices_IKJ();
	stop = (double)clock() / CLK_TCK;
	print_elapsed_time(parTag, stop - start, 0, file);
	try
	{
		verify();
	}
	catch (exception * e)
	{
		file << e->what() << endl;
		cout << e->what() << endl;
	}



	return 0;
}