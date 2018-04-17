
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
#define seqTag "Seq"
#define messagePattern "Czas przetwarzania %f dla macierzy o rozmiarze %d (rozdzielczoœæ %f)"

double start;
double stop;

static const int ROWS = MaxM;     // liczba wierszy macierzy
static const int COLUMNS = ROWS;  // lizba kolumn macierzy
static const double EPSILON = 0.00001;

float matrix_a[ROWS][COLUMNS];    // lewy operand 
float matrix_b[ROWS][COLUMNS];    // prawy operand
float matrix_r[ROWS][COLUMNS];    // wynik
float true_r[ROWS][COLUMNS];    // wynik

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
/// sekwencyjne mnozenie macierzy metoda trzech petli
void sequentially_multiply_matrices_IKJ(int max)
{
	for (int i = 0; i < max; i++)
		for (int k = 0; k < max; k++)
			for (int j = 0; j < max; j++)
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
void print_elapsed_time(double time, int size, ofstream &file)
{
	double elapsed;
	double resolution = 1.0 / CLK_TCK;

	char buff[500];
	sprintf(buff, messagePattern, time, size, resolution);
	string message(buff);
	file << message << endl;
	cout << message << endl;
}


int main()
{
	ofstream file;
	file.open("classic.txt", std::ofstream::out | std::ofstream::app);

	initialize_matrices();
	start = (double)clock() / CLK_TCK;
	sequentially_multiply_matrices_IKJ(N6b);
	stop = (double)clock() / CLK_TCK;
	print_elapsed_time(stop - start, N6b, file);

	initialize_matrices();
	start = (double)clock() / CLK_TCK;
	sequentially_multiply_matrices_IKJ(N6p);
	stop = (double)clock() / CLK_TCK;
	print_elapsed_time(stop - start, N6p, file);

	initialize_matrices();
	start = (double)clock() / CLK_TCK;
	sequentially_multiply_matrices_IKJ(N6cp);
	stop = (double)clock() / CLK_TCK;
	print_elapsed_time(stop - start, N6cp, file);

	initialize_matrices();
	start = (double)clock() / CLK_TCK;
	sequentially_multiply_matrices_IKJ(Ns);
	stop = (double)clock() / CLK_TCK;
	print_elapsed_time(stop - start, Ns, file);

	initialize_matrices();
	start = (double)clock() / CLK_TCK;
	sequentially_multiply_matrices_IKJ(Ncp);
	stop = (double)clock() / CLK_TCK;
	print_elapsed_time(stop - start, Ncp, file);

	initialize_matrices();
	start = (double)clock() / CLK_TCK;
	sequentially_multiply_matrices_IKJ(Np);
	stop = (double)clock() / CLK_TCK;
	print_elapsed_time(stop - start, Np, file);

	initialize_matrices();
	start = (double)clock() / CLK_TCK;
	sequentially_multiply_matrices_IKJ(Nb);
	stop = (double)clock() / CLK_TCK;
	print_elapsed_time(stop - start, Nb, file);

	return 0;
}