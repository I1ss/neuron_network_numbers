#include <cstdlib>
#include <fstream>
#include <random>
#include <time.h>
#include <ctime>
#include <cmath>
#include <windows.h>
#include <string>
#include <iostream>
using namespace std;

double sigmoid(double* weights, double* layer, int size, double bias) {
	double sum = 0;
	for (int i = 0; i < size; i++) {
		sum += weights[i] * layer[i];
	}
	return (1 / (1 + pow(2.71828, -sum+bias)));
}

double softmax(double** weights, double* layer, int size_i, int size_j, int iter, double* bias) {
	double* sum = new double[size_i] {0};
	double sum_arr = 0, answer = 0;
	for (int i = 0; i < size_i; i++) {
		for (int j = 0; j < size_j; j++) {
			sum[i] += weights[i][j] * layer[j];
		}
		sum[i] += bias[i];
		sum_arr += pow(2.71828, sum[i]);
	}
	answer = (pow(2.71828, sum[iter]) / sum_arr);
	delete[] sum;
	return answer;
}

struct nn {
public:
	double* bias1;
	double* bias2;
	double* bias3;
	double** weights12;
	double** weights23;
	double** weights34;
	double* layer1;
	double* layer2;
	double* layer3;
	double* layer4;
	double* error2;
	double* error3;
	double* error4;
	double* delta_weights2;
	double* delta_weights3;
	double* delta_weights4;
	double* expected;
	double learning_rate = 0.015;
	nn() {
		layer1 = new double[2500]{ 0 };
		layer2 = new double[80]{ 0 };
		layer3 = new double[32]{ 0 };
		layer4 = new double[10]{ 0 };
		error2 = new double[80]{ 0 };
		error3 = new double[32]{ 0 };
		error4 = new double[10]{ 0 };
		delta_weights2 = new double[80]{ 0 };
		delta_weights3 = new double[32]{ 0 };
		delta_weights4 = new double[10]{ 0 };
		expected = new double[10]{ 0 };
		bias1 = new double[80];
		for (int i = 0; i < 80; i++) {
			double actual = rand()%21 - 10;
			if (actual == 0) {
				actual += 0.1;
			}
			bias1[i] = actual;
		}
		bias2 = new double[32];
		for (int i = 0; i < 32; i++) {
			double actual = rand() % 21 - 10;
			if (actual == 0) {
				actual += 0.1;
			}
			bias2[i] = actual;
		}
		bias3 = new double[10];
		for (int i = 0; i < 10; i++) {
			double actual = rand() % 21 - 10;
			if (actual == 0) {
				actual += 0.1;
			}
			bias3[i] = actual;
		}
		weights12 = new double* [80];
		for (int i = 0; i < 80; i++) {
			weights12[i] = new double[2500];
		}
		for (int i = 0; i < 80; i++) {
			for (int j = 0; j < 2500; j++) {
				double actual = rand() % 21 - 10;
				if (actual == 0) {
					actual += 0.1;
				}
				weights12[i][j] = actual / 10;
			}
		}
		weights23 = new double* [32];
		for (int i = 0; i < 32; i++) {
			weights23[i] = new double[80];
		}
		for (int i = 0; i < 32; i++) {
			for (int j = 0; j < 80; j++) {
				double actual = rand() % 21 - 10;
				if (actual == 0) {
					actual += 0.1;
				}
				weights23[i][j] = actual / 10;
			}
		}
		weights34 = new double* [10];
		for (int i = 0; i < 10; i++) {
			weights34[i] = new double[32];
		}
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 32; j++) {
				double actual = rand() % 21 - 10;
				if (actual == 0) {
					actual += 0.1;
				}
				weights34[i][j] = actual / 10;
			}
		}
	}
	void set_actual_value(string number) {
		for (int i = 0; i < 10; i++) {
			expected[i] = 0;
		}
		string file = "test";
		file = file + number + ".txt";
		ifstream ifs;
		ifs.open(file);
		if (ifs.is_open() == false) {
			cout << "\n\n\n\n\n\nFILE IS NOT OPEN!\n\n\n\n\n\n";
		}
		for (int i = 0; i < 2500; i++) {
			double temp = 0;
			ifs >> temp;
			layer1[i] = floor(temp);
		}
		for (int i = 0; i < 80; i++) {
			layer2[i] = sigmoid(weights12[i], layer1, 2500, bias1[i]);
		}
		for (int i = 0; i < 32; i++) {
			layer3[i] = sigmoid(weights23[i], layer2, 80, bias2[i]);
		}
		for (int i = 0; i < 10; i++) {
			layer4[i] = softmax(weights34, layer3, 10, 32, i, bias3);
		}
		int check_ = stoi(number);
		if (check_ <= 100) {
			expected[0] = 1;
		}
		else if (check_ > 100 && check_ <= 200) {
			expected[1] = 1;
		}
		else if (check_ > 200 && check_ <= 300) {
			expected[2] = 1;
		}
		else if (check_ > 300 && check_ <= 400) {
			expected[3] = 1;
		}
		else if (check_ > 400 && check_ <= 500) {
			expected[4] = 1;
		}
		else if (check_ > 500 && check_ <= 600) {
			expected[5] = 1;
		}
		else if (check_ > 600 && check_ <= 700) {
			expected[6] = 1;
		}
		else if (check_ > 700 && check_ <= 800) {
			expected[7] = 1;
		}
		else if (check_ > 800 && check_ <= 900) {
			expected[8] = 1;
		}
		else if (check_ > 900 && check_ <= 1000) {
			expected[9] = 1;
		}
	}
	void search_error4() {
		for (int i = 0; i < 10; i++) {
			error4[i] = layer4[i] - expected[i];
		}
	}
	void search_delta_weights4() { // градиентный спуск.
		for (int i = 0; i < 10; i++) {
			delta_weights4[i] = error4[i] * layer4[i] * (1 - layer4[i]);
		}
	}
	void work_with_weights34() {
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 32; j++) {
				weights34[i][j] = weights34[i][j] - learning_rate * delta_weights4[i] * layer3[j];
			}
			bias3[i] -= learning_rate * delta_weights4[i];
		}
	}
	void search_error3() {
		for (int i = 0; i < 32; i++) {
			error3[i] = 0;
		}
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 32; j++) {
				error3[j] += weights34[i][j] * delta_weights4[i];
			}
		}
	}
	void search_delta_weights3() {
		for (int i = 0; i < 32; i++) {
			delta_weights3[i] = error3[i] * layer3[i] * (1 - layer3[i]);
		}
	}
	void work_with_weights23() {
		for (int i = 0; i < 32; i++) {
			for (int j = 0; j < 80; j++) {
				weights23[i][j] = weights23[i][j] - learning_rate * delta_weights3[i] * layer2[j];
			}
			bias2[i] -= learning_rate * delta_weights3[i];
		}
	}
	void search_error2() {
		for (int i = 0; i < 80; i++) {
			error2[i] = 0;
		}
		for (int i = 0; i < 32; i++) {
			for (int j = 0; j < 80; j++) {
				error2[j] += weights23[i][j] * delta_weights3[i];
			}
		}
	}
	void search_delta_weights2() {
		for (int i = 0; i < 80; i++) {
			delta_weights2[i] = error2[i] * layer2[i] * (1 - layer2[i]);
		}
	}
	void work_with_weights12() {
		for (int i = 0; i < 80; i++) {
			for (int j = 0; j < 2500; j++) {
				weights12[i][j] = weights12[i][j] - learning_rate * delta_weights2[i] * layer1[j];
			}
			bias1[i] -= learning_rate * delta_weights2[i];
		}
	}
	void again() {
		for (int i = 0; i < 80; i++) {
			layer2[i] = sigmoid(weights12[i], layer1, 2500, bias1[i]);
		}
		for (int i = 0; i < 32; i++) {
			layer3[i] = sigmoid(weights23[i], layer2, 80, bias2[i]);
		}
		for (int i = 0; i < 10; i++) {
			layer4[i] = softmax(weights34, layer3, 10, 32, i, bias3);
		}
	}
	string get_answer() {
		double answer = layer4[0], answer_iter = 0, temp_ = 0;
		for (int i = 1; i < 10; i++) {
			if (layer4[i] > answer) {
				answer = layer4[i];
				answer_iter = i;
			}
		}
		for (int i = 0; i < 10; i++) {
			temp_ += layer4[i];
			cout.precision(3);
			cout << fixed << layer4[i] << "\n";
		}
		cout << temp_ << "\n";
		if (answer_iter == 0) {
			return "Ноль";
		}
		else if (answer_iter == 1) {
			return "Один";
		}
		else if (answer_iter == 2) {
			return "Два";
		}
		else if (answer_iter == 3) {
			return "Три";
		}
		else if (answer_iter == 4) {
			return "Четыре";
		}
		else if (answer_iter == 5) {
			return "Пять";
		}
		else if (answer_iter == 6) {
			return "Шесть";
		}
		else if (answer_iter == 7) {
			return "Семь";
		}
		else if (answer_iter == 8) {
			return "Восемь";
		}
		else if (answer_iter == 9) {
			return "Девять";
		}
	}
};

int main()
{
	setlocale(LC_ALL, "ru");
	nn test;
	int checker = 0, task = 0;
	for (int i = 0; i < 300000; i++) {
		checker++;
		string temp = to_string(checker);	
		test.set_actual_value(temp);
		test.search_error4();
		test.search_delta_weights4();
		test.work_with_weights34();
		test.search_error3();	
		test.search_delta_weights3();
		test.work_with_weights23();
		test.search_error2();
		test.search_delta_weights2();
		test.work_with_weights12();
		test.again();
		if (checker == 1000) {
			checker = 0;
			task++;
			cout << "Пройден " << task << " круг \n";
		}
	}
	cout << "\n\n\n\nКонец\n\n\n\n";
	test.set_actual_value("1001");
	cout << test.get_answer() << endl;
	test.set_actual_value("1002");
	cout << test.get_answer() << endl;
	test.set_actual_value("1003");
	cout << test.get_answer() << endl;
	test.set_actual_value("1004");
	cout << test.get_answer() << endl;
	test.set_actual_value("1005");
	cout << test.get_answer() << endl;
	test.set_actual_value("1006");
	cout << test.get_answer() << endl;
	test.set_actual_value("1007");
	cout << test.get_answer() << endl;
	test.set_actual_value("1008");
	cout << test.get_answer() << endl;
	test.set_actual_value("1009");
	cout << test.get_answer() << endl;
	test.set_actual_value("1010");
	cout << test.get_answer() << endl;
	return 0;
}