#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
using namespace std;

typedef pair<int, double> IDPair;
struct Compare{
	bool operator()(const IDPair &a, const IDPair &b)
	{
		return a.second < b.second;
	}
};

//user数目
const int USER_SIZE = 943;
//item数目
const int ITEM_SIZE = 1682;
//user最近邻居数
const int USER_NEIGHBOR_SIZE = 10;
//item最近邻居数
const int ITEM_NEIGHBOR_SIZE = 15;
//每个user对每个item的评分表
vector<vector<double> > 
rate_table(USER_SIZE, vector<double>(ITEM_SIZE));
//每个user对所有item评分的均值表
vector<double>
rate_avg_table(USER_SIZE);
//user最近邻居及相似度表
vector<vector<IDPair> > 
user_neighbor_table(USER_SIZE, vector<IDPair>(USER_NEIGHBOR_SIZE));
//item最近邻居及相似度表
vector<vector<IDPair> >
item_neighbor_table(ITEM_SIZE, vector<IDPair>(ITEM_NEIGHBOR_SIZE));

//从文件读入评分矩阵
int input_data(const string &filename)
{
	ifstream file(filename);
	if(file)
	{
		string line, user, item, rate;
		while(getline(file, line))
		{
			istringstream iss(line);
			iss >> user >> item >> rate;
			int user_id = atoi(user.c_str());
			int item_id = atoi(item.c_str());
			double rates= atof(rate.c_str());
			rate_table[user_id-1][item_id-1] = rates;
			line.clear();
		}
		cout << "data input is finish." << endl;
		return 0;
	}
	else
	{
		cerr << "unable to open input file : " << filename << endl;
		return -1;
	}
}

//计算每个用户的平均评分
void calculate_avg_rate()
{
	for(int i = 0; i < USER_SIZE; ++i)
	{
		double sum = 0.0;
		for(int j = 0; j < ITEM_SIZE; ++j)
			sum += rate_table[i][j];
		rate_avg_table[i] = sum / ITEM_SIZE;
	}
	cout << "calculate_avg_rate is finish." << endl;
}

//计算两个向量的皮尔森相关系数
//COV(X, Y) = E(XY) - E(X)E(Y)
//D(X) = E(X^2) - (E(X))^2
double calculate_person_coefficient
(
	const vector<double> &xvec,
	const vector<double> &yvec
)
{
	int len = xvec.size();
	assert(len == yvec.size());
	double sum_xy = 0.0;
	double sum_x = 0.0, sum_y = 0.0;
	double sum_x2 = 0.0, sum_y2 = 0.0;
	for(int i = 0; i < len; ++i)
	{
		sum_x += xvec[i];
		sum_y += yvec[i];
		sum_x2 += xvec[i] * xvec[i];
		sum_y2 += yvec[i] * yvec[i];
		sum_xy += xvec[i] * yvec[i];
	}
	//mean
	double ex  = sum_x  / len;
	double ey  = sum_y  / len;
	double ex2 = sum_x2 / len;
	double ey2 = sum_y2 / len;
	double exy = sum_xy / len;

	//variance
	double dx = sqrt(ex2 - ex * ex);
	double dy = sqrt(ey2 - ey * ey);

	//assert(dx >  0.000001 && dy > 0.000001);
	if(dx <= 0.000001 || dy <= 0.000001)return 0;
	double coff = (exy - ex * ey) / (dx * dy);
	return coff;
}
//计算每个Item的最近邻
void calculate_item_neighbors()
{
	for(int i = 0; i < ITEM_SIZE; ++i)
	{
		vector<double> xvec(USER_SIZE);
		priority_queue<IDPair, vector<IDPair>, Compare> neighbor;
		for(int k = 0; k < USER_SIZE; ++k)
			xvec[k] = rate_table[k][i];
		for(int j = 0; j < ITEM_SIZE; ++j)
		{
			if(i == j)continue;
			vector<double> yvec(USER_SIZE);
			for(int k = 0; k < USER_SIZE; ++k)
				yvec[k] = rate_table[k][j];
			double coff = calculate_person_coefficient(xvec, yvec);
			neighbor.push({j, coff});
		}
		for(int j = 0; j < ITEM_NEIGHBOR_SIZE; ++j)
		{
			item_neighbor_table[i][j] = neighbor.top();
			neighbor.pop();
		}
	}
	cout << "calculate_item_neighbors is finish." << endl;
}

//预测user对未评分的item的评分
double predict_unrate(const vector<double> &user, int index)
{
	double sum_weigh = 0.0, sum_rate = 0.0;
	for(int i = 0; i < ITEM_NEIGHBOR_SIZE; ++i)
	{
		int nb_id = item_neighbor_table[index][i].first;
		double nb_sim = item_neighbor_table[index][i].second;
		sum_rate += nb_sim * user[nb_id];
		sum_weigh += fabs(nb_sim);
	}
	return sum_rate / sum_weigh;
}	

//计算两个用户的相似度
double calculate_user_sim(const vector<double> &user1, const vector<double> &user2)
{
	int len = user1.size();
	assert(len == user2.size());
	vector<double> vec1, vec2;
	for(int i = 0; i < len; ++i)
	{
		if(user1[i] == 0 && user2[i] == 0)continue;
		vec1.push_back(user1[i] != 0 ? user1[i] : predict_unrate(user1, i));
		vec2.push_back(user2[i] != 0 ? user2[i] : predict_unrate(user2, i));
	}
	return calculate_person_coefficient(vec1, vec2);
}

//计算每个user的最近邻
void calculate_user_neighbors()
{
	for(int i = 0; i < USER_SIZE; ++i)
	{
		vector<double> xvec(ITEM_SIZE);
		priority_queue<IDPair, vector<IDPair>, Compare> neighbor;
		for(int k = 0; k < ITEM_SIZE; ++k)
			xvec[k] = rate_table[i][k];
		for(int j = 0; j < USER_SIZE; ++j)
		{
			if(i == j)continue;
			vector<double> yvec(ITEM_SIZE);
			for(int k = 0; k < ITEM_SIZE; ++k)
				yvec[k] = rate_table[j][k];
			double sim = calculate_user_sim(xvec, yvec);
			neighbor.push({j, sim});
		}
		for(int j = 0; j < USER_NEIGHBOR_SIZE; ++j)
		{
			user_neighbor_table[i][j] = neighbor.top();
			neighbor.pop();
		}
	}
	cout << "calculate_user_neighbors is finish." << endl;
}

//产生推荐，预测用户对某项目的评分
double predict_rate(int user, int item)
{
	double sum_weigh = 0.0, sum_rate = 0.0;
	for(int i = 0; i < USER_NEIGHBOR_SIZE; ++i)
	{
		int nb_index = user_neighbor_table[user][i].first;
		double nb_sim = user_neighbor_table[user][i].second;
		sum_rate += nb_sim * (rate_table[nb_index][item] - rate_avg_table[nb_index]);
		sum_weigh += fabs(nb_sim);
	}
	return rate_avg_table[user] + sum_rate / sum_weigh;
}

int main(int args, char *arg[])
{
	string filename("ml-100k/u.data");
	if(input_data(filename) != 0)return -1;

	calculate_avg_rate();
	calculate_item_neighbors();
	calculate_user_neighbors();

	cout << "please input user id nad item id to predict:" << endl;
	int user, item;
	while(cin >> user >> item)
	{
		cout << "user : " << user << " item : " << item;
		cout << " rate : " << predict_rate(user, item) << endl;
	}
	return 0;
}