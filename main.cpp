/*Machine Learning en C++*/
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

using namespace std;
using Eigen::MatrixXf;
using Eigen::VectorXf;
float OLS_Cost(Eigen::MatrixXf X, Eigen::MatrixXf y, Eigen::MatrixXf theta);
tuple<Eigen::VectorXf, vector<float>> GradientDescent(Eigen::MatrixXf X, Eigen::MatrixXf y, Eigen::VectorXf theta, float alpha, int iters);
float RSquared(Eigen::MatrixXf y, Eigen::MatrixXf y_hat);

int main() {

    ifstream vcx;
    ifstream vcy;
    //Obtencion de datos.... 
    MatrixXf xori_train(274, 3), x1_train(274, 3);  // matrizx tipo eigen
    MatrixXf yori(274, 1), y(274, 1); // matrizy

    vcx.open("x.txt", ios::in);
    vcy.open("y.txt", ios::in);
    if (!vcx.eof()) {
        for (int i = 0; i < 274; i++)
            for (int j = 0; j < 3; j++) {
                vcx >> xori_train(i, j);
            }
        for (int k = 0; k < 274; k++) {
            vcy >> yori(k, 0);
        }
    }
    vcy.close();
    vcx.close();

    y = yori;
    x1_train = xori_train;

    //Normalizamos y: y=(y-u)/s
    VectorXf meany = y.colwise().mean();//hallar su promedio
    y.rowwise() -= meany.transpose();
    VectorXf stdy = (y.array().square().colwise().sum() / (y.rows() - 1)).sqrt();//Desviacion estandar de y
    y /= stdy(0);

    //Normalizamos x1_train;
    VectorXf meanx = x1_train.colwise().mean();
    x1_train.rowwise() -= meanx.transpose();
    VectorXf stdx = (x1_train.array().square().colwise().sum() / (x1_train.rows() - 1)).sqrt();
    for (int i = 0; i < x1_train.cols(); i++) {
        x1_train.col(i) /= stdx(i);
    }

    //Agregamos columna de coeficientes independientes
    MatrixXf x0 = MatrixXf::Ones(x1_train.rows(), 1);  // aumentar una columna de 1, multiplicacion de matrices,
    MatrixXf x(x1_train.rows(), x1_train.cols() + 1);//columna de x y columna de 1
    x << x0, x1_train; // Agregamos una columna de 1 a la matriz x1_train

    //Definimos objetos para guardar lo resultante de la gradiente dependiente
    Eigen::VectorXf theta = Eigen::VectorXf::Zero(x.cols());//valores arbitrarios de coeficiente inicial
    float alpha = 0.01;
    int iters = 1000;

    Eigen::VectorXf thetaOut;//coeficientes finales
    vector<float> cost;//funcion de costo por iteracion

    //TRAINING....
    tuple<Eigen::VectorXf, vector<float>> gd = GradientDescent(x, y, theta, alpha, iters);
    tie(thetaOut, cost) = gd;
    cout << "\n********Funci\xA2n costo**********\n";
    for (unsigned int i = 0; i < cost.size(); i++)
    { //con el mètodo .size() se obtiene el tamaño del vector
        cout << cost[i] << endl;
    }
    cout << "***********Par\xA0metros de la funci\xA2n objetivo********"<<endl;
    cout << thetaOut<<endl;

    //Obtenemos datos para el testing
    MatrixXf xori_test(65, 3), x1_test(65, 3);  // matrizx tipo eigen
    MatrixXf yori_test(65, 1), y_test(65, 1); // matrizy
    vcx.open("x_test.txt", ios::in);
    vcy.open("y_test.txt", ios::in);
    if (!vcx.eof()) {
        for (int i = 0; i < 65; i++)
            for (int j = 0; j < 3; j++) {
                vcx >> xori_test(i, j);
            }
        for (int k = 0; k < 65; k++) {
            vcy >> yori_test(k, 0);
        }
    }
    vcy.close();
    vcx.close();

    y_test = yori_test;
    x1_test = xori_test;
    //Normalizamos los datos del test
    VectorXf meanyt = y_test.colwise().mean();//hallar su promedio
    y_test.rowwise() -= meanyt.transpose();
    VectorXf stdyt = (y_test.array().square().colwise().sum() / (y_test.rows() - 1)).sqrt();//Desviacion estandar de y
    y_test /= stdyt(0);

    //Normalizamos x1_test;
    VectorXf meanxt = x1_test.colwise().mean();
    x1_test.rowwise() -= meanxt.transpose();
    VectorXf stdxt = (x1_test.array().square().colwise().sum() / (x1_test.rows() - 1)).sqrt();
    for (int i = 0; i < x1_test.cols(); i++) {
        x1_test.col(i) /= stdxt(i);
    }

    MatrixXf x0_test = MatrixXf::Ones(x1_test.rows(), 1);  // aumentar una columna de 1, multiplicacion de matrices,
    MatrixXf x_test(x1_test.rows(), x1_test.cols() + 1);//columna de x y columna de 1
    x_test << x0_test, x1_test; // Agregamos una columna de 1 a la matriz x1_train

    //TESTING

    Eigen::MatrixXf y_hat = (x_test * thetaOut * stdyt(0));
    y_hat.rowwise() += meanyt.transpose();
    cout << "\n*********Valores predichos vs  Valores Reales********" << endl<<endl;
    for (int i = 0; i < yori_test.rows(); i++) {
        cout << "\t\t"<<(int)y_hat(i) << "\t\t" << yori_test(i) << "\t\t" << endl;
    }
    float r2 = RSquared(yori_test, y_hat);
    cout <<endl<<"Coeficiente de determinaci\xA2n: " << abs(r2)<<endl;
    return 0;

}

float OLS_Cost(Eigen::MatrixXf X, Eigen::MatrixXf y, Eigen::MatrixXf theta) {

    Eigen::MatrixXf inner = pow(((X * theta) - y).array(), 2);

    return (inner.sum() / (2 * X.rows()));
}
tuple<Eigen::VectorXf, vector<float>> GradientDescent(Eigen::MatrixXf X, Eigen::MatrixXf y, Eigen::VectorXf theta, float alpha, int iters) {

    Eigen::MatrixXf temp = theta;

    int parameters = theta.rows();

    vector<float> cost;
    cost.push_back(OLS_Cost(X, y, theta));

    for (int i = 0; i < iters; ++i) {
        Eigen::MatrixXf error = X * theta - y;
        for (int j = 0; j < parameters;j++) {
            Eigen::MatrixXf X_i = X.col(j);
            Eigen::MatrixXf term = error.cwiseProduct(X_i);
            temp(j, 0) = theta(j, 0) - ((alpha / X.rows()) * term.sum());
        }
        theta = temp;
        cost.push_back(OLS_Cost(X, y, theta));
    }
    return make_tuple(theta, cost);
}

float RSquared(Eigen::MatrixXf y, Eigen::MatrixXf y_hat) {
    auto num = pow((y - y_hat).array(), 2).sum();
    auto den = pow(y.array() - y.mean(), 2).sum();

    return 1 - num / den;
}
