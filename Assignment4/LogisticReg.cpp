#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <ctime>
using namespace std;

#define VAL_PI 3.14159265358979323846
// Gaussian generator
double univariateGaussian(double mean,double variance){
    double u1 = ((double)rand()/RAND_MAX)*0.9999 + 1e-6;
    double u2 = ((double)rand()/RAND_MAX)*0.9999 + 1e-6;
    double z = sqrt(-2*log(u1))*cos(2*VAL_PI*u2);
    return z*sqrt(variance) + mean;
}
// Sigmoid
double sigmoid(double z){
    if(z >= 0)
        return 1.0 / (1.0 + exp(-z));
    else{
        double e = exp(z);
        return e / (1.0 + e);
    }
}
// Dot product
double dotProduct(vector<double>& a, vector<double>& b){
    double sum = 0;
    for(int i=0;i<a.size();i++)
        sum += a[i]*b[i];
    return sum;
}
//find the inverse of a matrix with Gauss Jordan method
vector<vector<double>> gaussJordanInverse(vector<vector<double>> &designOriginal){
    vector<vector<double>> design = designOriginal;
    int n = design.size();
    vector<vector<double>> identity (n, vector<double>(n,0)); // all elements are zero
    //made every diagonal element of identity to 1 
    for(int i=0;i<n;i++){
        identity[i][i]=1;
    }
    for(int i=0;i<n;i++){
        double diag = design[i][i];
        for(int j=0;j<n;j++){
            design[i][j] /= diag;
            identity[i][j] /= diag;
        }
        for(int k=0;k<n;k++){
            if(k==i) continue;
            double factor = design[k][i];
            for(int j=0;j<n;j++){
                design[k][j] -= factor*design[i][j];
                identity[k][j] -= factor*identity[i][j];
            }
        }
    } 
    return identity; 
}
//Check if determinant is =0 return false
bool getDeterminant(vector<vector<double>>& A){
    double det =
        A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]) -
        A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) +
        A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);

    if(fabs(det) < 1e-8) return false;
    return true;
}
// Generate data class 0 Y=1 and class 1 Y=0
void generateLogisticData(int N,vector<vector<double>>& X,vector<int>& Y,double mx1,double vx1,double my1,double vy1, double mx2,double vx2,double my2,double vy2){
    
    for(int i=0;i<N;i++){
        double x1 = univariateGaussian(mx1,vx1);
        double x2 = univariateGaussian(my1,vy1);
        X.push_back({1, x1, x2});
        Y.push_back(1);
    }
    for(int i=0;i<N;i++){
        double x1 = univariateGaussian(mx2,vx2);
        double x2 = univariateGaussian(my2,vy2);
        X.push_back({1, x1, x2});
        Y.push_back(0);
    }
}
// Gradient Descent
vector<double> trainGradient(vector<vector<double>>& X,vector<int>& Y,double learningRate,int max_iter){
    vector<double> w(3, 0.0);
    for(int iter=0; iter<max_iter; iter++){
        vector<double> grad(3,0.0);
        for(int i=0;i<X.size();i++){
            double z = dotProduct(w, X[i]);
            double p = sigmoid(z);
            for(int j=0;j<3;j++){
                grad[j] += (Y[i] - p) * X[i][j];
            }
        }
        // normalize gradient (important)
        for(int j=0;j<3;j++) grad[j] /= X.size();
        vector<double> w_new(3);
        for(int j=0;j<3;j++) w_new[j] = w[j] + learningRate * grad[j];

        // convergence
        double diff = 0;
        for(int j=0;j<3;j++) diff += pow(w_new[j] - w[j],2);

        if(sqrt(diff) < 1e-6) break;
        w = w_new;
    }

    return w;
}
//Train Newton method
vector<double> trainNewton(vector<vector<double>>& X,vector<int>& Y,double learningRate,int max_iter){
    vector<double> w(3, 0.0);
    for(int iter=0; iter<max_iter; iter++){
        vector<double> grad(3,0.0);
        vector<vector<double>> hessian(3, vector<double>(3,0.0));

        for(int i=0;i<X.size();i++){
            double z = dotProduct(w, X[i]);
            double p = sigmoid(z);
            //Gradient
            for(int j=0;j<3;j++){
                grad[j] += (Y[i] - p) * X[i][j];
            }
            //Hessian
            //H=−X^T*DX where D = P*(1-P);
            double d=p*(1-p);
            for(int j=0;j<3;j++){
                for(int k=0;k<3;k++){
                    hessian[j][k] += -d*X[i][j]*X[i][k];
                }
            }
        }

        vector<double> w_new(3);
        vector<vector<double>> H_inv;

        //If we can have Hessian
        if(getDeterminant(hessian)) {
            H_inv = gaussJordanInverse(hessian);
            for(int k=0;k<3;k++){
                double sum=0;
                for(int s=0;s<3;s++){
                    sum+=H_inv[k][s]*grad[s];
                }
                w_new[k] = w[k]-sum;
            }
        }else{
            // fallback to gradient descent
            for(int j=0;j<3;j++){
                w_new [j]= w[j]+learningRate*grad[j];
            }
        }
        // convergence
        double diff = 0;
        for(int j=0;j<3;j++) diff += pow(w_new[j] - w[j],2);

        if(sqrt(diff) < 1e-6) break;
        w = w_new;
    }
    return w;
}
// Evaluation
void evaluate(vector<vector<double>>& X,vector<int>& Y,vector<double>& w,string output){
    //Printing weights
    cout<<output<<"\nw\n";
    for(double val : w) cout << val << "\n";
    
    int TruePos=0,TrueNeg=0,FalsePos=0,FalseNeg=0;
    for(int i=0;i<X.size();i++){
        double p = sigmoid(dotProduct(w, X[i]));
        int pred = (p >= 0.5);
        if(pred==1 && Y[i]==1) TruePos++;
        else if(pred==0 && Y[i]==0) TrueNeg++;
        else if(pred==1 && Y[i]==0) FalsePos++;
        else FalseNeg++;
    }
    //Print confusion matrix
    cout<<"Confusion Matrix: "<<"\n"<<"\t    Predict Cluster 1"<<"\tPredict Cluster 2\n";
    cout<<"is cluster 1  "<<"\t"<<TruePos<<"\t\t    "<<FalseNeg<<"\nis cluster 2\t"<<FalsePos<<"\t\t    "<<TrueNeg<<endl;

    double sensitivity = (double)TruePos / (TruePos + FalseNeg);
    double specificity = (double)TrueNeg/ (TrueNeg + FalsePos);
    cout << "Sensitivity (Successfully predict cluster 1): " << sensitivity << endl;
    cout << "Specificity (Successfully predict cluster 2): " << specificity << endl;
}
// Save data
void savePoints(string filename,vector<vector<double>>& X,vector<int>& Y){
    ofstream file(filename);
    for(int i=0;i<X.size();i++){
        file << X[i][1] << " "
             << X[i][2] << " "
             << Y[i] << "\n";
    }
    file.close();
}
// Save boundary
void saveBoundary(string filename, vector<double>& w){
    ofstream file(filename);
    for(double x=-5;x<=5;x+=0.1){
        if(fabs(w[2]) < 1e-6) continue; // avoid division by zero
        double y = -(w[0] + w[1]*x)/w[2];
        file << x << " " << y << "\n";
    }

    file.close();
}

int main(){
    srand(time(NULL));
    vector<vector<double>> X;
    vector<int> Y;
    //Generate data
    int n=50,mx1=1,vx1=2,my1=1,vy1=2,mx2=3,vx2=4,my2=4,vy2=3;
    generateLogisticData(n, X, Y,mx1,vx1,my1,vy1,mx2,vx2,my2,vy2);
    //Train
    vector<double> wGradient = trainGradient(X, Y, 0.01, 20000);
    vector<double> wNewton = trainNewton(X, Y, 0.01, 20000);
    //Evaluate
    evaluate(X, Y, wGradient,"Gradient Descent");
    evaluate(X, Y, wNewton,"Newton's Method");
    //Save for plotting
    savePoints("data.dat", X, Y);
    saveBoundary("GradientMethod.dat", wGradient);
    saveBoundary("NewtonMethod.dat", wNewton);

    return 0;
}