#include <iostream>
#include <cmath>
#include<chrono>
#include<random>
#include <fstream>
using namespace std;
#define  VAL_PI  3.14159265358979323846

//Box Muller Method for data generation
double univariateGaussian(double mean,double varience){
    double u1,u2;
    u1 = ((double)rand()/RAND_MAX)*0.9999 + 1e-6;
    u2 = ((double)rand()/RAND_MAX)*0.9999 + 1e-6;
    double gaussianData;
    double firstTerm  = (sqrt(-2*log(u1))*cos(2*VAL_PI*u2));
    gaussianData = (firstTerm*sqrt(varience))+mean;
    return gaussianData;
}   
//Polynomial basis with weight matrix
vector<double> polynomialBasisLinear(vector<double> &weightMatrix,int n,double a){    
    double x = 2*(double)rand()/RAND_MAX-1;
    //Value of Noice generated via Gaussian Data generator
    double e=univariateGaussian(0,a);
    double y=0;
    //generating value of Y
    for(int i=0;i<n;i++){
        y +=pow(x,i)*weightMatrix[i];
    }
    y=y+e;
    return vector<double>{x,y};
}
//Adding two matrices
vector<vector<double>> addMatrices(vector<vector<double>> A, vector<vector<double>> B){
    int n =A.size(); //rows
    int m = A[0].size();//columns
    vector<vector<double>> result (n,vector<double>(m,0));
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            result[i][j]=A[i][j]+B[i][j];
        }
    }
    return result;
}
//Get an identity matrix of size M
vector<vector<double>> getIdentityMat(int degree){
    vector<vector<double>> identity (degree,vector<double>(degree,0));
    for(int i=0;i<degree;i++)identity[i][i]=1;
    return identity;
}
//Generate Phi matrix 
vector<double> getPhi(double x,double degree){
    //the phi vector will be of degree N and will have squared elements
    vector<double> phi(degree);
    for(int i=0;i<degree;i++){
        phi[i]=pow(x,i);
    }
    return phi;
}
//multiple two matrices
vector<vector<double>> matrixMultiply(vector<vector<double>> A ,vector<vector<double>> B ){
    int rowOfOne = A.size();
    int colOfTwo = B[0].size();
    int rowOfTwo = B.size();
    //mat created with row size of one and col size of two
    vector<vector<double>> multiple(rowOfOne, vector<double>(colOfTwo,0));

    for(int i=0;i<rowOfOne;i++){
        for(int j=0;j<colOfTwo;j++){
            for(int k=0;k<rowOfTwo;k++){
                multiple[i][j] += A[i][k]*B[k][j];
            }
        }
    }
    return multiple;
}
//Get transpose of a matrix
vector<vector<double>> getTranspose2D(vector<vector<double>> &A ){
    int row = A.size();
    int col = A[0].size();

    vector<vector<double>> T(col, vector<double>(row));

    for(int i=0;i<row;i++)
        for(int j=0;j<col;j++)
            T[j][i] = A[i][j];

    return T;
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
//Get scalar multiplication of a matrix with
vector<vector<double>> scalarMatMul(double digit , vector<vector<double>> matrix){
    for(auto &row:matrix)
        for(auto &v:row)
            v*=digit;
    return matrix;
}
//convert 1*M matrix into M*1
vector<vector<double>> getTranspose1D(vector<double> A){
    vector<vector<double>> result(A.size(),vector<double>(1)) ;
    for(int i=0;i<A.size();i++){
        result[i][0]=A[i];
    }
    return result;
}
//Add a scalar to the matrix
vector<vector<double>> addScalar(vector<vector<double>>matrix,double digit){
    for(int i=0;i<matrix.size();i++){
        for(int j=0;j<matrix[0].size();j++){
            matrix[i][j]+=digit;
        }
    }
    return matrix;
}
//Get Euclidian Norm of a Matrix -->>>the square root of the sum of the squares of all its elements
double checkConverge(vector<vector<double>> A,vector<vector<double>> B ){
    double sum = 0;
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < A[0].size(); j++)
            sum += pow(A[i][j] - B[i][j], 2);
    return sqrt(sum);
}
//Function for storing the seen data
void saveSeen(vector<vector<double>>&xSeen,vector<vector<double>>&ySeen){
    ofstream file("points.dat");
    for(int i=0;i<xSeen.size();i++){
        file<<xSeen[i][0]<<" "<<ySeen[i][0]<<"\n";
    }
    file.close();
}
//Function for storing the curve data
void saveCurve(string filename,vector<vector<double>>m,vector<vector<double>>s,vector<double>&trueWeights,int degree, double noisePrecision,bool groundTruth=false){
    ofstream file(filename);
    for(double x=-1;x<=1;x+=0.01){
        double mean,varience;
        if(groundTruth){
            mean=0;
            for(int i=0;i<trueWeights.size();i++){
                mean+=trueWeights[i]*pow(x,i);
            }
            varience=0.2;
        }else{
            auto phi = getTranspose1D(getPhi(x,degree));
            auto phiTranspose = getTranspose2D(phi);

            auto predMean = matrixMultiply(getTranspose2D(m),phi);
            auto predVar = matrixMultiply(matrixMultiply(phiTranspose,s),phi);
            predVar = addScalar(predVar,1/noisePrecision);
            mean=predMean[0][0];
            varience = predVar[0][0];
        }
        file<<x<<" "<<mean<<" "<<mean+sqrt(varience)<<" "<<mean-sqrt(varience)<<"\n";
    }
    file.close();
}
//The funtion for linear bayesian regression
void linearBayesianRegression(double degree,double priorPrecision,double noisePrecision,vector<double>&weightMatrix,vector<vector<vector<double>>> &graphData)
    {
    //Get the noise mattrix(from 0 and beta^-1*I matrix) -->add that matrix to the Y data generated Matrix-->make it Y matrix
    int iterations=0;
    vector<vector<double>> m = getTranspose1D(vector<double>(degree, 0));
    vector<vector<double>> S = scalarMatMul(1.0 / priorPrecision, getIdentityMat(degree));

    while(true){
        auto data = polynomialBasisLinear(weightMatrix,degree,1/noisePrecision);
        double x=data[0];
        double y=data[1];

        auto phi = getTranspose1D(getPhi(x,degree));
        auto phiT = getTranspose2D(phi);
        //////////////////////////////////////////////////
        auto s_inv = gaussJordanInverse(S);
        vector<vector<double>> SNew_inv = addMatrices(s_inv,scalarMatMul(noisePrecision,matrixMultiply(phi,phiT)));
        auto SNew = gaussJordanInverse(SNew_inv);
        //////////////////////////////////////////////////
        //Posterior Mean
        auto term1 = matrixMultiply(s_inv, m);
        auto term2 = scalarMatMul(noisePrecision*y,phi);
        auto mNew = matrixMultiply(SNew,addMatrices(term1,term2));
        //////////////////////////////////////////////////
        auto predMean = matrixMultiply( getTranspose2D(mNew),phi);
        auto predVar = matrixMultiply( matrixMultiply(phiT, SNew),phi);
        predVar = addScalar(predVar, 1/noisePrecision);
        ///////////////////////////////////////// Printing
        cout << "\nAdd data point (" << x << ", " << y << ")\n";
        cout << "\nPosterior mean\n";
        for (auto &row : mNew)
            cout << row[0] << "\n";
        cout << "\nPosterior covariance\n";
        for (auto &r : SNew)
        {
            for (auto &v : r)
                cout << v << " ";
            cout << "\n";
        }
        cout <<"\nPredictive distribution ~ N("<<predMean[0][0]<<" , "<<predVar[0][0]<<")";
        //----------------------------------
        graphData[0].push_back(vector<double>{x});
        graphData[1].push_back(vector<double>{y});
        
        if(iterations==10){
            graphData[2]=mNew;
            graphData[3]=SNew;
        }else if(iterations==50){
            graphData[4]=mNew;
            graphData[5]=SNew;
        }
        if (checkConverge(m, mNew)< 1e-6){
            //final prediction
            graphData[6]=mNew;
            graphData[7]=SNew;
            break;
        }
            
        m = mNew;
        S = SNew;

        iterations++;
    }
}
int main(int argc, char const *argv[])
{   
    srand(time(NULL));
    double noisePrecision,priorPrecision,degree;
    cout << "Enter noise Precision: ";
    cin >> noisePrecision;
    cout << "Enter Prior Precision: ";
    cin >> priorPrecision;
    cout << "Enter Degree of polynomial: ";
    cin >> degree;
    //weight matrix of size degree+1 
    vector<double> weightMatrix{1,2,3,4};
    //vectors to store the 50/10 iterations and seen data along with Final outcome.
    vector<vector<double>> xSeen; //0
    vector<vector<double>> ySeen; //1
    
    vector<vector<double>> m_10; //2
    vector<vector<double>> s_10; //3
     
    vector<vector<double>> m_50; //4
    vector<vector<double>> s_50; //5

    vector<vector<double>> mFinal; //6
    vector<vector<double>> sFinal; //7

    vector<vector<vector<double>>> graphData {xSeen,ySeen,m_10,s_10,m_50,s_50,mFinal,sFinal};

    linearBayesianRegression(degree,priorPrecision,noisePrecision,weightMatrix,graphData);

    //Save the graph data in files
    saveCurve("ground_truth.dat",graphData[6],graphData[7],weightMatrix,degree,noisePrecision,true);

    saveCurve("predict_10.dat",graphData[2],graphData[3],weightMatrix,degree,noisePrecision);

    saveCurve("predict_50.dat",graphData[4],graphData[5],weightMatrix,degree,noisePrecision);

    saveCurve("predict_final.dat",graphData[6],graphData[7],weightMatrix,degree,noisePrecision);
    saveSeen(graphData[0],graphData[1]);
    return 0;
}
