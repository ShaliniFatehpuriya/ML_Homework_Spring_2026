#include <iostream>
#include <cmath>
#include<chrono>
#include<random>
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

double polynomialBasisLinear(vector<double> &weightMatrix,int n,double a){    
    double x = 2*(double)rand()/RAND_MAX-1;
    //Value of Noice generated via Gaussian Data generator
    double e=univariateGaussian(0,a);
    double y=0;
    //generating value of Y
    for(int i=0;i<n;i++){
        y +=pow(x,i)*weightMatrix[i];
    }
    y=y+e;
    return y;
}
int sequentialEstimator(double mean,double varience,int prevM,int n,double firstMean,double firstVarience){
    double eps = 0.01;
    double newY=univariateGaussian(firstMean,firstVarience);
    n++;

    double newMean = mean+((newY-mean)/n);
    double M = prevM+((newY-mean)*(newY-newMean));
    double newVareince = M/n;


    cout<<" \nAdd data point: "<<newY;
    cout<<" \nMean: "<<newMean<<"   Varienece: "<<newVareince;

    if(abs(firstMean-newMean) < eps && abs(firstVarience-newVareince) < eps){
        return 0;
    }  
    return sequentialEstimator(newMean,newVareince,M,n,firstMean,firstVarience);
}


int main(int argc, char const *argv[])
{   
    srand(time(NULL));
    double trueMean;
    double trueVariance;
    double noiseVariance;

    cout << "Enter first mean: ";
    cin >> trueMean;

    cout << "Enter first variance: ";
    cin >> trueVariance;

    // cout << "Enter noise variance: ";
    // cin >> noiseVariance;
    vector<double> weightMatrix(5,0);
    for(int i=0;i<weightMatrix.size();i++){
        weightMatrix[i] = (double)rand()/RAND_MAX;
    }
    double polySample = polynomialBasisLinear(weightMatrix, 5, 2);
    sequentialEstimator(0,0,0,0,trueMean,trueVariance);

    return 0;
}
