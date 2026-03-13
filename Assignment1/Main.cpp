#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
using namespace std;

//Reading and storing the data
void readData(string fileName, vector<double> &x, vector <double>&y){
    ifstream infile(fileName);
    if (!infile.is_open()) { 
        cout << "Error opening file." << endl;
        return;
    }
    string line;
    while (getline(infile, line)) {
        int pos = line.find(',');
        //if no comma was found
        if(pos==string::npos)continue;
        x.push_back(stod(line.substr(0,pos)));
        y.push_back(stod(line.substr(pos+1)));
        //cout << line << endl;
    }
    infile.close();
}
//Create a design matrix of degree X^n
void createDesignMat(vector<double> &x , int degree,vector<vector<double>> &design){
    for(int i=0;i<x.size();i++){
        vector<double> temp;
        for(int j=degree;j>=0;j--){
            temp.push_back(pow(x[i],j));
        }
        design.push_back(temp);
    }
}
//Multiple Two matrices
vector<vector<double>> matrixMultiply(vector<vector<double>> &A ,vector<vector<double>> &B ){
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
//Find transpose of Matrix
void getTranspose(vector<vector<double>> &A , vector<vector<double>> &matrixTranspose){
    int row = A.size();
    int col = A[0].size();
    //get the colm to be the row
    for(int i=0;i<col;i++){
        vector<double> temp;
        for(int j=0;j<row;j++){
            temp.push_back(A[j][i]);
        }
        matrixTranspose.push_back(temp);
    }
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
//adding lambda for regularisation
void addLambda(vector<vector<double>> &design,double lambda){
    for(int i=0;i<design.size();i++){
        design[i][i] += lambda;
    }
}
//Prints the given matrix
void equationPrint(vector<vector<double>> &pos,int degree){
    cout<<"y = ";
    for(int i=0;i<pos.size();i++){
        if(degree==0){cout<<pos[i][0];continue;}
        else cout<<pos[i][0]<<" X^"<<degree--<<" + ";
    }
    cout<<endl;
}

//E(w)=∥Aw−y∥^2+λ∥w1∥
//w=w−η(2AT(Aw−y)+λ(w))
vector<vector<double>> steepestDescent(vector<vector<double>> &A,
    vector<vector<double>> &AT,
    vector<vector<double>> &y,
    vector<vector<double>> &weights,
    double lambda
    )
{
    vector<vector<double>> w = weights;
    double learningRate = 0.00000000003;
    int iterations = 1000;

    for(int it=0; it<iterations; it++)
    {    //Aw
        vector<vector<double>> Aw = matrixMultiply(A,w);
        // Aw - y
        vector<vector<double>> diff(Aw.size(), vector<double>(1));
        for(int i=0;i<Aw.size();i++)
            diff[i][0] = Aw[i][0] - y[i][0];
        // AT(Aw-y)
        vector<vector<double>> grad = matrixMultiply(AT,diff);
        // multiply by 2
        for(int i=0;i<grad.size();i++)
            grad[i][0] *= 2;
        // adding lambda L1 norm
        for(int i=0;i<w.size();i++)
        {
            if(w[i][0] > 0) grad[i][0] += lambda;
            else if(w[i][0] < 0) grad[i][0] -= lambda;
        }
        // update weights
        for(int i=0;i<w.size();i++)
            w[i][0] -= learningRate * grad[i][0];
    }
    return w;
}


int main(int argc, char const *argv[])
{   
    //all variables and DS
    vector<double> x;
    vector<double> y;
    vector<vector<double>> design;
    vector<vector<double>> designTranspose;
    vector<vector<double>> outputMatrixofY;;
    
    //Reading the file and creating data
    readData("data.txt",x,y);
    //Getting transpose, inverse and design Mat
    int n=2 ;//polynomial degree
    createDesignMat(x,n,design);
    for(double val : y){
        outputMatrixofY.push_back({val});
    }
    getTranspose(design,designTranspose);

    //(A^TA+λI)−1A^Ty
    vector<vector<double>> multiple =  matrixMultiply(designTranspose,design);
    addLambda(multiple,0);
    vector<vector<double>> firstTerm =  gaussJordanInverse(multiple);
    vector<vector<double>> secondTerm = matrixMultiply(designTranspose,outputMatrixofY);
    vector<vector<double>> weightLSE = matrixMultiply(firstTerm,secondTerm);
    equationPrint(weightLSE,n);

    vector<vector<double>> weightSteepDescent =
    steepestDescent(design, designTranspose, outputMatrixofY, weightLSE, 0.1);
    equationPrint(weightSteepDescent,n);

    
    //saving the predicted weights of LSE
    ofstream pred("predictedLSE.txt");
    for(double xEle:x){
        double yEle =0;
        for(int i=0;i<weightLSE.size();i++){
            int power =(weightLSE.size()-1)-i;
            yEle += weightLSE[i][0] * pow(xEle, power);
        }
    pred << xEle << " " << yEle << endl;
    }
    pred.close();

    //saving the predicted weights of SD
    ofstream predSD("predictedSD.txt");
    for(double xEle:x){
        double yEle =0;
        for(int i=0;i<weightSteepDescent.size();i++){
            int power =(weightSteepDescent.size()-1)-i;
            yEle += weightSteepDescent[i][0] * pow(xEle, power);
        }
    predSD << xEle << " " << yEle << endl;
    }
    predSD.close();
    return 0;
}